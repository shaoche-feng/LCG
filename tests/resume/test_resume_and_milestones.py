"""Step 4 (saving and resume): faithful epoch-boundary resume against an uninterrupted run,
immutable replay versions, atomic saves, and downstream milestones."""
import hashlib
import pickle
import random
import os
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from dm_control import suite
from omegaconf import OmegaConf

import exploration_trainer as trainer_module
from envs import DMControlEnv
from envs.dm_control_env import REWARD_LABEL_TASK_GROUPS
from data.milestone import downstream_output_dir, load_milestone, referenced_files
from exploration_trainer import ExplorationTrainer
from utils import rng_state

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("stand", "walk", "run")

# Real Walker, tiny models, CPU. Episodes are 10 env steps (time limit 0.49 s, action
# repeat 2), so the boundaries at 12, 18, 24 real steps fall inside unfinished episodes.
REAL = [
    "common.seed=11", "common.devices=cpu", "env.train.size=8", "env.train.time_limit=0.49",
    "env.train.task_names=[stand,walk,run]",
    "agent.denoiser.inner_model.num_steps_conditioning=2", "agent.denoiser.inner_model.cond_channels=16",
    "agent.denoiser.inner_model.depths=[1]", "agent.denoiser.inner_model.channels=[8]",
    "agent.denoiser.inner_model.attn_depths=[0]", "agent.actor_critic.channels=[8]", "agent.actor_critic.down=[0]",
    "agent.actor_critic.actor_hidden_dims=[16]", "agent.actor_critic.value_hidden_dims=[16]",
    "world_model_env.horizon=2", "world_model_env.num_batches_to_preload=1",
    "world_model_env.diffusion_sampler.num_steps_denoising=2", "training.compile_wm=false",
    "denoiser.training.batch_size=2", "actor_critic.training.batch_size=2",
    "denoiser.training.steps_first_epoch=2", "denoiser.training.steps_per_epoch=2",
    "actor_critic.training.steps_first_epoch=2", "actor_critic.training.steps_per_epoch=2",
    "intrinsic_reward.theta_s.include=[conv_out.*]", "intrinsic_reward.precision_reference_size=8",
    "intrinsic_reward.precision_num_mc=1", "intrinsic_reward.candidate_num_mc=2", "intrinsic_reward.candidate_chunk_size=4",
    "collection.train.first_epoch.min=12", "collection.train.steps_per_epoch=6", "collection.train.num_steps_total=30",
    "collection.test.num_episodes=1", "collection.test.num_final_episodes=1", "training.num_final_epochs=0",
    "evaluation.every=2", "milestones.at_steps=[12,24]",
]
UNIFORM = ["collection.train.mode=uniform", "intrinsic_reward.enabled=false"]


class Crash(Exception):
    pass


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        return hydra_compose(config_name="exploration", overrides=list(overrides))


def start(run_dir, monkeypatch, overrides, resume=False, resume_changes=None):
    """An ExplorationTrainer on CPU in run_dir; logs are captured as (epoch, row). A resume uses
    the run's saved config with resume_changes ({key: value}) applied, as resume.sh overrides do."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for method in ("init", "save"):
        monkeypatch.setattr(trainer_module.wandb, method, lambda *args, **kwargs: None)
    logs = []
    monkeypatch.setattr(trainer_module, "wandb_log", lambda rows, epoch: logs.extend((epoch, r) for r in rows))
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(run_dir)
    if resume:
        cfg = OmegaConf.load(run_dir / "config/exploration.yaml")
        cfg.common.resume = True
        for key, value in (resume_changes or {}).items():
            OmegaConf.update(cfg, key, value)
    else:
        keys = {o.split("=")[0] for o in overrides}  # a test's own value replaces REAL's
        cfg = compose([o for o in REAL if o.split("=")[0] not in keys] + list(overrides))
        (run_dir / ".hydra").mkdir()
        OmegaConf.save(cfg, run_dir / ".hydra/config.yaml")
    return ExplorationTrainer(cfg, ROOT), logs


def finish(trainer):
    try:
        trainer._run()
    finally:
        trainer._train_env.close()
        trainer._test_collector.close()


def numeric(rows, after_epoch=0):
    out = []
    for epoch, row in rows:
        values = {k: float(v) for k, v in row.items()
                  if k not in ("duration", "progress/train_env_steps")  # the per-row env-step axis
                  and not k.startswith(("resume/", "timing/")) and np.ndim(v) == 0}
        assert all(np.isfinite(v) for v in values.values()), values
        if epoch > after_epoch and values:
            out.append((epoch, values))
    return out


def dataset_contents(dataset):
    contents = []
    for i in range(dataset.num_episodes):
        ep = dataset.load_episode(i)
        contents.append([ep.obs, ep.act, ep.rew, ep.end, ep.trunc, ep.task_rew,
                         ep.info.get("final_observation", torch.zeros(0))])
    return contents


def assert_identical(x, y, where="state"):
    """Exact equality of nested states (tensors, arrays, dicts, lists, scalars)."""
    if torch.is_tensor(x):
        assert torch.is_tensor(y) and x.dtype == y.dtype and torch.equal(x, y), where
    elif isinstance(x, np.ndarray):
        assert np.array_equal(x, y), where
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), where
        for k in x:
            assert_identical(x[k], y[k], f"{where}.{k}")
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), where
        for i, (u, v) in enumerate(zip(x, y)):
            assert_identical(u, v, f"{where}[{i}]")
    else:
        assert x == y, where


def assert_same_run(a, b, logs_a, logs_b, after_epoch):
    """Bit-identical: weights, optimizers, counters, RNG, replay, collector, logs."""
    assert (a.epoch, a.num_epochs_collect) == (b.epoch, b.num_epochs_collect)
    assert a.num_batch_train.state_dict() == b.num_batch_train.state_dict()
    assert_identical(a.agent.state_dict(), b.agent.state_dict(), "agent")  # incl. PMPO's private RNG
    assert_identical(a.opt.state_dict(), b.opt.state_dict(), "optimizers")
    assert_identical(a.lr_sched.state_dict(), b.lr_sched.state_dict(), "schedulers")
    assert_identical(a._final_rng, b._final_rng, "global RNG")
    assert a.train_dataset.lengths.tolist() == b.train_dataset.lengths.tolist()
    for x, y in zip(dataset_contents(a.train_dataset), dataset_contents(b.train_dataset)):
        for u, v in zip(x, y):
            torch.testing.assert_close(v, u, rtol=0, atol=0)
    sa, sb = a._train_collector_state, b._train_collector_state
    assert sa.episode_ids == sb.episode_ids
    torch.testing.assert_close(sb.loop.obs, sa.loop.obs, rtol=0, atol=0)
    torch.testing.assert_close(sb.loop.history, sa.loop.history, rtol=0, atol=0)
    assert numeric(logs_b, after_epoch) == numeric(logs_a, after_epoch)


def run_uninterrupted(run_dir, monkeypatch, overrides):
    trainer, logs = start(run_dir, monkeypatch, overrides)
    finish(trainer)
    trainer._final_rng = rng_state()
    return trainer, logs


def file_hashes(paths):
    return {str(p): hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}


# ---------------------------------------------------------------- learned exploration

def test_learned_resume_matches_uninterrupted_run(tmp_path, monkeypatch):
    full, logs_full = run_uninterrupted(tmp_path / "full", monkeypatch, [])
    assert full.agent.actor_critic.updates.item() == 3 * 2 and full.train_dataset.num_steps == 30  # 3 epochs

    # Crash during epoch 2, after its collection wrote newer replay versions but before
    # its checkpoint: the epoch-1 checkpoint must restore its exact replay prefix.
    original = ExplorationTrainer.train_agent

    def crash_in_epoch_2(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash_in_epoch_2)
    crashed, _ = start(tmp_path / "split", monkeypatch, [])
    with pytest.raises(Crash):
        finish(crashed)
    assert crashed.train_dataset.num_steps == 24  # epoch 2 collected past the checkpoint (18)
    monkeypatch.setattr(ExplorationTrainer, "train_agent", original)

    resumed, logs_resumed = start(tmp_path / "split", monkeypatch, [], resume=True)
    assert resumed.epoch == 1 and resumed.train_dataset.num_steps == 18
    assert resumed._train_collector_state.episode_ids == [1]  # unfinished episode, 8 of 10 steps
    finish(resumed)
    resumed._final_rng = rng_state()
    # The resume is W&B summary metadata, not a history row.
    assert resumed._resume_event == {"from_epoch": 1, "count": 1}
    assert not any(k.startswith("resume/") for _, row in logs_resumed for k in row)
    assert_same_run(full, resumed, logs_full, logs_resumed, after_epoch=1)

    # Milestones: same contents in both runs; the epoch-1 one kept its 8-step version.
    for name in ("steps_000018", "steps_000024"):
        a = load_milestone(tmp_path / "full/milestones" / name, "walk")
        b = load_milestone(tmp_path / "split/milestones" / name, "walk")
        assert a.manifest["identity"] == b.manifest["identity"]
    early = load_milestone(tmp_path / "full/milestones/steps_000018", "walk")
    assert early.dataset.lengths.tolist() == [10, 8] and early.task_index == 1
    final_episode = full.train_dataset.load_episode(1)
    cut = early.dataset.load_episode(1)
    for field in ("obs", "act", "rew", "end", "trunc", "task_rew"):
        torch.testing.assert_close(getattr(cut, field), getattr(final_episode, field)[:8], rtol=0, atol=0)
    torch.testing.assert_close(early.successor_observations[1], final_episode.obs[8], rtol=0, atol=0)
    # Only versions referenced by the checkpoint or a milestone remain on disk.
    monkeypatch.chdir(tmp_path / "full")  # the run's dataset paths are relative to its folder
    on_disk = {p.resolve() for p in (tmp_path / "full/dataset/train").rglob("*.pt") if p.name != "info.pt"}
    referenced = {p.resolve() for p in full.train_dataset.episode_files()}
    for path in (tmp_path / "full/milestones").glob("steps_*"):
        referenced.update(referenced_files(path))
    assert on_disk == referenced and any(p.name == "1_8.pt" for p in on_disk)


def test_extended_budget_continues_exactly_like_a_longer_planned_run(tmp_path, monkeypatch):
    """A finished 30-step run, resumed with collection.train.num_steps_total=42, runs two more
    collection epochs and ends bit-identical to a run planned for 42 steps from the start. The
    milestone added for the extension (36) is published; earlier ones are untouched."""
    milestones = ["milestones.at_steps=[12,24,36]"]
    full, logs_full = run_uninterrupted(tmp_path / "full", monkeypatch, ["collection.train.num_steps_total=42", *milestones])
    assert full.epoch == full.num_epochs_collect == 5 and full.train_dataset.num_steps == 42

    short, _ = run_uninterrupted(tmp_path / "split", monkeypatch, milestones)
    assert short.epoch == short.num_epochs_collect == 3 and short.train_dataset.num_steps == 30
    before = file_hashes(sorted((tmp_path / "split/milestones").rglob("*.json")))
    extended, logs_extended = start(tmp_path / "split", monkeypatch, [], resume=True,
                                    resume_changes={"collection.train.num_steps_total": 42})
    extension = dict(at_epoch=3, from_steps=30, to_steps=42, added_epochs=2)
    assert extended.num_epochs_collect == 5 and extended.budget_extensions == [extension]
    assert extended._resume_event == {"from_epoch": 3, "count": 1, "budget_extension": extension}
    assert not extended.final_test_done  # the short run's final diagnostics ran at its old end (epoch 3)
    finish(extended)
    extended._final_rng = rng_state()
    assert extended.agent.actor_critic.updates.item() == 5 * 2 and extended.num_batch_train.denoiser == 5 * 2
    assert extended.final_test_done
    # Training is identical. Test diagnostics differ only because they draw on their own RNG stream and
    # test replay, which the short run's extra final diagnostics at epoch 3 advanced; the test-step
    # counter includes those extra episodes.
    assert extended.num_test_steps == full.num_test_steps + 10  # one more final test episode of 10 steps
    def training_rows(logs):
        return [(e, {k: v for k, v in row.items() if "test" not in k}) for e, row in logs]
    assert_same_run(full, extended, training_rows(logs_full), training_rows(logs_extended), after_epoch=3)
    assert file_hashes(sorted((tmp_path / "split/milestones").rglob("*.json"))).items() >= before.items()
    for name in ("steps_000024", "steps_000036"):
        a = load_milestone(tmp_path / "full/milestones" / name, "walk")
        b = load_milestone(tmp_path / "split/milestones" / name, "walk")
        assert a.manifest["identity"] == b.manifest["identity"]
    # The extension is checkpointed: a later resume at the new budget changes nothing more.
    again, _ = start(tmp_path / "split", monkeypatch, [], resume=True,
                     resume_changes={"collection.train.num_steps_total": 42})
    assert again.num_epochs_collect == 5 and again.budget_extensions == [extension]
    assert again._resume_event == {"from_epoch": 5, "count": 2}
    again._train_env.close()
    again._test_collector.close()


@pytest.mark.parametrize("total,match", [(24, "must not shrink"), (33, "must be a multiple of steps_per_epoch")])
def test_budget_changes_that_are_not_whole_extensions_are_refused(tmp_path, monkeypatch, total, match):
    run_uninterrupted(tmp_path / "run", monkeypatch, [])
    with pytest.raises(ValueError, match=match):
        start(tmp_path / "run", monkeypatch, [], resume=True, resume_changes={"collection.train.num_steps_total": total})


def test_milestones_are_immutable_movable_and_validated(tmp_path, monkeypatch):
    published = {}
    original = ExplorationTrainer.save_checkpoint

    def record(self):
        original(self)
        path = Path("milestones/steps_000018")
        if self.epoch == 1:  # right after publication
            published.update(file_hashes([*path.iterdir(), *referenced_files(path)]))

    monkeypatch.setattr(ExplorationTrainer, "save_checkpoint", record)
    trainer, _ = run_uninterrupted(tmp_path / "run", monkeypatch, [])
    monkeypatch.setattr(ExplorationTrainer, "save_checkpoint", original)
    # Later collection extended episode 1 and added episodes; the milestone is untouched.
    assert published and file_hashes(published) == published

    monkeypatch.chdir(tmp_path)
    moved = shutil.move(str(tmp_path / "run"), str(tmp_path / "moved"))
    milestone = load_milestone(Path(moved) / "milestones/steps_000018", "run")
    assert milestone.dataset.num_steps == 18 and milestone.task_index == 2
    assert all(Path(moved) in p.resolve().parents for p in milestone.dataset.episode_files())
    assert milestone.manifest["collected_steps"] == 18 and milestone.manifest["run_config"] == "../../config/exploration.yaml"
    assert (milestone.path / milestone.manifest["run_config"]).is_file()
    assert set(milestone.denoiser_state) == set(trainer.agent.denoiser.state_dict())
    with pytest.raises(Exception):
        milestone.dataset.add_episode(milestone.dataset.load_episode(0))  # read-only
    assert downstream_output_dir(milestone.path, "walk", 0) == Path(moved) / "downstream/steps_000018/walk/seed_0"

    with pytest.raises(ValueError, match="no reward labels for task 'jump'"):
        load_milestone(milestone.path, "jump")
    bad = Path(shutil.copytree(moved, tmp_path / "bad"))
    manifest = bad / "milestones/steps_000018/manifest.json"
    manifest.write_text(manifest.read_text(encoding="utf-8").replace('"collected_steps": 18', '"collected_steps": 19', 1),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="manifest says 19"):
        load_milestone(bad / "milestones/steps_000018", "walk")
    (bad / "milestones/steps_000024/denoiser.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="does not match its checksum"):
        load_milestone(bad / "milestones/steps_000024", "walk")
    shutil.rmtree(bad)
    # A same-size change to a task reward must still be caught (content checksum).
    tampered = Path(shutil.copytree(moved, tmp_path / "tampered"))
    episode_file = tampered / "dataset/train/000/00/1/1_8.pt"
    size = episode_file.stat().st_size
    data = torch.load(episode_file)
    data["task_rew"][0, 1] += 1.0
    torch.save(data, episode_file)
    assert episode_file.stat().st_size == size
    with pytest.raises(ValueError, match="1_8.pt does not match its checksum"):
        load_milestone(tampered / "milestones/steps_000018", "walk")
    shutil.rmtree(tampered)
    missing = Path(shutil.copytree(moved, tmp_path / "missing"))
    (missing / "dataset/train/000/00/1/1_8.pt").unlink()
    with pytest.raises(FileNotFoundError, match="missing episode file"):
        load_milestone(missing / "milestones/steps_000018", "walk")
    with pytest.raises(FileNotFoundError, match="No milestone manifest"):
        load_milestone(missing / "milestones/steps_000030", "walk")

    # Republishing the same name: identical state is kept, a different one is rejected.
    monkeypatch.chdir(moved)
    with pytest.raises(ValueError, match="already exists with a different training state"):
        trainer.publish_milestone("steps_000024")


# ---------------------------------------------------------------- uniform baseline, interrupted save

def test_uniform_resume_after_interrupted_save(tmp_path, monkeypatch):
    full, logs_full = run_uninterrupted(tmp_path / "full", monkeypatch, UNIFORM)
    assert full.agent.actor_critic.updates.item() == 0 and full.train_dataset.num_steps == 30

    original = trainer_module.save_atomically

    def interrupted(obj, path):
        if Path(path).name == "state.pt" and obj["epoch"] == 2:
            Path(path).with_suffix(".tmp").write_bytes(b"partial")  # torch.save cut short
            raise Crash()
        return original(obj, path)

    monkeypatch.setattr(trainer_module, "save_atomically", interrupted)
    crashed, _ = start(tmp_path / "split", monkeypatch, UNIFORM)
    with pytest.raises(Crash):
        finish(crashed)
    monkeypatch.setattr(trainer_module, "save_atomically", original)
    assert torch.load(tmp_path / "split/checkpoints/state.pt", weights_only=False)["epoch"] == 1

    resumed, logs_resumed = start(tmp_path / "split", monkeypatch, UNIFORM, resume=True)
    assert resumed.epoch == 1
    finish(resumed)
    resumed._final_rng = rng_state()
    assert_same_run(full, resumed, logs_full, logs_resumed, after_epoch=1)
    assert not list((tmp_path / "split/dataset/train").rglob("*.tmp"))


# ---------------------------------------------------------------- milestone retention

RETENTION = UNIFORM + ["milestones.at_steps=[12,18,24]", "milestones.keep_last=1"]


def crash_in_train_agent(monkeypatch, epoch):
    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == epoch:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    return original


def test_retention_continues_and_resumes_including_interrupted_retention(tmp_path, monkeypatch):
    full, logs_full = run_uninterrupted(tmp_path / "full", monkeypatch, RETENTION)
    # Boundaries at 18 (targets 12 and 18), 24 (target 24), 30: two milestones, one kept.
    assert [(m["name"], m["retired"]) for m in full.milestones] == [("steps_000018", True), ("steps_000024", False)]
    assert sorted(p.name for p in (tmp_path / "full/milestones").iterdir()) == ["steps_000024"]
    monkeypatch.chdir(tmp_path / "full")
    on_disk = {p.resolve() for p in Path("dataset/train").rglob("*.pt") if p.name != "info.pt"}
    assert on_disk == {p.resolve() for p in full.train_dataset.episode_files()} | set(
        referenced_files(Path("milestones/steps_000024")))

    # (a) Resume after retention deleted the older milestone.
    original = crash_in_train_agent(monkeypatch, epoch=3)
    crashed, _ = start(tmp_path / "a", monkeypatch, RETENTION)
    with pytest.raises(Crash):
        finish(crashed)
    monkeypatch.setattr(ExplorationTrainer, "train_agent", original)
    assert not (tmp_path / "a/milestones/steps_000018").exists()
    resumed, logs_resumed = start(tmp_path / "a", monkeypatch, RETENTION, resume=True)
    finish(resumed)
    resumed._final_rng = rng_state()
    assert_same_run(full, resumed, logs_full, logs_resumed, after_epoch=2)
    assert resumed.milestones == full.milestones

    # (b) Interrupted retention: crash after the epoch-2 commit and publication, while
    # deleting the retired milestone. Resume finishes the deletion and continues.
    real_rmtree = shutil.rmtree

    def rmtree(path, *args, **kwargs):
        if "milestones" in str(path):
            raise Crash()
        return real_rmtree(path, *args, **kwargs)

    monkeypatch.setattr(trainer_module.shutil, "rmtree", rmtree)
    crashed, _ = start(tmp_path / "b", monkeypatch, RETENTION)
    with pytest.raises(Crash):
        finish(crashed)
    monkeypatch.setattr(trainer_module.shutil, "rmtree", real_rmtree)
    assert torch.load(tmp_path / "b/checkpoints/state.pt", weights_only=False)["epoch"] == 2
    assert (tmp_path / "b/milestones/steps_000018").exists() and (tmp_path / "b/milestones/steps_000024").exists()
    resumed, logs_resumed = start(tmp_path / "b", monkeypatch, RETENTION, resume=True)
    assert not (tmp_path / "b/milestones/steps_000018").exists()
    finish(resumed)
    resumed._final_rng = rng_state()
    assert_same_run(full, resumed, logs_full, logs_resumed, after_epoch=2)


def test_keep_last_zero_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="keep_last must be null or at least 1"):
        start(tmp_path / "run", monkeypatch, UNIFORM + ["milestones.keep_last=0"])
    assert not (tmp_path / "run/checkpoints").exists()


def test_missing_milestones_republish_or_fail(tmp_path, monkeypatch):
    original = crash_in_train_agent(monkeypatch, epoch=3)
    crashed, _ = start(tmp_path / "run", monkeypatch, UNIFORM)
    with pytest.raises(Crash):
        finish(crashed)
    monkeypatch.setattr(ExplorationTrainer, "train_agent", original)
    milestones = tmp_path / "run/milestones"
    identity = load_milestone(milestones / "steps_000024", "walk").manifest["identity"]

    # The checkpoint's own milestone (a crash before publication) is rebuilt identically.
    shutil.rmtree(milestones / "steps_000024")
    resumed, _ = start(tmp_path / "run", monkeypatch, UNIFORM, resume=True)
    resumed._train_env.close()
    assert load_milestone(milestones / "steps_000024", "walk").manifest["identity"] == identity

    # An older milestone that was not retired cannot be rebuilt.
    shutil.rmtree(milestones / "steps_000018")
    with pytest.raises(RuntimeError, match="steps_000018 is missing but was not retired"):
        start(tmp_path / "run", monkeypatch, UNIFORM, resume=True)


# ---------------------------------------------------------------- incompatible restoration

@pytest.mark.parametrize("change,match", [
    (lambda s: s.pop("resume_format"), "resume state of format 3"),
    (lambda s: s["source"]["packages"].update(mujoco="0.0"), "package versions changed"),
    (lambda s: s["collection"]["env_config"].update(action_repeat=3), "different env.train config"),
    (lambda s: s["collection"].update(episode_ids=[0]), "not an unfinished episode"),
])
def test_incompatible_checkpoints_are_rejected(tmp_path, monkeypatch, change, match):
    original = ExplorationTrainer.train_agent

    def stop_in_epoch_2(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", stop_in_epoch_2)
    trainer, _ = start(tmp_path / "run", monkeypatch, UNIFORM)
    with pytest.raises(Crash):
        finish(trainer)
    path = tmp_path / "run/checkpoints/state.pt"
    state = torch.load(path, weights_only=False)
    change(state)
    torch.save(state, path)
    with pytest.raises(ValueError, match=match):
        resumed, _ = start(tmp_path / "run", monkeypatch, UNIFORM, resume=True)
        resumed._train_env.close()


# ---------------------------------------------------------------- episode files

def test_saved_frames_round_trip_exactly(tmp_path):
    from data import Episode
    levels = torch.arange(256, dtype=torch.uint8).reshape(1, 1, 16, 16).repeat(3, 3, 1, 1)
    obs = levels.float().div(255).mul(2).sub(1)  # as TorchEnv converts frames
    episode = Episode(obs, torch.zeros(3, 2), torch.zeros(3), torch.zeros(3, dtype=torch.long),
                      torch.zeros(3, dtype=torch.long), {})
    for i in range(3):  # repeated save/load, as when an unfinished episode is extended
        episode.save(tmp_path / f"{i}.pt")
        loaded = Episode.load(tmp_path / f"{i}.pt")
        torch.testing.assert_close(loaded.obs, obs, rtol=0, atol=0)
        episode = loaded


# ---------------------------------------------------------------- env state

def short_time_limit(domain, task):
    return 7.5 * suite.load(domain, task).control_timestep()


@pytest.mark.parametrize("domain,group", [(d, g) for d, gs in REWARD_LABEL_TASK_GROUPS.items() for g in gs],
                         ids=lambda x: x if isinstance(x, str) else "-".join(x))
def test_env_state_restores_exact_continuation(domain, group):
    time_limit = short_time_limit(domain, group[0])  # 8 control steps: 4 env steps per episode
    rng = np.random.default_rng(0)
    source = DMControlEnv(domain, list(group), size=16, action_repeat=2, time_limit=time_limit, seed=3)
    source.reset()
    low, high = source.action_low, source.action_high
    for _ in range(2):  # mid-episode
        source.step(rng.uniform(low, high).astype(np.float32))
    saved = pickle.loads(pickle.dumps(source.collection_state))  # as stored in a checkpoint
    target = DMControlEnv(domain, list(group), size=16, action_repeat=2, time_limit=time_limit, seed=99)
    target.reset()
    target.collection_state = saved
    for step in range(6):  # finish the episode, reset from the task RNG, continue
        action = rng.uniform(low, high).astype(np.float32)
        out_a, out_b = source.step(action), target.step(action)
        np.testing.assert_array_equal(out_a[0], out_b[0])
        assert out_a[1:4] == out_b[1:4]
        np.testing.assert_array_equal(out_a[4]["task_rewards"], out_b[4]["task_rewards"])
        if out_a[3]:
            np.testing.assert_array_equal(source.reset()[0], target.reset()[0])
        for name in ("qpos", "qvel", "act", "qacc_warmstart"):
            np.testing.assert_array_equal(getattr(source._dm_env.physics.data, name),
                                          getattr(target._dm_env.physics.data, name))


def test_env_state_rejects_a_different_model():
    walker = DMControlEnv("walker", ["walk"], size=8)
    walker.reset()
    hopper = DMControlEnv("hopper", ["hop"], size=8)
    with pytest.raises(ValueError, match="different MuJoCo model"):
        hopper.collection_state = walker.collection_state


@pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")
@pytest.mark.parametrize("entry, config", [("src/exploration_main.py", "config/exploration.yaml"),
                                           ("src/main.py", "config/trainer.yaml")])  # before the 2026-10-30 rename
def test_resume_script_runs_the_run_folders_own_entry_point_and_config(tmp_path, entry, config):
    """scripts/resume.sh with the run folder's copied entry point (a stub recording its arguments here)."""
    run = tmp_path / "run"
    for path in (entry, config, "checkpoints/state.pt", "dataset/train/info.pt"):
        (run / path).parent.mkdir(parents=True, exist_ok=True)
    (run / entry).write_text("import pathlib, sys\npathlib.Path('argv.txt').write_text(' '.join(sys.argv))\n")
    OmegaConf.save(OmegaConf.create(dict(common=dict(devices="cpu"), agent={}, env={}, static_dataset=dict(path=None))),
                   run / config)
    (run / "checkpoints/state.pt").write_bytes(b"x")
    (run / "dataset/train/info.pt").write_bytes(b"x")
    env = {k: v for k, v in os.environ.items() if k not in ("CUDA_VISIBLE_DEVICES", "MUJOCO_EGL_DEVICE_ID")}
    env["PYTHON"] = Path(sys.executable).as_posix()
    bash = shutil.which("bash")  # PATH's bash (on Windows a bare "bash" can resolve to WSL's in System32)
    result = subprocess.run([bash, (ROOT / "scripts/resume.sh").as_posix(), run.as_posix(), "pmpo_diagnostic.max_epochs=3"],
                            env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    argv = (run / "argv.txt").read_text()
    assert argv.split()[0] == entry and "common.resume=true" in argv and "hydra.run.dir=." in argv
    assert argv.split()[-1] == "pmpo_diagnostic.max_epochs=3" and f"command:       {env['PYTHON']} {entry}" in result.stdout
    (run / entry).unlink()
    result = subprocess.run([bash, (ROOT / "scripts/resume.sh").as_posix(), run.as_posix()], env=env, capture_output=True,
                            text=True)
    assert result.returncode != 0 and f"{entry} is missing" in result.stderr
