"""Step 4 (saving and resume): faithful epoch-boundary resume against an uninterrupted run,
immutable replay versions, atomic saves, and downstream milestones."""
import hashlib
import pickle
import random
import shutil
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
        return hydra_compose(config_name="trainer", overrides=list(overrides))


def start(run_dir, monkeypatch, overrides, resume=False):
    """An ExplorationTrainer on CPU in run_dir; logs are captured as (epoch, row)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    for method in ("init", "save"):
        monkeypatch.setattr(trainer_module.wandb, method, lambda *args, **kwargs: None)
    logs = []
    monkeypatch.setattr(trainer_module, "wandb_log", lambda rows, epoch: logs.extend((epoch, r) for r in rows))
    run_dir.mkdir(parents=True, exist_ok=True)
    monkeypatch.chdir(run_dir)
    if resume:
        cfg = OmegaConf.load(run_dir / "config/trainer.yaml")
        cfg.common.resume = True
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
        # NaN compares unequal to itself; the tiny test episodes give a NaN test loss.
        values = {k: "nan" if np.isnan(float(v)) else float(v) for k, v in row.items()
                  if k != "duration" and not k.startswith("resume/") and np.ndim(v) == 0}
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
    assert logs_resumed[0] == (2, {"resume/from_epoch": 1})
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
    assert milestone.manifest["collected_steps"] == 18 and milestone.manifest["run_config"] == "../../config/trainer.yaml"
    assert (milestone.path / milestone.manifest["run_config"]).is_file()
    assert set(milestone.denoiser_state) == set(trainer.agent.denoiser.state_dict())
    with pytest.raises(Exception):
        milestone.dataset.add_episode(milestone.dataset.load_episode(0))  # read-only
    assert downstream_output_dir(milestone.path, "walk", 0) == Path(moved) / "downstream/steps_000018/walk/seed_0"

    with pytest.raises(ValueError, match="no reward labels for task 'jump'"):
        load_milestone(milestone.path, "jump")
    bad = Path(shutil.copytree(moved, tmp_path / "bad"))
    with open(bad / "milestones/steps_000018/manifest.yaml", "a", encoding="utf-8") as f:
        f.write("collected_steps: 19\n")
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
    (lambda s: s.pop("resume_format"), "saved before Step 4"),
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
