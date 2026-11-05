"""Step 5 (first increment): downstream reward-model and PMPO training on a frozen milestone."""
import csv
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
import yaml
from hydra.utils import instantiate
from omegaconf import OmegaConf

import downstream_analysis as analysis_module
import downstream_main
import downstream_trainer as downstream_module
from coroutines.env_loop import make_env_loop
from data import Dataset, Episode
from data.milestone import discover_milestones, load_milestone, publish_milestone, read_manifest, write_json
from utils import file_sha256
from data.segment import SegmentId
from data.utils import make_task_reward_collate
from downstream_trainer import DownstreamTrainer
from models.diffusion import Denoiser

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("stand", "walk", "run")
TINY_DENOISER = [
    "env.train.size=8", "env.train.task_names=[stand,walk,run]", "env.train.domain_name=walker",
    "agent.denoiser.inner_model.num_steps_conditioning=2", "agent.denoiser.inner_model.cond_channels=16",
    "agent.denoiser.inner_model.depths=[1]", "agent.denoiser.inner_model.channels=[8]",
    "agent.denoiser.inner_model.attn_depths=[0]",
]
TINY_DOWNSTREAM = [
    "agent.rew_end_model.lstm_dim=16", "agent.rew_end_model.cond_channels=16", "agent.rew_end_model.depths=[1]",
    "agent.rew_end_model.channels=[8]", "agent.rew_end_model.attn_depths=[0]", "agent.actor_critic.channels=[8]",
    "agent.actor_critic.down=[0]", "agent.actor_critic.actor_hidden_dims=[16]",
    "agent.actor_critic.value_hidden_dims=[16]", "world_model_env.horizon=2", "world_model_env.num_batches_to_preload=1",
    "world_model_env.diffusion_sampler.num_steps_denoising=2", "rew_end_model.training.steps=4",
    "rew_end_model.training.batch_size=3", "rew_end_model.training.lr_warmup_steps=0",
    "actor_critic.training.steps=3", "actor_critic.training.batch_size=2", "evaluation.after_training=false",
]


def compose(config_name, overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        return hydra_compose(config_name=config_name, overrides=list(overrides))


def frames(n, generator):
    levels = torch.randint(0, 256, (n, 3, 8, 8), generator=generator)
    return levels.float().div(255).mul(2).sub(1)  # on the uint8 grid, as TorchEnv produces


def build_milestone(root, column_offsets=(0.0, 0.0, 0.0), early=()):
    """A labeled walker milestone: two finished (truncated) episodes and one unfinished
    episode at the cut. column_offsets shift each task's reward column. early: also publish
    milestones of the same run after that many finished episodes (1: steps_000010, 2: steps_000020)."""
    g = torch.Generator().manual_seed(0)
    run = root / "run"
    cfg = compose("exploration", TINY_DENOISER)
    denoiser_cfg = instantiate(cfg.agent.denoiser)
    denoiser_cfg.inner_model.continuous_action_dim = 6
    torch.manual_seed(1)
    denoiser = Denoiser(denoiser_cfg)

    def publish(dataset, successors, epoch):
        steps = dataset.num_steps
        manifest = dict(created_by="test", domain="walker", task_names=list(TASKS), collected_steps=steps,
                        num_episodes=dataset.num_episodes, epoch=epoch, counters=dict(denoiser_batches=5, pmpo_updates=0),
                        exploration=dict(collection_mode="uniform", intrinsic_reward_enabled=False),
                        model=dict(denoiser=OmegaConf.to_container(cfg.agent.denoiser, resolve=True),
                                   env=OmegaConf.to_container(cfg.env.train, resolve=True)),
                        run_config="../../config/exploration.yaml", source=dict(git_commit=None))
        return publish_milestone(run / "milestones", f"steps_{steps:06d}", denoiser.state_dict(), dataset, successors, manifest)

    dataset = Dataset(run / "dataset/train", "train_dataset")
    for k, (length, finished) in enumerate(((10, True), (10, True), (6, False))):
        if k in early:
            publish(dataset, {}, epoch=k)
        obs, act = frames(length, g), torch.rand(length, 6, generator=g) * 2 - 1
        base = torch.rand(length, generator=g)
        task_rew = torch.stack([base * (i + 1) + column_offsets[i] for i in range(3)], 1)
        trunc = torch.zeros(length, dtype=torch.long)
        trunc[-1] = int(finished)
        info = {"final_observation": frames(1, g)[0]} if finished else {}
        dataset.add_episode(Episode(obs, act, task_rew[:, 0].clone(), torch.zeros(length, dtype=torch.long), trunc,
                                    info, task_rew=task_rew, task_names=TASKS))
    successor = frames(1, g)[0]
    return publish(dataset, {2: successor}, epoch=2)


def make_trainer(milestone, monkeypatch, task="walk", extra=()):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = compose("downstream", [f"initialization.milestone_path={Path(milestone).as_posix()}",
                                 f"downstream.task_name={task}", *TINY_DOWNSTREAM, *extra])
    return DownstreamTrainer(cfg)


def tree_hashes(*roots):
    return {str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for root in roots for p in sorted(Path(root).rglob("*")) if p.is_file()}


def states_equal(a, b):
    return all(torch.equal(a[k], b[k]) for k in a if torch.is_tensor(a[k]))


def tensor_state(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items() if torch.is_tensor(v)}


# ---------------------------------------------------------------- loading and outputs

def test_exact_milestone_is_loaded_and_bad_inputs_rejected(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    trainer = make_trainer(milestone, monkeypatch)
    assert trainer.milestone.path == milestone and trainer.milestone.task_index == 1
    assert trainer.milestone.dataset.lengths.tolist() == [10, 10, 6]
    saved = torch.load(milestone / "denoiser.pt")
    assert states_equal(saved, trainer.agent.denoiser.state_dict())
    assert trainer.cfg.env.train.size == 8 and trainer.cfg.agent.denoiser.inner_model.channels == [8]
    with pytest.raises(ValueError, match="no reward labels for task 'jump'"):
        make_trainer(milestone, monkeypatch, task="jump")
    with pytest.raises(ValueError, match="rew_end_model.training.steps must be a positive integer"):
        make_trainer(milestone, monkeypatch, extra=["rew_end_model.training.steps=0"])
    with pytest.raises(FileNotFoundError, match="No milestone manifest"):
        make_trainer(tmp_path / "run/milestones/steps_999999", monkeypatch)


def test_outputs_are_separate_complete_and_never_overwritten(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    before = tree_hashes(tmp_path / "run/milestones", tmp_path / "run/dataset")
    trainer = make_trainer(milestone, monkeypatch, extra=["downstream.seed=3"])
    trainer.run()
    out = tmp_path / "run/downstream/steps_000026/walk/seed_3"
    assert trainer.output_dir == out and not (out.parent / ".seed_3.tmp").exists()
    assert sorted(p.name for p in out.iterdir()) == ["actor_critic.pt", "config.yaml", "metrics.jsonl",
                                                     "rew_end_model.pt", "summary.json"]
    summary = OmegaConf.load(out / "summary.json")
    assert summary.task_name == "walk" and summary.task_index == 1 and summary.seed == 3
    assert summary.num_updates == {"rew_end_model": 4, "actor_critic": 3}
    assert summary.milestone.path == "../../../../milestones/steps_000026"
    assert (out / summary.milestone.path).resolve() == milestone.resolve()
    assert OmegaConf.load(out / "config.yaml").downstream.seed == 3
    assert tree_hashes(tmp_path / "run/milestones", tmp_path / "run/dataset") == before  # sources untouched
    with pytest.raises(FileExistsError, match="already exists; not overwriting"):
        make_trainer(milestone, monkeypatch, extra=["downstream.seed=3"])
    make_trainer(milestone, monkeypatch, extra=["downstream.seed=4"])  # leaves an incomplete attempt
    with pytest.raises(FileExistsError, match="Incomplete downstream output"):
        make_trainer(milestone, monkeypatch, extra=["downstream.seed=4"])


# ---------------------------------------------------------------- training phases

def test_two_phases_keep_denoiser_frozen_and_reward_model_fixed_during_pmpo(tmp_path, monkeypatch):
    trainer = make_trainer(build_milestone(tmp_path), monkeypatch)
    denoiser_before = tensor_state(trainer.agent.denoiser)
    reward_before, actor_before = tensor_state(trainer.agent.rew_end_model), tensor_state(trainer.agent.actor_critic)
    assert not trainer.agent.denoiser.training and not any(p.requires_grad for p in trainer.agent.denoiser.parameters())
    optimized = {id(p) for group in trainer.rew_end_opt.param_groups for p in group["params"]}
    assert optimized == {id(p) for p in trainer.agent.rew_end_model.parameters()}  # no denoiser optimizer

    trainer.train_reward_model()
    reward_after = tensor_state(trainer.agent.rew_end_model)
    assert not states_equal(reward_before, reward_after) and trainer.num_updates["rew_end_model"] == 4
    assert not any(p.requires_grad for p in trainer.agent.rew_end_model.parameters())

    calls = []
    predict = trainer.rl_env.predict_rew_end
    trainer.rl_env.predict_rew_end = lambda next_obs: calls.append(1) or predict(next_obs)
    trainer.train_policy()
    assert calls and trainer.agent.actor_critic.intrinsic_reward_fn is None  # predicted task reward only
    assert trainer.rl_env.rew_end_model is trainer.agent.rew_end_model
    assert not states_equal(actor_before, tensor_state(trainer.agent.actor_critic))
    assert trainer.agent.actor_critic.updates.item() == 3
    assert states_equal(reward_after, tensor_state(trainer.agent.rew_end_model))
    assert states_equal(denoiser_before, tensor_state(trainer.agent.denoiser))


def test_reward_targets_come_only_from_the_selected_task(tmp_path, monkeypatch):
    def train(offsets, name):
        trainer = make_trainer(build_milestone(tmp_path / name, offsets), monkeypatch)
        trainer.run()
        rows = (trainer.output_dir / "metrics.jsonl").read_text().splitlines()
        return tensor_state(trainer.agent.rew_end_model), tensor_state(trainer.agent.actor_critic), rows

    base = train((0.0, 0.0, 0.0), "base")
    others_changed = train((5.0, 0.0, -3.0), "others")  # stand and run columns differ
    walk_changed = train((0.0, 2.0, 0.0), "walk")
    assert states_equal(base[0], others_changed[0]) and states_equal(base[1], others_changed[1])
    assert base[2] == others_changed[2]  # every logged loss and PMPO metric
    assert not states_equal(base[0], walk_changed[0])


def test_only_an_action_spec_lookup_touches_dm_control(tmp_path, monkeypatch):
    import coroutines.collector as collector
    import envs
    import intrinsic_reward.lcg as lcg

    def forbidden(*args, **kwargs):
        raise AssertionError("constructed during downstream training")

    for owner, name in ((lcg.LCGLifecycle, "__init__"), (collector, "make_collector"), (envs, "make_dm_control_env"),
                        (envs.DMControlEnv, "__init__")):
        monkeypatch.setattr(owner, name, forbidden)

    # The real dm_control load is allowed for the action spec; the returned environment
    # must never be reset or stepped, and must be closed.
    real_load = downstream_module.suite.load
    loaded = []

    def load(domain_name, task_name, *args, **kwargs):
        env = real_load(domain_name, task_name, *args, **kwargs)
        record = dict(domain=domain_name, task=task_name, spec_read=False, closed=False)
        loaded.append(record)
        action_spec, close = env.action_spec, env.close

        def read_spec():
            record["spec_read"] = True
            return action_spec()

        def closed():
            record["closed"] = True
            return close()

        def interaction(*args, **kwargs):
            raise AssertionError("real environment reset or stepped during downstream training")

        env.action_spec, env.close, env.reset, env.step = read_spec, closed, interaction, interaction
        return env

    monkeypatch.setattr(downstream_module.suite, "load", load)
    trainer = make_trainer(build_milestone(tmp_path), monkeypatch)
    assert loaded == [dict(domain="walker", task="walk", spec_read=True, closed=True)]
    trainer.run()  # reward-model and PMPO phases under the same guards
    assert trainer.num_updates == {"rew_end_model": 4, "actor_critic": 3} and len(loaded) == 1


# ---------------------------------------------------------------- batches and imagination

def test_task_reward_batches_keep_alignment_masks_and_successors(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    trainer = make_trainer(milestone, monkeypatch)
    dataset, m = trainer.milestone.dataset, trainer.milestone
    sources = [dataset.load_episode(i) for i in range(3)]
    copies = [(e.obs.clone(), e.rew.clone(), e.task_rew.clone(), dict(e.info)) for e in sources]
    collate = make_task_reward_collate(m.task_index, m.successor_observations)
    # Past the end of a finished episode, past the cut of the unfinished one, inside one,
    # and before an episode start.
    ids = [SegmentId(0, 7, 13), SegmentId(2, 3, 9), SegmentId(1, 2, 8), SegmentId(1, -3, 3)]
    segments = [dataset[i] for i in ids]
    batch = collate(segments)
    for i, segment in enumerate(segments):
        torch.testing.assert_close(batch.mask_padding[i], segment.mask_padding, rtol=0, atol=0)
        torch.testing.assert_close(batch.rew[i], segment.task_rew[:, 1], rtol=0, atol=0)  # walk column
    assert batch.task_rew is None and batch.info == [{}] * 4
    torch.testing.assert_close(batch.obs[0, :3], sources[0].obs[7:10], rtol=0, atol=0)
    torch.testing.assert_close(batch.obs[0, 3], sources[0].info["final_observation"], rtol=0, atol=0)
    assert (batch.obs[0, 4:] == 0).all()
    torch.testing.assert_close(batch.obs[1, 3], m.successor_observations[2], rtol=0, atol=0)
    torch.testing.assert_close(batch.obs[2], sources[1].obs[2:8], rtol=0, atol=0)
    assert batch.mask_padding[3].tolist() == [False] * 3 + [True] * 3
    for episode, (obs, rew, task_rew, info) in zip(sources, copies):  # sources untouched
        assert torch.equal(episode.obs, obs) and torch.equal(episode.rew, rew) and torch.equal(episode.task_rew, task_rew)
        assert episode.info.keys() == info.keys()


def test_imagination_uses_burn_in_horizon_truncation_and_bootstrap(tmp_path, monkeypatch):
    trainer = make_trainer(build_milestone(tmp_path), monkeypatch)
    env, model = trainer.rl_env, trainer.agent.actor_critic
    burn_in = []
    original = trainer.agent.rew_end_model.predict_rew_end

    def spy(obs, act, next_obs, hx_cx=None):
        burn_in.append(hx_cx is None)
        return original(obs, act, next_obs, hx_cx)

    trainer.agent.rew_end_model.predict_rew_end = spy
    prompts = []
    reset_dead = env.reset_dead
    env.reset_dead = lambda dead: prompts.append(reset_dead(dead)) or prompts[-1]
    loop = make_env_loop(env, model, store_policy_observations=True)
    with torch.no_grad():
        states, _, rew, end, trunc, _, _, bootstrap, infos = loop.send(3)
    loop.close()
    assert True in burn_in and False in burn_in  # prompt burn-in, then per-step predictions
    assert (end == 0).all() and trunc.tolist() == [[0, 1, 0]] * 2  # horizon 2, no sampled endings
    assert rew.abs().sum() > 0  # predicted reward, not a placeholder
    with torch.no_grad():
        successor = torch.cat((states[:, 1][:, 3:], infos[1]["final_observation"]), 1)
        torch.testing.assert_close(bootstrap[:, 1], model.value(successor).squeeze(-1), rtol=0, atol=0)
    # The new episode's policy stack repeats its start frame (imagination_history=repeat_anchor,
    # the reference controller's convention); the start frame is the new prompt's last frame,
    # which the world model keeps conditioning on with the whole prompt.
    assert model.cfg.imagination_history == "repeat_anchor"
    (frames_, valid), = prompts
    assert valid[:, -1].all()
    torch.testing.assert_close(states[:, 2], frames_[:, -1].repeat(1, 4, 1, 1), rtol=0, atol=0)
    torch.testing.assert_close(env.obs_buffer[:, :-1], frames_[:, 1:], rtol=0, atol=0)  # prompt advanced one step


# ---------------------------------------------------------------- entry point

def test_entry_point_runs_through_hydra(tmp_path):
    milestone = build_milestone(tmp_path)
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), f"initialization.milestone_path={milestone.as_posix()}",
               "downstream.task_name=run", "downstream.seed=1", "common.devices=cpu", *TINY_DOWNSTREAM]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    out = tmp_path / "run/downstream/steps_000026/run/seed_1"
    assert (out / "summary.json").is_file() and OmegaConf.load(out / "summary.json").task_index == 2
    assert not list(ROOT.glob("*.log"))  # no Hydra log file written into the repository
    config = OmegaConf.load(out / "config.yaml")  # a normal single-milestone configuration
    assert "milestones_dir" not in config.initialization and config.initialization.milestone_path == milestone.as_posix()


# ---------------------------------------------------------------- folder mode (downstream_main.py)

def fake_hydra(overrides, config_dirs=(), hydra_overrides=(), config_path=ROOT / "config", config_name="downstream"):
    """The parts of HydraConfig.get() that child_command_base reads, as `python src/downstream_main.py
    [--config-path=...] [--config-dir=...] --config-name=... <hydra_overrides> <overrides>` records them."""
    sources = [dict(path="hydra.conf", schema="pkg", provider="hydra"),
               dict(path=str(config_path), schema="file", provider="main"),
               *(dict(path=str(d), schema="file", provider="command-line") for d in config_dirs),
               dict(path="", schema="structured", provider="schema")]
    return OmegaConf.create(dict(runtime=dict(config_sources=sources), job=dict(config_name=config_name),
                                 overrides=dict(hydra=[*hydra_overrides, "hydra.mode=RUN"], task=list(overrides))))


def folder_cfg(milestones_dir, *extra, task="walk"):
    """The launcher's configuration, child command base and overrides, as downstream_main.main has them."""
    overrides = [f"initialization.milestones_dir={Path(milestones_dir).as_posix()}", f"downstream.task_name={task}",
                 "common.devices=cpu", *TINY_DOWNSTREAM, *extra]
    cfg = compose("downstream", overrides)
    base = downstream_main.child_command_base(fake_hydra(overrides))
    return cfg, Path(downstream_main.pop_milestone_source(cfg)[0]), base, overrides


class FakeRun:
    """Records child commands (and environments) instead of starting processes; exit codes in order (default 0)."""
    def __init__(self, *codes):
        self.codes, self.commands, self.envs = list(codes), [], []

    def __call__(self, command, env=None):
        self.commands.append(command)
        self.envs.append(env)
        return subprocess.CompletedProcess(command, self.codes.pop(0) if self.codes else 0)


def test_exactly_one_milestone_source_and_no_launcher_field_in_trainer_configs(tmp_path, monkeypatch):
    for overrides in (["initialization.milestone_path=x", "initialization.milestones_dir=y"], []):
        with pytest.raises(ValueError, match="exactly one of initialization.milestone_path"):
            downstream_main.pop_milestone_source(compose("downstream", ["downstream.task_name=walk", *overrides]))
    cfg = compose("downstream", ["initialization.milestone_path=x", "downstream.task_name=walk"])
    assert downstream_main.pop_milestone_source(cfg) == (None, None)
    assert "milestones_dir" not in cfg.initialization and "milestone_steps" not in cfg.initialization
    milestone = build_milestone(tmp_path)
    trainer = make_trainer(milestone, monkeypatch)  # composed directly: the field is dropped as well
    saved = OmegaConf.load(trainer._work_dir / "config.yaml").initialization
    for key in ("milestones_dir", "milestone_steps"):
        assert key not in trainer.cfg.initialization and key not in saved
    with pytest.raises(ValueError, match="takes one milestone"):
        make_trainer(milestone, monkeypatch, extra=["downstream.seed=5", "initialization.milestones_dir=y"])
    with pytest.raises(ValueError, match="takes one milestone; initialization.milestone_steps"):
        make_trainer(milestone, monkeypatch, extra=["downstream.seed=6", "initialization.milestone_steps=[26]"])


def test_discovery_orders_by_recorded_steps_and_rejects_bad_folders(tmp_path):
    milestones = build_milestone(tmp_path, early=(1, 2)).parent
    # Folder names that sort the other way round: the manifests' collected_steps decide.
    for old, new in (("steps_000010", "c_first"), ("steps_000020", "b_second"), ("steps_000026", "a_third")):
        (milestones / old).rename(milestones / new)
    found = discover_milestones(milestones)
    assert [(p.name, m["collected_steps"]) for p, m in found] == [("c_first", 10), ("b_second", 20), ("a_third", 26)]

    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_milestones(tmp_path / "nowhere")
    (tmp_path / "empty").mkdir()
    with pytest.raises(ValueError, match="contains no milestones"):
        discover_milestones(tmp_path / "empty")
    shutil.copytree(milestones / "b_second", milestones / "b_copy")
    with pytest.raises(ValueError, match="ambiguous duplicate step counts 20 recorded by b_copy, b_second"):
        discover_milestones(milestones)
    shutil.rmtree(milestones / "b_copy")
    for name, make_bad, message in (("notes", lambda p: p.mkdir(), "notes: No milestone manifest"),
                                    (".steps_000030.tmp", lambda p: p.mkdir(), "interrupted or in-progress"),
                                    ("stray.txt", lambda p: p.write_text("x"), "stray.txt: not a milestone folder")):
        make_bad(milestones / name)
        with pytest.raises(ValueError, match=f"Malformed milestones folder(.|\\n)*{message}"):
            discover_milestones(milestones)
        shutil.rmtree(milestones / name) if (milestones / name).is_dir() else (milestones / name).unlink()
    manifest = milestones / "a_third/manifest.json"
    text = manifest.read_text(encoding="utf-8")
    manifest.write_text(text.replace('"collected_steps": 26', '"collected_steps": "many"'), encoding="utf-8")
    with pytest.raises(ValueError, match="a_third: manifest has no integer collected_steps"):
        discover_milestones(milestones)


def test_preflight_rejects_tasks_files_conflicts_and_modes_before_any_launch(tmp_path):
    milestones = build_milestone(tmp_path, early=(2,)).parent
    run = FakeRun()

    def rejected(match, *extra, task="walk"):
        cfg, folder, base, overrides = folder_cfg(milestones, *extra, task=task)
        with pytest.raises(ValueError, match=match):
            downstream_main.run_milestones(cfg, folder, base, run=run)
        assert not run.commands and not (tmp_path / "run/downstream/steps_000020").exists()

    rejected("Nothing was started(.|\\n)*steps_000020 has no reward labels(.|\\n)*steps_000026 has no reward labels", task="jump")
    for mode in ("resume", "evaluate"):
        rejected(f"supports only downstream.mode=train, got '{mode}'. {mode} takes one milestone", f"downstream.mode={mode}")
    denoiser = milestones / "steps_000026/denoiser.pt"
    original = denoiser.read_bytes()
    denoiser.write_bytes(original + b"x")
    rejected("Nothing was started(.|\\n)*steps_000026: denoiser file .* does not match its checksum")
    denoiser.write_bytes(original)

    # Conflicts at the LAST milestone are found before the first one trains, and left untouched.
    later = tmp_path / "run/downstream/steps_000026/walk"
    (later / "seed_0").mkdir(parents=True)
    (later / ".seed_0.tmp").mkdir()
    (later / ".seed_0.tmp/metrics.jsonl").write_text("{}\n")
    before = tree_hashes(tmp_path / "run")
    rejected("Nothing was started. 2 problem(.|\\n)*steps_000026: downstream output .*seed_0 already exists; not overwriting"
             "(.|\\n)*steps_000026: incomplete downstream output .*\\.seed_0\\.tmp from an earlier attempt")
    assert tree_hashes(tmp_path / "run") == before and sorted(p.name for p in later.iterdir()) == [".seed_0.tmp", "seed_0"]
    cfg, folder, base, overrides = folder_cfg(milestones, "downstream.seed=1")  # another seed has no conflict
    assert downstream_main.run_milestones(cfg, folder, base, run=run) == 0 and len(run.commands) == 2


def test_children_get_their_own_milestone_with_identical_settings_and_failure_stops_the_batch(tmp_path, capsys):
    milestones = build_milestone(tmp_path, early=(1, 2)).parent
    (milestones / "steps_000010").rename(milestones / "z_steps_10")  # order comes from the manifests
    cfg, folder, base, overrides = folder_cfg(milestones, "downstream.seed=7", "rew_end_model.optimizer.lr=0.0003")
    run = FakeRun()
    assert downstream_main.run_milestones(cfg, folder, base, run=run) == 0
    assert [c[-2:] for c in run.commands] == [[f"initialization.milestone_path={(milestones / name).resolve().as_posix()}",
                                               "downstream.mode=train"] for name in ("z_steps_10", "steps_000020", "steps_000026")]
    assert [c[:-2] for c in run.commands] == [base] * 3
    assert base[:4] == [sys.executable, str(ROOT / "src/downstream_main.py"), f"--config-path={ROOT / 'config'}",
                        "--config-name=downstream"]
    # The same task, seed and settings; the folder fields are cleared, not just omitted.
    assert base[4:] == [*overrides[1:], "++initialization.milestones_dir=null", "++initialization.milestone_steps=null"]
    out = capsys.readouterr().out
    assert out.count("Downstream task adaptation") == 1 and "Downstream batch summary" not in out  # no success summary
    assert "Downstream batch: 3 milestones\n  Domain: walker; task: walk\n  Adaptation seed: 7\n  Milestones:\n" in out
    assert out.index("  1. z_steps_10 | 10 training env steps\n") < out.index("  2. steps_000020 | 20 training env steps\n") \
        < out.index("  3. steps_000026 | 26 training env steps\n") < out.index("\n\n=== Milestone 1/3: z_steps_10 ===\n") \
        < out.index("\n\n=== Milestone 2/3: steps_000020 ===\n") < out.index("\n\n=== Milestone 3/3: steps_000026 ===\n")
    # Children print only their milestone's lines (presentation only: not in the command or the config).
    assert all(env[downstream_main.BATCH_CHILD_ENV] == "1" for env in run.envs)
    assert not any(downstream_main.BATCH_CHILD_ENV in part for command in run.commands for part in command)

    # The second milestone fails after a checkpoint (as a crashed child leaves it); the first completed.
    completed = tmp_path / "run/downstream/z_steps_10/walk/seed_7"
    work = tmp_path / "run/downstream/steps_000020/walk/.seed_7.tmp"

    def crash_after_checkpoint(command):
        if len(run.commands) == 1:
            completed.mkdir(parents=True)
            (completed / "summary.json").write_text("done\n")
        else:
            (work / "checkpoints").mkdir(parents=True)
            (work / "checkpoints/state.pt").write_bytes(b"checkpoint")
        return subprocess.CompletedProcess(command, 0 if len(run.commands) == 1 else 3)

    run = FakeRun()
    calls = run.__call__
    run_and_crash = lambda command, env=None: (calls(command, env), crash_after_checkpoint(command))[1]
    assert downstream_main.run_milestones(cfg, folder, base, run=run_and_crash) == 1
    assert len(run.commands) == 2
    out = capsys.readouterr().out.split("Downstream batch summary")[1]
    assert "z_steps_10: completed" in out and "steps_000026: not started" in out
    assert "steps_000020: FAILED (exit code 3)" in out and f"resumable work folder {work} kept; resume it with:" in out
    resume = downstream_main.milestone_command(base, milestones / "steps_000020", "resume")
    assert downstream_main.shell(resume) in out and resume[-1] == "downstream.mode=resume" and "downstream.seed=7" in resume
    assert (completed / "summary.json").is_file() and (work / "checkpoints/state.pt").is_file()  # both preserved
    assert "Stopped at the failure; later milestones were not started." in out


def test_folder_mode_runs_independent_single_milestone_trainings(tmp_path):
    """Two real child processes; the later milestone's result equals the single-milestone command's."""
    milestones = build_milestone(tmp_path / "a", early=(2,)).parent
    shutil.copytree(tmp_path / "a/run", tmp_path / "b/run")
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), f"initialization.milestones_dir={milestones.as_posix()}",
               "downstream.task_name=run", "downstream.seed=1", "common.devices=cpu", *TINY_DOWNSTREAM]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    print(result.stdout)  # the banner, plan and milestone blocks (shown with pytest -s)
    assert "Downstream batch: 2 milestones" in result.stdout and "Downstream batch summary" not in result.stdout
    configs = {}
    for name, steps in (("steps_000020", 20), ("steps_000026", 26)):
        out = tmp_path / f"a/run/downstream/{name}/run/seed_1"
        summary = OmegaConf.load(out / "summary.json")
        assert (summary.milestone.collected_steps, summary.seed, summary.task_name) == (steps, 1, "run")
        assert (out / summary.milestone.path).resolve() == (milestones / name).resolve()
        assert summary.num_updates == {"rew_end_model": 4, "actor_critic": 3}
        config = OmegaConf.to_container(OmegaConf.load(out / "config.yaml"))
        assert "milestones_dir" not in config["initialization"] and "milestone_steps" not in config["initialization"]
        assert Path(config["initialization"].pop("milestone_path")).resolve() == (milestones / name).resolve()
        configs[name] = config
    assert configs["steps_000020"] == configs["steps_000026"]  # identical settings apart from the milestone

    # No warm start: the later milestone's run equals the single-milestone command on a copy of the run folder
    # (a fresh process as well: an in-process run can differ through process-global state left by other tests).
    single = [sys.executable, str(ROOT / "src/downstream_main.py"),
              f"initialization.milestone_path={(tmp_path / 'b/run/milestones/steps_000026').as_posix()}",
              "downstream.task_name=run", "downstream.seed=1", "common.devices=cpu", *TINY_DOWNSTREAM]
    result = subprocess.run(single, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    reference = tmp_path / "b/run/downstream/steps_000026/run/seed_1"
    batch = tmp_path / "a/run/downstream/steps_000026/run/seed_1"
    for name in ("rew_end_model.pt", "actor_critic.pt"):
        assert states_equal(torch.load(batch / name, weights_only=False), torch.load(reference / name, weights_only=False))
    assert (batch / "metrics.jsonl").read_bytes() == (reference / "metrics.jsonl").read_bytes()


def test_milestone_steps_must_be_a_folder_selection_of_distinct_positive_integers():
    def source(*overrides):
        return downstream_main.pop_milestone_source(compose("downstream", ["downstream.task_name=walk", *overrides]))

    folder = "initialization.milestones_dir=/runs/a/milestones"
    assert source(folder) == ("/runs/a/milestones", None)  # null: every milestone
    assert source(folder, "initialization.milestone_steps=[30,10]") == ("/runs/a/milestones", [30, 10])
    for value, match in (("20", "must be null or a list"), ("[]", "empty list; use null"), ("[10,true]", "positive integers"),
                         ("['10']", "positive integers"), ("[0,10]", "positive integers"), ("[-5]", "positive integers"),
                         ("[1.5]", "positive integers"), ("[10,20,10]", r"lists \[10\] more than once")):
        with pytest.raises(ValueError, match=match):
            source(folder, f"initialization.milestone_steps={value}")
    with pytest.raises(ValueError, match="milestone_steps selects milestones of a folder: it needs initialization.milestones_dir"):
        source("initialization.milestone_path=/runs/a/milestones/steps_000010", "initialization.milestone_steps=[10]")
    cfg = compose("downstream", ["downstream.task_name=walk", folder, "initialization.milestone_steps=[10]"])
    downstream_main.pop_milestone_source(cfg)
    assert "milestone_steps" not in cfg.initialization and "milestones_dir" not in cfg.initialization


def test_selection_uses_manifest_steps_in_ascending_order_and_ignores_unselected_milestones(tmp_path, capsys):
    milestones = build_milestone(tmp_path, early=(1, 2)).parent  # collected_steps 10, 20, 26
    # Folder names that point the wrong way: only the manifests decide.
    (milestones / "steps_000010").rename(milestones / "steps_000026_old")
    (milestones / "steps_000026").rename(milestones / "steps_000010_new")
    # The unselected milestone has a completed output, an incomplete attempt and a broken file:
    # none of it blocks the selected ones, and none of it is touched.
    unselected = tmp_path / "run/downstream/steps_000020/walk"
    (unselected / "seed_0").mkdir(parents=True)
    (unselected / "seed_0/summary.json").write_text("{}\n")
    (unselected / ".seed_0.tmp").mkdir()
    (milestones / "steps_000020/denoiser.pt").write_bytes(b"broken")
    before = tree_hashes(tmp_path / "run")
    cfg, folder, base, _ = folder_cfg(milestones)
    run = FakeRun()
    assert downstream_main.run_milestones(cfg, folder, base, run=run, milestone_steps=[26, 10]) == 0
    assert [c[-2] for c in run.commands] == [f"initialization.milestone_path={(milestones / name).resolve().as_posix()}"
                                             for name in ("steps_000026_old", "steps_000010_new")]
    assert all(c[:-2] == base and "++initialization.milestone_steps=null" in c for c in run.commands)
    out = capsys.readouterr().out
    assert ("Downstream batch: 2 milestones (selected by initialization.milestone_steps from 3 milestones in the folder)\n"
            in out)
    assert out.index("  1. steps_000026_old | 10 training env steps\n") < out.index("  2. steps_000010_new | 26 training env steps\n") \
        < out.index("=== Milestone 1/2: steps_000026_old ===") < out.index("=== Milestone 2/2: steps_000010_new ===")
    assert "steps_000020" not in out and "3." not in out.split("Milestones:")[1].split("===")[0]
    assert tree_hashes(tmp_path / "run") == before  # the fake children write nothing; nothing else was written

    # Every missing step is reported at once, together with the selected milestones' problems,
    # before any child starts; selecting the broken milestone reports it.
    run = FakeRun()
    with pytest.raises(ValueError, match="Nothing was started. 4 problem(.|\\n)*requests 30, 40, which no milestone records "
                                         "\\(recorded: 10, 20, 26\\)(.|\\n)*steps_000020: denoiser file .* does not match"
                                         "(.|\\n)*steps_000020: downstream output .*seed_0 already exists"
                                         "(.|\\n)*steps_000020: incomplete downstream output"):
        downstream_main.run_milestones(cfg, folder, base, run=run, milestone_steps=[40, 20, 30])
    with pytest.raises(ValueError, match="positive integers"):
        downstream_main.run_milestones(cfg, folder, base, run=run, milestone_steps=[10, True])
    assert not run.commands and tree_hashes(tmp_path / "run") == before


def test_selected_folder_run_adapts_only_selected_milestones_from_the_command_line(tmp_path):
    """Real children: only the selected milestone is adapted, its config has no launcher fields, and
    the unselected milestone and the selected milestone's inputs are unchanged."""
    milestones = build_milestone(tmp_path, early=(2,)).parent  # steps_000020, steps_000026
    before = tree_hashes(milestones)
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), f"initialization.milestones_dir={milestones.as_posix()}",
               "initialization.milestone_steps=[26]", "downstream.task_name=run", "downstream.seed=1", "common.devices=cpu",
               *TINY_DOWNSTREAM]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Downstream batch: 1 milestone (selected by initialization.milestone_steps from 2 milestones" in result.stdout
    assert "=== Milestone 1/1: steps_000026 ===" in result.stdout and "steps_000020" not in result.stdout
    assert sorted(p.name for p in (tmp_path / "run/downstream").iterdir()) == ["steps_000026"]
    out = tmp_path / "run/downstream/steps_000026/run/seed_1"
    assert OmegaConf.load(out / "summary.json").milestone.collected_steps == 26
    config = OmegaConf.load(out / "config.yaml").initialization
    assert "milestone_steps" not in config and "milestones_dir" not in config
    assert tree_hashes(milestones) == before  # milestone inputs (selected and unselected) unchanged


def test_selection_from_a_config_file_reaches_the_children_cleared(tmp_path):
    milestones = build_milestone(tmp_path, early=(2,)).parent
    yaml_batch_config(tmp_path / "conf/batch.yaml", milestones, extra=["initialization.milestone_steps=[20]"])
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), f"--config-dir={tmp_path / 'conf'}",
               "--config-name=batch", "common.devices=cpu"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Downstream batch: 1 milestone (selected by" in result.stdout and "=== Milestone 1/1: steps_000020 ===" in result.stdout
    assert sorted(p.name for p in (tmp_path / "run/downstream").iterdir()) == ["steps_000020"]
    config = OmegaConf.load(tmp_path / "run/downstream/steps_000020/run/seed_3/config.yaml")
    assert "milestone_steps" not in config.initialization and config.downstream.seed == 3


def test_child_command_keeps_the_configuration_source_and_clears_the_folder_field():
    hydra_cfg = fake_hydra(["+initialization.milestones_dir=/runs/a/milestones", "downstream.task_name=run",
                            "downstream.seed=4", "downstream.mode=train", "initialization.member_index=0",
                            "rew_end_model.training.steps=7"],
                           config_dirs=["/extra/conf"], hydra_overrides=["hydra.job.chdir=false"],
                           config_path="/custom/config", config_name="batch")
    base = downstream_main.child_command_base(hydra_cfg)
    assert base == [sys.executable, str(ROOT / "src/downstream_main.py"), f"--config-path={Path('/custom/config').resolve()}",
                    f"--config-dir={Path('/extra/conf').resolve()}", "--config-name=batch", "hydra.job.chdir=false",
                    "downstream.task_name=run", "downstream.seed=4", "initialization.member_index=0",
                    "rew_end_model.training.steps=7", "++initialization.milestones_dir=null", "++initialization.milestone_steps=null"]
    command = downstream_main.milestone_command(base, Path("/runs/a/milestones/steps_000020"), "evaluate")
    assert command[len(base):] == [f"initialization.milestone_path={Path('/runs/a/milestones/steps_000020').resolve().as_posix()}",
                                   "downstream.mode=evaluate"]
    assert downstream_main.hydra_quote(Path("/a b/it's")) == "'" + Path("/a b/it's").resolve().as_posix().replace("'", "\\'") + "'"


def yaml_batch_config(path, milestones, extra=()):
    """A full downstream config file that sets the folder, a non-default task and seed and the tiny
    training settings itself: the children only finish quickly if they read it."""
    data = yaml.safe_load((ROOT / "config/downstream.yaml").read_text(encoding="utf-8"))
    settings = OmegaConf.from_dotlist([*TINY_DOWNSTREAM, f"initialization.milestones_dir={milestones.as_posix()}",
                                       "downstream.task_name=run", "downstream.seed=3", "rew_end_model.optimizer.lr=0.0003",
                                       *extra])
    merged = OmegaConf.merge(OmegaConf.create({k: v for k, v in data.items() if k != "defaults"}), settings)
    text = yaml.safe_dump(dict(defaults=data["defaults"], **OmegaConf.to_container(merged)), sort_keys=False)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


@pytest.mark.parametrize("source", ["config-dir", "config-path"])
def test_folder_mode_from_a_config_file_with_a_custom_config_source(tmp_path, source):
    """milestones_dir set in YAML (not on the command line), loaded through --config-dir or --config-path."""
    milestones = build_milestone(tmp_path, early=(2,)).parent
    if source == "config-dir":
        yaml_batch_config(tmp_path / "conf/batch.yaml", milestones)  # env/agent groups come from the primary path
        args = [f"--config-dir={tmp_path / 'conf'}"]
    else:
        shutil.copytree(ROOT / "config", tmp_path / "conf")
        yaml_batch_config(tmp_path / "conf/batch.yaml", milestones)
        args = [f"--config-path={tmp_path / 'conf'}"]
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), *args, "--config-name=batch", "common.devices=cpu"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "  Domain: walker; task: run\n  Adaptation seed: 3\n" in result.stdout
    assert "Downstream batch: 2 milestones" in result.stdout and "=== Milestone 2/2: steps_000026 ===" in result.stdout
    for name in ("steps_000020", "steps_000026"):
        out = tmp_path / f"run/downstream/{name}/run/seed_3"
        config = OmegaConf.load(out / "config.yaml")
        # Exactly one milestone source: its own milestone_path, no folder field.
        assert "milestones_dir" not in config.initialization and "milestone_steps" not in config.initialization
        assert Path(config.initialization.milestone_path).resolve() == (milestones / name).resolve()
        assert (config.downstream.task_name, config.downstream.seed, config.rew_end_model.optimizer.lr,
                config.rew_end_model.training.steps, config.actor_critic.training.steps) == ("run", 3, 0.0003, 4, 3)
        assert OmegaConf.load(out / "summary.json").num_updates == {"rew_end_model": 4, "actor_critic": 3}


def test_failure_after_publication_reports_the_outputs_and_their_evaluate_command(tmp_path, capsys):
    """A child that fails after publishing its training outputs (as in a failed evaluation): the batch
    stops, nothing is deleted, and the printed evaluate command (same task, seed, member and
    settings) evaluates the published output."""
    milestones = build_milestone(tmp_path, early=(2,)).parent
    cfg, folder, base, _ = folder_cfg(milestones, "downstream.seed=2", "initialization.member_index=0",
                                      "evaluation.seeds=[9]", "evaluation.max_steps=3")
    output = tmp_path / "run/downstream/steps_000020/walk/seed_2"
    commands = []

    def train_then_fail(command, env=None):  # trains for real (evaluation.after_training=false), then reports a failure
        commands.append(command)
        result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True, env=env)
        assert result.returncode == 0, result.stderr[-3000:]
        return subprocess.CompletedProcess(command, 1)

    assert downstream_main.run_milestones(cfg, folder, base, run=train_then_fail) == 1
    assert len(commands) == 1 and not (tmp_path / "run/downstream/steps_000026").exists()
    out = capsys.readouterr().out.split("Downstream batch summary")[1]
    evaluate = downstream_main.milestone_command(base, milestones / "steps_000020", "evaluate")
    assert f"steps_000020: FAILED (exit code 1) after" in out and f"training outputs were published at {output}" in out
    assert downstream_main.shell(evaluate) in out and "steps_000026: not started" in out
    assert (output / "summary.json").is_file() and not (output / "evaluations").exists()  # nothing deleted
    assert evaluate[len(base) - 2:] == ["++initialization.milestones_dir=null", "++initialization.milestone_steps=null",
                                        f"initialization.milestone_path={(milestones / 'steps_000020').resolve().as_posix()}",
                                        "downstream.mode=evaluate"]
    assert {"downstream.task_name=walk", "downstream.seed=2", "initialization.member_index=0", "evaluation.seeds=[9]"} <= set(evaluate)
    result = subprocess.run(evaluate, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    (evaluation,) = (output / "evaluations").iterdir()
    assert OmegaConf.load(evaluation).seeds == [9]


def test_folder_mode_failing_child_gives_a_nonzero_exit_and_starts_nothing_later(tmp_path):
    """A real child that fails (rejected by the trainer, after the launcher's preflight passed)."""
    milestones = build_milestone(tmp_path, early=(2,)).parent
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), f"initialization.milestones_dir={milestones.as_posix()}",
               "downstream.task_name=walk", "common.devices=cpu", *TINY_DOWNSTREAM, "checkpointing.every=0"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and "checkpointing.every must be null or a positive integer" in result.stderr
    assert "steps_000020: FAILED (exit code 1)" in result.stdout and "no work folder was created" in result.stdout
    assert "steps_000026: not started" in result.stdout
    assert not (tmp_path / "run/downstream/steps_000026").exists()

# ---------------------------------------------------------------- terminal format

EVALUATED = [o for o in TINY_DOWNSTREAM if o != "evaluation.after_training=false"] + [
    "evaluation.seeds=[3001,3002]", "evaluation.max_steps=3"]
RULE = "=" * 78
# A closed bar: description, 100%, completed/total, [elapsed<remaining, rate]; the rate with a space
# before its unit, inverted below one per second (s/update, s/ep).
BAR = re.compile(r"^(?P<desc>.+): 100%\|[^|]*\| (?P<n>\d+)/(?P=n) \[[\d:]+<[\d:]+, "
                 r"(?:\d+\.\d\d (?P<plural>[a-z]+)/s|\d+\.\d\d s/(?P<singular>[a-z]+))\]$")


def screen_lines(raw):
    """What stays on screen: line endings normalized, and of a bar's in-place redraws (carriage
    returns) only the last, without trailing blanks (tqdm pads a redraw that is shorter than the
    previous one, e.g. "9.84 updates/s" after "10.12 updates/s", with spaces)."""
    text = raw.decode("utf-8").replace("\r\n", "\n")
    return [line.rsplit("\r", 1)[-1].rstrip(" ") for line in text.split("\n")]


def downstream_screen(*args):
    """A real downstream_main.py run (as typed in a terminal, not as a folder's child); its screen lines."""
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    env.pop(downstream_main.BATCH_CHILD_ENV, None)
    result = subprocess.run([sys.executable, str(ROOT / "src/downstream_main.py"), "common.devices=cpu", *args],
                            cwd=ROOT, capture_output=True, env=env)
    assert result.returncode == 0, result.stderr.decode("utf-8", "replace")[-3000:]
    return screen_lines(result.stdout)


def banner(label, source, root, training=True, seeds="2 seeds (3001\u20133002)"):
    lines = [RULE, "Downstream task adaptation", f"  {label}: {Path(source).resolve()}", "  Compute device: CPU", None]
    if training:
        lines.append("  Training budget: 4 reward-predictor updates + 3 policy updates (per milestone)")
    return lines + [f"  Evaluation: {seeds} | maximum 3 steps per episode", f"  Output root: {Path(root).resolve()}", RULE, ""]


def check_block(lines, output, seeds, training=True):
    """One milestone's lines after its heading, in order and without blank lines: the training bars
    (if any), the evaluation bar, the summary from the saved evaluation (max, median, mean, min, std)
    and the saved paths. Returns the lines after the block."""
    phases = [("Reward-predictor training", 4, "update"), ("Task-policy training", 3, "update")] if training else []
    phases.append(("Evaluation", len(seeds), "ep"))
    for line, (desc, n, unit) in zip(lines, phases):
        bar = BAR.match(line)
        assert bar and bar["desc"] == desc and int(bar["n"]) == n, line
        assert bar["plural"] == dict(update="updates", ep="ep")[unit] or bar["singular"] == unit, line
    (evaluation,) = [p for p in (output / "evaluations").iterdir() if OmegaConf.load(p).seeds == seeds]
    result = OmegaConf.to_container(OmegaConf.load(evaluation))
    returns = np.array([e["episode_return"] for e in result["episodes"]])  # per-seed returns stay in the file
    assert [e["seed"] for e in result["episodes"]] == seeds
    assert [result[f"return_{k}"] for k in ("max", "median", "mean", "min", "std")] == [
        float(returns.max()), float(np.median(returns)), float(returns.mean()), float(returns.min()), float(returns.std())]
    rest = lines[len(phases):]
    assert rest[0] == "Evaluation summary: " + " | ".join(
        f"{k} {result[f'return_{k}']:.2f}" for k in ("max", "median", "mean", "min", "std"))
    assert rest[1] == f"Evaluation saved: > {evaluation.resolve()}"
    if training:
        assert rest[2] == f"Downstream outputs saved: > {output.resolve()}"
    return rest[2 + training:]


def test_terminal_format_single_milestone_folder_and_evaluate_only(tmp_path):
    milestones = build_milestone(tmp_path, early=(2,)).parent
    root, single = tmp_path / "run/downstream", milestones / "steps_000026"

    lines = downstream_screen(f"initialization.milestone_path={single.as_posix()}", "downstream.task_name=walk", *EVALUATED)
    expected = banner("Milestone directory", single, root)
    assert lines[4].startswith("  Rendering device: ")
    assert lines[:len(expected)] == expected[:4] + [lines[4]] + expected[5:]
    rest = lines[len(expected):]
    assert rest[:6] == ["Downstream batch: 1 milestone", "  Domain: walker; task: walk", "  Adaptation seed: 0",
                        "  Milestones:", "  1. steps_000026 | 26 training env steps", ""]
    assert rest[6] == "=== Milestone 1/1: steps_000026 ==="
    assert check_block(rest[7:], root / "steps_000026/walk/seed_0", [3001, 3002]) == [""]  # nothing after the saved paths

    # Evaluate-only: the same banner without the training budget, then only the evaluation.
    lines = downstream_screen(f"initialization.milestone_path={single.as_posix()}", "downstream.task_name=walk",
                              "downstream.mode=evaluate", *EVALUATED, "evaluation.seeds=[5,9]")
    expected = banner("Milestone directory", single, root, training=False, seeds="2 seeds (5, 9)")
    assert lines[:4] + lines[5:len(expected)] == expected[:4] + expected[5:]
    rest = lines[len(expected):]
    assert rest[:4] == ["Downstream batch: 1 milestone", "  Domain: walker; task: walk", "  Adaptation seed: 0",
                        "  Mode: evaluate only (completed output)"]
    assert rest[4:8] == ["  Milestones:", "  1. steps_000026 | 26 training env steps", "", "=== Milestone 1/1: steps_000026 ==="]
    assert check_block(rest[8:], root / "steps_000026/walk/seed_0", [5, 9], training=False) == [""]

    # Folder: one banner and plan; one block per milestone, a blank line between them; the children's
    # output is not duplicated and no success summary follows the last saved paths.
    lines = downstream_screen(f"initialization.milestones_dir={milestones.as_posix()}", "downstream.task_name=walk",
                              "downstream.seed=1", *EVALUATED)
    expected = banner("Milestones folder", milestones, root)
    assert lines[:4] + lines[5:len(expected)] == expected[:4] + expected[5:]
    rest = lines[len(expected):]
    assert rest[:7] == ["Downstream batch: 2 milestones", "  Domain: walker; task: walk", "  Adaptation seed: 1", "  Milestones:",
                        "  1. steps_000020 | 20 training env steps", "  2. steps_000026 | 26 training env steps", ""]
    assert rest[7] == "=== Milestone 1/2: steps_000020 ==="
    rest = check_block(rest[8:], root / "steps_000020/walk/seed_1", [3001, 3002])
    assert rest[:2] == ["", "=== Milestone 2/2: steps_000026 ==="]
    assert check_block(rest[2:], root / "steps_000026/walk/seed_1", [3001, 3002]) == [""]
    assert sum(line == "Downstream task adaptation" for line in lines) == 1


def test_evaluate_only_needs_only_the_saved_output_not_the_milestone(tmp_path):
    """A completed output is evaluated after its source milestone and replay were moved away: the
    banner comes from the output's config.yaml and summary.json; nothing reads the milestone."""
    milestones = build_milestone(tmp_path).parent
    single, root = milestones / "steps_000026", tmp_path / "run/downstream"
    output = root / "steps_000026/walk/seed_0"
    downstream_screen(f"initialization.milestone_path={single.as_posix()}", "downstream.task_name=walk", *TINY_DOWNSTREAM)
    shutil.move(str(tmp_path / "run/milestones"), str(tmp_path / "moved_milestones"))
    shutil.move(str(tmp_path / "run/dataset"), str(tmp_path / "moved_dataset"))
    assert not single.exists() and (output / "summary.json").is_file()
    trained = tree_hashes(output)

    evaluate = [f"initialization.milestone_path={single.as_posix()}", "downstream.task_name=walk", "downstream.mode=evaluate",
                *EVALUATED, "evaluation.seeds=[5,9]"]
    lines = downstream_screen(*evaluate)
    expected = banner("Milestone directory", single, root, training=False, seeds="2 seeds (5, 9)")
    expected[2] += " (not found; evaluation uses the saved output)"
    assert lines[:4] + lines[5:len(expected)] == expected[:4] + expected[5:]
    rest = lines[len(expected):]
    assert rest[:7] == ["Downstream batch: 1 milestone", "  Domain: walker; task: walk", "  Adaptation seed: 0",
                        "  Mode: evaluate only (completed output)", "  Milestones:",
                        "  1. steps_000026 | 26 training env steps", ""]  # recorded in summary.json
    assert rest[7] == "=== Milestone 1/1: steps_000026 ==="
    assert check_block(rest[8:], output, [5, 9], training=False) == [""]
    assert {k: v for k, v in tree_hashes(output).items() if "evaluations" not in k} == trained  # outputs untouched

    # Overwrite protection: the same policy and protocol again is refused, the file is unchanged.
    (evaluation,) = (output / "evaluations").iterdir()
    before = evaluation.read_bytes()
    env = dict(os.environ, PYTHONIOENCODING="utf-8")
    result = subprocess.run([sys.executable, str(ROOT / "src/downstream_main.py"), "common.devices=cpu", *evaluate],
                            cwd=ROOT, capture_output=True, env=env)
    assert result.returncode != 0 and b"already exists; not overwriting it" in result.stderr
    assert evaluation.read_bytes() == before

    # Historical metadata not in the summary is labelled, not guessed.
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"))
    del summary["milestone"]["collected_steps"]
    (output / "summary.json").write_text(json.dumps(summary), encoding="utf-8")
    lines = downstream_screen(*evaluate[:-1], "evaluation.seeds=[7]")
    assert "  1. steps_000026 | training env steps not recorded in the saved output" in lines
    assert check_block(lines[lines.index("=== Milestone 1/1: steps_000026 ===") + 1:], output, [7], training=False) == [""]

    # No completed output: no banner, the existing error.
    result = subprocess.run([sys.executable, str(ROOT / "src/downstream_main.py"), "common.devices=cpu", *evaluate,
                             "downstream.seed=5"], cwd=ROOT, capture_output=True, env=env)
    assert result.returncode != 0 and b"No completed downstream output" in result.stderr
    assert b"Downstream task adaptation" not in result.stdout


def test_folder_plan_singular_and_member_lines(tmp_path, capsys):
    milestones = build_milestone(tmp_path).parent  # one milestone
    cfg, folder, base, _ = folder_cfg(milestones, "downstream.seed=4", "initialization.member_index=0")
    assert downstream_main.run_milestones(cfg, folder, base, run=FakeRun()) == 0
    out = capsys.readouterr().out
    assert f"  Milestones folder: {milestones.resolve()}\n" in out and "Milestone directory" not in out
    assert ("  Evaluation: none after training (evaluation.after_training=false)\n" in out
            and "Downstream batch: 1 milestone\n  Domain: walker; task: walk\n  Adaptation seed: 4\n  Milestones:\n"
                "  1. steps_000026 | 26 training env steps\n\n=== Milestone 1/1: steps_000026 ===\n" in out)
    assert "Ensemble member" not in out  # single-model milestone, member 0
    assert out.endswith("=== Milestone 1/1: steps_000026 ===\n")  # the (fake) child printed nothing; no summary


# ---------------------------------------------------------------- W&B logging (fake W&B module)

class FakeWandb:
    """Stands in for the wandb module of downstream_trainer: init kwargs, axes, (step, row) logs, summary
    and finish codes. starting_step and summary model what the W&B server holds for a resumed run. With a
    server (dict: run id -> dict(rows, summary)) one fake serves several processes of the same runs, as W&B
    does: a resumed run starts after its last step and keeps its summary; a step below that is dropped."""

    def __init__(self, run_id="dsrun1", starting_step=0, offline=False, summary=None, server=None, lose=None,
                 api_error=None):
        self.run, self.run_id, self.starting_step, self.offline = None, run_id, starting_step, offline
        self.inits, self.metrics, self.hidden, self.logged, self.finished = [], [], [], [], []
        self.summary = dict(summary or {})
        self.server = server
        self.lose = lose  # (step, row) -> True: logged (queued) but never reaches the server (died before upload)
        self.api_error = api_error  # the server's history cannot be read (wandb.Api raises it)

    def Api(self, timeout=None):
        """The public API's view of the server's history (scan_history of a run's rows)."""
        if self.api_error is not None:
            raise self.api_error
        if self.server is None:
            raise RuntimeError("this fake has no server history")
        server = self.server

        class Run:
            def __init__(self, rows):
                self.rows = rows

            def scan_history(self, min_step=None, max_step=None):
                return [dict(r, _step=s) for s, r in self.rows
                        if (min_step is None or s >= min_step) and (max_step is None or s < max_step)]

        class Api:
            def run(self, path):
                return Run(server[path.rsplit("/", 1)[1]]["rows"])
        return Api()

    def Settings(self, **kwargs):
        return SimpleNamespace(**kwargs)

    def init(self, **kwargs):
        self.inits.append(kwargs)
        run_id = kwargs.get("id") or self.run_id
        starting_step = self.starting_step
        if self.server is not None:
            held = self.server.setdefault(run_id, dict(rows=[], summary={}))
            starting_step = max([s for s, _ in held["rows"]], default=-1) + 1
            self.summary = held["summary"]
        self.run = SimpleNamespace(id=run_id, project=kwargs.get("project"),
                                   entity=kwargs.get("entity"), name=kwargs.get("name"), group=kwargs.get("group"),
                                   url=f"https://wandb.test/{run_id}", starting_step=starting_step, offline=self.offline)

    def define_metric(self, name, step_metric=None, hidden=None):
        self.metrics.append((name, step_metric))
        if hidden:
            self.hidden.append(name)

    def log(self, row, step=None):
        self.logged.append((step, dict(row)))
        if self.server is not None and not (self.lose and self.lose(step, row)):
            rows = self.server[self.run.id]["rows"]
            assert not rows or step > rows[-1][0], f"step {step} after {rows[-1][0]}: W&B would drop it"
            rows.append((step, dict(row)))

    def finish(self, exit_code=0):
        self.finished.append(exit_code)
        self.run = None


def shared_record(run, task="walk", seed=0):
    """The shared W&B run record of an exploration run, task and adaptation seed."""
    return json.loads((Path(run) / f"downstream/wandb_{task}_seed_{seed}.json").read_text())


ONLINE = ["wandb.mode=online", "wandb.project=test-project"]
REWARD_DEBUG_KEYS = {f"reward_model/train/debug/{k}" for k in ("loss_total", "loss_end", "grad_norm", "prediction_mae")}


def downstream_ac_keys(ac_schema, action_dim, debug):
    """The downstream actor_critic/train/ keys: the accepted exploration schema without the advantage-weight
    statistics (sign-only PMPO) and intrinsic rewards, with the predicted task rewards instead."""
    keys = {k for k in ac_schema(action_dim, debug) if not k.startswith(("weights/", "intrinsic/", "debug/weight_",
                                                                        "debug/intrinsic_"))}
    keys |= {"task_reward/reward_mean", "task_reward/reward_max"}
    if debug:
        keys |= {f"debug/task_reward_{s}" for s in ("std", "min", "p10", "median", "p90", "p99")}
    return {f"actor_critic/train/{k}" for k in keys}


def logged_run(milestone, monkeypatch, *extra, fake=None):
    fake = fake or FakeWandb()
    monkeypatch.setattr(downstream_module, "wandb", fake)
    trainer = make_trainer(milestone, monkeypatch, extra=[*ONLINE, *extra])
    output = trainer.run()
    return trainer, output, fake


@pytest.mark.parametrize("debug", [False, True])
def test_wandb_rows_keys_axes_and_evaluation(tmp_path, monkeypatch, ac_schema, debug):
    milestone = build_milestone(tmp_path)
    flags = [f"rew_end_model.debug_metrics={debug}", f"agent.actor_critic.debug_metrics={debug}"]
    trainer, output, fake = logged_run(milestone, monkeypatch, *flags, "evaluation.after_training=true",
                                       "evaluation.seeds=[3001,3002,3005]", "evaluation.max_steps=3")
    steps = [step for step, _ in fake.logged]
    assert steps == list(range(4 + 3 + 1))  # one row per update (reward model, then PMPO), then the evaluation
    reward, policy, (evaluation,) = [r for _, r in fake.logged[:4]], [r for _, r in fake.logged[4:7]], [r for _, r in fake.logged[7:]]

    reward_keys = {f"reward_model/train/core/{k}" for k in ("loss_reward", "predicted_reward_mean", "target_reward_mean",
                                                             "num_updates", "cumulative_updates")} | \
        (REWARD_DEBUG_KEYS if debug else set()) | {"exploration/train_env_steps"}
    assert all(set(r) == reward_keys for r in reward) and [r[downstream_module.REWARD_AXIS] for r in reward] == [1, 2, 3, 4]
    assert [r[downstream_module.REWARD_CUMULATIVE_AXIS] for r in reward] == [1, 2, 3, 4]  # the run's first milestone
    ac_keys = downstream_ac_keys(ac_schema, 6, debug)
    assert len(ac_keys) == (132 if debug else 39)
    assert all(set(r) == ac_keys | {downstream_module.POLICY_CUMULATIVE_AXIS, "exploration/train_env_steps"}
               for r in policy) and [r[downstream_module.POLICY_AXIS] for r in policy] == [1, 2, 3]
    assert all(r["exploration/train_env_steps"] == 26 for r in reward + policy)  # identifies the milestone
    published = set().union(*map(set, (r for _, r in fake.logged)))
    assert {k.split("/", 1)[0] for k in published} == {"reward_model", "actor_critic", "eval", "exploration"}
    assert not any("intrinsic" in k or "weights/" in k or "weight_" in k for k in published)

    # loss_reward is loss_rew (the reward MSE), not loss_total; values equal metrics.jsonl's.
    rows = [json.loads(line) for line in (output / "metrics.jsonl").read_text().splitlines()]
    assert [r["reward_model/train/core/loss_reward"] for r in reward] == [r["loss_rew"] for r in rows[:4]]
    assert all(r["loss_rew"] != r["loss_total"] for r in rows[:4])
    if debug:
        assert [r["reward_model/train/debug/grad_norm"] for r in reward] == [r["grad_norm_before_clip"] for r in rows[:4]]

    # Evaluation: one point at the milestone's recorded data budget, under its protocol; seed keys from the
    # configured seeds.
    (file,) = (output / "evaluations").iterdir()
    saved = OmegaConf.to_container(OmegaConf.load(file))
    # The run's primary protocol publishes directly under eval/.
    protocol = "protocol-" + analysis_module.protocol_id(file)
    assert evaluation == {**{f"eval/seed_{e['seed']}/return": e["episode_return"] for e in saved["episodes"]},
                          **{f"eval/return_{k}": saved[f"return_{k}"] for k in ("max", "median", "mean", "min", "std")},
                          "exploration/train_env_steps": 26}
    assert sorted(k for k in evaluation if "/seed_" in k) == [f"eval/seed_{s}/return" for s in (3001, 3002, 3005)]
    assert fake.summary[f"evaluation_protocols/{protocol}"] == dict(seeds=[3001, 3002, 3005], max_steps=3, namespace="eval")
    assert saved["protocol_id"] == analysis_module.protocol_id(file)  # in the local report too

    # Axes: training metrics on the cumulative update axes, evaluations on the training env steps (all hidden).
    assert fake.metrics == [(downstream_module.REWARD_CUMULATIVE_AXIS, None),
                            ("reward_model/train/*", downstream_module.REWARD_CUMULATIVE_AXIS),
                            (downstream_module.POLICY_CUMULATIVE_AXIS, None),
                            ("actor_critic/train/*", downstream_module.POLICY_CUMULATIVE_AXIS),
                            ("exploration/train_env_steps", None), ("eval/*", "exploration/train_env_steps"),
                            ("eval_protocols/*", "exploration/train_env_steps")]
    assert fake.hidden == [downstream_module.REWARD_CUMULATIVE_AXIS, downstream_module.POLICY_CUMULATIVE_AXIS,
                           "exploration/train_env_steps"]

    # Run identity (the shared run's record), names and provenance; status; the identity file travels with the output.
    record = shared_record(tmp_path / "run")
    (init,) = fake.inits
    default = "uniform-walker-exploration-seed-unrecorded-walk-adaptation-seed0"  # the manifest records no seed
    assert (init["id"], init["resume"], init["name"], init["group"], init["job_type"], init["mode"]) == (
        record["identity"]["id"], "allow", default, default, "adaptation", "online")
    provenance = init["config"]["provenance"]
    assert provenance["exploration"]["method"] == "not recorded" and provenance["exploration"]["collection_mode"] == "uniform"
    assert provenance["exploration"]["seed"].startswith("not recorded")
    assert "milestone" not in provenance and "milestone_path" not in init["config"]["initialization"]  # per milestone:
    milestone_summary = fake.summary["milestones/steps_000026"]                                       # in the summary
    assert milestone_summary["train_env_steps"] == 26 and milestone_summary["member_index"] == 0
    assert milestone_summary["identity"] == trainer.milestone.manifest["identity"] and milestone_summary["status"] == "trained"
    assert milestone_summary["member_sha256"] == OmegaConf.load(output / "summary.json").milestone.member_sha256
    assert provenance["controller"]["actor_objective"] == {"mode": "sign"} and provenance["evaluation"]["seeds"] == [3001, 3002, 3005]
    identity = json.loads((output / "wandb_run.json").read_text())
    assert identity["id"] == record["identity"]["id"] and identity["shared_run"] == "../../../wandb_walk_seed_0.json"
    assert record["milestones"]["steps_000026"]["first_step"] == 0 and record["next_step"] == 8
    assert record["primary_protocol"] == dict(id=analysis_module.protocol_id(file), seeds=[3001, 3002, 3005], max_steps=3)
    assert fake.summary["evaluation_protocols/primary"] == record["primary_protocol"]
    assert init["config"]["provenance"]["evaluation"]["protocol"] == record["primary_protocol"]["id"]
    assert record["evaluations"] == {f"steps_000026/walk/seed_0/evaluations/{file.name}": dict(
        step=7, original_step=7, kind="after_training", status="unverified", attempted=True,
        namespace="eval", protocol=record["primary_protocol"],
        verification="unavailable (RuntimeError: this fake has no server history)")}
    assert fake.summary["status"] == "completed" and fake.finished == [0]
    assert fake.summary[f"evaluations/{file.stem}"]["kind"] == "after_training"


def test_reward_statistics_use_the_masked_transitions_and_task_rewards_pmpo_used(tmp_path, monkeypatch):
    trainer = make_trainer(build_milestone(tmp_path), monkeypatch)
    model = trainer.agent.rew_end_model
    model.debug_metrics = True
    batch = next(trainer._reward_batches)
    mask = batch.mask_padding[:, :-1]
    assert not mask.all()  # the windows reach beyond episode ends: padding to exclude
    with torch.no_grad():
        _, metrics = model(batch)
        pred, _, _ = model.predict_rew_end(batch.obs[:, :-1], batch.act[:, :-1], batch.obs[:, 1:])
    pred, target = pred[mask], batch.rew[:, :-1][mask]
    torch.testing.assert_close(metrics["loss_rew"], ((pred - target) ** 2).mean())
    torch.testing.assert_close(metrics["predicted_reward_mean"], pred.mean())
    torch.testing.assert_close(metrics["target_reward_mean"], target.mean())
    torch.testing.assert_close(metrics["prediction_mae"], (pred - target).abs().mean())
    model.debug_metrics = False
    assert "prediction_mae" not in model(batch)[1]  # skipped when off

    # Task rewards: exactly the tensor PMPO gave the lambda returns.
    import models.pmpo_beta as pmpo_module
    given, original = [], pmpo_module.compute_lambda_returns
    monkeypatch.setattr(pmpo_module, "compute_lambda_returns", lambda rew, *a: given.append(rew.clone()) or original(rew, *a))
    trainer.agent.actor_critic.debug_metrics = True
    trainer.phase = "actor_critic"
    trainer.policy_update()
    row = json.loads((trainer._work_dir / "metrics.jsonl").read_text().splitlines()[-1])
    (rew,) = given
    assert rew.shape == (2, 2)  # batch x horizon, every imagined step
    assert row["task_reward_mean"] == pytest.approx(rew.mean().item()) and row["task_reward_max"] == pytest.approx(rew.max().item())
    assert row["task_reward_p99"] == pytest.approx(torch.quantile(rew.flatten(), 0.99).item())
    assert row["task_reward_median"] == pytest.approx(torch.quantile(rew.flatten(), 0.5).item())


ONE_EVALUATION = ["evaluation.after_training=true", "evaluation.seeds=[3001]", "evaluation.max_steps=3"]


class InProcessChildren:
    """Runs the folder launcher's child commands in this process (their overrides composed as the child's Hydra
    would), against one fake W&B server, as separate processes would reach the same W&B runs in turn."""

    def __init__(self, monkeypatch, server, fail=None):
        self.monkeypatch, self.server, self.fail, self.fakes, self.commands = monkeypatch, server, fail, [], []

    def __call__(self, command, env=None):
        self.commands.append(command)
        overrides = [a for a in command[2:] if not a.startswith("--")]
        fake = FakeWandb(server=self.server)
        self.fakes.append(fake)
        self.monkeypatch.setattr(downstream_module, "wandb", fake)
        self.monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
        cfg = compose("downstream", overrides)
        try:
            if self.fail is not None:
                self.fail(len(self.commands))
            DownstreamTrainer(cfg).run()
        except Exception:  # a failed child process: nonzero exit
            return subprocess.CompletedProcess(command, 1)
        return subprocess.CompletedProcess(command, 0)


def folder_launch(milestones, monkeypatch, server, *extra, fail=None):
    """downstream_main.run_milestones with in-process children; the launcher itself must not start W&B."""
    overrides = [f"initialization.milestones_dir={Path(milestones).as_posix()}", "downstream.task_name=walk",
                 "common.devices=cpu", *TINY_DOWNSTREAM, *ONLINE, *ONE_EVALUATION, *extra]
    cfg = compose("downstream", overrides)
    folder, steps = downstream_main.pop_milestone_source(cfg)
    base = downstream_main.child_command_base(fake_hydra(overrides))
    parent = FakeWandb(server=server)
    parent.init = lambda **kwargs: pytest.fail("the folder launcher started a W&B run")
    monkeypatch.setattr(downstream_module, "wandb", parent)
    children = InProcessChildren(monkeypatch, server, fail)
    code = downstream_main.run_milestones(cfg, Path(folder), base, run=children, milestone_steps=steps)
    return code, children


def eval_points(rows):
    """(train env steps, mean return) of every evaluation row of a run's history, in step order."""
    return [(r["exploration/train_env_steps"], next(v for k, v in r.items() if k.endswith("/return_mean")))
            for _, r in rows if any(k.startswith(("eval/", "eval_protocols/")) for k in r)]


@pytest.mark.parametrize("selection", ["all", "subset", "single"])
def test_selected_milestones_share_one_run_with_cumulative_axes(tmp_path, monkeypatch, selection):
    """One W&B run for every milestone of one invocation (all of a folder, a milestone_steps subset, or one
    milestone_path); each milestone's children attach in turn and none gets a run of its own. Each milestone
    has its own block of steps, then one evaluation point at its training env steps."""
    milestones = build_milestone(tmp_path, early=(1, 2)).parent  # 10, 20, 26 steps
    server = {}
    if selection == "single":
        fake = FakeWandb(server=server)
        logged_run(milestones / "steps_000020", monkeypatch, *ONE_EVALUATION, fake=fake)
        fakes, expected = [fake], [20]
    else:
        extra = ["initialization.milestone_steps=[10,26]"] if selection == "subset" else []
        code, children = folder_launch(milestones, monkeypatch, server, *extra)
        assert code == 0
        fakes, expected = children.fakes, [10, 26] if selection == "subset" else [10, 20, 26]
    record = shared_record(tmp_path / "run")
    run_id = record["identity"]["id"]
    assert list(server) == [run_id]  # one run on the server, attached once per milestone
    assert [f.inits[0]["id"] for f in fakes] == [run_id] * len(expected) and all(f.finished == [0] for f in fakes)
    rows = server[run_id]["rows"]
    assert [s for s, _ in rows] == list(range(8 * len(expected)))  # each block: 4 + 3 updates, then its evaluation
    assert [steps for steps, _ in eval_points(rows)] == expected  # one point per selected milestone
    assert sorted(e["collected_steps"] for e in record["milestones"].values()) == expected
    reward = [r for _, r in rows if downstream_module.REWARD_AXIS in r]
    policy = [r for _, r in rows if downstream_module.POLICY_AXIS in r]
    # Cumulative axes across milestones; own counters restart (every milestone trains fresh models).
    assert [r[downstream_module.REWARD_CUMULATIVE_AXIS] for r in reward] == list(range(1, 4 * len(expected) + 1))
    assert [r[downstream_module.REWARD_AXIS] for r in reward] == [1, 2, 3, 4] * len(expected)
    assert [r[downstream_module.POLICY_CUMULATIVE_AXIS] for r in policy] == list(range(1, 3 * len(expected) + 1))
    assert [r["exploration/train_env_steps"] for r in policy] == [s for s in expected for _ in range(3)]
    # Every adaptation keeps its own local outputs and counters.
    for steps in expected:
        output = tmp_path / f"run/downstream/steps_{steps:06d}/walk/seed_0"
        assert json.loads((output / "summary.json").read_text())["num_updates"] == {"rew_end_model": 4, "actor_critic": 3}
        assert [json.loads(line)["update"] for line in (output / "metrics.jsonl").read_text().splitlines()] == [1, 2, 3, 4, 1, 2, 3]
        assert json.loads((output / "wandb_run.json").read_text())["id"] == run_id
    assert sorted(p.name for p in (tmp_path / "run/downstream").iterdir()) == sorted(
        [f"steps_{s:06d}" for s in expected] + ["wandb_walk_seed_0.json"])


def test_explicit_run_names_are_used_exactly_and_a_different_one_is_refused(tmp_path, monkeypatch):
    milestones = build_milestone(tmp_path, early=(1, 2)).parent
    server = {}
    names = ["wandb.name=lcg-walker--downstream-walk-es0-ds0", "wandb.group=lcg-walker--downstream-walk-es0-ds0"]
    code, children = folder_launch(milestones, monkeypatch, server, *names, "initialization.milestone_steps=[10,20]")
    assert code == 0
    assert {(f.inits[0]["name"], f.inits[0]["group"]) for f in children.fakes} == {
        ("lcg-walker--downstream-walk-es0-ds0", "lcg-walker--downstream-walk-es0-ds0")}  # no suffixes appended
    record = shared_record(tmp_path / "run")
    assert (record["identity"]["name"], record["identity"]["group"], record["identity"]["project"]) == (
        "lcg-walker--downstream-walk-es0-ds0", "lcg-walker--downstream-walk-es0-ds0", "test-project")
    # A later invocation for the same exploration run, task and seed joins that run; another explicit name is
    # refused before anything is trained.
    before = tree_hashes(tmp_path / "run")
    with pytest.raises(ValueError, match="records the W&B run 'lcg-walker--downstream-walk-es0-ds0'.*wandb.name='other'"):
        folder_launch(milestones, monkeypatch, server, "wandb.name=other", "initialization.milestone_steps=[26]")
    assert tree_hashes(tmp_path / "run") == before
    code, children = folder_launch(milestones, monkeypatch, server, *names, "initialization.milestone_steps=[26]")
    assert code == 0 and children.fakes[0].inits[0]["id"] == record["identity"]["id"]
    assert [steps for steps, _ in eval_points(server[record["identity"]["id"]]["rows"])] == [10, 20, 26]


def test_shared_logging_changes_no_training(tmp_path, monkeypatch):
    """Two milestones adapted into one shared run versus with W&B off: the later milestone's weights, metrics
    and evaluation are identical (no warm start, no RNG draw by logging)."""
    on = build_milestone(tmp_path / "on", early=(2,)).parent
    off = build_milestone(tmp_path / "off", early=(2,)).parent
    assert folder_launch(on, monkeypatch, {})[0] == 0
    cfg, folder, base, _ = folder_cfg(off, *ONE_EVALUATION)
    monkeypatch.setattr(downstream_module, "wandb", FakeWandb())
    assert downstream_main.run_milestones(cfg, folder, base, run=InProcessChildren(monkeypatch, {})) == 0
    assert not (tmp_path / "off/run/downstream/wandb_walk_seed_0.json").exists()
    for name in ("steps_000020", "steps_000026"):
        a, b = (root / f"run/downstream/{name}/walk/seed_0" for root in (tmp_path / "on", tmp_path / "off"))
        for weights in ("rew_end_model.pt", "actor_critic.pt"):
            assert states_equal(torch.load(a / weights, weights_only=False), torch.load(b / weights, weights_only=False))
        assert (a / "metrics.jsonl").read_bytes() == (b / "metrics.jsonl").read_bytes()
        (x,), (y,) = (a / "evaluations").iterdir(), (b / "evaluations").iterdir()
        assert x.name == y.name and x.read_bytes() == y.read_bytes()


def test_evaluate_only_adds_to_the_adaptation_run_without_the_milestone(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    trainer, output, first = logged_run(milestone, monkeypatch, "evaluation.after_training=true",
                                        "evaluation.seeds=[3001,3002]", "evaluation.max_steps=3")
    shutil.move(str(tmp_path / "run/milestones"), str(tmp_path / "moved_milestones"))
    shutil.move(str(tmp_path / "run/dataset"), str(tmp_path / "moved_dataset"))
    cfg = compose("downstream", [f"initialization.milestone_path={milestone.as_posix()}", "downstream.task_name=walk",
                                 "downstream.mode=evaluate", *TINY_DOWNSTREAM, *ONLINE, "evaluation.seeds=[5,9]",
                                 "evaluation.max_steps=3"])
    fake = FakeWandb(starting_step=8)
    monkeypatch.setattr(downstream_module, "wandb", fake)
    path = downstream_module.evaluate_saved_output(cfg)
    record = shared_record(tmp_path / "run")
    (init,) = fake.inits
    assert init["id"] == record["identity"]["id"] and init["resume"] == "allow"  # the shared run
    ((step, row),) = fake.logged
    saved = OmegaConf.to_container(OmegaConf.load(path))
    first = "protocol-" + analysis_module.protocol_id(next(p for p in (output / "evaluations").iterdir() if p != path))
    second = "protocol-" + analysis_module.protocol_id(path)
    assert first != second  # another protocol: its own namespace, never the primary eval/ curves
    assert step == 8 and row["exploration/train_env_steps"] == 26
    assert row[f"eval_protocols/{second}/seed_9/return"] == saved["episodes"][1]["episode_return"]
    assert not any(k.startswith("eval/") for k in row)
    assert fake.summary[f"evaluations/{path.stem}"]["kind"] == "evaluate_only" and fake.finished == [0]
    assert fake.summary[f"evaluation_protocols/{second}"]["namespace"] == f"eval_protocols/{second}"
    assert record["evaluations"][f"steps_000026/walk/seed_0/evaluations/{path.name}"] == dict(
        step=8, original_step=8, kind="evaluate_only", status="unverified", attempted=True,
        namespace=f"eval_protocols/{second}", protocol=downstream_module.configured_protocol(cfg),
        verification="unavailable (RuntimeError: this fake has no server history)") and record["next_step"] == 9
    with pytest.raises(FileExistsError, match="not overwriting"):  # local overwrite protection unchanged
        downstream_module.evaluate_saved_output(cfg)

    # An output without a W&B identity (e.g. trained with W&B off): a new, labelled evaluate-only run.
    (output / "wandb_run.json").unlink()
    (tmp_path / "run/downstream/wandb_walk_seed_0.json").unlink()
    cfg.evaluation.seeds = [7]
    fake = FakeWandb(run_id="evalonly")
    monkeypatch.setattr(downstream_module, "wandb", fake)
    downstream_module.evaluate_saved_output(cfg)
    (init,) = fake.inits
    assert "id" not in init and init["job_type"] == "evaluate-only" and init["name"] == (  # milestone and run configuration gone: only the saved domain is known
        "unrecorded-walker-exploration-seed-unrecorded-walk-adaptation-seed0-evaluate-only")
    assert init["config"]["provenance"]["exploration"]["method"] == "not recorded in the saved output"
    assert init["config"]["provenance"]["milestone"]["collected_steps"] == 26
    assert fake.logged[0][0] == 7 + 2  # after the training updates and the two earlier evaluation files



class EvaluationFailure(Exception):
    pass


def failing_evaluation(monkeypatch):
    def fail(*args, **kwargs):
        raise EvaluationFailure("evaluation crashed")
    monkeypatch.setattr(downstream_module, "evaluate_policy", fail)


def evaluate_only_cfg(milestone, *extra):
    return compose("downstream", [f"initialization.milestone_path={milestone.as_posix()}", "downstream.task_name=walk",
                                  "downstream.mode=evaluate", *TINY_DOWNSTREAM, *ONLINE, "evaluation.max_steps=3", *extra])


def history(summary):
    return [v for k, v in sorted(summary.items()) if k.startswith("status_history/")]


def test_evaluate_only_status_after_a_failed_final_evaluation_of_an_adaptation_run(tmp_path, monkeypatch):
    """The adaptation run's final evaluation fails (outputs published, run failed); evaluate-only fails
    once more, then succeeds: the run ends completed with the evaluation, both failures kept as history."""
    milestone = build_milestone(tmp_path)
    original = downstream_module.evaluate_policy
    failing_evaluation(monkeypatch)
    first = FakeWandb()
    monkeypatch.setattr(downstream_module, "wandb", first)
    trainer = make_trainer(milestone, monkeypatch, extra=[*ONLINE, "evaluation.after_training=true",
                                                          "evaluation.seeds=[3001,3002]", "evaluation.max_steps=3"])
    with pytest.raises(EvaluationFailure):
        trainer.run()
    output = tmp_path / "run/downstream/steps_000026/walk/seed_0"
    assert (output / "summary.json").is_file() and not (output / "evaluations").exists()
    assert (first.summary["status"], first.summary["activity"], first.finished) == ("failed", "adaptation", [1])
    assert "evaluation crashed" in first.summary["error"]

    # Evaluate-only fails too (the server's summary is the first run's): failed again, the earlier one is history.
    second = FakeWandb(starting_step=7, summary=first.summary)
    monkeypatch.setattr(downstream_module, "wandb", second)
    with pytest.raises(EvaluationFailure):
        downstream_module.evaluate_saved_output(evaluate_only_cfg(milestone, "evaluation.seeds=[3001,3002]"))
    run_id = shared_record(tmp_path / "run")["identity"]["id"]
    assert second.inits[0]["id"] == run_id and second.finished == [1] and not second.logged
    assert (second.summary["status"], second.summary["activity"]) == ("failed", "evaluate_only")
    assert history(second.summary) == [dict(status="failed", activity="adaptation", error=first.summary["error"])]

    # The retry succeeds: completed, the evaluation recorded, no current error, both failures in the history.
    monkeypatch.setattr(downstream_module, "evaluate_policy", original)
    third = FakeWandb(starting_step=7, summary=second.summary)
    monkeypatch.setattr(downstream_module, "wandb", third)
    path = downstream_module.evaluate_saved_output(evaluate_only_cfg(milestone, "evaluation.seeds=[3001,3002]"))
    assert (third.summary["status"], third.summary["activity"], third.finished) == ("completed", "evaluate_only", [0])
    assert "error" not in third.summary
    assert [h["activity"] for h in history(third.summary)] == ["adaptation", "evaluate_only"]
    ((step, row),) = third.logged
    # The next free step of the shared run (the failed after-training evaluation left its reserved step 7 unused);
    # the same protocol as the run's primary one: under eval/.
    assert step == 8 and row["exploration/train_env_steps"] == 26
    assert set(row) >= {"eval/seed_3001/return", "eval/return_mean"}
    assert third.summary[f"evaluations/{path.stem}"]["kind"] == "evaluate_only"

    # The same policy and protocol again: refused before anything runs; W&B is not touched.
    fourth = FakeWandb(summary=third.summary)
    monkeypatch.setattr(downstream_module, "wandb", fourth)
    with pytest.raises(FileExistsError, match="not overwriting"):
        downstream_module.evaluate_saved_output(evaluate_only_cfg(milestone, "evaluation.seeds=[3001,3002]"))
    assert not fourth.inits


def test_standalone_evaluate_only_run_status(tmp_path, monkeypatch):
    """An output without a W&B identity: each evaluate-only is its own run, running then failed or completed."""
    milestone = build_milestone(tmp_path)
    make_trainer(milestone, monkeypatch).run()  # W&B off: no wandb_run.json
    original = downstream_module.evaluate_policy
    failing_evaluation(monkeypatch)
    failed = FakeWandb(run_id="standalone1")
    monkeypatch.setattr(downstream_module, "wandb", failed)
    with pytest.raises(EvaluationFailure):
        downstream_module.evaluate_saved_output(evaluate_only_cfg(milestone, "evaluation.seeds=[5]"))
    assert failed.inits[0]["job_type"] == "evaluate-only" and "id" not in failed.inits[0]
    assert (failed.summary["status"], failed.summary["activity"], failed.finished) == ("failed", "evaluate_only", [1])
    assert "evaluation crashed" in failed.summary["error"] and not failed.logged

    monkeypatch.setattr(downstream_module, "evaluate_policy", original)
    retry = FakeWandb(run_id="standalone2")
    monkeypatch.setattr(downstream_module, "wandb", retry)
    downstream_module.evaluate_saved_output(evaluate_only_cfg(milestone, "evaluation.seeds=[5]"))
    assert (retry.summary["status"], retry.summary["activity"], retry.finished) == ("completed", "evaluate_only", [0])
    assert "error" not in retry.summary and not history(retry.summary) and len(retry.logged) == 1

# ---------------------------------------------------------------- metadata format (JSON only)

def test_milestone_metadata_is_json_only(tmp_path):
    milestones = build_milestone(tmp_path, early=(2,)).parent
    new = milestones / "steps_000026"
    assert sorted(p.name for p in new.iterdir()) == ["dataset.pt", "denoiser.pt", "manifest.json"]
    manifest = json.loads((new / "manifest.json").read_text(encoding="utf-8"))  # standard JSON
    assert manifest == read_manifest(new) and manifest["collected_steps"] == 26
    assert load_milestone(new, "walk").manifest == manifest
    assert [(p.name, m["collected_steps"]) for p, m in discover_milestones(milestones)] == [("steps_000020", 20),
                                                                                            ("steps_000026", 26)]
    cfg, folder, base, _ = folder_cfg(milestones)
    run = FakeRun()
    assert downstream_main.run_milestones(cfg, folder, base, run=run) == 0 and len(run.commands) == 2

    # A milestone with only generated YAML metadata (written before 2026-10-29) is not read: a clear
    # error, and the old file is left as it is.
    old = milestones / "steps_000020"
    data = json.loads((old / "manifest.json").read_text(encoding="utf-8"))
    (old / "manifest.yaml").write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    (old / "manifest.json").unlink()
    before = tree_hashes(old)
    with pytest.raises(FileNotFoundError, match="No milestone manifest at .*manifest.json \\(only manifest.yaml, "
                                                "generated YAML metadata from before 2026-10-29, which is no longer read\\)"):
        read_manifest(old)
    with pytest.raises(ValueError, match="Malformed milestones folder(.|\\n)*steps_000020: No milestone manifest"):
        discover_milestones(milestones)
    with pytest.raises(FileNotFoundError, match="No milestone manifest"):
        load_milestone(old, "walk")
    assert tree_hashes(old) == before
    shutil.rmtree(old)

    # Malformed or non-standard JSON is rejected; checksums are still checked.
    text = (new / "manifest.json").read_text(encoding="utf-8")
    for bad, message in (('{"schema_version": 1,', "is not valid JSON"),
                         (text.replace('"epoch": 2', '"epoch": NaN', 1), "non-standard JSON constant NaN"),
                         ("[1, 2]", "does not hold a JSON object")):
        (new / "manifest.json").write_text(bad, encoding="utf-8")
        with pytest.raises(ValueError, match=message):
            read_manifest(new)
    (new / "manifest.json").write_text(text, encoding="utf-8")
    (new / "denoiser.pt").write_bytes((new / "denoiser.pt").read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match its checksum"):
        load_milestone(new, "walk")
    with pytest.raises(ValueError, match="Out of range float values are not JSON compliant"):
        write_json(tmp_path / "x.json", dict(value=float("inf")))  # never written as non-standard JSON
    assert not (tmp_path / "x.json").exists()


def test_downstream_outputs_are_json_only(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    trainer, output, first = logged_run(milestone, monkeypatch, "evaluation.after_training=true",
                                        "evaluation.seeds=[5,6]", "evaluation.max_steps=3")
    (report,) = (output / "evaluations").iterdir()
    assert report.suffix == ".json" and sorted(p.name for p in output.iterdir()) == [
        "actor_critic.pt", "config.yaml", "evaluations", "metrics.jsonl", "rew_end_model.pt", "summary.json", "wandb_run.json"]
    stored = json.loads(report.read_text(encoding="utf-8"), parse_constant=lambda name: 1 / 0)
    assert stored["seeds"] == [5, 6] and len(stored["episodes"]) == 2 and stored["policy_sha256_prefix"] in report.stem
    summary = json.loads((output / "summary.json").read_text(encoding="utf-8"), parse_constant=lambda name: 1 / 0)
    assert summary["num_updates"] == {"rew_end_model": 4, "actor_critic": 3}

    # Evaluate-only from the JSON output, with the source milestone and replay moved away.
    shutil.move(str(tmp_path / "run/milestones"), str(tmp_path / "moved_milestones"))
    shutil.move(str(tmp_path / "run/dataset"), str(tmp_path / "moved_dataset"))
    cfg = compose("downstream", [f"initialization.milestone_path={milestone.as_posix()}", "downstream.task_name=walk",
                                 "downstream.mode=evaluate", *TINY_DOWNSTREAM, *ONLINE, "evaluation.max_steps=3",
                                 "evaluation.seeds=[5,6]"])
    assert downstream_main.saved_output_milestone(cfg, milestone) == dict(domain="walker", collected_steps=26)
    assert "training outputs were published" in downstream_main.failure_state(milestone, output, ["python"])
    fake = FakeWandb(starting_step=8)
    monkeypatch.setattr(downstream_module, "wandb", fake)
    before = report.read_bytes()
    with pytest.raises(FileExistsError, match=f"{report.name} already exists; not overwriting"):
        downstream_module.evaluate_saved_output(cfg)  # the same evaluation: refused before W&B starts
    assert not fake.inits and report.read_bytes() == before
    cfg.evaluation.seeds = [7]
    path = downstream_module.evaluate_saved_output(cfg)
    assert path.suffix == ".json" and fake.logged[0][0] == 7 + 1  # after the one earlier report
    assert fake.summary["status"] == "completed"

    # Malformed summary.json: an error; an output with only a generated summary.yaml is not a completed output.
    text = (output / "summary.json").read_text(encoding="utf-8")
    (output / "summary.json").write_text("{", encoding="utf-8")
    cfg.evaluation.seeds = [8]
    with pytest.raises(ValueError, match="summary.json is not valid JSON"):
        downstream_module.evaluate_saved_output(cfg)
    (output / "summary.yaml").write_text(yaml.safe_dump(json.loads(text)), encoding="utf-8")
    (output / "summary.json").unlink()
    with pytest.raises(FileNotFoundError, match="No completed downstream output at .* \\(no summary.json\\)"):
        downstream_module.evaluate_saved_output(cfg)
    assert downstream_main.saved_output_milestone(cfg, milestone) is None
    assert sorted(p.name for p in (output / "evaluations").iterdir()) == sorted([report.name, path.name])

# ---------------------------------------------------------------- result collection and comparison (CPU only)

EVAL_SEEDS = list(range(3001, 3031))  # the production 30-seed protocol
METHOD_KEYS = {"uniform": None, "lcg": "lcg", "full-ensemble": "ensemble_disagreement",
               "laplace-ensemble": "laplace_disagreement", "plan2explore": "plan2explore_disagreement"}


def synthetic_returns(steps, n):
    """Deterministic returns per milestone, spread over the thresholds."""
    return [float((steps // 10 + 37 * i) % 700) + 0.25 for i in range(n)]


def synthetic_output(run, steps, returns, seeds=EVAL_SEEDS, max_steps=500, task="walk", seed=0, member=0,
                     settings_steps=5, action_repeat=2):
    """One milestone of a synthetic exploration run and its completed downstream output (JSON metadata,
    saved config and one evaluation report), as the trainers write them."""
    milestone = run / f"milestones/steps_{steps:06d}"
    milestone.mkdir(parents=True, exist_ok=True)
    env = dict(domain_name="walker", task_names=["stand", "walk", "run"], size=64, camera_id=0, action_repeat=action_repeat,
               time_limit=None)
    denoiser = dict(architecture="synthetic")
    identity = dict(denoiser_sha256=f"denoiser-{steps}", dataset_sha256=f"index-{steps}", collected_steps=steps)
    if member:
        identity["member_sha256"] = [identity["denoiser_sha256"], *(f"denoiser-{steps}-{k}" for k in range(1, member + 1))]
    if not (milestone / "manifest.json").exists():
        method = METHOD_KEYS[run.parents[1].name]
        write_json(milestone / "manifest.json", dict(schema_version=1, collected_steps=steps, identity=identity,
                   domain="walker", task_names=env["task_names"], model=dict(env=env, denoiser=denoiser),
                   exploration=dict(intrinsic_reward_enabled=method is not None, method=method or "lcg")))
    elif member:  # this fixture can add a selected member's result to an existing synthetic milestone
        manifest = read_manifest(milestone)
        manifest["identity"] = identity
        write_json(milestone / "manifest.json", manifest)
    output = downstream_module.downstream_output_dir(milestone, task, seed, member)
    output.mkdir(parents=True, exist_ok=True)
    member_sha = identity.get("member_sha256", [identity["denoiser_sha256"]])[member]
    write_json(output / "summary.json", dict(milestone=dict(collected_steps=steps, identity=identity, member_index=member,
                                             member_sha256=member_sha, path=Path(os.path.relpath(milestone, output)).as_posix()),
                                             task_name=task, seed=seed, num_updates=dict(rew_end_model=5, actor_critic=5)))
    OmegaConf.save(OmegaConf.create(dict(env=dict(train=env), agent=dict(denoiser=denoiser, actor_critic=dict(actor_objective="sign",
                                         actor_distribution="beta")), downstream=dict(task_name=task, seed=seed),
                                         initialization=dict(member_index=member),
                                         world_model_env=dict(horizon=10),
                                         rew_end_model=dict(training=dict(steps=settings_steps), optimizer=dict(lr=1e-4)),
                                         actor_critic=dict(training=dict(steps=5)), checkpointing=dict(every=100))),
                   output / "config.yaml")
    values = np.array(returns, dtype=float)
    result = dict(task_name=task, domain_name="walker", seeds=list(seeds), max_steps=max_steps,
                  action_repeat=action_repeat, policy="deterministic (Beta mean)",
                  return_convention=downstream_module.RETURN_CONVENTION,
                  episodes=[dict(seed=s, episode_return=float(r), length=max_steps) for s, r in zip(seeds, values)],
                  return_mean=float(values.mean()), return_median=float(np.median(values)), return_std=float(values.std()),
                  return_min=float(values.min()), return_max=float(values.max()))
    policy_state = {"w": torch.tensor([steps, member])}
    torch.save(policy_state, output / "actor_critic.pt")
    path = downstream_module.evaluation_path(output, task, seeds, max_steps, policy_state)
    result["policy_sha256_prefix"] = path.stem.split("_policy-")[1].split("_protocol-")[0]
    path.parent.mkdir(exist_ok=True)
    write_json(path, result)
    return output, path


def synthetic_run(root, method, steps=(300, 400, 500), exploration_seed=0, **kwargs):
    run = root / "outputs" / method / "walker" / f"seed_{exploration_seed}"
    (run / "config").mkdir(parents=True, exist_ok=True)
    env = dict(domain_name="walker", task_names=["stand", "walk", "run"], size=64, camera_id=0,
               action_repeat=kwargs.get("action_repeat", 2), time_limit=None)
    OmegaConf.save(OmegaConf.create(dict(common=dict(seed=exploration_seed), env=dict(train=env),
                                         agent=dict(denoiser=dict(architecture="synthetic")),
                                         intrinsic_reward=dict(enabled=METHOD_KEYS[method] is not None,
                                                               method=METHOD_KEYS[method] or "lcg"))),
                   run / "config/exploration.yaml")
    for s in steps:
        synthetic_output(run, s, synthetic_returns(s, len(kwargs.get("seeds", EVAL_SEEDS))), **kwargs)
    return run


def test_result_table_statistics_strict_thresholds_and_30_seeds(tmp_path):
    run = synthetic_run(tmp_path, "plan2explore", steps=(500, 300))
    at_threshold = [200.0, 400.0, 500.0, 600.0, 200.5, 600.25] + [100.0] * 24  # returns exactly at the thresholds
    synthetic_output(run, 400, at_threshold)
    before = tree_hashes(run)
    path, rows, missing = analysis_module.collect_results(run, "walk", 0)
    assert path == run / "result/walk/seed_0/result.csv" and not missing
    with open(path, encoding="utf-8", newline="") as f:
        header = next(csv.reader(f))
    assert header == analysis_module.RESULT_COLUMNS + [f"return_seed_{s}" for s in EVAL_SEEDS] + ["source_evaluation_file"]
    table = analysis_module.read_csv(path)
    assert [r["train_env_steps"] for r in table] == [300, 400, 500] and [r["milestone"] for r in table] == [
        "steps_000300", "steps_000400", "steps_000500"]
    row = table[1]
    assert (row["method"], row["domain"], row["exploration_seed"], row["task"], row["adaptation_seed"],
            row["member_index"], row["num_eval_episodes"]) == ("plan2explore", "walker", 0, "walk", 0, 0, 30)
    values = np.array(at_threshold)
    assert row["return_mean"] == float(values.mean()) and row["return_std"] == float(values.std())
    assert (row["return_median"], row["return_min"], row["return_max"]) == (100.0, 100.0, 600.25)
    # Strictly greater: 200 is not over 200, 600 is not over 600.
    assert (row["frac_over_200"], row["frac_over_400"], row["frac_over_500"], row["frac_over_600"]) == (
        5 / 30, 3 / 30, 2 / 30, 1 / 30)
    assert [row[f"return_seed_{s}"] for s in EVAL_SEEDS] == at_threshold
    report = run / row["source_evaluation_file"]
    assert report.is_file() and report.suffix == ".json" and row["evaluation_max_steps"] == 500
    assert {k: v for k, v in tree_hashes(run).items() if "result" not in Path(k).relative_to(run).parts} == before

    # Saved aggregates that disagree with the episodes are rejected.
    data = json.loads(report.read_text(encoding="utf-8"))
    data["return_mean"] += 1.0
    report.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(ValueError, match="saved return_mean .* differs from the episodes"):
        analysis_module.collect_results(run, "walk", 0)


def test_result_collection_missing_ambiguous_protocols_settings_and_members(tmp_path):
    run = synthetic_run(tmp_path, "uniform", steps=(300, 400, 500, 600), seeds=[3001, 3002], max_steps=50)
    shutil.rmtree(run / "downstream/steps_000400")  # never adapted
    work = run / "downstream/steps_000500/walk"
    shutil.move(str(work / "seed_0"), str(work / ".seed_0.tmp"))  # interrupted
    path, rows, missing = analysis_module.collect_results(run, "walk", 0)
    assert [r["train_env_steps"] for r in rows] == [300, 600]
    assert [(m["milestone"], m["reason"]) for m in missing] == [("steps_000400", "no downstream output"),
                                                                ("steps_000500", "incomplete (work folder .seed_0.tmp only)")]
    # A second evaluation protocol on one milestone: ambiguous until one is selected.
    output, extra = synthetic_output(run, 600, [1.0, 2.0, 3.0], seeds=[1, 2, 3], max_steps=50)
    with pytest.raises(ValueError, match="several evaluation protocols .*select one with analysis.protocol"):
        analysis_module.collect_results(run, "walk", 0)
    chosen = analysis_module.protocol_id(extra)
    path, rows, missing = analysis_module.collect_results(run, "walk", 0, protocol=chosen)
    assert [r["train_env_steps"] for r in rows] == [600] and [r["return_seed_2"] for r in rows] == [2.0]
    assert ("steps_000300", f"completed, but no evaluation with protocol {chosen}") in [(m["milestone"], m["reason"])
                                                                                       for m in missing]
    # Denoiser members are separate tables; different downstream settings are never combined.
    synthetic_output(run, 300, [5.0, 6.0], seeds=[3001, 3002], max_steps=50, member=1)
    path, rows, _ = analysis_module.collect_results(run, "walk", 0, member_index=1)
    assert path == run / "result/walk/seed_0_member_1/result.csv" and [r["member_index"] for r in rows] == [1]
    other = synthetic_run(tmp_path / "b", "uniform", steps=(300,), seeds=[3001, 3002], max_steps=50)
    synthetic_output(other, 400, [1.0, 2.0], seeds=[3001, 3002], max_steps=50, settings_steps=6)
    with pytest.raises(ValueError, match="not comparable: .* differ in rew_end_model.training.steps: 5 vs 6"):
        analysis_module.collect_results(other, "walk", 0)


@pytest.mark.parametrize("change", ["action_repeat", "exploration_action_repeat", "return_mean", "frac_over_600", "policy_weights",
                                  "report_task", "report_protocol", "report_policy", "milestone", "missing_report",
                                  "second_policy"])
def test_comparison_revalidates_sources_after_collection(tmp_path, monkeypatch, change):
    runs = [synthetic_run(tmp_path, method, steps=(300,)) for method in ("uniform", "plan2explore")]
    for run in runs:
        analysis_module.collect_results(run, "walk", 0)
    run = runs[1]
    output = run / "downstream/steps_000300/walk/seed_0"
    csv_path = run / "result/walk/seed_0/result.csv"
    (report_path,) = (output / "evaluations").iterdir()
    if change in ("action_repeat", "exploration_action_repeat"):
        config_path = output / "config.yaml" if change == "action_repeat" else run / "config/exploration.yaml"
        cfg = OmegaConf.load(config_path)
        cfg.env.train.action_repeat = 4
        OmegaConf.save(cfg, config_path)
    elif change in ("return_mean", "frac_over_600"):
        rows = analysis_module.read_csv(csv_path)
        rows[0][change] = 9999. if change == "return_mean" else 1.
        analysis_module.write_csv(csv_path, rows, list(rows[0]))
    elif change == "policy_weights":
        torch.save({"w": torch.tensor([-1, -1])}, output / "actor_critic.pt")
    elif change.startswith("report_"):
        report = json.loads(report_path.read_text(encoding="utf-8"))
        report[dict(report_task="task_name", report_protocol="max_steps", report_policy="policy_sha256_prefix")[change]] = \
            dict(report_task="stand", report_protocol=499, report_policy="000000000000")[change]
        write_json(report_path, report)
    elif change == "milestone":
        path = run / "milestones/steps_000300/manifest.json"
        manifest = read_manifest(path.parent)
        manifest["identity"]["denoiser_sha256"] = "another-model"
        write_json(path, manifest)
    elif change == "missing_report":
        report_path.unlink()
    else:
        extra = downstream_module.evaluation_path(output, "walk", EVAL_SEEDS, 500, {"w": torch.tensor([-9])})
        write_json(extra, json.loads(report_path.read_text(encoding="utf-8")))
    before = tree_hashes(*runs)
    monkeypatch.setattr(analysis_module, "plot_comparison", lambda spec: pytest.fail("published invalid comparison"))
    with pytest.raises(ValueError, match="stale or inconsistent result table .*recollect with"):
        analysis_module.compare_methods(tmp_path / "outputs", "walker", 0, "walk", 0)
    assert tree_hashes(*runs) == before
    assert not (tmp_path / "outputs/comparisons").exists()


def test_method_comparison_files_inputs_and_checks(tmp_path):
    root = tmp_path / "outputs"
    uniform = synthetic_run(tmp_path, "uniform", steps=(300, 400, 500))
    p2e = synthetic_run(tmp_path, "plan2explore", steps=(300, 400, 500))
    shutil.rmtree(p2e / "downstream/steps_000400")  # a milestone without a result: recorded, not interpolated
    for run in (uniform, p2e):
        assert analysis_module.collect_results(run, "walk", 0)[0] is not None
    before = tree_hashes(uniform, p2e)
    command = [sys.executable, str(ROOT / "src/downstream_analysis.py"), "downstream.mode=compare",
               f"analysis.output_root={root.as_posix()}", "analysis.domain=walker", "analysis.exploration_seed=0",
               "downstream.task_name=walk", "downstream.seed=0"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    out = root / "comparisons/walker/exploration_seed_0/walk/adaptation_seed_0"
    assert sorted(p.name for p in out.iterdir()) == ["analysis.json", "comparison.csv", "fraction_over_thresholds.pdf",
                                                    "fraction_over_thresholds.png", "task_return.pdf", "task_return.png"]
    assert all((out / name).stat().st_size > 1000 for name in ("task_return.png", "fraction_over_thresholds.pdf"))
    assert "Comparison: uniform, plan2explore | 1 missing results" in result.stdout
    table = analysis_module.read_csv(out / "comparison.csv")
    assert [(r["method"], r["train_env_steps"]) for r in table] == [("uniform", 300), ("uniform", 400), ("uniform", 500),
                                                                    ("plan2explore", 300), ("plan2explore", 500)]
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["methods"] == ["uniform", "plan2explore"]
    assert analysis["missing"] == [dict(method="plan2explore", train_env_steps=400, reason="no result row")]
    assert analysis["evaluation_protocol"]["seeds"] == EVAL_SEEDS and analysis["evaluation_protocol"]["max_steps"] == 500
    assert analysis["selection"] == dict(output_root=str(root.resolve()), domain="walker", exploration_seed=0, task="walk",
                                         adaptation_seed=0, member_index=0, protocol=None)
    p2e_input = next(i for i in analysis["inputs"] if i["method"] == "plan2explore")
    csv_path = p2e / "result/walk/seed_0/result.csv"
    assert p2e_input["result_csv_sha256"] == file_sha256(csv_path) and len(p2e_input["evaluations"]) == 2
    assert tree_hashes(uniform, p2e) == before  # analysis only reads its inputs

    # Methods come from the saved configuration: a second run recording the same method is ambiguous.
    shutil.copytree(uniform, root / "lcg/walker/seed_0")
    with pytest.raises(ValueError, match="Two runs record method uniform"):
        analysis_module.compare_methods(root, "walker", 0, "walk", 0)
    shutil.rmtree(root / "lcg")
    # Different evaluation protocols or downstream settings are not compared.
    synthetic_output(p2e, 600, [1.0, 2.0], seeds=[3001, 3002], max_steps=50)
    analysis_module.collect_results(p2e, "walk", 0, protocol=analysis_module.protocol_id(
        next((p2e / "downstream/steps_000600/walk/seed_0/evaluations").iterdir())))
    with pytest.raises(ValueError, match="different evaluation protocols"):
        analysis_module.compare_methods(root, "walker", 0, "walk", 0)


def test_comparison_needs_the_same_effective_environment_evaluation_and_training_settings(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis_module, "plot_comparison", lambda spec: None)
    root = tmp_path / "outputs"

    def compare_with(change):
        shutil.rmtree(root, ignore_errors=True)
        uniform, p2e = synthetic_run(tmp_path, "uniform"), synthetic_run(tmp_path, "plan2explore")
        for output in sorted(p2e.glob("downstream/*/walk/seed_0")):
            change(output)
        for run in (uniform, p2e):
            analysis_module.collect_results(run, "walk", 0)
        return analysis_module.compare_methods(root, "walker", 0, "walk", 0)

    def config_change(**changes):
        def change(output):
            cfg = OmegaConf.load(output / "config.yaml")
            for key, value in changes.items():
                OmegaConf.update(cfg, key, value, force_add=True)
            OmegaConf.save(cfg, output / "config.yaml")
        return change

    def report_change(key, value):
        def change(output):
            (report,) = (output / "evaluations").iterdir()
            data = json.loads(report.read_text(encoding="utf-8"))
            data[key] = value
            report.write_text(json.dumps(data), encoding="utf-8")
        return change

    for change, key in ((config_change(**{"env.train.action_repeat": 4}), "env.train.action_repeat"),
                        (config_change(**{"env.train.time_limit": 100}), "env.train.time_limit"),
                        (config_change(**{"env.train.camera_id": 1}), "env.train.camera_id"),
                        (config_change(**{"env.train.size": 32}), "env.train.size"),
                        (config_change(**{"world_model_env.horizon": 15}), "world_model_env.horizon"),
                        (config_change(**{"agent.actor_critic.actor_objective": "advantage_weighted"}),
                         "agent.actor_critic.actor_objective"),
                        (report_change("return_convention", "another convention"), "evaluation.return_convention"),
                        (report_change("action_repeat", 4), "evaluation.action_repeat")):
        with pytest.raises(ValueError, match=f"not comparable: .* differ in .*{re.escape(key)}"):
            compare_with(change)
    # Logging/diagnostics, checkpoint cadence and W&B alone do not make identical science incompatible; resolved
    # references compare by their effective values.
    def logging_only(output):
        config_change(**{"agent.actor_critic.debug_metrics": True, "agent.actor_critic.log_diagnostics": True,
                         "rew_end_model.debug_metrics": True, "checkpointing.every": 2, "wandb.mode": "online",
                         "common.action_repeat": 2})(output)
        cfg = OmegaConf.load(output / "config.yaml")
        cfg.env.train.action_repeat = "${common.action_repeat}"  # the same value through a reference
        OmegaConf.save(cfg, output / "config.yaml")
    out = compare_with(logging_only)
    assert json.loads((out / "analysis.json").read_text())["methods"] == ["uniform", "plan2explore"]

    # Within one run, mixed settings are refused, naming the setting; a table from before the environment and
    # evaluation were part of the fingerprint is not compared.
    run = synthetic_run(tmp_path / "mixed", "uniform")
    config_change(**{"env.train.action_repeat": 4})(run / "downstream/steps_000400/walk/seed_0")
    with pytest.raises(ValueError, match="not comparable: .* differ in env.train.action_repeat: 2 vs 4"):
        analysis_module.collect_results(run, "walk", 0)
    compare_with(lambda output: None)
    csv_path = root / "uniform/walker/seed_0/result/walk/seed_0/result.csv"
    csv_path.write_text(csv_path.read_text(encoding="utf-8").replace("compatibility_sha256", "downstream_settings_sha256"),
                        encoding="utf-8")
    with pytest.raises(ValueError, match="collected before the compatibility fingerprint .* collect it again"):
        analysis_module.compare_methods(root, "walker", 0, "walk", 0)


@pytest.mark.parametrize("order", ["sorts_before", "sorts_after"])
def test_two_policies_under_the_selected_protocol_are_refused(tmp_path, order):
    run = synthetic_run(tmp_path, "uniform", steps=(300, 400), seeds=[3001, 3002], max_steps=50)
    path, rows, _ = analysis_module.collect_results(run, "walk", 0)
    assert len(rows) == 2  # one unambiguous report per milestone: collected
    before = path.read_bytes()
    output = run / "downstream/steps_000400/walk/seed_0"
    (original,) = (output / "evaluations").iterdir()
    report = json.loads(original.read_text(encoding="utf-8"))
    for k in range(1000):  # another policy's report of the same protocol, named before or after the original
        other = downstream_module.evaluation_path(output, "walk", [3001, 3002], 50, {"w": torch.tensor([k, 99])})
        if (other.name < original.name) == (order == "sorts_before"):
            break
    write_json(other, dict(report, return_mean=report["return_mean"]))
    assert analysis_module.protocol_id(other) == analysis_module.protocol_id(original) and other != original
    with pytest.raises(ValueError, match=f"2 evaluation reports of different policies with protocol .*"
                                         f"{re.escape(min(original.name, other.name))}, {re.escape(max(original.name, other.name))}"):
        analysis_module.collect_results(run, "walk", 0)
    assert path.read_bytes() == before  # the refused collection left the table as it was
    # One policy with two protocols still needs a selection.
    other.unlink()
    synthetic_output(run, 400, [1.0, 2.0, 3.0], seeds=[1, 2, 3], max_steps=50)
    with pytest.raises(ValueError, match="several evaluation protocols"):
        analysis_module.collect_results(run, "walk", 0)


def test_plot_grid_keeps_milestones_without_results_as_gaps(tmp_path, monkeypatch):
    specs = []
    monkeypatch.setattr(analysis_module, "plot_comparison", specs.append)
    root = tmp_path / "outputs"
    uniform = synthetic_run(tmp_path, "uniform", steps=(300, 400, 500, 600))
    p2e = synthetic_run(tmp_path, "plan2explore", steps=(300, 400, 500, 600))
    for run in (uniform, p2e):  # no method has a result at 400
        shutil.rmtree(run / "downstream/steps_000400")
    shutil.rmtree(uniform / "downstream/steps_000500")  # only uniform lacks 500
    for run in (uniform, p2e):
        analysis_module.collect_results(run, "walk", 0)
    out = analysis_module.compare_methods(root, "walker", 0, "walk", 0)
    (spec,) = specs
    assert spec["grid"] == [300, 400, 500, 600]
    value = lambda run, s, key: analysis_module.read_csv(run / "result/walk/seed_0/result.csv")[
        [300, 500, 600].index(s) if run == p2e else [300, 600].index(s)][key]
    for key in ("return_mean", "frac_over_200", "frac_over_400", "frac_over_500", "frac_over_600"):
        assert spec["series"]["uniform"][key] == [value(uniform, 300, key), None, None, value(uniform, 600, key)]
        assert spec["series"]["plan2explore"][key] == [value(p2e, 300, key), None, value(p2e, 500, key), value(p2e, 600, key)]
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["plot_grid"] == [300, 400, 500, 600]
    assert sorted((m["method"], m["train_env_steps"]) for m in analysis["missing"]) == [
        ("plan2explore", 400), ("uniform", 400), ("uniform", 500)]


# Runs an entry point as __main__ and reports, at its exit, what the process loaded and whether CUDA was initialized.
ENTRY_POINT_PROBE = """
import atexit, json, pathlib, runpy, sys
def report():
    import torch
    print("PROBE " + json.dumps(dict(cuda=torch.cuda.is_initialized(), loaded=sorted(
        m for m in ("agent", "dm_control", "downstream_main", "downstream_trainer", "envs", "exploration_trainer", "matplotlib",
                    "models", "mujoco") if m in sys.modules))))
atexit.register(report)
script, sys.argv = sys.argv[1], sys.argv[1:]
sys.path.insert(0, str(pathlib.Path(script).parent))
runpy.run_path(script, run_name="__main__")
"""


def analysis_command(*overrides):
    return [sys.executable, "-c", ENTRY_POINT_PROBE, str(ROOT / "src/downstream_analysis.py"), *overrides]


def test_analysis_entry_point_collects_and_compares_without_gpu_or_training_code(tmp_path):
    root = tmp_path / "outputs"
    runs = [synthetic_run(tmp_path, method, steps=(300, 400)) for method in ("uniform", "plan2explore")]
    before = tree_hashes(*runs)
    select = ["downstream.task_name=walk", "downstream.seed=0"]
    for run in runs:
        result = subprocess.run(analysis_command("downstream.mode=collect", f"analysis.run_dir={run.as_posix()}", *select),
                                cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr[-3000:]
        assert "Result: 2 milestones" in result.stdout and "Result saved: > " in result.stdout
        assert 'PROBE {"cuda": false, "loaded": []}' in result.stdout
    after = tree_hashes(*runs)
    assert {k: v for k, v in after.items() if "result" not in Path(k).relative_to(root).parts} == before  # sources only read
    result = subprocess.run(analysis_command("downstream.mode=compare", f"analysis.output_root={root.as_posix()}",
                                             "analysis.domain=walker", "analysis.exploration_seed=0", *select),
                            cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-3000:]
    assert "Comparison: uniform, plan2explore | 0 missing results" in result.stdout
    assert 'PROBE {"cuda": false, "loaded": []}' in result.stdout  # the plots are drawn in their own process
    assert (root / "comparisons/walker/exploration_seed_0/walk/adaptation_seed_0/task_return.pdf").is_file()


@pytest.mark.parametrize("script, mode, expected", [
    ("downstream_main.py", "collect", "src/downstream_main.py takes downstream.mode=train, resume or evaluate, got 'collect'"),
    ("downstream_main.py", "compare", "src/downstream_main.py takes downstream.mode=train, resume or evaluate, got 'compare'"),
    ("downstream_main.py", "other", "src/downstream_main.py takes downstream.mode=train, resume or evaluate, got 'other'"),
    ("downstream_analysis.py", "train", "src/downstream_analysis.py takes downstream.mode=collect or compare, got 'train'"),
    ("downstream_analysis.py", "resume", "src/downstream_analysis.py takes downstream.mode=collect or compare, got 'resume'"),
    ("downstream_analysis.py", "evaluate", "src/downstream_analysis.py takes downstream.mode=collect or compare, got 'evaluate'"),
    ("downstream_analysis.py", "other", "src/downstream_analysis.py takes downstream.mode=collect or compare, got 'other'")])
def test_entry_points_reject_each_others_modes_showing_the_right_command(tmp_path, script, mode, expected):
    command = [sys.executable, str(ROOT / "src" / script), f"downstream.mode={mode}", "downstream.task_name=walk",
               f"analysis.run_dir={(tmp_path / 'absent').as_posix()}", "common.devices=cpu"]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 1 and expected in result.stderr
    assert "python src/downstream_analysis.py downstream.mode=collect analysis.run_dir=<exploration run>" in result.stderr
    assert "python src/downstream_analysis.py downstream.mode=compare analysis.domain=<domain>" in result.stderr
    if script == "downstream_analysis.py":
        chosen = mode if mode != "other" else "train"
        assert f"python src/downstream_main.py downstream.mode={chosen} initialization.milestone_path=" in result.stderr
    assert not list(tmp_path.iterdir())


def test_analysis_arguments_are_required_and_responsibilities_stay_separate(tmp_path):
    base = ["downstream.task_name=walk", "downstream.seed=0"]
    with pytest.raises(ValueError, match="downstream.mode=collect needs analysis.run_dir"):
        analysis_module.analyze(compose("downstream", ["downstream.mode=collect", *base]))
    with pytest.raises(ValueError, match="downstream.mode=compare needs analysis.domain and analysis.exploration_seed"):
        analysis_module.analyze(compose("downstream", ["downstream.mode=compare", *base]))
    run = synthetic_run(tmp_path, "uniform", steps=(300,))
    shutil.rmtree(run / "downstream")
    with pytest.raises(ValueError, match="No completed downstream results"):
        analysis_module.analyze(compose("downstream", ["downstream.mode=collect", f"analysis.run_dir={run.as_posix()}", *base]))
    # One definition of what training and analysis share; analysis names live in one module only.
    import data.milestone as milestone_module
    import exploration_trainer
    for name in ("RETURN_CONVENTION", "check_evaluation_config", "deterministic_policy_description", "evaluation_path",
                 "exploration_identity", "method_label"):
        assert getattr(downstream_module, name) is getattr(milestone_module, name)
    assert exploration_trainer.method_label is milestone_module.method_label
    assert exploration_trainer.OUTPUT_LABELS is milestone_module.OUTPUT_LABELS
    for name in ("analyze", "collect_results", "compare_methods", "plot_comparison", "RESULT_COLUMNS", "THRESHOLDS"):
        assert not hasattr(downstream_module, name) and not hasattr(downstream_main, name)
    # Training never needs the analysis module.
    code = "import sys; import downstream_main; print('downstream_analysis' in sys.modules, 'matplotlib' in sys.modules)"
    result = subprocess.run([sys.executable, "-c", code], cwd=ROOT / "src", capture_output=True, text=True)
    assert result.returncode == 0 and result.stdout.strip().endswith("False False"), result.stderr[-2000:]


def test_default_wandb_names_identify_the_exploration_run(tmp_path):
    def names(run, milestone="steps_000300", member=0, members=None, *extra):
        cfg = compose("downstream", ["initialization.milestone_path=x", "downstream.task_name=walk", "downstream.seed=0",
                                     *extra])
        return downstream_module.run_names(cfg, run / "milestones" / milestone, member, members)

    lcg = synthetic_run(tmp_path, "lcg", steps=(300, 400))
    # One run for the milestones of one exploration run, task and seed: no milestone in the name; group = name.
    assert names(lcg) == ("lcg-walker-exploration-seed0-walk-adaptation-seed0",
                          "lcg-walker-exploration-seed0-walk-adaptation-seed0")
    assert names(lcg, "steps_000400") == names(lcg)
    others = [synthetic_run(tmp_path, "uniform"), synthetic_run(tmp_path, "lcg", exploration_seed=1)]
    cheetah = synthetic_run(tmp_path / "c", "lcg")
    cfg = OmegaConf.load(cheetah / "config/exploration.yaml")
    cfg.env.train.domain_name = "cheetah"
    OmegaConf.save(cfg, cheetah / "config/exploration.yaml")
    groups = {names(r)[1] for r in [lcg, *others, cheetah]}
    assert groups == {"lcg-walker-exploration-seed0-walk-adaptation-seed0", "uniform-walker-exploration-seed0-walk-adaptation-seed0",
                      "lcg-walker-exploration-seed1-walk-adaptation-seed0", "lcg-cheetah-exploration-seed0-walk-adaptation-seed0"}
    assert names(lcg, "steps_000300", 2, 5)[1] == "lcg-walker-exploration-seed0-walk-adaptation-seed0-member2"
    # Explicit names are used exactly, nothing appended; an explicit name alone is also the group.
    assert names(lcg, "steps_000300", 0, None, "wandb.name=mine", "wandb.group=chosen") == ("mine", "chosen")
    assert names(lcg, "steps_000300", 0, None, "wandb.name=lcg-walker--downstream-walk-es0-ds0") == (
        "lcg-walker--downstream-walk-es0-ds0", "lcg-walker--downstream-walk-es0-ds0")
    # Without the exploration run's saved configuration: what the manifest records, the rest labelled.
    (lcg / "config/exploration.yaml").unlink()
    assert downstream_module.exploration_label(lcg, dict(domain="walker", exploration=dict(method="lcg"))) == \
        "lcg-walker-exploration-seed-unrecorded"
    assert downstream_module.exploration_label(lcg, None, "walker") == "unrecorded-walker-exploration-seed-unrecorded"

    # An existing run keeps its identity, name and group (resume, evaluate-only).
    fake = FakeWandb()
    cfg = compose("downstream", ["initialization.milestone_path=x", "downstream.task_name=walk", *ONLINE])
    downstream_module.wandb, saved = fake, downstream_module.wandb
    try:
        downstream_module.init_wandb(cfg, dict(id="old1", name="old-name", group="old-group"), "new-name", "new-group",
                                     "adaptation", torch.device("cpu"))
    finally:
        downstream_module.wandb = saved
    assert {k: fake.inits[0][k] for k in ("id", "name", "group", "resume")} == dict(
        id="old1", name="old-name", group="old-group", resume="allow")

# ---------------------------------------------------------------- Gaussian H10 EMA-critic controller

def test_downstream_controller_is_a_fresh_sign_only_h10_controller(tmp_path, monkeypatch):
    cfg = compose("downstream", ["initialization.milestone_path=x", "downstream.task_name=walk"])
    ac = cfg.agent.actor_critic
    assert (ac.actor_objective, ac.actor_distribution, ac.slow_critic_coef, ac.slow_critic_rate,
            ac.imagination_history, cfg.world_model_env.horizon, cfg.rew_end_model.training.seq_length) == (
        "sign", "tanh_gaussian", 1.0, 0.02, "repeat_anchor", 10, 19)
    trainer = make_trainer(build_milestone(tmp_path), monkeypatch)
    model = trainer.agent.actor_critic
    assert model.objective_settings() == dict(mode="sign") and model.gaussian and model.repeats_imagined_anchor
    # Fresh: no update, prior and diagnostic snapshot equal the new actor, the slow critic the
    # new critic, empty optimizer state. The milestone holds a denoiser and replay only.
    assert int(model.updates) == 0 and not trainer.actor_critic_opt.actor.state and not trainer.actor_critic_opt.value.state
    for copy, source in ((model.prior_actor, model.actor), (model.initial_actor, model.actor), (model.slow_value, model.value)):
        assert states_equal(copy.state_dict(), source.state_dict())
    assert not any("actor" in p.name or "value" in p.name for p in Path(trainer.milestone.path).rglob("*"))
    with pytest.raises(ValueError, match="Downstream PMPO is sign-only"):
        make_trainer(build_milestone(tmp_path / "other"), monkeypatch,
                     extra=["agent.actor_critic.actor_objective=advantage_weighted"])
