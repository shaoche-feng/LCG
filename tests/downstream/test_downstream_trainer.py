"""Step 5 (first increment): downstream reward-model and PMPO training on a frozen milestone."""
import hashlib
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

import downstream_trainer as downstream_module
from coroutines.env_loop import make_env_loop
from data import Dataset, Episode
from data.milestone import publish_milestone
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
    "actor_critic.training.steps=3", "actor_critic.training.batch_size=2",
]


def compose(config_name, overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        return hydra_compose(config_name=config_name, overrides=list(overrides))


def frames(n, generator):
    levels = torch.randint(0, 256, (n, 3, 8, 8), generator=generator)
    return levels.float().div(255).mul(2).sub(1)  # on the uint8 grid, as TorchEnv produces


def build_milestone(root, column_offsets=(0.0, 0.0, 0.0)):
    """A labeled walker milestone: two finished (truncated) episodes and one unfinished
    episode at the cut. column_offsets shift each task's reward column."""
    g = torch.Generator().manual_seed(0)
    run = root / "run"
    dataset = Dataset(run / "dataset/train", "train_dataset")
    for length, finished in ((10, True), (10, True), (6, False)):
        obs, act = frames(length, g), torch.rand(length, 6, generator=g) * 2 - 1
        base = torch.rand(length, generator=g)
        task_rew = torch.stack([base * (i + 1) + column_offsets[i] for i in range(3)], 1)
        trunc = torch.zeros(length, dtype=torch.long)
        trunc[-1] = int(finished)
        info = {"final_observation": frames(1, g)[0]} if finished else {}
        dataset.add_episode(Episode(obs, act, task_rew[:, 0].clone(), torch.zeros(length, dtype=torch.long), trunc,
                                    info, task_rew=task_rew, task_names=TASKS))
    successor = frames(1, g)[0]
    cfg = compose("trainer", TINY_DENOISER)
    denoiser_cfg = instantiate(cfg.agent.denoiser)
    denoiser_cfg.inner_model.continuous_action_dim = 6
    torch.manual_seed(1)
    denoiser = Denoiser(denoiser_cfg)
    manifest = dict(created_by="test", domain="walker", task_names=list(TASKS), collected_steps=26, num_episodes=3,
                    epoch=2, counters=dict(denoiser_batches=5, pmpo_updates=0),
                    exploration=dict(collection_mode="uniform", intrinsic_reward_enabled=False),
                    model=dict(denoiser=OmegaConf.to_container(cfg.agent.denoiser, resolve=True),
                               env=OmegaConf.to_container(cfg.env.train, resolve=True)),
                    run_config="../../config/trainer.yaml", source=dict(git_commit=None))
    return publish_milestone(run / "milestones", "steps_000026", denoiser.state_dict(), dataset, {2: successor}, manifest)


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
                                                     "rew_end_model.pt", "summary.yaml"]
    summary = OmegaConf.load(out / "summary.yaml")
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
    import lcg

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
    # The new episode's stack is the latest valid prompt frames, the earliest valid one
    # repeated before them (prompts may be padded before an episode start).
    (frames_, valid), = prompts
    first = valid.shape[1] - valid.sum(1, keepdim=True)
    index = torch.maximum(torch.arange(valid.shape[1] - 4, valid.shape[1]), first)
    expected = frames_[torch.arange(len(frames_))[:, None], index].flatten(1, 2)
    torch.testing.assert_close(states[:, 2], expected, rtol=0, atol=0)


# ---------------------------------------------------------------- entry point

def test_entry_point_runs_through_hydra(tmp_path):
    milestone = build_milestone(tmp_path)
    command = [sys.executable, str(ROOT / "src/downstream_main.py"), f"initialization.milestone_path={milestone.as_posix()}",
               "downstream.task_name=run", "downstream.seed=1", "common.devices=cpu", *TINY_DOWNSTREAM]
    result = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr[-2000:]
    out = tmp_path / "run/downstream/steps_000026/run/seed_1"
    assert (out / "summary.yaml").is_file() and OmegaConf.load(out / "summary.yaml").task_index == 2
    assert not list(ROOT.glob("*.log"))  # no Hydra log file written into the repository
