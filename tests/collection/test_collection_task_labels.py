"""Step 2 (data collection): uniform/policy collection modes and per-task reward labels."""
from collections import Counter
from pathlib import Path
import random

import numpy as np
import pytest
import torch
from dm_control import suite
from omegaconf import OmegaConf

from coroutines.collector import ActionSource, make_collector, NumToCollect
from data import Dataset, Episode, collate_segments_to_batch
from data.segment import SegmentId
from data.utils import make_segment
from envs import DMControlEnv, TorchEnv, make_dm_control_env
from envs.dm_control_env import REWARD_LABEL_TASK_GROUPS, validate_task_names
from models.pmpo_beta import PMPOBeta, PMPOBetaConfig

WALKER_TASKS = ("stand", "walk", "run")
VERIFIED_GROUPS = [(domain, group) for domain, groups in REWARD_LABEL_TASK_GROUPS.items() for group in groups]
GROUP_IDS = [f"{domain}-{'-'.join(group)}" for domain, group in VERIFIED_GROUPS]
# Same-domain combinations that are not verified (see docs/DM_CONTROL_TASK_COMPATIBILITY.md).
UNSUPPORTED_COMBINATIONS = [
    ("cartpole", ["balance", "swingup"]),  # different reset distributions
    ("cartpole", ["balance", "balance_sparse", "swingup"]),  # spans two groups
    ("cartpole", ["swingup", "two_poles"]),  # different models
    ("point_mass", ["easy", "hard"]),  # hard randomizes actuator gains at reset
    ("reacher", ["easy", "hard"]),  # target size written into the model at reset
    ("finger", ["turn_easy", "turn_hard"]),  # target size written into the model at reset
    ("finger", ["spin", "turn_easy"]),
    ("fish", ["upright", "swim"]),
    ("dog", ["stand", "walk"]),  # reward reads task state set only by its own reset
    ("dog", ["walk", "trot"]),  # floor size differs
    ("quadruped", ["walk", "run"]),  # floor size differs
    ("manipulator", ["bring_ball", "insert_ball"]),
    ("stacker", ["stack_2", "stack_4"]),
    ("swimmer", ["swimmer6", "swimmer15"]),
    ("lqr", ["lqr_2_1", "lqr_6_2"]),
]


def frame(value, batch=1, size=8):
    return torch.full((batch, 3, size, size), float(value))


def make_policy(action_dim=2, low=(-3., 0.5), high=(2., 7.), size=8):
    return PMPOBeta(PMPOBetaConfig(img_channels=3, img_size=size, channels=[8], down=[0],
                                   continuous_action_dim=action_dim, action_low=list(low), action_high=list(high),
                                   actor_hidden_dims=(16,), value_hidden_dims=(16,), imagination_horizon=3))


def labeled_episode(length, names=("a", "b"), offset=0., dead=False):
    t = torch.arange(length).float() + offset
    trunc = torch.zeros(length, dtype=torch.long)
    trunc[-1] = int(dead)
    return Episode(frame(0).repeat(length, 1, 1, 1), torch.zeros(length, 2), t, torch.zeros(length, dtype=torch.long),
                   trunc, {}, task_rew=torch.stack([t, -t], 1)[:, :len(names)], task_names=names)


# ---------------------------------------------------------------- task selection / config

def test_task_selection_validation():
    assert validate_task_names("walker", ["stand", "walk", "run"]) == WALKER_TASKS
    assert validate_task_names("walker", OmegaConf.create({"t": ["run", "walk"]}).t) == ("run", "walk")
    assert validate_task_names("cheetah", ["run"]) == ("run",)  # a single task needs no group
    for names, match in [("walk", "list of task names"), (None, "list of task names"),
                         ([], "one or more strings"), (["walk", 1], "one or more strings"),
                         (["walk", "walk"], "duplicates"), (["walk", "fly"], r"Unknown tasks for walker: \['fly'\]")]:
        with pytest.raises(ValueError, match=match):
            validate_task_names("walker", names)
    assert validate_task_names("cartpole", ["swingup_sparse", "swingup"]) == ("swingup_sparse", "swingup")
    with pytest.raises(ValueError, match="Unknown tasks for walker"):
        validate_task_names("walker", ["stand", "swingup"])
    # The vector factory rejects before spawning any worker.
    with pytest.raises(ValueError, match="Unsupported task combination for cartpole"):
        make_dm_control_env(num_envs=1, device=torch.device("cpu"), domain_name="cartpole",
                            task_names=["balance", "swingup"], size=8, camera_id=0, action_repeat=2, time_limit=None)


def compose(overrides=()):
    from hydra import compose as hydra_compose, initialize_config_dir
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        return hydra_compose(config_name="trainer", overrides=list(overrides))


def test_config_interpolation_and_collection_modes():
    cfg = compose()
    OmegaConf.resolve(cfg)
    assert cfg.env.train.domain_name == cfg.env.test.domain_name == "walker"
    assert list(cfg.env.train.task_names) == list(cfg.env.test.task_names) == list(WALKER_TASKS)
    assert (cfg.collection.train.first_epoch.mode, cfg.collection.train.mode) == ("uniform", "policy")
    cfg = compose(["env.train.task_names=[walk,run]", "env.train.action_repeat=3"])
    OmegaConf.resolve(cfg)
    assert list(cfg.env.test.task_names) == ["walk", "run"] and cfg.env.test.action_repeat == 3
    with pytest.raises(ValueError, match="Collection mode"):
        NumToCollect(steps=1, mode="random")


@pytest.mark.parametrize("key", ["collection.train.first_epoch.mode", "collection.train.mode"])
def test_trainer_rejects_unknown_collection_mode_before_any_work(key, tmp_path, monkeypatch):
    import exploration_trainer as trainer_module
    cfg = compose(["common.devices=cpu", f"{key}=epsilon"])
    monkeypatch.setattr(trainer_module.wandb, "init", lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    with pytest.raises(ValueError, match=key.replace(".", r"\.")):
        trainer_module.ExplorationTrainer(cfg, Path(__file__).resolve().parents[2])
    assert not (tmp_path / "checkpoints").exists()


def test_trainer_passes_initial_and_later_collection_modes(monkeypatch):
    import exploration_trainer as trainer_module

    class Collector:
        def __init__(self, dataset):
            self.dataset, self.modes = dataset, []
        def send(self, n):
            self.modes.append((n.mode, n.steps))
            self.dataset.num_steps += n.steps
            return []

    class Data:
        name, num_steps, counts_end = "train_dataset", 0, [0, 0]
        def state_dict(self):
            return {}
        def load_state_dict(self, state):
            pass

    t = trainer_module.ExplorationTrainer.__new__(trainer_module.ExplorationTrainer)
    t._cfg = OmegaConf.create(dict(
        collection=dict(train=dict(num_steps_total=40, steps_per_epoch=10, mode="policy",
                                   first_epoch=dict(min=20, mode="uniform"))),
        training=dict(should=False, num_final_epochs=0), evaluation=dict(should=False, every=1)))
    t._rank, t._is_static_dataset, t._pmpo_diagnostic, t.epoch, t.num_epochs_collect = 0, False, None, 0, None
    t._should_test = False  # evaluation.should=False
    t._use_cuda, t._epoch_summary = False, False  # no GPU timing, no epoch summary row
    t.log_rows = lambda rows: None
    t.train_dataset = Data()
    t._train_collector = Collector(t.train_dataset)
    t.save_checkpoint = lambda: None
    t.collect_test = lambda final=False: []
    monkeypatch.setattr(trainer_module, "wandb_log", lambda *args: None)
    t._run()
    assert t._train_collector.modes == [("uniform", 20), ("policy", 10), ("policy", 10)]


@pytest.mark.parametrize("domain,tasks", UNSUPPORTED_COMBINATIONS)
def test_unsupported_task_combinations_are_rejected(domain, tasks):
    for name in tasks:  # each task alone is valid
        assert validate_task_names(domain, [name]) == (name,)
    with pytest.raises(ValueError, match=f"Unsupported task combination for {domain}"):
        validate_task_names(domain, tasks)


# ---------------------------------------------------------------- collector modes (fake env)

class LabeledFakeEnv:
    """TorchEnv-like: 1 env, asymmetric bounds, episodes of 5 steps, task labels in info."""
    num_envs = 1
    action_dim = 2
    action_low = torch.tensor([-3., .5])
    action_high = torch.tensor([2., 7.])
    task_names = ("a", "b")

    def __init__(self):
        self.actions = []

    def reset(self, **kwargs):
        self.t = 0
        return frame(0), {}

    def step(self, action):
        self.actions.append(action.clone())
        self.t += 1
        dead = self.t == 5
        rew = action.sum(1)
        info = {"task_rewards": torch.stack([rew, 10 * rew + self.t], 1)}
        if dead:
            info["final_observation"] = frame(self.t)
            self.t = 0
        return frame(self.t), rew, torch.tensor([0]), torch.tensor([int(dead)]), info


def test_uniform_collection_never_evaluates_policy_respects_bounds_then_policy_continues(tmp_path):
    torch.manual_seed(0)
    env = LabeledFakeEnv()
    model = make_policy()
    calls = Counter()
    for name in ("predict_act_value", "sample_action"):
        original = getattr(model, name)
        def counted(*args, _name=name, _original=original, **kwargs):
            calls[_name] += 1
            return _original(*args, **kwargs)
        setattr(model, name, counted)
    model.actor.register_forward_pre_hook(lambda *args: calls.update(["actor"]))
    model.value.register_forward_pre_hook(lambda *args: calls.update(["value"]))
    dataset = Dataset(tmp_path / "d", cache_in_ram=True, save_on_disk=False)
    collector = make_collector(env, model, dataset, epsilon=0., verbose=False)

    collector.send(NumToCollect(steps=403, mode="uniform"))
    assert sum(calls.values()) == 0
    actions = torch.cat(env.actions)
    assert ((actions >= env.action_low) & (actions <= env.action_high)).all()
    for d in range(2):  # Independent dims spread over each full interval.
        width = env.action_high[d] - env.action_low[d]
        assert actions[:, d].min() < env.action_low[d] + .05 * width
        assert actions[:, d].max() > env.action_high[d] - .05 * width
    # torch, not numpy: numpy's MKL BLAS aborts beside torch's OpenMP in this Windows env.
    assert abs(torch.corrcoef(actions.T)[0, 1]) < .2
    assert dataset.num_steps == 403 and len(dataset.load_episode(80)) == 3  # 80 full episodes + partial

    collector.send(NumToCollect(steps=4, mode="policy"))
    assert calls["predict_act_value"] > 0 and calls["actor"] > 0
    episode = dataset.load_episode(80)  # the partial uniform episode continued under the policy
    assert len(episode) == 5 and episode.trunc[-1] == 1 and dataset.num_episodes == 82
    torch.testing.assert_close(episode.act, torch.cat(env.actions[400:405]), rtol=0, atol=0)
    for i in range(dataset.num_episodes):
        ep = dataset.load_episode(i)
        assert ep.task_names == ("a", "b") and ep.task_rew.shape == (len(ep), 2)
        torch.testing.assert_close(ep.task_rew[:, 0], ep.rew, rtol=0, atol=0)
        torch.testing.assert_close(ep.task_rew[:, 1], 10 * ep.rew + torch.arange(1, len(ep) + 1), rtol=0, atol=0)
    collector.close()


def test_action_source_uniform_outputs_are_placeholders_only():
    env = LabeledFakeEnv()
    source = ActionSource(env, make_policy())
    source.uniform = True
    logits, value = source.predict_act_value(torch.zeros(3, 12, 8, 8))
    assert logits.shape == (3, 0) and value.shape == (3,)
    assert source.sample_action(logits).shape == (3, 2)


# ---------------------------------------------------------------- TorchEnv autoreset labels

def test_torch_env_takes_labels_of_finished_envs_from_final_info():
    import gymnasium

    class Vector(gymnasium.Env):
        single_action_space = gymnasium.spaces.Box(-1., 1., (2,), np.float32)
        observation_space = gymnasium.spaces.Box(0, 255, (2, 8, 8, 3), np.uint8)
        action_space = single_action_space
        labels_missing = False
        def reset(self, **kwargs):
            return np.zeros((2, 8, 8, 3), np.uint8), {}
        def step(self, actions):
            # gymnasium 0.29 AsyncVectorEnv layout: env 0 finished and autoreset (its top-level
            # info is the reset's, without labels); env 1 continues.
            final_obs, final_info = np.empty(2, object), np.empty(2, object)
            final_obs[0], final_info[0] = np.full((8, 8, 3), 255, np.uint8), {"task_rewards": np.array([.5, .25])}
            labels, mask = np.empty(2, object), np.array([False, not self.labels_missing])
            labels[1] = np.array([1.5, 3.])
            return (np.zeros((2, 8, 8, 3), np.uint8), np.array([.5, 1.5]), np.array([False, False]),
                    np.array([True, False]), {"final_observation": final_obs, "_final_observation": np.array([True, False]),
                                              "final_info": final_info, "_final_info": np.array([True, False]),
                                              "task_rewards": labels, "_task_rewards": mask})

    env = TorchEnv(Vector(), torch.device("cpu"), ("x", "y"))
    _, rew, _, trunc, info = env.step(torch.zeros(2, 2))
    torch.testing.assert_close(info["task_rewards"], torch.tensor([[.5, .25], [1.5, 3.]]), rtol=0, atol=0)
    assert "_task_rewards" not in info and trunc.tolist() == [1, 0]
    Vector.labels_missing = True
    with pytest.raises(RuntimeError, match="no task reward labels"):
        env.step(torch.zeros(2, 2))


# ---------------------------------------------------------------- storage

def test_episode_labels_extend_slice_batch_and_reject_mismatches():
    first, second = labeled_episode(3), labeled_episode(2, offset=3., dead=True)
    joined = first + second
    assert joined.task_names == ("a", "b")
    torch.testing.assert_close(joined.task_rew[:, 0], joined.rew, rtol=0, atol=0)
    with pytest.raises(ValueError, match="Cannot extend"):
        first + labeled_episode(2, names=("a",))
    with pytest.raises(ValueError, match="given together"):
        Episode(*[getattr(first, k) for k in ("obs", "act", "rew", "end", "trunc")], {}, task_rew=first.task_rew)
    with pytest.raises(ValueError, match="does not match"):
        Episode(*[getattr(first, k) for k in ("obs", "act", "rew", "end", "trunc")], {},
                task_rew=first.task_rew[:2], task_names=("a", "b"))
    # Padding on both sides keeps labels aligned with rew and mask_padding.
    left = make_segment(joined, SegmentId(0, -2, 3))
    right = make_segment(joined, SegmentId(0, 3, 8))
    for segment in (left, right):
        torch.testing.assert_close(segment.task_rew[:, 0], segment.rew, rtol=0, atol=0)
        assert (segment.task_rew[~segment.mask_padding] == 0).all()
    batch = collate_segments_to_batch([left, right])
    assert batch.task_rew.shape == (2, 5, 2) and batch.to("cpu").task_rew is not None
    unlabeled = make_segment(Episode(joined.obs, joined.act, joined.rew, joined.end, joined.trunc, {}), SegmentId(0, 0, 5))
    assert unlabeled.task_rew is None and collate_segments_to_batch([unlabeled]).task_rew is None
    with pytest.raises(ValueError, match="with and without"):
        collate_segments_to_batch([left, unlabeled])


def test_dataset_label_save_load_and_legacy_files(tmp_path):
    dataset = Dataset(tmp_path / "d", cache_in_ram=False)
    dataset.add_episode(labeled_episode(3))
    dataset.add_episode(labeled_episode(4, offset=7., dead=True))
    with pytest.raises(ValueError, match="do not match"):
        dataset.add_episode(labeled_episode(2, names=("b", "a")))
    dataset.save_to_default_path()
    restored = Dataset(tmp_path / "d")
    restored.load_from_default_path()
    assert restored.task_names == ("a", "b")
    for i, length in enumerate((3, 4)):
        # torch.load's weights_only default reads the stored task names.
        loaded = restored.load_episode(i)
        assert loaded.task_names == ("a", "b") and len(loaded) == length
        torch.testing.assert_close(loaded.task_rew, dataset.load_episode(i).task_rew, rtol=0, atol=0)
    # Files written before task labels existed load as unlabeled.
    legacy_dir = tmp_path / "legacy"
    legacy = Dataset(legacy_dir)
    legacy.add_episode(Episode(frame(0), torch.zeros(1, 2), torch.ones(1), torch.zeros(1, dtype=torch.long),
                               torch.ones(1, dtype=torch.long), {}))
    path = legacy._get_episode_path(0)
    raw = torch.load(path)
    torch.save({k: v for k, v in raw.items() if k not in ("task_rew", "task_names")}, path)
    state = legacy.state_dict()
    del state["task_names"]
    torch.save(state, legacy_dir / "info.pt")
    reloaded = Dataset(legacy_dir)
    reloaded.load_from_default_path()
    assert reloaded.task_names is None and reloaded.load_episode(0).task_rew is None
    reloaded.clear()
    reloaded.add_episode(labeled_episode(2))
    assert reloaded.task_names == ("a", "b")


# ---------------------------------------------------------------- real dm_control

def raw_task_envs(domain, tasks, seed, time_limit):
    """Each task's own independent environment."""
    return [suite.load(domain, t, task_kwargs={"random": seed, "time_limit": time_limit}) for t in tasks]


def short_time_limit(domain, task):
    # 7.5 control steps: every episode ends after 8, robust to float rounding.
    return 7.5 * suite.load(domain, task).control_timestep()


def raw_step(envs, action, action_repeat):
    """Independent per-task truth: each task's own environment, stepped in lockstep."""
    sums, substeps = np.zeros(len(envs)), 0
    for _ in range(action_repeat):
        steps = [env.step(action.astype(np.float64)) for env in envs]
        sums += [s.reward for s in steps]
        substeps += 1
        if steps[0].last():
            assert all(s.last() for s in steps)
            break
    return sums, substeps


def assert_same_physics(env, raw_envs):
    for raw in raw_envs:
        for field in ("qpos", "qvel", "act"):
            np.testing.assert_array_equal(getattr(env._dm_env.physics.data, field), getattr(raw.physics.data, field))


def set_default_pose(all_physics):
    # Random resets can leave e.g. hopper lying down with zero reward; the model's default
    # pose gives informative labels. Applied identically to every environment.
    for physics in all_physics:
        with physics.reset_context():
            physics.data.qpos[:] = physics.model.qpos0
            physics.data.qvel[:] = 0


@pytest.mark.parametrize("domain,group", VERIFIED_GROUPS, ids=GROUP_IDS)
def test_group_labels_equal_independent_task_rewards_including_action_repeat_sums(domain, group):
    # 8 control steps with action_repeat=3: 3, 3, then only 2 substeps.
    time_limit = short_time_limit(domain, group[0])
    env = DMControlEnv(domain, list(group), size=8, action_repeat=3, time_limit=time_limit, seed=11)
    raws = raw_task_envs(domain, group, 11, time_limit)
    rng = np.random.default_rng(0)
    labels = []
    for episode in range(3):  # later episodes follow unseeded resets, as vector autoreset does
        env.reset()
        for raw in raws:
            raw.reset()
        if episode == 0:
            set_default_pose([env._dm_env.physics] + [raw.physics for raw in raws])
        assert_same_physics(env, raws)
        for step in range(3):
            action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
            _, reward, end, trunc, info = env.step(action)
            expected, substeps = raw_step(raws, action, 3)
            assert substeps == (2 if step == 2 else 3) and trunc == (step == 2) and not end
            np.testing.assert_array_equal(info["task_rewards"], expected)
            assert info["task_rewards"][0] == reward
            assert_same_physics(env, raws)
            labels.append(info["task_rewards"])
    labels = np.array(labels)
    informative = (labels != 0).any(0)
    assert informative.all(), f"labels never non-zero for {np.array(group)[~informative]}"
    assert (labels.max(1) != labels.min(1)).any()  # the tasks' rewards genuinely differ


@pytest.mark.parametrize("domain,group", VERIFIED_GROUPS, ids=GROUP_IDS)
def test_extra_labels_leave_trajectory_and_random_streams_unchanged(domain, group):
    time_limit = short_time_limit(domain, group[0])  # 4 steps per episode at action_repeat=2

    def rollout(tasks):
        state = (random.getstate(), np.random.get_state()[1].copy(), torch.get_rng_state())
        env = DMControlEnv(domain, list(tasks), size=16, action_repeat=2, time_limit=time_limit, seed=5)
        frames, steps = [env.reset()[0]], []
        rng = np.random.default_rng(1)
        for _ in range(2):
            for _ in range(4):
                action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
                obs, reward, end, trunc, info = env.step(action)
                frames.append(obs)
                steps.append((reward, end, trunc, dict(zip(tasks, info["task_rewards"].tolist()))))
            # Unseeded reset: draws from the collection task's RandomState.
            frames.append(env.reset()[0])
        after = (random.getstate(), np.random.get_state()[1], torch.get_rng_state())
        assert after[0] == state[0] and np.array_equal(after[1], state[1]) and torch.equal(after[2], state[2])
        return frames, steps, env._dm_env.task.random.get_state()[1]

    first = group[0]
    alone = rollout([first])
    labeled = rollout(group)
    reordered = rollout(group[::-1])  # a different task drives collection
    for run in (labeled, reordered):
        np.testing.assert_array_equal(run[2], alone[2])
        for frame_alone, frame_run in zip(alone[0], run[0]):
            np.testing.assert_array_equal(frame_alone, frame_run)
    for a, b, c in zip(alone[1], labeled[1], reordered[1]):
        assert a[1:3] == b[1:3] == c[1:3]
        assert a[3] == {first: a[0]} and b[3][first] == a[0] == b[0]
        assert b[3] == c[3] and c[0] == c[3][group[-1]]


def test_vector_collection_labels_stay_aligned_across_resets_extension_and_save_load(tmp_path):
    seeding = 5
    env = make_dm_control_env(num_envs=2, device=torch.device("cpu"), domain_name="walker",
                              task_names=list(WALKER_TASKS), size=8, camera_id=0, action_repeat=3, time_limit=0.2)
    try:
        assert env.task_names == WALKER_TASKS
        model = make_policy(6, [-1.] * 6, [1.] * 6)
        calls = Counter()
        model.actor.register_forward_pre_hook(lambda *args: calls.update(["actor"]))
        dataset = Dataset(tmp_path / "train", cache_in_ram=False)
        collector = make_collector(env, model, dataset, epsilon=0., verbose=False)
        random.seed(seeding)
        collector.send(NumToCollect(steps=4, mode="uniform"))  # 2 of 3 steps per env: partial
        assert calls["actor"] == 0 and dataset.lengths.tolist() == [2, 2]
        collector.send(NumToCollect(steps=6, mode="policy"))  # finish, autoreset, 2 more steps
        assert calls["actor"] > 0
        collector.close()
    finally:
        env.close()
    dataset.save_to_default_path()
    restored = Dataset(tmp_path / "train")
    restored.load_from_default_path()
    assert restored.task_names == WALKER_TASKS and restored.lengths.tolist() == [3, 3, 2, 2]

    # Replay each env's stored actions in independent per-task environments, seeded as
    # make_env_loop seeds the vector env (random.randint, then +i per env).
    first_seed = random.Random(seeding).randint(0, 2**31 - 1)
    for i, episode_ids in enumerate(([0, 2], [1, 3])):
        raws = raw_task_envs("walker", WALKER_TASKS, first_seed + i, 0.2)
        for episode_id in episode_ids:
            for raw in raws:
                raw.reset()
            episode = restored.load_episode(episode_id)
            assert episode.task_names == WALKER_TASKS and episode.task_rew.dtype == torch.float32
            torch.testing.assert_close(episode.task_rew[:, 0], episode.rew, rtol=0, atol=0)
            for t in range(len(episode)):
                expected, _ = raw_step(raws, episode.act[t].numpy(), 3)
                torch.testing.assert_close(episode.task_rew[t], torch.tensor(expected, dtype=torch.float32),
                                           rtol=0, atol=0)
            complete = episode_id < 2
            assert episode.trunc[-1] == int(complete) and ("final_observation" in episode.info) == complete
