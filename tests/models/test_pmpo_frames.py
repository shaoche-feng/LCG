"""Exact frame history through the shared real/imagination coroutine."""
from copy import deepcopy

import pytest
import torch

from coroutines.env_loop import make_env_loop
from coroutines.frame_history import FrameHistory
from test_pmpo_beta import make_controller


def frame(value, batch=1):
    return torch.full((batch, 3, 8, 8), float(value))


def stack(values):
    return torch.cat([frame(v) for v in values], 1)


def test_initial_early_rolling_and_immutable_storage():
    initial = frame(0)
    history = FrameHistory(initial, 4)
    retained = [history.state]
    initial.fill_(99)
    expected = [[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 1, 2], [0, 1, 2, 3], [1, 2, 3, 4]]
    for value in range(1, 5):
        history.advance(frame(value), torch.tensor([False]))
        retained.append(history.state)
    history.advance(frame(9), torch.tensor([True]))
    torch.testing.assert_close(history.state, stack([9, 9, 9, 9]), rtol=0, atol=0)
    for actual, wanted in zip(retained, expected):
        torch.testing.assert_close(actual, stack(wanted), rtol=0, atol=0)


class CounterEnv:
    num_envs = 2
    def __init__(self, ending=None):
        self.ending = ending
    def reset(self, **kwargs):
        self.t = 0
        self.obs = frame(0, 2)
        return self.obs, {}
    def step(self, action):
        self.t += 1
        self.obs.fill_(self.t)
        end, trunc = torch.zeros(2, dtype=torch.long), torch.zeros(2, dtype=torch.long)
        info = {}
        if self.t == 2 and self.ending is not None:
            (end if self.ending == "end" else trunc)[0] = 1
            info["final_observation"] = self.obs[:1].clone()
            self.obs[0].fill_(10)
            # Stacks deliberately ignore the WM recurrent burn-in history.
            info["burnin_obs"] = torch.full((1, 3, 3, 8, 8), -100.)
        return self.obs, torch.ones(2), end, trunc, info


@pytest.mark.parametrize("ending", [None, "end", "trunc"])
def test_action_state_identity_reset_successor_and_rollout_cut(ending):
    model = make_controller()
    states, sampling_raw = [], []
    original = model.predict_act_value
    def observe(obs):
        states.append(obs.clone())
        return original(obs)
    original_sample = model.sample_action
    def sample(raw, deterministic=False):
        sampling_raw.append(raw.clone())
        return original_sample(raw, deterministic)
    model.predict_act_value, model.sample_action = observe, sample
    loop = make_env_loop(CounterEnv(ending), model, store_policy_observations=True)
    with torch.no_grad():
        first = loop.send(2)
        snapshot = first[0].clone()
        second = loop.send(2)
    torch.testing.assert_close(first[0], snapshot, rtol=0, atol=0)
    for t in range(2):
        torch.testing.assert_close(model.actor(first[0][:, t]), sampling_raw[t], rtol=0, atol=0)
    for t in range(2):
        torch.testing.assert_close(model.actor(second[0][:, t]), sampling_raw[t + 2], rtol=0, atol=0)
    torch.testing.assert_close(first[0][0, 0], stack([0, 0, 0, 0])[0], rtol=0, atol=0)
    torch.testing.assert_close(first[0][0, 1], stack([0, 0, 0, 1])[0], rtol=0, atol=0)
    expected_successor = stack([0, 0, 1, 2])
    # Even an environment truncation bootstraps the pre-reset successor.
    # Match the production batch shape (two live rows vs one final/reset row).
    bootstrap_batch = expected_successor.repeat(2 if ending is None else 1, 1, 1, 1)
    torch.testing.assert_close(first[7][0, -1], model.value(bootstrap_batch)[0, 0], rtol=0, atol=0)
    expected_next = stack([0, 0, 1, 2] if ending is None else [10, 10, 10, 10])
    torch.testing.assert_close(second[0][0, 0], expected_next[0], rtol=0, atol=0)
    # The other environment never resets; history is independent by batch row.
    torch.testing.assert_close(second[0][1, 0], stack([0, 0, 1, 2])[0], rtol=0, atol=0)
    torch.testing.assert_close(second[0][1, 1], stack([0, 1, 2, 3])[0], rtol=0, atol=0)
    loop.close()


def test_real_collector_keeps_rgb_but_policy_receives_stack(tmp_path):
    from coroutines.collector import make_collector, NumToCollect
    from data import Dataset
    model = make_controller()
    shapes = []
    handle = model.actor.register_forward_pre_hook(lambda m, inputs: shapes.append(inputs[0].shape))
    dataset = Dataset(tmp_path / "real", cache_in_ram=True, save_on_disk=False)
    collector = make_collector(CounterEnv(), model, dataset, verbose=False)
    collector.send(NumToCollect(steps=8))
    assert shapes and all(s[1] == 12 for s in shapes)
    assert dataset.load_episode(0).obs.shape == (4, 3, 8, 8)
    handle.remove()
    collector.close()


def test_actor_value_prior_share_identical_state_representation():
    model = make_controller()
    observations = torch.randn(5, 12, 8, 8)
    assert model.actor(observations).shape == model.prior_actor(observations).shape == (5, 4)
    assert model.value(observations).shape == (5, 1)
    torch.testing.assert_close(model.actor(observations), model.prior_actor(observations), rtol=0, atol=0)
    with pytest.raises(ValueError, match="four-frame"):
        model.predict_act_value(observations[:, -3:])


def test_imagined_likelihood_replay_equals_generation_and_fixed_stacks_vary():
    from test_pmpo_integration import make_system
    model, _ = make_system()
    rollout = model.collect_imagination()
    states, _, _, _, _, raw, values, _, _ = rollout
    assert states.shape == (2, 3, 12, 8, 8)
    with torch.no_grad():
        for t in range(states.shape[1]):
            torch.testing.assert_close(model.actor(states[:, t]), raw[:, t], rtol=0, atol=0)
            torch.testing.assert_close(model.value(states[:, t]).squeeze(-1), values[:, t], rtol=0, atol=0)
    model()
    frames = model._fixed_obs.reshape(-1, 4, 3, 8, 8)
    assert (frames[:, 1:] != frames[:, :-1]).any()


def test_collector_continues_partial_episode_across_epochs_and_loads_old_dataset(tmp_path):
    from collections import Counter
    from coroutines.collector import make_collector, NumToCollect
    from data import Dataset

    class LongEpisode(CounterEnv):
        num_envs = 1
        action_dim = 2
        action_low = torch.tensor([-3., 0.5])
        action_high = torch.tensor([2., 7.])

        def reset(self, **kwargs):
            self.t = 0
            return frame(0), {}

        def step(self, action):
            assert ((action >= self.action_low) & (action <= self.action_high)).all()
            self.t += 1
            dead = self.t == 5
            final = frame(self.t)
            obs = frame(0) if dead else final
            return obs, torch.tensor([2.5]), torch.tensor([0]), torch.tensor([int(dead)]), (
                {"final_observation": final} if dead else {})

    dataset = Dataset(tmp_path / "real", cache_in_ram=True, save_on_disk=False)
    collector = make_collector(LongEpisode(), make_controller(), dataset, epsilon=1., verbose=False)
    collector.send(NumToCollect(steps=3))
    assert len(dataset.load_episode(0)) == 3
    collector.send(NumToCollect(steps=3))
    assert dataset.num_steps == 6 and dataset.num_episodes == 2
    episode = dataset.load_episode(0)
    assert len(episode) == 5 and episode.compute_metrics()["return"] == 12.5
    assert episode.trunc[-1] == 1
    torch.testing.assert_close(episode.obs[:, 0, 0, 0], torch.arange(5).float(), rtol=0, atol=0)
    torch.testing.assert_close(episode.info["final_observation"], frame(5)[0], rtol=0, atol=0)
    old = dict(dataset.state_dict(), counter_rew=Counter({1: 6}))
    restored = Dataset(tmp_path / "restored", save_on_disk=False)
    restored.load_state_dict(old)
    assert "counter_rew" in old and "counter_rew" not in restored.state_dict()
    assert restored.num_steps == 6 and restored.lengths.tolist() == [5, 1]
    collector.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_dataset_loads_cuda_final_observation_metadata_onto_cpu(tmp_path):
    from data import Dataset, Episode
    directory = tmp_path / "replay"
    dataset = Dataset(directory)
    episode = Episode(frame(0), torch.zeros(1, 2), torch.ones(1), torch.zeros(1, dtype=torch.long),
                      torch.ones(1, dtype=torch.long), {"final_observation": frame(1)[0].cuda()})
    dataset.add_episode(episode)
    dataset.save_to_default_path()
    restored = Dataset(directory)
    restored.load_from_default_path()
    loaded = restored.load_episode(0)
    assert loaded.obs.device.type == loaded.act.device.type == loaded.info["final_observation"].device.type == "cpu"
    torch.testing.assert_close(loaded.info["final_observation"], frame(1)[0], rtol=0, atol=0)


@pytest.mark.parametrize("discount,terminated,truncated", [(0., True, False), (1., False, True)])
def test_dm_control_action_repeat_reward_aggregation_and_done(discount, terminated, truncated):
    import numpy as np
    import gymnasium
    from types import SimpleNamespace
    from envs.dm_control_env import DMControlEnv

    class Physics:
        calls = 0
        def step(self, action):
            self.calls += 1
            np.testing.assert_array_equal(action, np.array([0.25, -0.5], dtype=np.float64))
            return SimpleNamespace(reward=0.75, discount=discount, last=lambda: self.calls == 2)

    env = DMControlEnv.__new__(DMControlEnv)
    env.action_low, env.action_high = np.full(2, -1.), np.ones(2)
    env.action_space = gymnasium.spaces.Box(-1., 1., (2,), np.float32)
    env._action_dtype, env.action_repeat = np.float64, 4
    env._dm_env = Physics()
    env._render = lambda: np.zeros((8, 8, 3), np.uint8)
    _, reward, end, trunc, info = env.step(np.array([0.25, -0.5]))
    assert reward == 1.5 and env._dm_env.calls == 2
    assert (end, trunc) == (terminated, truncated) and info["discount"] == discount


def test_torch_env_normalizes_final_observation_and_preserves_action_bounds():
    import gymnasium
    import numpy as np
    from envs import TorchEnv

    class Vector(gymnasium.Env):
        single_action_space = gymnasium.spaces.Box(np.array([-3., .5], np.float32),
                                                   np.array([2., 7.], np.float32))
        observation_space = gymnasium.spaces.Box(0, 255, (2, 8, 8, 3), np.uint8)
        action_space = single_action_space
        def reset(self, **kwargs):
            return np.zeros((2, 8, 8, 3), np.uint8), {}
        def step(self, actions):
            np.testing.assert_array_equal(actions, np.array([[-3., .5], [2., 7.]], np.float32))
            final = np.empty(2, object)
            final[0], final[1] = np.full((8, 8, 3), 255, np.uint8), None
            return (np.zeros((2, 8, 8, 3), np.uint8), np.array([.25, 1.5]),
                    np.array([False, False]), np.array([True, False]), {"final_observation": final})

    env = TorchEnv(Vector(), torch.device("cpu"))
    obs, _ = env.reset()
    assert obs.shape == (2, 3, 8, 8) and (obs == -1).all()
    _, rew, end, trunc, info = env.step(torch.tensor([[-3., .5], [2., 7.]]))
    assert (info["final_observation"] == 1).all()
    assert info["final_observation"].shape == (1, 3, 8, 8)
    torch.testing.assert_close(rew, torch.tensor([.25, 1.5]), rtol=0, atol=0)
    assert end.tolist() == [0, 0] and trunc.tolist() == [1, 0]


def test_viewer_uses_stacked_beta_policy_and_records_final_observation(tmp_path, monkeypatch):
    from types import SimpleNamespace
    from game.play_env import PlayEnv, NamedEnv

    class ViewerEnv:
        def reset(self):
            self.t = 0
            self.obs = frame(0)
            return self.obs, {}
        def step(self, action):
            self.t += 1
            dead = self.t == 2
            self.obs.fill_(0.1 * self.t)
            final = frame(.7)
            return self.obs, torch.tensor([.75]), torch.tensor([0]), torch.tensor([int(dead)]), (
                {"final_observation": final} if dead else {})

    monkeypatch.chdir(tmp_path)
    model = make_controller()
    observed = []
    hook = model.actor.register_forward_pre_hook(lambda m, args: observed.append(args[0].clone()))
    viewer = PlayEnv(SimpleNamespace(actor_critic=model), [NamedEnv("test", ViewerEnv())],
                     ["noop"], {}, True, False)
    viewer.reset()
    assert viewer.rec_dataset._directory.as_posix() == "dataset/rec_test_π"
    viewer.step(0)
    viewer.step(0)
    assert observed[0].shape == (1, 12, 8, 8)
    torch.testing.assert_close(observed[1], torch.cat([frame(0)] * 3 + [frame(.1)], 1), rtol=0, atol=0)
    episode = viewer.rec_dataset.load_episode(0)
    torch.testing.assert_close(episode.info["final_observation"], frame(.7)[0], rtol=0, atol=0)
    assert episode.act.shape == (2, 2) and episode.compute_metrics()["return"] == 1.5
    # Retained pre-action frame must survive in-place env observation mutation.
    # Dataset serialization retains the original uint8 RGB quantization.
    quantized_zero = torch.tensor(127).div(255).mul(2).sub(1).item()
    torch.testing.assert_close(episode.obs[0], frame(quantized_zero)[0], rtol=0, atol=0)
    viewer.print_controls()
    assert viewer.next_mode() is False and viewer.next_axis_2() is True
    hook.remove()
