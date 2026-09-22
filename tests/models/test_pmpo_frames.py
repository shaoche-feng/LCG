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
    def observe(obs, hx_cx):
        states.append(obs.clone())
        return original(obs, hx_cx)
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
        model.predict_act_value(observations[:, -3:], (None, None))


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
