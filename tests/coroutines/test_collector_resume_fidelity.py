"""Tests for coroutines.collector.make_collector's checkpoint-safe collection mode
(flush_before_reset + reset_every_collect + reset_seed_state) and
coroutines.env_loop.EnvResetSeedState -- the fix for real-env-collection checkpoint/resume
fidelity (previously: collector-local Python state -- buffer, episode_ids, dead -- plus the
env_loop coroutine's own hx/cx and the real env's own internal state were never checkpointed at
all, so an uninterrupted run and a checkpoint->resume run could silently collect different next
real transitions even with every model/optimizer state restored).

CPU-only, fast; uses fake deterministic stand-ins for the real env and the policy model (not
mocks of the collector itself) so the actual coroutines.collector/coroutines.env_loop code runs
unmodified, exactly as production does.
"""
import numpy as np
import torch

from coroutines.collector import make_collector, NumToCollect
from coroutines.env_loop import EnvResetSeedState
from data import Dataset
from data.batch_sampler import COMPONENT_SEED_ID
from utils import derive_component_seed


class _DeterministicFakeEnv:
    """A minimal real-env stand-in whose entire trajectory (observations, rewards, episode
    length) is a PURE deterministic function of the seed(s) passed to reset() -- no hidden
    state beyond a per-env numpy Generator seeded exactly there, and (like a real
    gymnasium AsyncVectorEnv) auto-resets a dead env's OWN generator-derived stream internally
    on termination, without ever being re-seeded from outside except at construction/explicit
    reset() calls. This lets the test isolate exactly what make_collector's checkpoint-safe mode
    claims: reproducing collection results requires ONLY the reset-seed stream's position, not
    any other collector-local or environment-internal state, PROVIDED the env itself is a
    deterministic function of its reset seed (true for MuJoCo/dm_control physics; modeled
    directly here rather than depending on a slow, heavier real dm_control instance)."""

    def __init__(self, num_envs, img_channels, img_size, episode_len):
        self.num_envs = num_envs
        self.is_discrete = False
        self.action_dim = 2
        self.action_low = torch.tensor([-1.0, -1.0])
        self.action_high = torch.tensor([1.0, 1.0])
        self._img_channels = img_channels
        self._img_size = img_size
        self._episode_len = episode_len
        self._rngs = [None] * num_envs
        self._step_count = [0] * num_envs

    def _obs(self, i):
        val = float(self._rngs[i].random())
        return torch.full((self._img_channels, self._img_size, self._img_size), val)

    def reset(self, seed):
        self._rngs = [np.random.default_rng(s) for s in seed]
        self._step_count = [0] * self.num_envs
        obs = torch.stack([self._obs(i) for i in range(self.num_envs)])
        return obs, {}

    def step(self, act):
        rew = torch.tensor([float(self._rngs[i].random()) for i in range(self.num_envs)])
        for i in range(self.num_envs):
            self._step_count[i] += 1
        trunc = torch.tensor([self._step_count[i] >= self._episode_len for i in range(self.num_envs)])
        end = torch.zeros(self.num_envs, dtype=torch.bool)
        next_obs = torch.stack([self._obs(i) for i in range(self.num_envs)])
        info = {}
        dead_idx = torch.nonzero(trunc).flatten().tolist()
        if dead_idx:
            info["final_observation"] = next_obs[trunc]
            for i in dead_idx:
                # Auto-reset on termination, like a real vector env -- a new episode begins
                # immediately from this SAME env's own ongoing generator, not re-seeded
                # externally (only the very first reset() call's seed is externally controlled).
                self._step_count[i] = 0
                next_obs[i] = self._obs(i)
        return next_obs, rew, end, trunc, info


class _ConstantActionModel:
    """Fully deterministic policy stand-in (no RNG at all) -- isolates the test to the
    collector/env-reset-seed mechanism, not any model-side exploration randomness (already
    covered by DrQActorCritic's own RNG-isolation tests)."""

    def __init__(self, lstm_dim, action_dim):
        self.lstm_dim = lstm_dim
        self.device = torch.device("cpu")
        self._action_dim = action_dim

    def predict_act_value(self, obs, hx_cx):
        hx, cx = hx_cx
        dist_params = torch.zeros(obs.size(0), self._action_dim)
        val = torch.zeros(obs.size(0))
        return dist_params, val, (hx, cx)

    def sample_action(self, dist_params):
        return torch.full((dist_params.size(0), self._action_dim), 0.3), None


IMG_C, IMG_S = 3, 8
LSTM_DIM = IMG_C * IMG_S * IMG_S
BASE_SEED = 12345


def _make_collector_and_dataset(tmp_path, name, episode_len, num_envs=1):
    env = _DeterministicFakeEnv(num_envs, IMG_C, IMG_S, episode_len)
    model = _ConstantActionModel(LSTM_DIM, action_dim=2)
    reset_state = EnvResetSeedState(
        np.random.default_rng(derive_component_seed(BASE_SEED, COMPONENT_SEED_ID["train_collector_reset"]))
    )
    ds = Dataset(tmp_path / name, name, cache_in_ram=True)
    collector = make_collector(
        env, model, ds, epsilon=0.0, reset_every_collect=True, flush_before_reset=True,
        reset_seed_state=reset_state, verbose=False,
    )
    return collector, ds, reset_state


def test_uninterrupted_vs_resumed_next_real_collection_is_identical(tmp_path):
    """The required fidelity test: an uninterrupted run and a save -> recreate objects -> load
    checkpoint -> collect run must produce bit-identical next-real-collection results.
    episode_len=17 > steps_per_batch=5 is chosen deliberately: under the OLD (pre-fix) collector
    behavior this env's episode would have stayed partially open (in-flight, un-checkpointed
    collector-local state) across the epoch/checkpoint boundary -- exactly the scenario that
    used to silently diverge between an uninterrupted run and a resumed one."""
    STEPS_PER_BATCH = 5
    EPISODE_LEN = 17

    # Run A: uninterrupted, three batches straight through in the same objects.
    collector_a, ds_a, reset_state_a = _make_collector_and_dataset(tmp_path, "a", EPISODE_LEN)
    collector_a.send(NumToCollect(steps=STEPS_PER_BATCH))  # batch 1
    collector_a.send(NumToCollect(steps=STEPS_PER_BATCH))  # batch 2
    checkpoint = reset_state_a.state_dict()  # "checkpoint" taken right here
    collector_a.send(NumToCollect(steps=STEPS_PER_BATCH))  # batch 3: the "next real collection"
    ep_a = ds_a.load_episode(ds_a.num_episodes - 1)

    # Run B: entirely fresh objects (simulating process recreation), loading ONLY the
    # checkpointed reset-seed state -- deliberately WITHOUT replaying batches 1-2, to prove no
    # other state (buffer, episode_ids, hx/cx, env internals) needs to survive.
    collector_b, ds_b, reset_state_b = _make_collector_and_dataset(tmp_path, "b", EPISODE_LEN)
    reset_state_b.load_state_dict(checkpoint)
    collector_b.send(NumToCollect(steps=STEPS_PER_BATCH))  # the same "next real collection"
    ep_b = ds_b.load_episode(ds_b.num_episodes - 1)

    assert len(ep_a) == STEPS_PER_BATCH and len(ep_b) == STEPS_PER_BATCH
    assert torch.equal(ep_a.obs, ep_b.obs)
    assert torch.equal(ep_a.act, ep_b.act)
    assert torch.equal(ep_a.rew, ep_b.rew)
    assert torch.equal(ep_a.end, ep_b.end)
    assert torch.equal(ep_a.trunc, ep_b.trunc)
    # And a sanity check that this isn't trivially true because every batch looks the same
    # regardless of seed: batch 3's reset seed must differ from batch 1's/2's, so its
    # observations differ from an EARLIER batch's in the SAME run.
    ep_a_batch1 = ds_a.load_episode(0)
    assert not torch.equal(ep_a.obs, ep_a_batch1.obs)


def test_resumed_run_does_not_require_replaying_earlier_batches(tmp_path):
    """Stronger form of the fidelity test: run B collects ONLY the post-checkpoint batch (never
    runs batches 1-2 at all, not even into a throwaway dataset) and still matches run A's batch
    3 exactly -- confirming the checkpoint-safe boundary genuinely carries no dependency on
    collection history, only on the checkpointed reset-seed stream's position."""
    STEPS_PER_BATCH = 4
    EPISODE_LEN = 9

    collector_a, ds_a, reset_state_a = _make_collector_and_dataset(tmp_path, "a2", EPISODE_LEN)
    for _ in range(4):
        collector_a.send(NumToCollect(steps=STEPS_PER_BATCH))
    checkpoint = reset_state_a.state_dict()
    collector_a.send(NumToCollect(steps=STEPS_PER_BATCH))
    ep_a = ds_a.load_episode(ds_a.num_episodes - 1)

    collector_b, ds_b, reset_state_b = _make_collector_and_dataset(tmp_path, "b2", EPISODE_LEN)
    reset_state_b.load_state_dict(checkpoint)
    collector_b.send(NumToCollect(steps=STEPS_PER_BATCH))
    ep_b = ds_b.load_episode(ds_b.num_episodes - 1)

    assert torch.equal(ep_a.obs, ep_b.obs)
    assert torch.equal(ep_a.act, ep_b.act)
    assert torch.equal(ep_a.rew, ep_b.rew)


def test_natural_mid_batch_termination_not_lost_duplicated_or_miscounted(tmp_path):
    """episode_len=3 < steps_per_batch=5: within a single collection batch, the env naturally
    terminates (trunc=True) partway through, auto-resets, and then gets forcibly cut off again
    at the batch boundary -- two episode entries from one batch. Verifies no partial episode is
    silently lost, duplicated, or counted differently: total steps across all stored episodes
    must equal exactly the number of real env steps taken, and episode boundaries must line up
    with the natural termination points."""
    STEPS_PER_BATCH = 5
    EPISODE_LEN = 3

    collector, ds, _ = _make_collector_and_dataset(tmp_path, "c", EPISODE_LEN)
    collector.send(NumToCollect(steps=STEPS_PER_BATCH))

    total_steps_stored = sum(ds.lengths.tolist())
    assert total_steps_stored == STEPS_PER_BATCH  # no loss, no duplication

    # Two episodes: [3 steps, naturally truncated] + [2 steps, forcibly cut at the boundary].
    assert ds.num_episodes == 2
    ep0, ep1 = ds.load_episode(0), ds.load_episode(1)
    assert len(ep0) == 3 and bool(ep0.trunc[-1].item()) is True
    assert len(ep1) == 2 and bool(ep1.trunc[-1].item()) is False


def test_env_reset_seed_state_checkpoint_round_trip():
    """EnvResetSeedState in isolation: save/restore reproduces the exact same future seed
    sequence, and draws before the save do NOT reappear after a restore to that point."""
    rng = np.random.default_rng(derive_component_seed(BASE_SEED, COMPONENT_SEED_ID["train_collector_reset"]))
    state = EnvResetSeedState(rng)

    first = state.next_seeds(2)
    checkpoint = state.state_dict()
    expected_next = state.next_seeds(2)

    # A fresh state loaded from the checkpoint reproduces the same "next" draw exactly.
    rng2 = np.random.default_rng(0)  # deliberately different initial seed -- load must override it
    state2 = EnvResetSeedState(rng2)
    state2.load_state_dict(checkpoint)
    actual_next = state2.next_seeds(2)
    assert actual_next == expected_next
    assert actual_next != first
