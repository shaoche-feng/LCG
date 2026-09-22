"""Tests for the resume-fidelity work: component-local BatchSampler RNG streams, and
WorldModelEnv's preload-block replay / live-rollout-buffer resume mechanism. CPU-only (tiny
real Denoiser/RewEndModel, not mocks -- same code paths as production, just small), fast.
"""
import numpy as np
import torch
from torch.utils.data import DataLoader

from coroutines.env_loop import make_env_loop, RolloutHxCxState
from data import BatchSampler, collate_segments_to_batch, Dataset, Episode
from data.batch_sampler import COMPONENT_SEED_ID
from envs.world_model_env import WorldModelEnv, WorldModelEnvConfig
from models.actor_critic import ActorCritic, ActorCriticConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig
from models.diffusion.inner_model import InnerModelConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from utils import derive_component_seed

IMG_CHANNELS = 3
IMG_SIZE = 8
NUM_STEPS_COND = 1
ACTION_DIM = 2
SEQ_LENGTH = NUM_STEPS_COND + 1
BATCH_SIZE = 4
NUM_BATCHES_TO_PRELOAD = 3  # -> 12 items/block, 3 reset_dead(4)-sized draws per block


def make_tiny_denoiser():
    torch.manual_seed(0)
    inner_cfg = InnerModelConfig(
        img_channels=IMG_CHANNELS, num_steps_conditioning=NUM_STEPS_COND, cond_channels=16,
        depths=[1], channels=[8], attn_depths=[False], continuous_action_dim=ACTION_DIM,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.1)
    d = Denoiser(cfg)
    d.eval()
    return d


def make_tiny_rew_end_model():
    cfg = RewEndModelConfig(
        lstm_dim=8, img_channels=IMG_CHANNELS, img_size=IMG_SIZE, cond_channels=8,
        depths=[1], channels=[8], attn_depths=[False], continuous_action_dim=ACTION_DIM, continuous_reward=True,
    )
    m = RewEndModel(cfg)
    m.eval()
    return m


def make_dataset(tmp_path, n_episodes=6, ep_len=20, name="ds"):
    ds = Dataset(tmp_path / name, name, cache_in_ram=True)
    torch.manual_seed(1)
    for _ in range(n_episodes):
        ds.add_episode(Episode(
            obs=torch.rand(ep_len, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
            act=torch.randn(ep_len, ACTION_DIM),
            rew=torch.zeros(ep_len),
            end=torch.zeros(ep_len, dtype=torch.uint8),
            trunc=torch.zeros(ep_len, dtype=torch.uint8),
            info={},
        ))
    return ds


def make_component_sampler(dataset, component_name, base_seed=123):
    return BatchSampler(
        dataset, 0, 1, BATCH_SIZE, SEQ_LENGTH,
        rng=derive_component_seed(base_seed, COMPONENT_SEED_ID[component_name]),
    )


def make_world_model_env(dataset, rew_end_model, denoiser, base_seed=123):
    sampler = make_component_sampler(dataset, "actor_critic", base_seed)
    data_loader = DataLoader(dataset=dataset, batch_sampler=sampler, collate_fn=collate_segments_to_batch)
    wm_cfg = WorldModelEnvConfig(
        horizon=5, num_batches_to_preload=NUM_BATCHES_TO_PRELOAD,
        diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=1),
    )
    return WorldModelEnv(denoiser, rew_end_model, data_loader, wm_cfg), sampler


# ---------------------------------------------------------------------------------------------
# Component RNG isolation
# ---------------------------------------------------------------------------------------------

def test_component_samplers_draw_different_sequences_from_same_base_seed(tmp_path):
    ds = make_dataset(tmp_path)
    samplers = {name: make_component_sampler(ds, name, base_seed=42) for name in COMPONENT_SEED_ID}
    draws = {name: s.sample() for name, s in samplers.items()}
    seqs = list(draws.values())
    assert seqs[0] != seqs[1] and seqs[1] != seqs[2] and seqs[0] != seqs[2]


def test_component_seed_derivation_is_deterministic_not_python_hash(tmp_path):
    ds = make_dataset(tmp_path)
    a = make_component_sampler(ds, "denoiser", base_seed=7).sample()
    b = make_component_sampler(ds, "denoiser", base_seed=7).sample()
    assert a == b, "same base_seed + same component must always derive the same stream"


def test_global_numpy_rng_unaffected_by_component_local_sampling(tmp_path):
    ds = make_dataset(tmp_path)
    np.random.seed(999)
    before = np.random.get_state()
    for name in COMPONENT_SEED_ID:
        make_component_sampler(ds, name, base_seed=1).sample()
    after = np.random.get_state()
    assert before[1].tolist() == after[1].tolist() and before[2] == after[2]


def test_actor_critic_preload_burst_does_not_change_other_component_streams(tmp_path):
    """Direct regression test for the root cause found in the resume-fidelity integration
    test: with isolated streams, consuming a large actor_critic preload burst must leave
    denoiser/rew_end_model's OWN samplers completely unaffected."""
    ds = make_dataset(tmp_path)
    denoiser_sampler = make_component_sampler(ds, "denoiser", base_seed=55)
    rew_end_sampler = make_component_sampler(ds, "rew_end_model", base_seed=55)
    ac_sampler = make_component_sampler(ds, "actor_critic", base_seed=55)

    denoiser_before = denoiser_sampler.state_dict()
    rew_end_before = rew_end_sampler.state_dict()

    for _ in range(50):  # simulate a large actor_critic preload burst
        ac_sampler.sample()

    assert denoiser_sampler.state_dict()["bit_generator_state"] == denoiser_before["bit_generator_state"]
    assert rew_end_sampler.state_dict()["bit_generator_state"] == rew_end_before["bit_generator_state"]


# ---------------------------------------------------------------------------------------------
# BatchSampler state round-trip
# ---------------------------------------------------------------------------------------------

def test_batch_sampler_state_round_trip(tmp_path):
    ds = make_dataset(tmp_path)
    sampler_a = make_component_sampler(ds, "actor_critic", base_seed=3)
    sampler_a.sample()
    sampler_a.sample()
    sd = sampler_a.state_dict()
    ref = sampler_a.sample()

    sampler_b = make_component_sampler(ds, "actor_critic", base_seed=999)  # deliberately different seed
    sampler_b.load_state_dict(sd)
    got = sampler_b.sample()
    assert got == ref


# ---------------------------------------------------------------------------------------------
# WorldModelEnv preload-block replay
# ---------------------------------------------------------------------------------------------

def test_preload_resume_mid_block_matches_uninterrupted_continuation(tmp_path):
    ds = make_dataset(tmp_path)
    denoiser = make_tiny_denoiser()
    rew_end_model = make_tiny_rew_end_model()

    env_a, sampler_a = make_world_model_env(ds, rew_end_model, denoiser, base_seed=123)
    env_a.reset()  # consumes block items [0:4] of a fresh 12-item block, cursor -> 4

    # Save state at cursor=4 (mid-block: 4 of 12 consumed).
    preload_sd = env_a.preload_state_dict()
    rollout_sd = env_a.rollout_state_dict()
    sampler_sd = sampler_a.state_dict()
    assert preload_sd["preload_cursor"] == 4

    # Reference continuation: env_a's OWN next draw, items [4:8].
    ref_obs, ref_act, (ref_hx, ref_cx) = env_a.generator_init.send(4)

    # Fresh process simulation: brand new WorldModelEnv/sampler, resume-state injected BEFORE
    # any real .send().
    env_b, sampler_b = make_world_model_env(ds, rew_end_model, denoiser, base_seed=42)  # different seed on purpose
    sampler_b.load_state_dict(sampler_sd)
    env_b.load_preload_state_dict(preload_sd)
    env_b.load_rollout_state_dict(rollout_sd)
    got_obs, got_act, (got_hx, got_cx) = env_b.generator_init.send(4)

    assert torch.equal(got_obs, ref_obs)
    assert torch.equal(got_act, ref_act)
    assert torch.equal(got_hx, ref_hx)
    assert torch.equal(got_cx, ref_cx)
    assert env_b._preload_cursor == 8


def test_preload_resume_exactly_at_refill_boundary_matches_fresh_block_draw(tmp_path):
    ds = make_dataset(tmp_path)
    denoiser = make_tiny_denoiser()
    rew_end_model = make_tiny_rew_end_model()

    env_a, sampler_a = make_world_model_env(ds, rew_end_model, denoiser, base_seed=77)
    env_a.reset()  # [0:4], cursor -> 4
    env_a.generator_init.send(4)  # [4:8], cursor -> 8
    env_a.generator_init.send(4)  # [8:12], cursor -> 12 == exhausted

    preload_sd = env_a.preload_state_dict()
    rollout_sd = env_a.rollout_state_dict()
    sampler_sd = sampler_a.state_dict()
    assert preload_sd["preload_cursor"] == 12  # exactly exhausted: the boundary case

    # Reference: env_a's next draw must fetch a brand-new block (fresh sampler draws).
    ref_obs, ref_act, (ref_hx, ref_cx) = env_a.generator_init.send(4)

    env_b, sampler_b = make_world_model_env(ds, rew_end_model, denoiser, base_seed=1000)
    sampler_b.load_state_dict(sampler_sd)
    env_b.load_preload_state_dict(preload_sd)
    env_b.load_rollout_state_dict(rollout_sd)
    got_obs, got_act, (got_hx, got_cx) = env_b.generator_init.send(4)

    assert torch.equal(got_obs, ref_obs)
    assert torch.equal(got_act, ref_act)
    assert torch.equal(got_hx, ref_hx)
    assert torch.equal(got_cx, ref_cx)


def test_preload_state_dict_none_before_any_use(tmp_path):
    ds = make_dataset(tmp_path)
    denoiser = make_tiny_denoiser()
    rew_end_model = make_tiny_rew_end_model()
    env, _ = make_world_model_env(ds, rew_end_model, denoiser)
    assert env.rollout_state_dict() is None  # reset() never called yet


# ---------------------------------------------------------------------------------------------
# RolloutHxCxState
# ---------------------------------------------------------------------------------------------

def test_rollout_hx_cx_state_round_trip():
    state_a = RolloutHxCxState()
    assert not state_a.initialized
    state_a.hx = torch.randn(4, 8)
    state_a.cx = torch.randn(4, 8)
    state_a.initialized = True
    sd = state_a.state_dict()

    state_b = RolloutHxCxState()
    state_b.load_state_dict(sd)
    assert state_b.initialized
    assert torch.equal(state_b.hx, state_a.hx)
    assert torch.equal(state_b.cx, state_a.cx)


# ---------------------------------------------------------------------------------------------
# Backward compatibility
# ---------------------------------------------------------------------------------------------

class _FakeDeterministicEnv:
    """Minimal env_loop-compatible env: never truncates/ends (isolates hx_cx_state's handoff
    timing from death-handling entirely), and returns a FIXED, precomputed observation
    sequence (not random) so two independently-constructed rollouts fed the same seed of
    actions see bit-identical observations -- required to compare an uninterrupted
    continuation against a resumed one call-for-call."""

    def __init__(self, num_envs, obs_shape, num_calls_worth):
        self.num_envs = num_envs
        self.obs_shape = obs_shape
        self.is_discrete = False
        self.action_dim = 2
        self.action_low = torch.tensor([-1.0, -1.0])
        self.action_high = torch.tensor([1.0, 1.0])
        g = torch.Generator().manual_seed(0)
        self._obs_sequence = [torch.rand(num_envs, *obs_shape, generator=g) for _ in range(num_calls_worth)]
        self._t = 0

    def reset(self, seed=None):
        self._t = 0
        return self._obs_sequence[0], {}

    def step(self, act):
        self._t += 1
        obs = self._obs_sequence[self._t % len(self._obs_sequence)]
        rew = torch.zeros(self.num_envs)
        end = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rew, end, trunc, {}

    def prime_for_resume(self, t, obs):
        """Simulate a WorldModelEnv whose rollout buffers were already restored from a
        checkpoint before make_env_loop resumes: set the step counter so this fake env's
        deterministic sequence continues from the right point (instead of restarting at t=0
        via reset()), and expose `obs_buffer` -- make_env_loop's resume path checks
        `hasattr(env, "obs_buffer")` and, when true, reads the current observation from
        `env.obs_buffer[:, -1]` instead of calling env.reset(). Without this, the fake env
        has no obs_buffer attribute, the hasattr check fails, and the resume path silently
        falls through to a fresh reset() -- which is a gap in this test double, not a bug in
        make_env_loop itself, since the real WorldModelEnv always has obs_buffer by the time
        this coroutine resumes."""
        self._t = t
        self.obs_buffer = obs.unsqueeze(1)


def make_tiny_ac(action_dim=2, lstm_dim=8, img_size=8):
    cfg = ActorCriticConfig(
        lstm_dim=lstm_dim, img_channels=3, img_size=img_size, channels=[4], down=[0],
        continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim,
    )
    return ActorCritic(cfg)


def test_env_loop_hx_cx_state_resume_matches_uninterrupted_continuation():
    """Regression test for the off-by-one timing bug the resume-fidelity integration test
    itself caught: a checkpoint taken right after call N must let a FRESH env_loop's call
    N+1 reproduce exactly what the ORIGINAL, uninterrupted env_loop's call N+1 produces --
    not what its call N produced (the pre-fix bug: hx_cx_state was written at the TOP of
    each call, capturing the value that call STARTED from, one call stale)."""
    torch.manual_seed(0)
    model_ref = make_tiny_ac()
    model_ref.critic_linear.weight.data.normal_(0, 0.1)  # break the all-zero init so hx/cx
    model_ref.actor_linear.weight.data.normal_(0, 0.1)   # actually evolve call-to-call

    # Second, identically-initialized model (state_dict copy) for the resumed arm.
    model_resumed = make_tiny_ac()
    model_resumed.load_state_dict(model_ref.state_dict())

    env_ref = _FakeDeterministicEnv(num_envs=3, obs_shape=(3, 8, 8), num_calls_worth=5)
    state_ref = RolloutHxCxState()
    loop_ref = make_env_loop(env_ref, model_ref, hx_cx_state=state_ref)
    loop_ref.send(4)  # call 1
    loop_ref.send(4)  # call 2 -- checkpoint taken right after this

    saved_sd = state_ref.state_dict()
    saved_torch_rng = torch.get_rng_state()  # what Trainer's RNGState would also snapshot here
    # Snapshot the checkpoint-time position BEFORE running call 3 on loop_ref -- env_ref._t
    # keeps advancing once call 3 runs, so reading it afterward would capture call 3's ending
    # position instead of the checkpoint position, silently priming the resumed arm one call
    # too far ahead.
    checkpoint_t = env_ref._t
    checkpoint_obs = env_ref._obs_sequence[checkpoint_t % len(env_ref._obs_sequence)]
    ref_call3_output = loop_ref.send(4)  # call 3 (uninterrupted reference continuation)

    env_resumed = _FakeDeterministicEnv(num_envs=3, obs_shape=(3, 8, 8), num_calls_worth=5)
    # Mirror what Trainer's resume path does for a real WorldModelEnv: restore its rollout
    # buffers (here, just the deterministic sequence position + current obs) BEFORE the
    # env_loop coroutine's first resumed .send() -- without this, env_resumed has no
    # obs_buffer, make_env_loop's `hasattr(env, "obs_buffer")` check fails, and it falls
    # through to a fresh env.reset(), restarting the deterministic sequence at t=0 instead of
    # continuing from where env_ref left off.
    env_resumed.prime_for_resume(t=checkpoint_t, obs=checkpoint_obs)
    state_resumed = RolloutHxCxState()
    state_resumed.load_state_dict(saved_sd)
    torch.set_rng_state(saved_torch_rng)  # mirrors RNGState.load_state_dict's restoration
    loop_resumed = make_env_loop(env_resumed, model_resumed, hx_cx_state=state_resumed)
    resumed_call_output = loop_resumed.send(4)  # the FIRST call after "resume" == call 3

    # Compare the val (critic output) tensor from each -- a function of the model forward
    # pass on (obs, hx, cx), so equal iff hx/cx (and everything else) matched exactly.
    ref_val = ref_call3_output[6]
    resumed_val = resumed_call_output[6]
    assert torch.equal(ref_val, resumed_val)


def test_resume_fidelity_state_load_tolerates_missing_keys(tmp_path):
    """Simulates loading an old-format checkpoint dict (predates this feature): the inner
    load_state_dict must not raise even with an empty/partial dict -- Trainer's own
    load_state_checkpoint() is what actually prints the user-facing warning and injects a
    full placeholder; this test covers ResumeFidelityState's own defensive handling of the
    same scenario in isolation."""
    from trainer import ResumeFidelityState

    ds = make_dataset(tmp_path)
    sampler = make_component_sampler(ds, "actor_critic", base_seed=5)
    state = ResumeFidelityState(batch_samplers={"actor_critic": sampler}, world_model_env=None, rollout_hx_cx_state=None)
    state.load_state_dict({})  # old-format: no keys at all -- must not raise
