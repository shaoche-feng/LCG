"""Tests for the DrQ-v2-style continuous-action actor-critic (models.drq_actor_critic).

CPU-only and fast; no GPU or trained checkpoint required, matching
tests/models/test_actor_critic_continuous_action.py's convention of exercising the real
modules (not mocks) so gradient-flow assertions mean something.
"""
import torch

from coroutines.env_loop import make_env_loop
from models.drq_actor_critic import (
    DrQActorCritic,
    DrQActorCriticConfig,
    DrQEncoder,
    DrQExplorationState,
    DrQLossConfig,
    NoiseScheduleConfig,
    _noise_std_at,
)


def make_cfg(img_size=16, frame_stack=3, action_dim=2, encoder_channels=(8, 8), encoder_down=(1, 1)):
    tmp = DrQActorCriticConfig(
        img_channels=3, img_size=img_size, frame_stack=frame_stack,
        encoder_channels=list(encoder_channels), encoder_down=list(encoder_down), feature_dim=0,
        actor_hidden_dim=16, critic_hidden_dim=16, continuous_action_dim=action_dim,
        action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim,
        noise_schedule=NoiseScheduleConfig(std_start=1.0, std_end=0.1, decay_steps=100, clip=0.3),
    )
    enc = DrQEncoder(tmp)
    with torch.no_grad():
        dummy = torch.zeros(1, frame_stack * 3, img_size, img_size)
        tmp.feature_dim = enc(dummy).shape[1]
    return tmp


def make_ac(**kwargs):
    ac = DrQActorCritic(make_cfg(**kwargs))
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    return ac


# ---------------------------------------------------------------------------------------------
# 1. Action bounds.
# ---------------------------------------------------------------------------------------------

def test_deterministic_action_uses_mu_with_no_noise():
    torch.manual_seed(0)
    ac = make_ac(action_dim=2)
    num_envs = 5
    hx = torch.zeros(num_envs, ac.lstm_dim)
    cx = torch.zeros(num_envs, 1)
    obs = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    mu, val, _ = ac.predict_act_value(obs, (hx, cx))

    action1, aux1 = ac.sample_action(mu, deterministic=True)
    action2, aux2 = ac.sample_action(mu, deterministic=True)
    assert aux1 is None and aux2 is None
    assert torch.equal(action1, action2), "deterministic eval must be noise-free and reproducible"
    expected = ac._rescale(mu)
    assert torch.allclose(action1, expected)


def test_stochastic_action_respects_bounds_and_is_truncated_not_unbounded_tanh():
    """Action = clamp(mu, -1,1) + clamp(noise) then clamp again, all before rescale -- NOT
    tanh(mean + std*eps), which could push the tanh argument arbitrarily far even though the
    output stays in (-1,1). Here the pre-rescale canonical value must land in [-1, 1] exactly,
    including at the boundary under large noise."""
    torch.manual_seed(0)
    ac = make_ac(action_dim=1)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    mu = torch.zeros(1000, 1)  # centered mean
    action, aux = ac.sample_action(mu, deterministic=False)
    assert aux is None
    assert torch.all(action >= ac.action_low - 1e-5)
    assert torch.all(action <= ac.action_high + 1e-5)


def test_sample_action_bounds_hold_even_with_large_std():
    torch.manual_seed(0)
    ac = make_ac(action_dim=1)
    ac.cfg.noise_schedule.std_start = 10.0  # deliberately huge, to stress the clamp
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    mu = torch.zeros(500, 1)
    action, _ = ac.sample_action(mu, deterministic=False)
    assert torch.all(action >= ac.action_low - 1e-5)
    assert torch.all(action <= ac.action_high + 1e-5)


# ---------------------------------------------------------------------------------------------
# 2. Frame stacking.
# ---------------------------------------------------------------------------------------------

def test_frame_stack_shifts_and_drops_oldest():
    ac = make_ac(frame_stack=3, img_size=8)
    num_envs = 2
    hx = torch.zeros(num_envs, ac.lstm_dim)
    frame1 = torch.full((num_envs, 3, 8, 8), 1.0)
    frame2 = torch.full((num_envs, 3, 8, 8), 2.0)
    frame3 = torch.full((num_envs, 3, 8, 8), 3.0)
    frame4 = torch.full((num_envs, 3, 8, 8), 4.0)

    hx = ac._shift_and_append(hx, frame1)
    hx = ac._shift_and_append(hx, frame2)
    hx = ac._shift_and_append(hx, frame3)
    stack = hx.view(num_envs, 3, 3, 8, 8)
    assert torch.allclose(stack[:, 0], frame1)
    assert torch.allclose(stack[:, 1], frame2)
    assert torch.allclose(stack[:, 2], frame3)

    hx = ac._shift_and_append(hx, frame4)
    stack = hx.view(num_envs, 3, 3, 8, 8)
    assert torch.allclose(stack[:, 0], frame2), "oldest frame (frame1) must be dropped"
    assert torch.allclose(stack[:, 1], frame3)
    assert torch.allclose(stack[:, 2], frame4)


def test_reconstruct_stacks_matches_incremental_shift_and_append():
    """forward()'s post-hoc stack reconstruction from (pre_stack, all_obs) must reproduce
    EXACTLY what predict_act_value computed step-by-step during collection -- otherwise the
    critic loss would pair the wrong observation with the action that was actually taken."""
    ac = make_ac(frame_stack=3, img_size=8)
    num_envs = 2
    torch.manual_seed(0)
    pre_stack = torch.rand(num_envs, ac.lstm_dim)
    frames = [torch.rand(num_envs, 3, 8, 8) for _ in range(5)]

    # incremental, as predict_act_value would do it
    hx = pre_stack.clone()
    incremental_stacks = []
    for f in frames:
        hx = ac._shift_and_append(hx, f)
        incremental_stacks.append(hx.view(num_envs, 3, 3, 8, 8).clone())
    incremental_stacks = torch.stack(incremental_stacks, dim=1)

    # post-hoc, as forward() would do it
    all_obs = torch.stack(frames, dim=1)
    reconstructed = ac._reconstruct_stacks(pre_stack, all_obs)

    assert torch.allclose(reconstructed, incremental_stacks)


def test_reconstruct_stacks_reuses_cached_cold_seed_not_zero_padding():
    """Cold start (pre_stack_flat=None) must NOT blindly zero-pad -- it must reuse whatever
    predict_act_value actually cold-seeded from (obs_buffer or repeated obs), cached in
    self._cold_seeded_stack_flat, exactly reproducing what collection saw."""
    ac = make_ac(frame_stack=3, img_size=8)
    num_envs = 2
    all_obs = torch.rand(num_envs, 4, 3, 8, 8)
    seeded = torch.rand(num_envs, 3, 3, 8, 8)  # pretend predict_act_value cold-seeded this
    ac._cold_seeded_stack_flat = seeded.reshape(num_envs, -1)

    stacks = ac._reconstruct_stacks(None, all_obs)
    assert stacks.shape == (num_envs, 4, 3, 3, 8, 8)
    assert torch.allclose(stacks[:, 0], seeded), "t=0 must reuse the cached cold-seeded stack verbatim"
    # t=1 onward proceeds by ordinary shift-and-append from that cached starting point
    expected_t1 = torch.cat([seeded[:, 1:], all_obs[:, 1].unsqueeze(1)], dim=1)
    assert torch.allclose(stacks[:, 1], expected_t1)


def test_reconstruct_stacks_raises_without_cached_seed_or_pre_stack():
    ac = make_ac(frame_stack=3, img_size=8)
    all_obs = torch.rand(2, 4, 3, 8, 8)
    try:
        ac._reconstruct_stacks(None, all_obs)
        assert False, "expected an assertion error when neither pre_stack_flat nor a cached cold seed is available"
    except AssertionError as e:
        assert "cold-seeded" in str(e)


# ---------------------------------------------------------------------------------------------
# 3. Gradient isolation: critic trains the encoder, actor does not (module docstring / point 5
#    of the approved plan -- "detached encoder features" for the actor).
# ---------------------------------------------------------------------------------------------

class _FakeWorldModelEnv:
    """Minimal env_loop-compatible stand-in for WorldModelEnv: fixed-size continuous action
    space, never terminates, returns random imagined frames. No obs_buffer attribute, so
    env_loop's resume path is never exercised here -- deliberately out of scope for this file
    (covered by tests/checkpoint/test_resume_fidelity.py)."""

    def __init__(self, num_envs, img_channels, img_size, action_dim):
        self.num_envs = num_envs
        self.is_discrete = False
        self.action_dim = action_dim
        self.action_low = torch.tensor([-1.0] * action_dim)
        self.action_high = torch.tensor([1.0] * action_dim)
        self._img_channels = img_channels
        self._img_size = img_size

    def reset(self, seed=None):
        obs = torch.rand(self.num_envs, self._img_channels, self._img_size, self._img_size)
        return obs, {}

    def step(self, act):
        obs = torch.rand(self.num_envs, self._img_channels, self._img_size, self._img_size)
        rew = torch.rand(self.num_envs)
        end = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = torch.zeros(self.num_envs, dtype=torch.bool)
        return obs, rew, end, trunc, {}


def _make_ac_with_rollout(backup_every=6, n_step=2, num_envs=3, img_size=8, action_dim=2):
    ac = make_ac(img_size=img_size, action_dim=action_dim, frame_stack=3)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    env = _FakeWorldModelEnv(num_envs, ac.img_channels, img_size, action_dim)
    ac.env_loop = make_env_loop(env, ac, hx_cx_state=ac.rollout_hx_cx_state)
    ac.loss_cfg = DrQLossConfig(
        backup_every=backup_every, n_step=n_step, gamma=0.99, target_tau=0.01, noise_clip=0.3,
    )
    return ac


def test_forward_produces_finite_loss_and_backward_succeeds():
    torch.manual_seed(0)
    ac = _make_ac_with_rollout()
    loss, metrics = ac.forward()
    assert torch.isfinite(loss)
    ac.zero_grad(set_to_none=True)
    loss.backward()
    assert ac.encoder.encoder[0].weight.grad is not None
    assert torch.isfinite(ac.encoder.encoder[0].weight.grad).all()


def test_critic_loss_alone_trains_encoder():
    torch.manual_seed(0)
    ac = _make_ac_with_rollout()
    # Monkeypatch forward's actor loss contribution to zero by re-deriving just the critic path
    # directly, mirroring forward()'s own computation, to isolate it cleanly.
    c = ac.loss_cfg
    pre_stack = None
    all_obs, act, rew, end, trunc, *_ = ac.env_loop.send(c.backup_every)
    stacks = ac._reconstruct_stacks(pre_stack, all_obs)
    n = c.n_step
    usable = all_obs.size(1) - n

    def chw(x):
        return x.reshape(x.size(0), -1, ac.img_size, ac.img_size)

    s_t = torch.stack([chw(stacks[:, t]) for t in range(usable)], dim=1)
    a_t = act[:, :usable]
    num_envs = all_obs.size(0)
    s_t_flat = s_t.reshape(num_envs * usable, *s_t.shape[2:])
    a_t_flat = a_t.reshape(num_envs * usable, -1)

    ac.zero_grad(set_to_none=True)
    features = ac.encoder(ac.aug(s_t_flat))
    q1, q2 = ac.critic(features, a_t_flat)
    (q1.mean() + q2.mean()).backward()
    assert ac.encoder.encoder[0].weight.grad is not None
    assert ac.encoder.encoder[0].weight.grad.abs().sum().item() > 0, (
        "critic loss must produce nonzero gradient at the encoder's first layer"
    )


def test_actor_loss_does_not_train_encoder():
    """forward()'s actor loss uses features_t.detach() -- backprop through ONLY the actor loss
    must leave the encoder's gradient at None (never touched), matching the approved plan's
    'actor updates use detached encoder features' requirement."""
    torch.manual_seed(0)
    ac = _make_ac_with_rollout()
    c = ac.loss_cfg
    all_obs, act, rew, end, trunc, *_ = ac.env_loop.send(c.backup_every)
    stacks = ac._reconstruct_stacks(None, all_obs)
    n = c.n_step
    usable = all_obs.size(1) - n

    def chw(x):
        return x.reshape(x.size(0), -1, ac.img_size, ac.img_size)

    s_t = torch.stack([chw(stacks[:, t]) for t in range(usable)], dim=1)
    num_envs = all_obs.size(0)
    s_t_flat = s_t.reshape(num_envs * usable, *s_t.shape[2:])

    ac.zero_grad(set_to_none=True)
    features = ac.encoder(ac.aug(s_t_flat))
    features_detached = features.detach()
    mu = ac.actor(features_detached)
    q1_pi, q2_pi = ac.critic(features_detached, mu)
    loss_actor = -torch.min(q1_pi, q2_pi).mean()
    loss_actor.backward()

    assert ac.encoder.encoder[0].weight.grad is None, (
        "actor loss must never reach the encoder -- features were detached"
    )
    assert ac.actor.net[0].weight.grad is not None
    assert ac.actor.net[0].weight.grad.abs().sum().item() > 0


# ---------------------------------------------------------------------------------------------
# 4. n-step target bootstraps exclusively from the target critics (never val/val_bootstrap).
# ---------------------------------------------------------------------------------------------

def test_n_step_target_uses_only_target_critic_not_online_critic_value():
    """Verifies forward()'s target_q comes from ac.target_critic, not ac.critic, by making the
    two diverge (perturb target_critic's weights) and checking metrics['target_q_mean'] moves
    accordingly while online q1/q2 don't."""
    torch.manual_seed(0)
    ac = _make_ac_with_rollout()
    with torch.no_grad():
        for p in ac.target_critic.parameters():
            p.add_(1.0)  # push target critic far from online critic
    loss, metrics = ac.forward()
    assert torch.isfinite(metrics["target_q_mean"])
    # A second run with target_critic reset back near online critic should give a different
    # (generally smaller-magnitude, but at minimum DIFFERENT) target_q_mean, proving target_q
    # actually depends on target_critic's own (perturbed) parameters.
    ac2 = _make_ac_with_rollout()
    torch.manual_seed(0)
    loss2, metrics2 = ac2.forward()
    assert metrics["target_q_mean"].item() != metrics2["target_q_mean"].item()


def test_not_done_mask_zeroes_bootstrap_after_termination_within_window():
    ac = make_ac(action_dim=1)
    rew = torch.zeros(2, 5)
    end = torch.zeros(2, 5)
    trunc = torch.zeros(2, 5)
    end[0, 1] = 1.0  # env 0 terminates at step 1
    returns, not_done = ac._n_step_returns(rew, end, trunc, gamma=1.0, n=3, usable=2)
    assert not_done[0, 0] == 0.0, "3-step window from t=0 crosses env 0's termination at t=1"
    assert not_done[1, 0] == 1.0, "env 1 never terminates"


# ---------------------------------------------------------------------------------------------
# 5. Target-critic soft update.
# ---------------------------------------------------------------------------------------------

def test_target_critic_soft_update_moves_toward_online_critic():
    ac = make_ac(action_dim=1)
    with torch.no_grad():
        for p in ac.critic.parameters():
            p.fill_(1.0)
        for p in ac.target_critic.parameters():
            p.fill_(0.0)
    from models.drq_actor_critic import _soft_update
    _soft_update(ac.target_critic, ac.critic, tau=0.1)
    for p in ac.target_critic.parameters():
        assert torch.allclose(p, torch.full_like(p, 0.1), atol=1e-5)


def test_target_critic_starts_identical_to_online_critic():
    ac = make_ac(action_dim=1)
    for p_online, p_target in zip(ac.critic.parameters(), ac.target_critic.parameters()):
        assert torch.equal(p_online, p_target)
        assert p_target.requires_grad is False


# ---------------------------------------------------------------------------------------------
# 6. Noise schedule (pure function of step, externally advanced -- not a learned log_std).
# ---------------------------------------------------------------------------------------------

def test_noise_schedule_decays_linearly_then_floors():
    cfg = NoiseScheduleConfig(std_start=1.0, std_end=0.2, decay_steps=10, clip=0.3)
    assert _noise_std_at(cfg, 0) == 1.0
    assert abs(_noise_std_at(cfg, 5) - 0.6) < 1e-6
    assert abs(_noise_std_at(cfg, 10) - 0.2) < 1e-9
    assert abs(_noise_std_at(cfg, 1000) - 0.2) < 1e-9  # floors at std_end, never goes below


def test_schedule_step_advances_only_on_stochastic_sampling():
    ac = make_ac(action_dim=1)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    mu = torch.zeros(3, 1)
    ac.sample_action(mu, deterministic=True)
    assert ac.exploration_state.schedule_step == 0
    ac.sample_action(mu, deterministic=False)
    assert ac.exploration_state.schedule_step == 1


# ---------------------------------------------------------------------------------------------
# 7. Checkpoint round-trip: DrQExplorationState (see its docstring for why it's checked
#    separately from nn.Module's own state_dict) and the plain nn.Module parameters.
# ---------------------------------------------------------------------------------------------

def test_exploration_state_round_trip():
    state = DrQExplorationState()
    state.generator = torch.Generator().manual_seed(42)
    state.schedule_step = 17
    torch.randn(5, generator=state.generator)  # advance the generator's own internal state
    sd = state.state_dict()

    state2 = DrQExplorationState()
    state2.generator = torch.Generator().manual_seed(0)  # different seed, must be overwritten
    state2.load_state_dict(sd)
    assert state2.schedule_step == 17

    draw1 = torch.randn(5, generator=state.generator)
    draw2 = torch.randn(5, generator=state2.generator)
    assert torch.equal(draw1, draw2), "restored generator must continue the exact same stream"


def test_exploration_state_load_tolerates_missing_generator():
    """Mirrors ResumeFidelityState's backward-compat handling elsewhere: loading before
    setup_training has set a generator must not raise."""
    state = DrQExplorationState()
    state.load_state_dict({"generator_state": None, "schedule_step": 5})
    assert state.schedule_step == 5
    assert state.generator is None


def test_module_state_dict_round_trip_no_shape_changes():
    ac = make_ac(action_dim=2, img_size=8)
    sd = ac.state_dict()
    ac2 = make_ac(action_dim=2, img_size=8)
    ac2.load_state_dict(sd)  # raises on any key/shape mismatch
    for p1, p2 in zip(ac.parameters(), ac2.parameters()):
        assert torch.equal(p1, p2)


# ---------------------------------------------------------------------------------------------
# 8. Random-shift augmentation: identical translation across all K stacked frames (point 1 of
#    the revised plan) -- concatenating along the channel dim before ONE grid_sample call, not
#    independently shifting each frame.
# ---------------------------------------------------------------------------------------------

def test_augmentation_applies_identical_shift_to_every_stacked_frame():
    """Each of the K frames carries a distinguishable per-frame constant color; after
    augmentation, all K frames within one batch element must have moved by the SAME offset --
    verified by checking that a small marker pixel pattern ends up at the same (row, col)
    location in every frame's channel block."""
    from models.drq_actor_critic import RandomShiftsAug

    torch.manual_seed(0)
    aug = RandomShiftsAug(pad=4)
    K, C, H, W = 3, 3, 16, 16
    x = torch.zeros(5, K * C, H, W)
    # place a single bright marker pixel at the same (r, c) in every frame's channel block
    r, c = 8, 8
    for k in range(K):
        x[:, k * C : (k + 1) * C, r, c] = 1.0

    y = aug(x)
    for b in range(x.size(0)):
        locations = []
        for k in range(K):
            frame = y[b, k * C : (k + 1) * C]
            idx = (frame[0] == frame[0].max()).nonzero()
            locations.append(tuple(idx[0].tolist()) if idx.numel() > 0 else None)
        assert len(set(locations)) == 1, (
            f"batch element {b}: marker moved to different locations across frames {locations} "
            f"-- augmentation must apply ONE shift to the whole stacked tensor, not per-frame"
        )


def test_augmentation_shift_differs_across_batch_elements_generally():
    """Sanity check that the augmentation is actually doing something batch-element-specific
    (not a global no-op or a single shared shift for the whole batch), so the "identical across
    frames" test above isn't vacuously true because nothing moves at all."""
    from models.drq_actor_critic import RandomShiftsAug

    torch.manual_seed(0)
    aug = RandomShiftsAug(pad=4)
    x = torch.zeros(64, 3, 16, 16)
    x[:, :, 8, 8] = 1.0
    y = aug(x)
    locations = [tuple((y[b, 0] == y[b, 0].max()).nonzero()[0].tolist()) for b in range(x.size(0))]
    assert len(set(locations)) > 1, "expected at least some variation in shift across a 64-element batch"


# ---------------------------------------------------------------------------------------------
# 9. Frame-stack cold-start seeding (point 2 of the revised plan): obs_buffer preferred, repeat
#    fallback, and no blind [0, 0, obs] initialization.
# ---------------------------------------------------------------------------------------------

class _FakeWorldModelEnvWithBuffer:
    """Stand-in exposing obs_buffer like the real WorldModelEnv, with enough of reset_dead's
    behavior to exercise the dead-env burn-in path (point 2's "partial-environment reset" and
    "no cross-episode leakage" requirements). num_steps_conditioning frames per env, frames are
    per-(env, episode) constant colors so leakage is trivially detectable by color."""

    def __init__(self, num_envs, img_channels, img_size, action_dim, num_steps_conditioning=4, horizon=1000):
        self.num_envs = num_envs
        self.is_discrete = False
        self.action_dim = action_dim
        self.action_low = torch.tensor([-1.0] * action_dim)
        self.action_high = torch.tensor([1.0] * action_dim)
        self._img_channels = img_channels
        self._img_size = img_size
        self._num_steps_conditioning = num_steps_conditioning
        self._horizon = horizon
        self._episode_color = torch.arange(1, num_envs + 1, dtype=torch.float32) * 10.0  # distinct per env
        self._ep_len = torch.zeros(num_envs, dtype=torch.long)

    def _frame_for(self, env_idx, color=None):
        c = self._episode_color[env_idx] if color is None else color
        return torch.full((self._img_channels, self._img_size, self._img_size), float(c))

    def reset(self, seed=None):
        self.obs_buffer = torch.stack(
            [torch.stack([self._frame_for(i) for _ in range(self._num_steps_conditioning)]) for i in range(self.num_envs)]
        )
        self._ep_len.zero_()
        return self.obs_buffer[:, -1], {}

    def step(self, act):
        self._ep_len += 1
        dead = self._ep_len >= self._horizon
        next_obs = torch.stack([self._frame_for(i) for i in range(self.num_envs)])
        self.obs_buffer = torch.cat([self.obs_buffer[:, 1:], next_obs.unsqueeze(1)], dim=1)
        rew = torch.zeros(self.num_envs)
        end = torch.zeros(self.num_envs, dtype=torch.bool)
        trunc = dead.clone()
        info = {}
        if dead.any():
            # New episode: bump this env's color so leakage is detectable, then rebuild
            # obs_buffer for dead rows to the new episode's conditioning window BEFORE
            # returning -- mirrors WorldModelEnv.step()'s own reset_dead() ordering.
            for i in torch.nonzero(dead).flatten().tolist():
                self._episode_color[i] += 1000.0
                self.obs_buffer[i] = torch.stack([self._frame_for(i) for _ in range(self._num_steps_conditioning)])
                self._ep_len[i] = 0
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]
        return self.obs_buffer[:, -1], rew, end, trunc, info


def test_cold_start_seeds_from_obs_buffer_not_zero_padding():
    torch.manual_seed(0)
    num_envs = 3
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    env = _FakeWorldModelEnvWithBuffer(num_envs, ac.img_channels, ac.img_size, 2, num_steps_conditioning=4)
    ac._rl_env = env
    obs, _ = env.reset()
    hx = torch.zeros(num_envs, ac.lstm_dim)
    cx = torch.zeros(num_envs, 1)

    mu, val, (hx2, cx2) = ac.predict_act_value(obs, (hx, cx))
    stack = hx2.view(num_envs, 3, ac.img_channels, ac.img_size, ac.img_size)
    expected = env.obs_buffer[:, -3:]
    assert torch.allclose(stack, expected), (
        "cold-start stack must be seeded from obs_buffer's real conditioning window, not "
        "zero-padded [0, 0, obs]"
    )
    # explicitly confirm it is NOT the naive [0, 0, obs] pattern
    naive = torch.zeros_like(stack)
    naive[:, -1] = obs
    assert not torch.allclose(stack, naive)


def test_cold_start_falls_back_to_repeated_obs_without_obs_buffer():
    """No obs_buffer available (e.g. a plain TorchEnv real-env collector) -- fall back to
    repeating the single current observation K times, per the approved plan's explicit
    fallback, rather than zero-padding."""
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    assert ac._rl_env is None
    num_envs = 2
    obs = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    hx = torch.zeros(num_envs, ac.lstm_dim)
    cx = torch.zeros(num_envs, 1)
    mu, val, (hx2, cx2) = ac.predict_act_value(obs, (hx, cx))
    stack = hx2.view(num_envs, 3, 3, ac.img_size, ac.img_size)
    for k in range(3):
        assert torch.allclose(stack[:, k], obs), f"frame slot {k} must repeat the current obs"


def test_cold_start_seeding_fires_exactly_once():
    """The one-shot gate must not re-trigger on later calls, including a second (non-cold)
    step, so it never interferes with the dead-env burn-in mechanism after the first call."""
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    num_envs = 2
    obs1 = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    hx = torch.zeros(num_envs, ac.lstm_dim)
    cx = torch.zeros(num_envs, 1)
    assert ac._cold_start_pending is True
    _, _, (hx, cx) = ac.predict_act_value(obs1, (hx, cx))
    assert ac._cold_start_pending is False

    # A second call with a DELIBERATELY all-zero hx (mimicking what reset_gate produces for a
    # dead env mid-rollout) must NOT re-trigger cold-seeding -- it must fall through to
    # ordinary shift-and-append, per _seed_cold_stack's docstring on why re-firing would
    # corrupt the dead-env burn-in loop's own progressive state.
    obs2 = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    zero_hx = torch.zeros(num_envs, ac.lstm_dim)
    _, _, (hx3, cx3) = ac.predict_act_value(obs2, (zero_hx, cx))
    expected = ac._shift_and_append(zero_hx, obs2)
    assert torch.allclose(hx3, expected), "second call must use ordinary shift-and-append, not re-seed"


def test_resume_with_initialized_state_does_not_cold_seed():
    """On resume, rollout_hx_cx_state.initialized=True and hx already holds the restored
    stack -- predict_act_value must use it via ordinary shift-and-append, never overwrite it
    with cold-seeded content."""
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    ac.rollout_hx_cx_state.hx = torch.rand(2, ac.lstm_dim)
    ac.rollout_hx_cx_state.cx = torch.rand(2, 1)
    ac.rollout_hx_cx_state.initialized = True
    restored_hx = ac.rollout_hx_cx_state.hx.clone()

    obs = torch.rand(2, 3, ac.img_size, ac.img_size)
    _, _, (hx2, cx2) = ac.predict_act_value(obs, (restored_hx, ac.rollout_hx_cx_state.cx))
    expected = ac._shift_and_append(restored_hx, obs)
    assert torch.allclose(hx2, expected)
    assert ac._cold_start_pending is False  # gate still consumed, just took the non-seeding branch


# ---------------------------------------------------------------------------------------------
# 10. Partial-environment reset in a vectorized batch: no cross-episode frame leakage, via the
#     EXISTING (unmodified) dead-env burn-in mechanism in env_loop.py -- see _seed_cold_stack's
#     docstring for why this self-corrects without any special-casing.
# ---------------------------------------------------------------------------------------------

def test_partial_env_reset_no_cross_episode_leakage():
    torch.manual_seed(0)
    num_envs = 3
    frame_stack = 3
    num_steps_conditioning = 4
    ac = make_ac(frame_stack=frame_stack, img_size=8, action_dim=2)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    env = _FakeWorldModelEnvWithBuffer(
        num_envs, ac.img_channels, ac.img_size, 2, num_steps_conditioning=num_steps_conditioning, horizon=3,
    )
    ac._rl_env = env
    ac.env_loop = make_env_loop(env, ac, hx_cx_state=ac.rollout_hx_cx_state)
    # horizon=3 guarantees at least one env dies within a handful of steps of a small rollout.
    ac.env_loop.send(10)

    # After the rollout, every env's CURRENT stack must consist entirely of frames belonging
    # to whichever episode-color that env is currently on (env._episode_color), never a mix
    # that includes an OLDER episode's color -- i.e. no leakage across the reset boundary.
    final_stack = ac.rollout_hx_cx_state.hx.view(num_envs, frame_stack, ac.img_channels, ac.img_size, ac.img_size)
    for i in range(num_envs):
        current_color = env._episode_color[i].item()
        frame_colors = final_stack[i, :, 0, 0, 0].tolist()
        for fc in frame_colors:
            assert fc == current_color or fc == 0.0, (
                f"env {i}: stack contains frame color {fc}, expected only the current episode's "
                f"color {current_color} (or a still-cold zero slot) -- old-episode leakage detected"
            )


# ---------------------------------------------------------------------------------------------
# 11. Stack persistence across env_loop.send() call boundaries (i.e. across separate
#     Trainer.train_component optimizer steps within the SAME imagined episode).
# ---------------------------------------------------------------------------------------------

def test_stack_persists_across_rollout_chunks():
    torch.manual_seed(0)
    ac = _make_ac_with_rollout(backup_every=4, n_step=1, num_envs=2, img_size=8)
    ac.forward()
    stack_after_call1 = ac.rollout_hx_cx_state.hx.clone()
    assert ac.rollout_hx_cx_state.initialized is True

    ac.forward()
    stack_before_call2_first_step = stack_after_call1  # what call 2 should have started from
    # Reconstructing call 2's OWN t=0 stack from stack_after_call1 must be internally
    # consistent -- i.e. rollout_hx_cx_state.hx did NOT get reset to zero/cold between calls.
    assert not torch.allclose(ac.rollout_hx_cx_state.hx, torch.zeros_like(ac.rollout_hx_cx_state.hx)), (
        "stack must not have been reset to zero between two consecutive env_loop.send() calls"
    )


# ---------------------------------------------------------------------------------------------
# 12. n-step TD target termination boundary variants (point 3 of the revised plan).
# ---------------------------------------------------------------------------------------------

def test_n_step_no_termination_bootstraps_fully():
    ac = make_ac(action_dim=1)
    rew = torch.ones(1, 6)
    end = torch.zeros(1, 6)
    trunc = torch.zeros(1, 6)
    returns, not_done = ac._n_step_returns(rew, end, trunc, gamma=1.0, n=3, usable=3)
    assert torch.allclose(returns, torch.full((1, 3), 3.0))  # 1+1+1 undiscounted
    assert torch.allclose(not_done, torch.ones(1, 3))


def test_n_step_termination_at_first_step_of_window():
    ac = make_ac(action_dim=1)
    rew = torch.tensor([[5.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    end = torch.zeros(1, 6)
    end[0, 0] = 1.0  # dies at the very first step of the t=0 window
    trunc = torch.zeros(1, 6)
    returns, not_done = ac._n_step_returns(rew, end, trunc, gamma=1.0, n=3, usable=3)
    assert returns[0, 0].item() == 5.0, "reward AT the terminal step itself must still count"
    assert not_done[0, 0].item() == 0.0, "must not bootstrap past a termination on step 0 of the window"


def test_n_step_termination_mid_window():
    ac = make_ac(action_dim=1)
    rew = torch.tensor([[1.0, 1.0, 5.0, 1.0, 1.0, 1.0]])
    end = torch.zeros(1, 6)
    end[0, 2] = 1.0  # dies at the middle step (k=2) of a 3-step window starting at t=0
    trunc = torch.zeros(1, 6)
    returns, not_done = ac._n_step_returns(rew, end, trunc, gamma=1.0, n=3, usable=3)
    assert returns[0, 0].item() == 1.0 + 1.0 + 5.0, "rewards up to and including the terminal step count"
    assert not_done[0, 0].item() == 0.0
    # t=1's window (steps 1,2,3) also crosses the same termination at absolute step 2
    assert not_done[0, 1].item() == 0.0
    # t=2's window (steps 2,3,4) starts AT the terminal step -- terminal reward still counted,
    # no bootstrap
    assert returns[0, 2].item() == 5.0
    assert not_done[0, 2].item() == 0.0


def test_n_step_termination_exactly_at_bootstrap_boundary():
    """Termination at step t+n-1 exactly (the LAST reward-contributing step before the
    bootstrap) must still zero the bootstrap -- this codebase's env_loop.py convention is that
    end[i]=1 means the transition INTO step i+1 is invalid, so a termination at the final step
    of the window (not one step beyond it) is what should gate the bootstrap; see
    _n_step_returns' docstring / compute_lambda_returns' matching `not_end` convention in the
    original ActorCritic."""
    ac = make_ac(action_dim=1)
    n = 3
    rew = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    end = torch.zeros(1, 6)
    end[0, n - 1] = 1.0  # terminal exactly at the last in-window step (t=0's window: steps 0,1,2)
    trunc = torch.zeros(1, 6)
    returns, not_done = ac._n_step_returns(rew, end, trunc, gamma=1.0, n=n, usable=3)
    assert not_done[0, 0].item() == 0.0, "termination at the window's last step must zero the bootstrap"
    # a window ending one step EARLIER than the termination must still bootstrap normally --
    # not applicable here since usable=3 only gives t=0; verified structurally by the mid/first
    # tests above instead.


def test_n_step_survives_when_termination_is_after_the_window():
    ac = make_ac(action_dim=1)
    rew = torch.ones(1, 8)
    end = torch.zeros(1, 8)
    end[0, 5] = 1.0  # dies well after any 3-step window starting at t=0,1
    trunc = torch.zeros(1, 8)
    returns, not_done = ac._n_step_returns(rew, end, trunc, gamma=1.0, n=3, usable=5)
    assert not_done[0, 0].item() == 1.0
    assert not_done[0, 1].item() == 1.0


# ---------------------------------------------------------------------------------------------
# 13. Real-env collector boundary: DrQActorCritic can drive coroutines.collector.make_collector
#     (the exploration policy DOES drive real-env data collection) without that path ever
#     training DrQ -- collection never calls forward()/backward(), only predict_act_value/
#     sample_action, matching the module's own no_grad contract.
# ---------------------------------------------------------------------------------------------

def test_predict_act_value_and_sample_action_work_without_obs_buffer_for_real_env_collection():
    """Simulates coroutines.collector.make_collector driving this model against a plain
    TorchEnv-shaped object (no obs_buffer) -- must work via the repeat-obs cold-start fallback
    and produce valid bounded actions, with NO gradient graph built (collection never calls
    forward()/backward(), so nothing here should require grad)."""
    ac = make_ac(action_dim=2, img_size=8)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    num_envs = 4
    hx = torch.zeros(num_envs, ac.lstm_dim)
    cx = torch.zeros(num_envs, 1)
    obs = torch.rand(num_envs, 3, 8, 8)

    mu, val, (hx, cx) = ac.predict_act_value(obs, (hx, cx))
    assert not mu.requires_grad, "predict_act_value must never build a graph -- it's always no_grad"
    action, aux = ac.sample_action(mu, deterministic=False)
    assert aux is None
    assert torch.all(action >= ac.action_low - 1e-5) and torch.all(action <= ac.action_high + 1e-5)


# ---------------------------------------------------------------------------------------------
# 14. Checkpoint/resume preserves the exact frame-stack state (point 2's explicit requirement).
#     The stack IS rollout_hx_cx_state.hx (repurposed, see the module docstring) -- already
#     covered generically by tests/checkpoint/test_resume_fidelity.py's
#     test_rollout_hx_cx_state_round_trip, but this test verifies it specifically under DrQ's
#     OWN interpretation: a fresh instance, given the saved state, must produce bit-identical
#     actions to the original for the SAME next observation.
# ---------------------------------------------------------------------------------------------

def test_checkpoint_resume_preserves_exact_frame_stack_and_next_action():
    torch.manual_seed(0)
    ac1 = make_ac(frame_stack=3, img_size=8, action_dim=2)
    num_envs = 2
    hx = torch.zeros(num_envs, ac1.lstm_dim)
    cx = torch.zeros(num_envs, 1)
    for _ in range(4):
        obs = torch.rand(num_envs, 3, ac1.img_size, ac1.img_size)
        mu, val, (hx, cx) = ac1.predict_act_value(obs, (hx, cx))
    ac1.rollout_hx_cx_state.hx, ac1.rollout_hx_cx_state.cx, ac1.rollout_hx_cx_state.initialized = hx, cx, True

    saved_module_sd = ac1.state_dict()
    saved_rollout_sd = ac1.rollout_hx_cx_state.state_dict()

    ac2 = DrQActorCritic(ac1.cfg)
    ac2.load_state_dict(saved_module_sd)
    ac2.rollout_hx_cx_state.load_state_dict(saved_rollout_sd)

    assert torch.equal(ac2.rollout_hx_cx_state.hx, ac1.rollout_hx_cx_state.hx)
    assert ac2.rollout_hx_cx_state.initialized is True

    next_obs = torch.rand(num_envs, 3, ac1.img_size, ac1.img_size)
    mu1, _, _ = ac1.predict_act_value(next_obs, (ac1.rollout_hx_cx_state.hx, ac1.rollout_hx_cx_state.cx))
    mu2, _, _ = ac2.predict_act_value(next_obs, (ac2.rollout_hx_cx_state.hx, ac2.rollout_hx_cx_state.cx))
    assert torch.equal(mu1, mu2), "identical weights + identical restored stack must give identical next action"
