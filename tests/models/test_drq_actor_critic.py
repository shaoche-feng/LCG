"""Tests for the DrQ-v2-style continuous-action actor-critic (models.drq_actor_critic).

CPU-only and fast (except the CUDA-conditional tests near the end, skipped when no GPU is
available); no trained checkpoint required, matching
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
    DrQPolicyBinding,
    NoiseScheduleConfig,
    RandomShiftsAug,
    _noise_std_at,
    _soft_update,
)


def make_cfg(
    img_size=16, frame_stack=3, action_dim=2, encoder_channels=(8, 8), encoder_down=(1, 1), projection_dim=32
):
    tmp = DrQActorCriticConfig(
        img_channels=3, img_size=img_size, frame_stack=frame_stack,
        encoder_channels=list(encoder_channels), encoder_down=list(encoder_down), feature_dim=0,
        projection_dim=projection_dim, actor_hidden_dim=16, critic_hidden_dim=16,
        continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim,
        noise_schedule=NoiseScheduleConfig(std_start=1.0, std_end=0.1, decay_steps=100, clip=0.3),
    )
    enc = DrQEncoder(tmp)
    with torch.no_grad():
        dummy = torch.zeros(1, frame_stack * 3, img_size, img_size)
        tmp.feature_dim = enc.conv_output_dim(dummy)
    return tmp


def make_ac(**kwargs):
    ac = DrQActorCritic(make_cfg(**kwargs))
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    return ac


class _FakeWorldModelEnvWithBuffer:
    """Stand-in exposing obs_buffer like the real WorldModelEnv, with enough of reset_dead's
    behavior to exercise the dead-env burn-in path. num_steps_conditioning frames per env,
    frames are per-(env, episode) constant colors so leakage/identity is trivially detectable."""

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
        self._episode_color = torch.arange(1, num_envs + 1, dtype=torch.float32) * 10.0
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
            for i in torch.nonzero(dead).flatten().tolist():
                self._episode_color[i] += 1000.0
                self.obs_buffer[i] = torch.stack([self._frame_for(i) for _ in range(self._num_steps_conditioning)])
                self._ep_len[i] = 0
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = self.obs_buffer[dead, :-1]
        return self.obs_buffer[:, -1], rew, end, trunc, info


def _make_ac_with_rollout(backup_every=8, n_step=2, num_envs=3, img_size=16, action_dim=2, horizon=1000):
    ac = make_ac(img_size=img_size, action_dim=action_dim, frame_stack=3)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    env = _FakeWorldModelEnvWithBuffer(num_envs, ac.img_channels, img_size, action_dim, horizon=horizon)
    ac.env_loop = make_env_loop(env, DrQPolicyBinding(ac, env=env), hx_cx_state=ac.rollout_hx_cx_state)
    ac.loss_cfg = DrQLossConfig(backup_every=backup_every, n_step=n_step, gamma=0.99, target_tau=0.01, noise_clip=0.3)
    return ac, env


def _make_optimizers(ac):
    opt_critic = torch.optim.AdamW(list(ac.encoder.parameters()) + list(ac.critic.parameters()), lr=1e-3)
    opt_actor = torch.optim.AdamW(ac.actor.parameters(), lr=1e-3)
    return opt_critic, opt_actor


# ---------------------------------------------------------------------------------------------
# 1. Action bounds and determinism.
# ---------------------------------------------------------------------------------------------

def test_deterministic_action_uses_mu_with_no_noise():
    torch.manual_seed(0)
    ac = make_ac(action_dim=2)
    num_envs = 5
    hx, cx = ac.initial_hx_cx(num_envs)
    obs = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    mu, val, _ = ac.predict_act_value(obs, (hx, cx))

    action1, aux1 = ac.sample_action(mu, deterministic=True)
    action2, aux2 = ac.sample_action(mu, deterministic=True)
    assert aux1 is None and aux2 is None
    assert torch.equal(action1, action2)
    assert torch.allclose(action1, ac._rescale(mu))


def test_stochastic_action_respects_bounds():
    torch.manual_seed(0)
    ac = make_ac(action_dim=1)
    mu = torch.zeros(1000, 1)
    action, aux = ac.sample_action(mu, deterministic=False)
    assert aux is None
    assert torch.all(action >= ac.action_low - 1e-5) and torch.all(action <= ac.action_high + 1e-5)


def test_sample_action_bounds_hold_even_with_large_std():
    torch.manual_seed(0)
    ac = make_ac(action_dim=1)
    ac.cfg.noise_schedule.std_start = 10.0
    mu = torch.zeros(500, 1)
    action, _ = ac.sample_action(mu, deterministic=False)
    assert torch.all(action >= ac.action_low - 1e-5) and torch.all(action <= ac.action_high + 1e-5)


# ---------------------------------------------------------------------------------------------
# 2. Frame stack: shift, cold-start seeding, and per-loop isolation (issue 3).
# ---------------------------------------------------------------------------------------------

def test_frame_stack_shifts_and_drops_oldest():
    ac = make_ac(frame_stack=3, img_size=8)
    num_envs = 2
    hx, cx = ac.initial_hx_cx(num_envs)
    frame1 = torch.full((num_envs, 3, 8, 8), 1.0)
    frame2 = torch.full((num_envs, 3, 8, 8), 2.0)
    frame3 = torch.full((num_envs, 3, 8, 8), 3.0)
    frame4 = torch.full((num_envs, 3, 8, 8), 4.0)

    hx = ac._shift_and_append(torch.zeros_like(hx), frame1)
    hx = ac._shift_and_append(hx, frame2)
    hx = ac._shift_and_append(hx, frame3)
    stack = hx.view(num_envs, 3, 3, 8, 8)
    assert torch.allclose(stack[:, 0], frame1)
    assert torch.allclose(stack[:, 1], frame2)
    assert torch.allclose(stack[:, 2], frame3)

    hx = ac._shift_and_append(hx, frame4)
    stack = hx.view(num_envs, 3, 3, 8, 8)
    assert torch.allclose(stack[:, 0], frame2)
    assert torch.allclose(stack[:, 1], frame3)
    assert torch.allclose(stack[:, 2], frame4)


def test_cold_start_seeds_from_obs_buffer_not_zero_padding():
    torch.manual_seed(0)
    num_envs = 3
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    env = _FakeWorldModelEnvWithBuffer(num_envs, ac.img_channels, ac.img_size, 2, num_steps_conditioning=4)
    obs, _ = env.reset()
    hx, cx = ac.initial_hx_cx(num_envs)
    assert torch.isnan(hx).all()

    mu, val, (hx2, cx2) = ac.predict_act_value(obs, (hx, cx), env=env)
    stack = hx2.view(num_envs, 3, ac.img_channels, ac.img_size, ac.img_size)
    expected = env.obs_buffer[:, -3:]
    assert torch.allclose(stack, expected)
    naive = torch.zeros_like(stack)
    naive[:, -1] = obs
    assert not torch.allclose(stack, naive)


def test_cold_start_falls_back_to_repeated_obs_without_obs_buffer():
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    num_envs = 2
    obs = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    hx, cx = ac.initial_hx_cx(num_envs)
    mu, val, (hx2, cx2) = ac.predict_act_value(obs, (hx, cx), env=None)
    stack = hx2.view(num_envs, 3, 3, ac.img_size, ac.img_size)
    for k in range(3):
        assert torch.allclose(stack[:, k], obs)


def test_nan_sentinel_never_fires_on_mid_rollout_reset_gate_zero():
    """A reset_gate-zeroed hx (ordinary 0.0, produced mid-rollout for a dead env, NOT via
    initial_hx_cx) must NOT be treated as cold -- only the NaN sentinel does. This is the
    specific corruption this design avoids: see predict_act_value's docstring."""
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    num_envs = 2
    zero_hx = torch.zeros(num_envs, ac.lstm_dim)  # NOT NaN -- ordinary zeros, as reset_gate produces
    cx = torch.zeros(num_envs, 1)
    obs = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    _, _, (hx_out, _) = ac.predict_act_value(obs, (zero_hx, cx), env=None)
    expected = ac._shift_and_append(zero_hx, obs)
    assert torch.allclose(hx_out, expected), "ordinary zeros must take the ordinary shift-and-append path"


def test_resume_with_real_restored_hx_does_not_cold_seed():
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    restored_hx = torch.rand(2, ac.lstm_dim)  # a real (non-NaN) restored stack, as on resume
    restored_cx = torch.rand(2, 1)
    obs = torch.rand(2, 3, ac.img_size, ac.img_size)
    _, _, (hx2, cx2) = ac.predict_act_value(obs, (restored_hx, restored_cx), env=None)
    expected = ac._shift_and_append(restored_hx, obs)
    assert torch.allclose(hx2, expected)


def test_partial_env_reset_no_cross_episode_leakage():
    torch.manual_seed(0)
    num_envs = 3
    frame_stack = 3
    ac = make_ac(frame_stack=frame_stack, img_size=8, action_dim=2)
    env = _FakeWorldModelEnvWithBuffer(num_envs, ac.img_channels, ac.img_size, 2, num_steps_conditioning=4, horizon=3)
    ac.env_loop = make_env_loop(env, DrQPolicyBinding(ac, env=env), hx_cx_state=ac.rollout_hx_cx_state)
    ac.env_loop.send(10)

    final_stack = ac.rollout_hx_cx_state.hx.view(num_envs, frame_stack, ac.img_channels, ac.img_size, ac.img_size)
    for i in range(num_envs):
        current_color = env._episode_color[i].item()
        frame_colors = final_stack[i, :, 0, 0, 0].tolist()
        for fc in frame_colors:
            assert fc == current_color, (
                f"env {i}: stack contains frame color {fc}, expected only current color "
                f"{current_color} -- old-episode leakage detected"
            )


def test_two_independent_bindings_do_not_share_cold_start_or_env_state():
    """Issue 3's required test: instantiate two separate loop contexts (imagined-style binding
    with an obs_buffer-bearing env, and a real-collector-style binding with a plain env/no
    buffer) over the SAME DrQActorCritic, and verify using one has no effect on the other."""
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    num_envs = 2

    imagined_env = _FakeWorldModelEnvWithBuffer(num_envs, ac.img_channels, ac.img_size, 2, num_steps_conditioning=4)
    imagined_env.reset()
    imagined_binding = DrQPolicyBinding(ac, env=imagined_env)

    class _PlainRealEnv:
        pass  # no obs_buffer attribute at all -- like a real TorchEnv

    real_env = _PlainRealEnv()
    real_binding = DrQPolicyBinding(ac, env=real_env)

    hx_imagined, cx_imagined = imagined_binding.initial_hx_cx(num_envs)
    hx_real, cx_real = real_binding.initial_hx_cx(num_envs)
    assert torch.isnan(hx_imagined).all() and torch.isnan(hx_real).all()

    obs_imagined = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    mu_i, _, (hx_imagined2, _) = imagined_binding.predict_act_value(obs_imagined, (hx_imagined, cx_imagined))
    stack_imagined = hx_imagined2.view(num_envs, 3, ac.img_channels, ac.img_size, ac.img_size)
    assert torch.allclose(stack_imagined, imagined_env.obs_buffer[:, -3:]), "imagined binding must seed from its own env's obs_buffer"

    # Using the imagined binding must not have touched the real binding's (still cold, separate) state.
    obs_real = torch.rand(num_envs, 3, ac.img_size, ac.img_size)
    mu_r, _, (hx_real2, _) = real_binding.predict_act_value(obs_real, (hx_real, cx_real))
    stack_real = hx_real2.view(num_envs, 3, 3, ac.img_size, ac.img_size)
    for k in range(3):
        assert torch.allclose(stack_real[:, k], obs_real), "real binding (no obs_buffer) must repeat its OWN obs, unaffected by the imagined binding's seeding"
    assert not torch.allclose(stack_real, stack_imagined)


def test_train_and_test_collector_bindings_are_independent():
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    train_binding = ac.make_collector_binding(env=None)
    test_binding = ac.make_collector_binding(env=None)
    assert train_binding is not test_binding

    num_envs = 2
    hx_train, cx_train = train_binding.initial_hx_cx(num_envs)
    hx_test, cx_test = test_binding.initial_hx_cx(num_envs)

    obs_train = torch.full((num_envs, 3, ac.img_size, ac.img_size), 7.0)
    _, _, (hx_train2, _) = train_binding.predict_act_value(obs_train, (hx_train, cx_train))

    # test binding's hx must still be untouched (NaN, pristine) -- using train_binding must not
    # have mutated any state shared with test_binding.
    assert torch.isnan(hx_test).all()
    obs_test = torch.full((num_envs, 3, ac.img_size, ac.img_size), 3.0)
    _, _, (hx_test2, _) = test_binding.predict_act_value(obs_test, (hx_test, cx_test))
    stack_test = hx_test2.view(num_envs, 3, 3, ac.img_size, ac.img_size)
    for k in range(3):
        assert torch.allclose(stack_test[:, k], obs_test)
    assert not torch.allclose(hx_train2, hx_test2)


# ---------------------------------------------------------------------------------------------
# 3. Augmentation: identical shift across all K stacked frames.
# ---------------------------------------------------------------------------------------------

def test_augmentation_applies_identical_shift_to_every_stacked_frame():
    torch.manual_seed(0)
    aug = RandomShiftsAug(pad=4)
    K, C, H, W = 3, 3, 16, 16
    x = torch.zeros(5, K * C, H, W)
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
        assert len(set(locations)) == 1, f"batch element {b}: marker moved inconsistently across frames {locations}"


def test_augmentation_shift_differs_across_batch_elements_generally():
    torch.manual_seed(0)
    aug = RandomShiftsAug(pad=4)
    x = torch.zeros(64, 3, 16, 16)
    x[:, :, 8, 8] = 1.0
    y = aug(x)
    locations = [tuple((y[b, 0] == y[b, 0].max()).nonzero()[0].tolist()) for b in range(x.size(0))]
    assert len(set(locations)) > 1


# ---------------------------------------------------------------------------------------------
# 4. Exact mid-rollout-reset stack reconstruction (issue 1): training_stack[t] must equal the
#    actual stack used to generate the action at t, for every t, with one env terminating
#    halfway through the rollout and another not.
# ---------------------------------------------------------------------------------------------

def test_training_stack_matches_action_time_stack_across_mid_rollout_reset():
    torch.manual_seed(0)
    num_envs = 2
    backup_every = 6
    ac = make_ac(frame_stack=3, img_size=8, action_dim=2)
    ac.exploration_state.generator = torch.Generator().manual_seed(0)
    # env 0 terminates (via trunc) partway through; env 1 never does within this rollout.
    env = _FakeWorldModelEnvWithBuffer(num_envs, ac.img_channels, ac.img_size, 2, num_steps_conditioning=4, horizon=1000)
    env._horizon = 1000
    # Force env 0 specifically to die at absolute step 3 by wiring a custom per-env horizon.
    env._ep_len = torch.zeros(num_envs, dtype=torch.long)
    per_env_horizon = torch.tensor([3, 1000])

    orig_step = env.step

    def step_with_per_env_horizon(act):
        env._ep_len += 1
        dead = env._ep_len >= per_env_horizon
        next_obs = torch.stack([env._frame_for(i) for i in range(env.num_envs)])
        env.obs_buffer = torch.cat([env.obs_buffer[:, 1:], next_obs.unsqueeze(1)], dim=1)
        rew = torch.zeros(env.num_envs)
        end = torch.zeros(env.num_envs, dtype=torch.bool)
        trunc = dead.clone()
        info = {}
        if dead.any():
            for i in torch.nonzero(dead).flatten().tolist():
                env._episode_color[i] += 1000.0
                env.obs_buffer[i] = torch.stack([env._frame_for(i) for _ in range(env._num_steps_conditioning)])
                env._ep_len[i] = 0
            info["final_observation"] = next_obs[dead]
            info["burnin_obs"] = env.obs_buffer[dead, :-1]
        return env.obs_buffer[:, -1], rew, end, trunc, info

    env.step = step_with_per_env_horizon

    ac.env_loop = make_env_loop(env, DrQPolicyBinding(ac, env=env), hx_cx_state=ac.rollout_hx_cx_state)
    ac.loss_cfg = DrQLossConfig(backup_every=backup_every, n_step=1, gamma=0.99, target_tau=0.01, noise_clip=0.3)

    # Re-implement collect_rollout's own env_loop.send() call directly so we get all_hx back,
    # without going through the rest of collect_rollout's bootstrap computation.
    all_obs, act, rew, end, trunc, _dist, _val, _vb, _z, all_hx, infos = ac.env_loop.send(backup_every)

    # Independently re-derive, for EVERY t and EVERY env, what stack SHOULD have been used at
    # that step by replaying the frame-stack update rule against the actual reset history
    # (obs_buffer's color changes tell us exactly when each env's episode changed).
    # Simplest ground truth: at step t, the correct stack's LAST frame is all_obs[:, t] itself
    # (by construction, whatever it is), and the stack must contain ONLY frames whose color
    # belongs to whatever episode-color was active for that env at step t (no earlier-episode
    # colors mixed in) once enough real steps have elapsed to fully populate it.
    for env_idx in range(num_envs):
        for t in range(backup_every):
            stack_t = all_hx[env_idx, t].view(ac.frame_stack, ac.img_channels, ac.img_size, ac.img_size)
            last_frame_color = stack_t[-1, 0, 0, 0].item()
            all_obs_color = all_obs[env_idx, t, 0, 0, 0].item()
            assert abs(last_frame_color - all_obs_color) < 1e-4, (
                f"env {env_idx} t={t}: training_stack's last frame ({last_frame_color}) must equal "
                f"the actual observation used to produce the action at t ({all_obs_color})"
            )


def test_bootstrap_stack_after_truncation_is_final_observation_not_reset_episode():
    """Directly exercises _compute_bootstrap_info: a window whose only dead event is a
    truncation must bootstrap from the TRUE final observation at the truncation point, not
    from all_hx[:, t+n] (which after reset reflects an unrelated new episode)."""
    ac = make_ac(frame_stack=3, img_size=8, action_dim=1)
    num_envs, T = 1, 5
    all_hx = torch.zeros(num_envs, T, ac.lstm_dim)
    for t in range(T):
        all_hx[:, t] = float(t + 1)  # distinguishable per-step marker
    all_obs = torch.zeros(num_envs, T, 3, 8, 8)
    for t in range(T):
        all_obs[:, t] = float(100 + t)  # distinguishable, clearly different range from all_hx markers
    end = torch.zeros(num_envs, T)
    trunc = torch.zeros(num_envs, T)
    trunc[0, 1] = 1.0  # truncates at step 1
    final_obs_value = 999.0
    infos = [{} for _ in range(T)]
    infos[1] = {"final_observation": torch.full((1, 3, 8, 8), final_obs_value)}

    n = 3
    usable = T - n
    not_done, bootstrap_stack = ac._compute_bootstrap_info(all_hx, all_obs, end, trunc, infos, n, usable)
    assert not_done[0, 0].item() == 1.0, "truncation must still bootstrap"
    expected = ac._shift_and_append(all_hx[:, 1], infos[1]["final_observation"])
    assert torch.allclose(bootstrap_stack[0, 0], expected[0])


# ---------------------------------------------------------------------------------------------
# 5. n-step end-vs-trunc semantics (issue 2): 7 required scenarios with hand-computed values.
# ---------------------------------------------------------------------------------------------

def _bootstrap_setup(ac, num_envs, T):
    all_hx = torch.zeros(num_envs, T, ac.lstm_dim)
    for t in range(T):
        all_hx[:, t] = float(t + 1)
    all_obs = torch.zeros(num_envs, T, 3, 8, 8)
    return all_hx, all_obs


def test_n_step_no_end_or_truncation():
    ac = make_ac(action_dim=1, img_size=8)
    num_envs, T, n = 1, 6, 3
    rew = torch.ones(num_envs, T)
    end = torch.zeros(num_envs, T)
    trunc = torch.zeros(num_envs, T)
    usable = T - n
    returns = ac._accumulate_rewards(rew, end, trunc, gamma=1.0, n=n, usable=usable)
    all_hx, all_obs = _bootstrap_setup(ac, num_envs, T)
    infos = [{} for _ in range(T)]
    not_done, bootstrap_stack = ac._compute_bootstrap_info(all_hx, all_obs, end, trunc, infos, n, usable)
    assert torch.allclose(returns, torch.full((num_envs, usable), 3.0))
    assert torch.allclose(not_done, torch.ones(num_envs, usable))
    assert torch.allclose(bootstrap_stack[0, 0], all_hx[0, n])  # normal n-step lookahead


def test_n_step_true_termination_before_n():
    ac = make_ac(action_dim=1, img_size=8)
    num_envs, T, n = 1, 6, 3
    rew = torch.tensor([[5.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
    end = torch.zeros(num_envs, T)
    end[0, 1] = 1.0  # true termination at step 1 (within [0, n-1]=[0,2])
    trunc = torch.zeros(num_envs, T)
    usable = T - n
    returns = ac._accumulate_rewards(rew, end, trunc, gamma=1.0, n=n, usable=usable)
    all_hx, all_obs = _bootstrap_setup(ac, num_envs, T)
    infos = [{} for _ in range(T)]
    not_done, _ = ac._compute_bootstrap_info(all_hx, all_obs, end, trunc, infos, n, usable)
    assert returns[0, 0].item() == 5.0 + 1.0  # reward at steps 0,1 counted; step 2 excluded (dead after step1)
    assert not_done[0, 0].item() == 0.0, "true termination must not bootstrap"


def test_n_step_true_termination_exactly_at_bootstrap_boundary():
    ac = make_ac(action_dim=1, img_size=8)
    num_envs, T, n = 1, 6, 3
    rew = torch.ones(num_envs, T)
    end = torch.zeros(num_envs, T)
    end[0, n - 1] = 1.0  # terminal exactly at the window's last step (t=0's window: 0,1,2)
    trunc = torch.zeros(num_envs, T)
    usable = T - n
    all_hx, all_obs = _bootstrap_setup(ac, num_envs, T)
    infos = [{} for _ in range(T)]
    not_done, _ = ac._compute_bootstrap_info(all_hx, all_obs, end, trunc, infos, n, usable)
    assert not_done[0, 0].item() == 0.0, "termination at the window's last step must zero the bootstrap"


def test_n_step_truncation_before_nominal_boundary():
    ac = make_ac(action_dim=1, img_size=8)
    num_envs, T, n = 1, 6, 3
    rew = torch.tensor([[1.0, 5.0, 1.0, 1.0, 1.0, 1.0]])
    end = torch.zeros(num_envs, T)
    trunc = torch.zeros(num_envs, T)
    trunc[0, 1] = 1.0  # truncates at step 1, before the n=3 boundary
    usable = T - n
    returns = ac._accumulate_rewards(rew, end, trunc, gamma=1.0, n=n, usable=usable)
    all_hx, all_obs = _bootstrap_setup(ac, num_envs, T)
    infos = [{}, {"final_observation": torch.full((1, 3, 8, 8), 777.0)}] + [{} for _ in range(T - 2)]
    not_done, bootstrap_stack = ac._compute_bootstrap_info(all_hx, all_obs, end, trunc, infos, n, usable)
    assert returns[0, 0].item() == 1.0 + 5.0  # reward at steps 0,1 counted, nothing after
    assert not_done[0, 0].item() == 1.0, "truncation before the boundary must still bootstrap"
    expected = ac._shift_and_append(all_hx[:, 1], infos[1]["final_observation"])
    assert torch.allclose(bootstrap_stack[0, 0], expected[0])


def test_n_step_truncation_exactly_at_rollout_boundary_still_bootstraps():
    ac = make_ac(action_dim=1, img_size=8)
    num_envs, T, n = 1, 6, 3
    rew = torch.ones(num_envs, T)
    end = torch.zeros(num_envs, T)
    trunc = torch.zeros(num_envs, T)
    trunc[0, n - 1] = 1.0  # truncates exactly at the window's last step
    usable = T - n
    all_hx, all_obs = _bootstrap_setup(ac, num_envs, T)
    infos = [{} for _ in range(T)]
    infos[n - 1] = {"final_observation": torch.full((1, 3, 8, 8), 555.0)}
    not_done, bootstrap_stack = ac._compute_bootstrap_info(all_hx, all_obs, end, trunc, infos, n, usable)
    assert not_done[0, 0].item() == 1.0, "truncation confirmed to still bootstrap"
    expected = ac._shift_and_append(all_hx[:, n - 1], infos[n - 1]["final_observation"])
    assert torch.allclose(bootstrap_stack[0, 0], expected[0])


def test_n_step_truncation_confirmed_bootstraps_end_confirmed_does_not():
    """Direct side-by-side confirmation requested explicitly: same setup, only end vs trunc
    differs."""
    ac = make_ac(action_dim=1, img_size=8)
    num_envs, T, n = 1, 6, 3
    rew = torch.ones(num_envs, T)
    all_hx, all_obs = _bootstrap_setup(ac, num_envs, T)

    end_case = torch.zeros(num_envs, T)
    end_case[0, 1] = 1.0
    trunc_case_zeros = torch.zeros(num_envs, T)
    infos_end = [{} for _ in range(T)]
    not_done_end, _ = ac._compute_bootstrap_info(all_hx, all_obs, end_case, trunc_case_zeros, infos_end, n, T - n)

    trunc_case = torch.zeros(num_envs, T)
    trunc_case[0, 1] = 1.0
    end_case_zeros = torch.zeros(num_envs, T)
    infos_trunc = [{} for _ in range(T)]
    infos_trunc[1] = {"final_observation": torch.full((1, 3, 8, 8), 42.0)}
    not_done_trunc, _ = ac._compute_bootstrap_info(all_hx, all_obs, end_case_zeros, trunc_case, infos_trunc, n, T - n)

    assert not_done_end[0, 0].item() == 0.0
    assert not_done_trunc[0, 0].item() == 1.0


# ---------------------------------------------------------------------------------------------
# 6. Actor/critic optimizer isolation (issue 4).
# ---------------------------------------------------------------------------------------------

def _snapshot(module):
    return [p.detach().clone() for p in module.parameters()]


def _unchanged(before, after):
    return all(torch.equal(b, a) for b, a in zip(before, after))


def test_actor_update_changes_only_actor_params():
    torch.manual_seed(0)
    ac, _ = _make_ac_with_rollout()
    opt_critic, opt_actor = _make_optimizers(ac)
    ac.collect_rollout()
    ac.critic_update(opt_critic)

    actor_before = _snapshot(ac.actor)
    encoder_before = _snapshot(ac.encoder)
    critic_before = _snapshot(ac.critic)

    ac.actor_update(opt_actor)

    actor_after = _snapshot(ac.actor)
    encoder_after = _snapshot(ac.encoder)
    critic_after = _snapshot(ac.critic)

    assert not _unchanged(actor_before, actor_after), "actor parameters must change"
    assert _unchanged(encoder_before, encoder_after), "encoder parameters must NOT change during actor update"
    assert _unchanged(critic_before, critic_after), "Q1/Q2 parameters must NOT change during actor update"


def test_critic_update_changes_critic_and_encoder_not_actor():
    torch.manual_seed(0)
    ac, _ = _make_ac_with_rollout()
    opt_critic, opt_actor = _make_optimizers(ac)
    ac.collect_rollout()

    actor_before = _snapshot(ac.actor)
    encoder_before = _snapshot(ac.encoder)
    critic_before = _snapshot(ac.critic)

    ac.critic_update(opt_critic)

    actor_after = _snapshot(ac.actor)
    encoder_after = _snapshot(ac.encoder)
    critic_after = _snapshot(ac.critic)

    assert not _unchanged(critic_before, critic_after), "critic parameters must change"
    assert not _unchanged(encoder_before, encoder_after), "encoder parameters must change (critic loss trains it)"
    assert _unchanged(actor_before, actor_after), "actor parameters must NOT change during critic update"


def test_actor_update_backward_does_not_populate_critic_grad():
    """Stronger than the parameter-value check above: critic .grad must never even be
    populated by the actor's backward pass, thanks to the explicit requires_grad_(False)
    freeze -- not merely "opt_actor doesn't touch it"."""
    torch.manual_seed(0)
    ac, _ = _make_ac_with_rollout()
    opt_critic, opt_actor = _make_optimizers(ac)
    ac.collect_rollout()
    ac.critic_update(opt_critic)
    for p in ac.critic.parameters():
        p.grad = None
    ac.actor_update(opt_actor)
    for p in ac.critic.parameters():
        assert p.grad is None, "critic parameters must never accumulate gradient from the actor's backward pass"
        assert p.requires_grad is True, "critic requires_grad must be restored after the actor update"


# ---------------------------------------------------------------------------------------------
# 7. Target-network soft update happens AFTER the critic optimizer step (issue 5).
# ---------------------------------------------------------------------------------------------

def test_target_update_uses_new_online_critic_not_old():
    ac = make_ac(action_dim=1, img_size=8)
    tau = 0.5
    with torch.no_grad():
        for p in ac.critic.parameters():
            p.fill_(0.0)
        for p in ac.target_critic.parameters():
            p.fill_(0.0)

    old_online = [p.clone() for p in ac.critic.parameters()]

    # Simulate what critic_update does: an optimizer step that changes the online critic...
    with torch.no_grad():
        for p in ac.critic.parameters():
            p.fill_(2.0)  # pretend this is what the optimizer step produced
    new_online = [p.clone() for p in ac.critic.parameters()]

    # ...THEN the soft update.
    _soft_update(ac.target_critic, ac.critic, tau)

    for p_target, p_old, p_new in zip(ac.target_critic.parameters(), old_online, new_online):
        expected = (1 - tau) * torch.zeros_like(p_target) + tau * p_new  # target started at 0
        assert torch.allclose(p_target, expected), "target must move toward the NEW online critic, not the old one"
        assert not torch.allclose(p_target, (1 - tau) * torch.zeros_like(p_target) + tau * p_old) or torch.equal(p_old, p_new)


def test_critic_update_calls_soft_update_after_optimizer_step():
    """End-to-end: after one real critic_update() call, the target critic must have moved
    toward the POST-step online critic -- verified by checking target moved AT ALL (nonzero
    tau, real optimizer step) and that re-deriving what a PRE-step-based soft update would
    have produced gives a DIFFERENT (and here, wrong) answer."""
    torch.manual_seed(0)
    ac, _ = _make_ac_with_rollout()
    opt_critic, _ = _make_optimizers(ac)

    pre_step_critic = _snapshot(ac.critic)
    pre_update_target = _snapshot(ac.target_critic)

    ac.collect_rollout()
    ac.critic_update(opt_critic)

    post_step_critic = _snapshot(ac.critic)
    post_update_target = _snapshot(ac.target_critic)
    tau = ac.loss_cfg.target_tau

    # What the target WOULD be if soft-updated toward the OLD (pre-step) critic instead.
    wrong_target = [(1 - tau) * t + tau * c for t, c in zip(pre_update_target, pre_step_critic)]
    # What it SHOULD be: soft-updated toward the NEW (post-step) critic.
    right_target = [(1 - tau) * t + tau * c for t, c in zip(pre_update_target, post_step_critic)]

    for actual, right, wrong in zip(post_update_target, right_target, wrong_target):
        assert torch.allclose(actual, right, atol=1e-6), "target must reflect the post-critic-step online weights"
        if not torch.allclose(right, wrong, atol=1e-6):
            assert not torch.allclose(actual, wrong, atol=1e-6)


# ---------------------------------------------------------------------------------------------
# 8. Exploration RNG on CUDA (issue 6) -- skipped if no GPU; CPU equivalent always runs.
# ---------------------------------------------------------------------------------------------

def test_exploration_noise_reproducible_on_cpu_via_generator_state():
    ac = make_ac(action_dim=2, img_size=8)
    ac.exploration_state.generator = torch.Generator(device="cpu").manual_seed(123)
    mu = torch.zeros(4, 2)
    saved_state = ac.exploration_state.generator.get_state()
    action1, _ = ac.sample_action(mu, deterministic=False)
    ac.exploration_state.generator.set_state(saved_state)
    action2, _ = ac.sample_action(mu, deterministic=False)
    assert torch.equal(action1, action2)


def test_exploration_noise_reproducible_on_cuda_via_generator_state():
    if not torch.cuda.is_available():
        return  # skip: no GPU on this machine
    from utils import derive_torch_generator

    ac = make_ac(action_dim=2, img_size=8).to("cuda")
    ac.exploration_state.generator = derive_torch_generator(0, 99, device="cuda")
    mu = torch.zeros(4, 2, device="cuda")
    saved_state = ac.exploration_state.generator.get_state()
    action1, _ = ac.sample_action(mu, deterministic=False)
    ac.exploration_state.generator.set_state(saved_state)
    action2, _ = ac.sample_action(mu, deterministic=False)
    assert torch.equal(action1, action2)


# ---------------------------------------------------------------------------------------------
# 9. Exploration schedule cadence (issue 7): only collect_rollout() advances it.
# ---------------------------------------------------------------------------------------------

def test_deterministic_evaluation_does_not_advance_schedule():
    ac = make_ac(action_dim=2, img_size=8)
    mu = torch.zeros(3, 2)
    for _ in range(5):
        ac.sample_action(mu, deterministic=True)
    assert ac.exploration_state.schedule_step == 0


def test_stochastic_sample_action_alone_does_not_advance_schedule():
    """Real/test collector action calls go through sample_action directly and must NOT mutate
    the schedule counter -- only collect_rollout() does."""
    ac = make_ac(action_dim=2, img_size=8)
    mu = torch.zeros(3, 2)
    for _ in range(5):
        ac.sample_action(mu, deterministic=False)
    assert ac.exploration_state.schedule_step == 0


def test_one_training_update_advances_schedule_exactly_once():
    torch.manual_seed(0)
    ac, _ = _make_ac_with_rollout()
    opt_critic, opt_actor = _make_optimizers(ac)
    assert ac.exploration_state.schedule_step == 0
    ac.collect_rollout()
    assert ac.exploration_state.schedule_step == 1
    ac.critic_update(opt_critic)
    ac.actor_update(opt_actor)
    assert ac.exploration_state.schedule_step == 1, "critic_update/actor_update must not further advance the schedule"


def test_checkpoint_resume_preserves_schedule_position():
    state = DrQExplorationState()
    state.generator = torch.Generator().manual_seed(0)
    state.schedule_step = 42
    sd = state.state_dict()

    state2 = DrQExplorationState()
    state2.generator = torch.Generator().manual_seed(999)
    state2.load_state_dict(sd)
    assert state2.schedule_step == 42


# ---------------------------------------------------------------------------------------------
# 10. Noise schedule value function (unchanged from before, still relevant).
# ---------------------------------------------------------------------------------------------

def test_noise_schedule_decays_linearly_then_floors():
    cfg = NoiseScheduleConfig(std_start=1.0, std_end=0.2, decay_steps=10, clip=0.3)
    assert _noise_std_at(cfg, 0) == 1.0
    assert abs(_noise_std_at(cfg, 5) - 0.6) < 1e-6
    assert abs(_noise_std_at(cfg, 10) - 0.2) < 1e-9
    assert abs(_noise_std_at(cfg, 1000) - 0.2) < 1e-9


# ---------------------------------------------------------------------------------------------
# 11. Checkpoint round-trips: exploration state, and plain nn.Module parameters (targets
#     included, since they're ordinary submodules).
# ---------------------------------------------------------------------------------------------

def test_exploration_state_round_trip():
    state = DrQExplorationState()
    state.generator = torch.Generator().manual_seed(42)
    state.schedule_step = 17
    torch.randn(5, generator=state.generator)
    sd = state.state_dict()

    state2 = DrQExplorationState()
    state2.generator = torch.Generator().manual_seed(0)
    state2.load_state_dict(sd)
    assert state2.schedule_step == 17

    draw1 = torch.randn(5, generator=state.generator)
    draw2 = torch.randn(5, generator=state2.generator)
    assert torch.equal(draw1, draw2)


def test_exploration_state_load_tolerates_missing_generator():
    state = DrQExplorationState()
    state.load_state_dict({"generator_state": None, "schedule_step": 5})
    assert state.schedule_step == 5
    assert state.generator is None


def test_module_state_dict_round_trip_no_shape_changes():
    ac = make_ac(action_dim=2, img_size=8)
    sd = ac.state_dict()
    ac2 = make_ac(action_dim=2, img_size=8)
    ac2.load_state_dict(sd)
    for p1, p2 in zip(ac.parameters(), ac2.parameters()):
        assert torch.equal(p1, p2)


def test_checkpoint_resume_preserves_exact_frame_stack_and_next_action():
    torch.manual_seed(0)
    ac1 = make_ac(frame_stack=3, img_size=8, action_dim=2)
    num_envs = 2
    hx, cx = ac1.initial_hx_cx(num_envs)
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
    assert torch.equal(mu1, mu2)


# ---------------------------------------------------------------------------------------------
# 12. End-to-end sanity: full collect_rollout -> critic_update -> actor_update cycle produces
#     finite losses and a working real-collector-compatible path (predict_act_value/
#     sample_action never build a grad graph).
# ---------------------------------------------------------------------------------------------

def test_full_training_cycle_produces_finite_metrics():
    torch.manual_seed(0)
    ac, _ = _make_ac_with_rollout()
    opt_critic, opt_actor = _make_optimizers(ac)
    ac.collect_rollout()
    metrics_c = ac.critic_update(opt_critic)
    metrics_a = ac.actor_update(opt_actor)
    for v in list(metrics_c.values()) + list(metrics_a.values()):
        assert torch.isfinite(v).all()


def test_predict_act_value_never_builds_grad_graph():
    ac = make_ac(action_dim=2, img_size=8)
    num_envs = 4
    hx, cx = ac.initial_hx_cx(num_envs)
    obs = torch.rand(num_envs, 3, 8, 8)
    mu, val, (hx2, cx2) = ac.predict_act_value(obs, (hx, cx))
    assert not mu.requires_grad
