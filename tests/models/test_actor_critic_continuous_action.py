"""Regression tests for the continuous-action actor-critic log-probability fix.

Background (see docs of the investigation in the conversation/commit this accompanies):
`log_prob_and_entropy` used to reconstruct the pre-tanh sample `z` from the *stored, replayed*
action via `atanh()`. Whenever the true `z` pushed `tanh(z)` into float32 saturation (which
happens routinely once `mean` drifts past ~7.25 while `std` collapses toward its floor -- both
individually within their own clamps), the reconstructed `z` got silently pinned at
`atanh(1-ATANH_EPS) ~= 7.2477` regardless of the true `z`'s magnitude, producing a systematic
`(mean - recovered_z)` gap that a small `std` divided into log-probabilities of +-tens of
thousands. The fix carries the *true* sampled `z` through the rollout (`sample_action` ->
`env_loop` -> `forward`) instead of reconstructing it, and removes the log_prob clamp that used
to mask (without fixing) the symptom.

These tests are CPU-only and fast; no GPU or trained checkpoint required.
"""
import math

import torch
import torch.nn.functional as F
from torch.distributions import Normal, TransformedDistribution
from torch.distributions.transforms import AffineTransform, TanhTransform

from models.actor_critic import ActorCritic, ActorCriticConfig, ATANH_EPS


def make_ac(action_dim=2, lstm_dim=8, img_size=8, continuous=True, num_actions=4):
    """Tiny (CPU-fast) ActorCritic, real encoder+LSTM so gradient-flow tests exercise the
    actual parameters, not a mock."""
    if continuous:
        cfg = ActorCriticConfig(
            lstm_dim=lstm_dim, img_channels=3, img_size=img_size, channels=[4], down=[0],
            continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim,
        )
    else:
        cfg = ActorCriticConfig(
            lstm_dim=lstm_dim, img_channels=3, img_size=img_size, channels=[4], down=[0],
            num_actions=num_actions,
        )
    return ActorCritic(cfg)


def old_atanh_reconstruction(ac, action):
    """Reproduces the REMOVED reconstruction path exactly, for comparison only."""
    scale = 0.5 * (ac.action_high - ac.action_low)
    tanh_z = (action - ac.action_low) / scale - 1.0
    return torch.atanh(tanh_z.clamp(-1 + ATANH_EPS, 1 - ATANH_EPS))


# ---------------------------------------------------------------------------------------------
# 1. The decisive test: saturated policy, old route vs new route.
# ---------------------------------------------------------------------------------------------

def test_saturated_policy_old_route_explodes_new_route_is_finite():
    ac = make_ac(action_dim=1)
    mean = torch.tensor([[8.7]])       # within MEAN_ABS_MAX=10, but beyond atanh(1-ATANH_EPS)~=7.2477
    log_std = torch.tensor([[-5.0]])   # exactly at LOG_STD_MIN (std floor, ~0.0067)
    std = log_std.exp()
    # A very plausible real sample from N(8.7, 0.0067) -- about 1.5 std away, nothing exotic.
    z_true = torch.tensor([[8.71]])
    action = ac._squash_and_rescale(z_true)

    # --- OLD route: reconstruct z via atanh(action) ---
    z_recovered = old_atanh_reconstruction(ac, action)
    assert abs(z_recovered.item() - math.atanh(1 - ATANH_EPS)) < 0.05, (
        f"expected the old reconstruction to pin near atanh(1-ATANH_EPS)={math.atanh(1 - ATANH_EPS):.4f}, "
        f"got {z_recovered.item():.4f}"
    )
    lp_old = ac._tanh_affine_log_prob(mean, std, z_recovered)
    assert lp_old.item() < -1000, f"expected the old route to produce an extreme negative log_prob, got {lp_old.item()}"

    # --- NEW route: use the true z directly ---
    lp_new = ac._tanh_affine_log_prob(mean, std, z_true)
    assert torch.isfinite(lp_new).all()
    assert -50 < lp_new.item() < 50, f"expected a reasonable log_prob from the true z, got {lp_new.item()}"


def test_saturated_policy_via_log_prob_and_entropy_public_api():
    """Same scenario, through the actual public entry point forward() uses."""
    ac = make_ac(action_dim=1)
    mean = torch.tensor([[8.7]])
    log_std = torch.tensor([[-5.0]])
    dist_params = torch.cat([mean, log_std], dim=-1)
    z_true = torch.tensor([[8.71]])
    action = ac._squash_and_rescale(z_true).detach()

    log_prob, entropy = ac.log_prob_and_entropy(dist_params, action, z_true)
    assert torch.isfinite(log_prob).all()
    assert -50 < log_prob.item() < 50


# ---------------------------------------------------------------------------------------------
# 2. Gradient-flow tests: mean/log_std receive gradient via the actor loss; z is detached;
#    entropy's z stays reparameterized/connected.
# ---------------------------------------------------------------------------------------------

def test_policy_z_is_detached_and_gradients_reach_actor_linear():
    torch.manual_seed(0)
    ac = make_ac(action_dim=2, img_size=8)
    obs = torch.randn(3, 3, 8, 8)
    hx = torch.zeros(3, ac.lstm_dim)
    cx = torch.zeros(3, ac.lstm_dim)
    logits_act, val, (hx, cx) = ac.predict_act_value(obs, (hx, cx))

    action, z = ac.sample_action(logits_act)
    assert z.requires_grad is False, "sample_action must return a detached z"
    assert action.requires_grad is False

    log_prob, entropy = ac.log_prob_and_entropy(logits_act, action, z)
    assert log_prob.requires_grad is True, "log_prob must still be connected to actor_linear via mean/std"

    advantage = torch.full((3,), 2.5)
    loss_actions = (-log_prob * advantage).mean()
    ac.zero_grad(set_to_none=True)
    loss_actions.backward()

    assert ac.actor_linear.weight.grad is not None
    grad_norm = ac.actor_linear.weight.grad.norm().item()
    assert grad_norm > 0, "expected nonzero gradient at actor_linear from the policy loss"

    # Split mean-head vs log_std-head rows (actor_linear outputs [mean; log_std] concatenated).
    n_out = ac.actor_linear.weight.shape[0]
    mean_rows_grad = ac.actor_linear.weight.grad[: n_out // 2]
    logstd_rows_grad = ac.actor_linear.weight.grad[n_out // 2 :]
    assert mean_rows_grad.norm().item() > 0, "gradient must reach the mean head"
    assert logstd_rows_grad.norm().item() > 0, "gradient must reach the log_std head"


def test_entropy_sample_is_reparameterized_and_connected():
    torch.manual_seed(1)
    ac = make_ac(action_dim=2, img_size=8)
    obs = torch.randn(3, 3, 8, 8)
    hx = torch.zeros(3, ac.lstm_dim)
    cx = torch.zeros(3, ac.lstm_dim)
    logits_act, val, (hx, cx) = ac.predict_act_value(obs, (hx, cx))

    entropy = ac._continuous_entropy_estimate(logits_act)
    assert entropy.requires_grad is True

    ac.zero_grad(set_to_none=True)
    entropy.sum().backward()
    assert ac.actor_linear.weight.grad is not None
    assert ac.actor_linear.weight.grad.norm().item() > 0, "entropy gradient must reach actor_linear"


# ---------------------------------------------------------------------------------------------
# 3. Action sent to the environment is unchanged for a given sampled z.
# ---------------------------------------------------------------------------------------------

def test_action_formula_unchanged_for_given_z():
    ac = make_ac(action_dim=3)
    ac.action_low = torch.tensor([-1.0, -2.0, 0.0])
    ac.action_high = torch.tensor([1.0, 2.0, 4.0])
    z = torch.tensor([[0.3, -1.2, 5.0]])
    action = ac._squash_and_rescale(z)
    scale = 0.5 * (ac.action_high - ac.action_low)
    expected = ac.action_low + (torch.tanh(z) + 1) * scale
    assert torch.allclose(action, expected)


# ---------------------------------------------------------------------------------------------
# 4. Discrete-action / real-environment-collection compatibility.
# ---------------------------------------------------------------------------------------------

def test_discrete_action_sample_action_returns_none_z():
    ac = make_ac(continuous=False, num_actions=4, img_size=8)
    logits = torch.randn(3, 4)
    action, z = ac.sample_action(logits)
    assert z is None
    assert action.shape == (3,)


def test_discrete_action_log_prob_and_entropy_ignores_z():
    ac = make_ac(continuous=False, num_actions=4, img_size=8)
    logits = torch.randn(3, 4)
    action, z = ac.sample_action(logits)
    log_prob, entropy = ac.log_prob_and_entropy(logits, action, z)
    assert torch.isfinite(log_prob).all()
    assert torch.isfinite(entropy).all()


# ---------------------------------------------------------------------------------------------
# 5. Checkpoint compatibility: no parameter shape changes.
# ---------------------------------------------------------------------------------------------

def test_no_parameter_shape_changes():
    ac = make_ac(action_dim=2, lstm_dim=8, img_size=8)
    sd = ac.state_dict()
    assert sd["actor_linear.weight"].shape == (4, 8)   # 2*action_dim x lstm_dim
    assert sd["actor_linear.bias"].shape == (4,)
    assert sd["critic_linear.weight"].shape == (1, 8)
    assert sd["critic_linear.bias"].shape == (1,)
    assert sd["action_low"].shape == (2,)
    assert sd["action_high"].shape == (2,)
    # A fresh instance with the same config must be able to load this state_dict verbatim --
    # the actual guarantee existing checkpoints depend on.
    ac2 = make_ac(action_dim=2, lstm_dim=8, img_size=8)
    ac2.load_state_dict(sd)  # raises on any key/shape mismatch


# ---------------------------------------------------------------------------------------------
# 6. Cross-check against PyTorch's own TransformedDistribution, at *unsaturated* values (sanity
#    that the hand-rolled tanh+affine log-prob formula itself is correct, independent of the z
#    bug this file is otherwise about).
# ---------------------------------------------------------------------------------------------

def test_non_finite_log_prob_raises_fail_fast():
    """_tanh_affine_log_prob must raise FloatingPointError (not silently clamp, not print-and-
    continue) if it ever produces a NaN/Inf, so a corrupted value can never reach backward()/
    optimizer.step(). Inputs (mean, std, z) here are individually finite/valid Normal
    parameters -- the overflow happens inside the computation itself (std=1e-30 pushes
    (z-mean)^2/std^2 past float32 range), matching how this would occur in practice rather
    than testing an input PyTorch's own Distribution validation would already reject."""
    ac = make_ac(action_dim=1)
    mean = torch.tensor([[0.0]])
    std = torch.tensor([[1e-30]])
    z = torch.tensor([[1.0]])
    try:
        ac._tanh_affine_log_prob(mean, std, z)
        assert False, "expected FloatingPointError, no exception was raised"
    except FloatingPointError as e:
        msg = str(e)
        assert "non-finite" in msg
        # Useful tensor statistics must actually be present, not just a bare message.
        for token in ("mean[", "std[", "z[", "base_log_prob[", "log_abs_det["):
            assert token in msg, f"expected diagnostic field {token!r} in exception message"


def test_tanh_affine_log_prob_matches_pytorch_reference():
    ac = make_ac(action_dim=1)
    mean = torch.tensor([[0.3]])
    std = torch.tensor([[0.5]])
    z = torch.tensor([[0.4]])
    ours = ac._tanh_affine_log_prob(mean, std, z)

    low, high = -1.0, 1.0
    scale = (high - low) / 2
    action = low + (torch.tanh(z) + 1) * scale
    ref_dist = TransformedDistribution(
        Normal(mean, std), [TanhTransform(), AffineTransform(loc=low + scale, scale=scale)]
    )
    ref = ref_dist.log_prob(action).sum(dim=-1)
    assert torch.allclose(ours, ref, atol=1e-4)
