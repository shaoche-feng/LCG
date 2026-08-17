#! /usr/bin/env python
"""
Focused correction test: continuous-policy entropy gradient estimator.

Background: the original Stage 6 implementation computed the continuous entropy
bonus as `-log_prob(detached_action)`, reusing the same (detached) action used
for the REINFORCE actor loss. That is a valid single-sample *value* estimate of
the entropy, but backpropagating through it only yields the "direct" term of
the entropy gradient; the score-function identity E[d/dtheta log pi(a)] = 0
means that term alone has *zero expectation* -- i.e. it is not a valid gradient
estimator for maximizing entropy. The fix (ActorCritic._continuous_entropy_estimate)
draws a fresh, non-detached rsample() and evaluates -log_prob at that sample
instead, giving a proper reparameterized Monte Carlo estimate of the transformed
distribution's entropy gradient.

This script verifies:
  - the continuous entropy estimate is finite;
  - its gradient w.r.t. mean/log_std is finite;
  - that gradient is genuinely non-zero *in expectation* (not just for one lucky
    sample -- the buggy detached version can also produce a nonzero single-sample
    gradient despite having zero expectation, so we compare averaged gradients
    across many independent resamples against their standard error);
  - entropy responds sensibly (monotonically increasing) to increasing log_std
    away from saturation;
  - Atari's Categorical.entropy() is byte-for-byte unchanged.

Usage:
    python scripts/test_actor_critic_entropy_correction.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import math

import torch
from torch.distributions.categorical import Categorical

from models.actor_critic import ActorCritic, ActorCriticConfig


IMG_CHANNELS = 3
IMG_SIZE = 8
LSTM_DIM = 32


def make_model(num_actions=None, continuous_action_dim=None, action_low=None, action_high=None) -> ActorCritic:
    cfg = ActorCriticConfig(
        lstm_dim=LSTM_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE,
        channels=[8, 8], down=[1, 0],
        num_actions=num_actions, continuous_action_dim=continuous_action_dim,
        action_low=action_low, action_high=action_high,
    )
    return ActorCritic(cfg)


def old_buggy_entropy(model: ActorCritic, dist_params: torch.Tensor) -> torch.Tensor:
    """Reproduces the pre-fix behavior for comparison: entropy = -log_prob(detached_action)."""
    action = model.sample_action(dist_params)  # detached, as in the original implementation
    log_prob, _ = model.log_prob_and_entropy(dist_params, action)
    return -log_prob


def check_finite_and_shape(device) -> None:
    print(f"\n--- Entropy estimate finite, correct shape [{device}] ---")
    action_dim = 4
    model = make_model(continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim).to(device)
    dist_params = torch.randn(8, 2 * action_dim, device=device)
    entropy = model._continuous_entropy_estimate(dist_params)
    assert entropy.shape == (8,), f"expected (B,), got {entropy.shape}"
    assert torch.isfinite(entropy).all(), "entropy estimate not finite"
    print(f"  entropy shape={tuple(entropy.shape)}, finite, sample values={entropy[:3].tolist()}  OK")


def check_gradient_finite(device) -> None:
    print(f"\n--- Entropy gradient finite w.r.t. mean/log_std [{device}] ---")
    action_dim = 4
    model = make_model(continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim).to(device)
    mean = (torch.rand(16, action_dim, device=device) - 0.5).requires_grad_(True)
    log_std = torch.zeros(16, action_dim, device=device, requires_grad=True)
    dist_params = torch.cat([mean, log_std], dim=-1)

    entropy = model._continuous_entropy_estimate(dist_params)
    loss = entropy.mean()
    g_mean, g_log_std = torch.autograd.grad(loss, [mean, log_std])
    assert torch.isfinite(g_mean).all() and torch.isfinite(g_log_std).all()
    print(f"  grad(entropy.mean(), mean) finite, norm={g_mean.norm().item():.4f}")
    print(f"  grad(entropy.mean(), log_std) finite, norm={g_log_std.norm().item():.4f}  OK")


def check_gradient_nonzero_in_expectation(device) -> None:
    print(f"\n--- Entropy gradient genuinely non-zero in expectation (old vs fixed estimator) [{device}] ---")
    torch.manual_seed(0)
    action_dim = 2
    model = make_model(continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim).to(device)

    B = 64
    N_TRIALS = 400
    # Asymmetric mean: at mean=0 with symmetric bounds, d(entropy)/d(mean) is *actually* zero by
    # symmetry regardless of estimator correctness, which would confound this test. mean=0.5 breaks
    # that symmetry so a genuinely nonzero gradient is the correct expectation to check for.
    base_mean = torch.full((B, action_dim), 0.5, device=device)
    base_log_std = torch.zeros(B, action_dim, device=device)  # std=1, away from saturation

    def one_trial(entropy_fn):
        mean = base_mean.clone().requires_grad_(True)
        log_std = base_log_std.clone().requires_grad_(True)
        dist_params = torch.cat([mean, log_std], dim=-1)
        loss = entropy_fn(model, dist_params).mean()
        g_mean, g_log_std = torch.autograd.grad(loss, [mean, log_std])
        return g_mean.mean().item(), g_log_std.mean().item()

    def mean_and_sem(values):
        n = len(values)
        mean_v = sum(values) / n
        var = sum((v - mean_v) ** 2 for v in values) / (n - 1)
        sem = (var / n) ** 0.5
        return mean_v, sem

    old_mean_grads, old_log_std_grads = [], []
    new_mean_grads, new_log_std_grads = [], []
    for _ in range(N_TRIALS):
        gm, gs = one_trial(old_buggy_entropy)
        old_mean_grads.append(gm)
        old_log_std_grads.append(gs)
        gm2, gs2 = one_trial(lambda m, dp: m._continuous_entropy_estimate(dp))
        new_mean_grads.append(gm2)
        new_log_std_grads.append(gs2)

    old_mean_avg, old_mean_sem = mean_and_sem(old_mean_grads)
    new_mean_avg, new_mean_sem = mean_and_sem(new_mean_grads)
    old_ls_avg, old_ls_sem = mean_and_sem(old_log_std_grads)
    new_ls_avg, new_ls_sem = mean_and_sem(new_log_std_grads)

    print(f"  d(entropy)/d(mean):    old (detached) avg={old_mean_avg:+.5f} (SEM={old_mean_sem:.5f}, "
          f"{abs(old_mean_avg)/old_mean_sem:.2f} SEM from 0)")
    print(f"                         new (fixed)    avg={new_mean_avg:+.5f} (SEM={new_mean_sem:.5f}, "
          f"{abs(new_mean_avg)/new_mean_sem:.2f} SEM from 0)")
    print(f"  d(entropy)/d(log_std): old (detached) avg={old_ls_avg:+.5f} (SEM={old_ls_sem:.5f}, "
          f"{abs(old_ls_avg)/old_ls_sem:.2f} SEM from 0)")
    print(f"                         new (fixed)    avg={new_ls_avg:+.5f} (SEM={new_ls_sem:.5f}, "
          f"{abs(new_ls_avg)/new_ls_sem:.2f} SEM from 0)")

    # The old (detached) estimator's averaged gradient must be statistically consistent with zero.
    assert abs(old_mean_avg) < 3 * old_mean_sem, "old estimator's d/d(mean) unexpectedly far from 0"
    assert abs(old_ls_avg) < 3 * old_ls_sem, "old estimator's d/d(log_std) unexpectedly far from 0"

    # The fixed estimator's averaged gradient must be clearly, statistically bounded away from zero.
    assert abs(new_mean_avg) > 5 * new_mean_sem, "fixed estimator's d/d(mean) not clearly non-zero"
    assert abs(new_ls_avg) > 5 * new_ls_sem, "fixed estimator's d/d(log_std) not clearly non-zero"
    print("  old estimator's averaged gradient ~ 0 (as predicted); fixed estimator's is clearly non-zero  OK")


def check_entropy_response_to_log_std(device) -> None:
    print(f"\n--- Entropy response to increasing log_std, away from saturation [{device}] ---")
    torch.manual_seed(1)
    action_dim = 2
    model = make_model(continuous_action_dim=action_dim, action_low=[-1.0] * action_dim, action_high=[1.0] * action_dim).to(device)

    N = 20000
    mean = torch.zeros(N, action_dim, device=device)
    log_stds = [-1.6, -0.7, 0.0]  # std ~= 0.2, 0.5, 1.0: comfortably away from tanh saturation
    avg_entropies = []
    for ls in log_stds:
        log_std = torch.full((N, action_dim), ls, device=device)
        dist_params = torch.cat([mean, log_std], dim=-1)
        with torch.no_grad():
            entropy = model._continuous_entropy_estimate(dist_params)
        avg_entropies.append(entropy.mean().item())

    print(f"  log_std={log_stds} -> mean entropy={[f'{e:.4f}' for e in avg_entropies]}")
    assert avg_entropies[0] < avg_entropies[1] < avg_entropies[2], (
        "entropy did not increase monotonically with log_std in the non-saturating regime"
    )
    print("  entropy increases monotonically with log_std in the non-saturating regime  OK")


def check_atari_entropy_unchanged(device) -> None:
    print(f"\n--- Atari Categorical.entropy() unchanged [{device}] ---")
    num_actions = 18
    model = make_model(num_actions=num_actions).to(device)
    dist_params = torch.randn(8, num_actions, device=device)
    act = Categorical(logits=dist_params).sample()

    log_prob, entropy = model.log_prob_and_entropy(dist_params, act)
    d_ref = Categorical(logits=dist_params)
    assert torch.equal(entropy, d_ref.entropy()), "Atari entropy path changed"
    assert torch.equal(log_prob, d_ref.log_prob(act))
    print(f"  log_prob_and_entropy(discrete) exactly matches Categorical(logits).entropy()/.log_prob()  OK")


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"Devices under test: {[str(d) for d in devices]}")

    for device in devices:
        check_finite_and_shape(device)
        check_gradient_finite(device)
        check_gradient_nonzero_in_expectation(device)
        check_entropy_response_to_log_std(device)
        check_atari_entropy_unchanged(device)

    print("\nAll entropy-correction checks passed.")


if __name__ == "__main__":
    main()
