#! /usr/bin/env python
"""
Stage 6 verification: continuous bounded-action ActorCritic.

Exercises ActorCritic directly (no env_loop, no Trainer, no real env/replay
data) via predict_act_value / sample_action / log_prob_and_entropy /
compute_lambda_returns, plus a real Agent/Hydra regression check for the
existing Atari path.

Usage:
    python scripts/test_actor_critic_continuous.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import math

import torch
from torch.distributions.categorical import Categorical

from models.actor_critic import ActorCritic, ActorCriticConfig, compute_lambda_returns


IMG_CHANNELS = 3
IMG_SIZE = 8
LSTM_DIM = 32
BATCH_SIZE = 4
T = 5  # rollout length for forward()-style loss tests


def make_model(num_actions=None, continuous_action_dim=None, action_low=None, action_high=None,
                continuous_reward=False) -> ActorCritic:
    cfg = ActorCriticConfig(
        lstm_dim=LSTM_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE,
        channels=[8, 8], down=[1, 0],
        num_actions=num_actions, continuous_action_dim=continuous_action_dim,
        action_low=action_low, action_high=action_high, continuous_reward=continuous_reward,
    )
    return ActorCritic(cfg)


def make_hx_cx(model, batch_size, device):
    hx = torch.zeros(batch_size, model.lstm_dim, device=device)
    cx = torch.zeros(batch_size, model.lstm_dim, device=device)
    return hx, cx


def make_obs(batch_size, device):
    return torch.rand(batch_size, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device) * 2 - 1


# ---------------------------------------------------------------------------
# Atari regression
# ---------------------------------------------------------------------------

def check_atari_regression(device) -> None:
    print(f"\n--- Atari discrete regression [{device}] ---")
    num_actions = 18
    model = make_model(num_actions=num_actions).to(device)
    assert model.actor_linear.out_features == num_actions
    assert model.continuous_action is False

    obs = make_obs(BATCH_SIZE, device)
    hx_cx = make_hx_cx(model, BATCH_SIZE, device)
    logits_act, val, (hx, cx) = model.predict_act_value(obs, hx_cx)
    assert logits_act.shape == (BATCH_SIZE, num_actions)
    assert val.shape == (BATCH_SIZE,)
    print(f"  predict_act_value: logits_act={tuple(logits_act.shape)}, val={tuple(val.shape)}  OK")

    # action shape/dtype unchanged: Categorical over logits, int64 class indices
    act = Categorical(logits=logits_act).sample()
    assert act.shape == (BATCH_SIZE,) and act.dtype == torch.int64
    print(f"  Categorical(logits_act).sample() shape={tuple(act.shape)} dtype={act.dtype}  OK")

    # log_prob_and_entropy must be identical to constructing Categorical directly
    log_prob, entropy = model.log_prob_and_entropy(logits_act, act)
    d_ref = Categorical(logits=logits_act)
    assert torch.allclose(log_prob, d_ref.log_prob(act))
    assert torch.allclose(entropy, d_ref.entropy())
    print(f"  log_prob_and_entropy matches Categorical directly: log_prob shape={tuple(log_prob.shape)}  OK")

    # sample_action helper: stochastic matches Categorical family, deterministic == argmax
    det_act = model.sample_action(logits_act, deterministic=True)
    assert torch.equal(det_act, logits_act.argmax(dim=-1))
    print(f"  sample_action(deterministic=True) == argmax(logits)  OK")

    # lambda returns: Atari still uses reward sign
    rew = torch.tensor([[-3.0, 0.0, 5.0, -0.2, 2.0]] * BATCH_SIZE, device=device)
    end = torch.zeros(BATCH_SIZE, T, dtype=torch.int64, device=device)
    trunc = torch.zeros(BATCH_SIZE, T, dtype=torch.int64, device=device)
    trunc[:, -1] = 1
    val_bootstrap = torch.rand(BATCH_SIZE, T, device=device)
    lr_default = compute_lambda_returns(rew, end, trunc, val_bootstrap, gamma=0.99, lambda_=0.95)
    lr_explicit_false = compute_lambda_returns(rew, end, trunc, val_bootstrap, gamma=0.99, lambda_=0.95, continuous_reward=False)
    lr_signed_manually = compute_lambda_returns(rew.sign(), end, trunc, val_bootstrap, gamma=0.99, lambda_=0.95, continuous_reward=True)
    assert torch.allclose(lr_default, lr_explicit_false)
    assert torch.allclose(lr_default, lr_signed_manually), "Atari lambda returns must still be reward-sign based"
    print(f"  compute_lambda_returns (Atari) uses rew.sign() by default, matches continuous_reward=False  OK")


# ---------------------------------------------------------------------------
# Continuous policy
# ---------------------------------------------------------------------------

def check_continuous_policy(device, action_dim: int, low, high) -> None:
    tag = f"action_dim={action_dim}, bounds=[{low},{high}]"
    print(f"\n--- Continuous policy: {tag} [{device}] ---")
    action_low = [low] * action_dim
    action_high = [high] * action_dim
    model = make_model(continuous_action_dim=action_dim, action_low=action_low, action_high=action_high).to(device)
    assert model.continuous_action is True
    assert model.actor_linear.out_features == 2 * action_dim, "actor head must not hard-code action_dim"

    obs = make_obs(BATCH_SIZE, device)
    hx_cx = make_hx_cx(model, BATCH_SIZE, device)
    dist_params, val, (hx, cx) = model.predict_act_value(obs, hx_cx)
    assert dist_params.shape == (BATCH_SIZE, 2 * action_dim)
    mean, log_std = model._split_dist_params(dist_params)
    assert mean.shape == (BATCH_SIZE, action_dim) and log_std.shape == (BATCH_SIZE, action_dim)
    print(f"  predict_act_value: dist_params={tuple(dist_params.shape)} -> mean/log_std={tuple(mean.shape)}  OK")

    # sampled actions: shape (B, d), inside [low, high]
    act = model.sample_action(dist_params)
    assert act.shape == (BATCH_SIZE, action_dim), f"expected (B,d), got {act.shape}"
    assert act.dtype == torch.float32
    lo_t = torch.tensor(action_low, device=device)
    hi_t = torch.tensor(action_high, device=device)
    assert (act >= lo_t).all() and (act <= hi_t).all(), "sampled action outside configured bounds"
    print(f"  sample_action shape={tuple(act.shape)}, within bounds [{low},{high}]  OK")

    # deterministic action also stays inside bounds
    det_act = model.sample_action(dist_params, deterministic=True)
    assert det_act.shape == (BATCH_SIZE, action_dim)
    assert (det_act >= lo_t).all() and (det_act <= hi_t).all(), "deterministic action outside bounds"
    print(f"  deterministic sample_action within bounds  OK")

    # many samples across a wide range of dist_params never escape bounds
    torch.manual_seed(0)
    wide_dist_params = torch.randn(256, 2 * action_dim, device=device) * 5
    wide_act = model.sample_action(wide_dist_params)
    assert (wide_act >= lo_t - 1e-4).all() and (wide_act <= hi_t + 1e-4).all(), "bound violation under extreme dist params"
    print(f"  256 samples from extreme (mean,log_std) still respect bounds  OK")

    # log_prob finite and scalar-per-sample
    log_prob, entropy = model.log_prob_and_entropy(dist_params, act)
    assert log_prob.shape == (BATCH_SIZE,), f"expected (B,), got {log_prob.shape}"
    assert torch.isfinite(log_prob).all()
    assert entropy.shape == (BATCH_SIZE,)
    assert torch.isfinite(entropy).all()
    print(f"  log_prob shape={tuple(log_prob.shape)} finite, entropy shape={tuple(entropy.shape)} finite  OK")

    # near-boundary actions (numerical stability check)
    boundary_act = torch.stack([lo_t, hi_t] * (BATCH_SIZE // 2)).to(device)
    lp_boundary, ent_boundary = model.log_prob_and_entropy(dist_params, boundary_act)
    assert torch.isfinite(lp_boundary).all(), "log_prob blew up at exact action bounds"
    assert torch.isfinite(ent_boundary).all()
    print(f"  log_prob finite even for actions exactly at bounds (atanh clamp working)  OK")

    # gradients: log_prob/entropy depend differentiably on dist_params (hence on actor_linear).
    # NOTE: entropy is defined as -log_prob here (see log_prob_and_entropy docstring), so a loss
    # of the form `-log_prob.mean() - entropy.mean()` algebraically cancels to 0 and would give a
    # false "no gradient" reading; mirror the real advantage-weighted actor loss shape instead
    # (as ActorCritic.forward() does) to avoid that trivial cancellation.
    model.zero_grad()
    dist_params_g, val_g, _ = model.predict_act_value(obs, hx_cx)
    act_g = model.sample_action(dist_params_g)  # detached, used as replay data
    lp_g, ent_g = model.log_prob_and_entropy(dist_params_g, act_g)
    fake_advantage = torch.randn(BATCH_SIZE, device=device)
    loss = (-lp_g * fake_advantage).mean() - 0.001 * ent_g.mean() + val_g.mean()
    loss.backward()
    grad = model.actor_linear.weight.grad
    assert grad is not None, "actor_linear received no gradient"
    assert torch.isfinite(grad).all()
    assert grad.norm().item() > 0, "actor_linear gradient is exactly zero"
    print(f"  actor_linear.weight.grad norm={grad.norm().item():.6f} (finite, non-zero)  OK")

    # actor + critic loss backprop end-to-end via a synthetic forward()-style computation
    check_continuous_loss_backward(model, action_dim, device)


def check_continuous_loss_backward(model, action_dim, device) -> None:
    obs = make_obs(BATCH_SIZE, device)
    hx_cx = make_hx_cx(model, BATCH_SIZE, device)

    dist_params_seq, val_seq, act_seq = [], [], []
    hx, cx = hx_cx
    for _ in range(T):
        dist_params, val, (hx, cx) = model.predict_act_value(obs, (hx, cx))
        act = model.sample_action(dist_params)
        dist_params_seq.append(dist_params)
        val_seq.append(val)
        act_seq.append(act)
    dist_params_t = torch.stack(dist_params_seq, dim=1)  # (B, T, 2d)
    val_t = torch.stack(val_seq, dim=1)  # (B, T)
    act_t = torch.stack(act_seq, dim=1)  # (B, T, d)

    log_prob, entropy = model.log_prob_and_entropy(dist_params_t, act_t)
    assert log_prob.shape == (BATCH_SIZE, T)
    assert entropy.shape == (BATCH_SIZE, T)

    rew = (torch.rand(BATCH_SIZE, T, device=device) - 0.5)  # raw continuous reward, e.g. [-0.25, 0.13, ...]
    end = torch.zeros(BATCH_SIZE, T, dtype=torch.int64, device=device)
    trunc = torch.zeros(BATCH_SIZE, T, dtype=torch.int64, device=device)
    trunc[:, -1] = 1
    val_bootstrap = val_t.detach().clone()

    lambda_returns = compute_lambda_returns(rew, end, trunc, val_bootstrap, gamma=0.99, lambda_=0.95, continuous_reward=True)

    loss_actions = (-log_prob * (lambda_returns - val_t).detach()).mean()
    loss_values = torch.nn.functional.mse_loss(val_t, lambda_returns)
    loss_entropy = -0.001 * entropy.mean()
    loss = loss_actions + loss_values + loss_entropy

    assert torch.isfinite(loss)
    model.zero_grad()
    loss.backward()
    n_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_total = sum(1 for _ in model.parameters())
    assert n_with_grad == n_total, f"only {n_with_grad}/{n_total} params received gradients"
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    print(f"  actor+critic loss={loss.item():.4f} (finite), full backward OK, all {n_total} params got finite grads  OK")


def check_continuous_reward_lambda_returns(device) -> None:
    print(f"\n--- Continuous reward lambda returns (raw values, not sign) [{device}] ---")
    rew = torch.tensor([[-0.25, 0.13, 0.47, 0.92, -0.6]], device=device)
    end = torch.zeros(1, 5, dtype=torch.int64, device=device)
    trunc = torch.zeros(1, 5, dtype=torch.int64, device=device)
    trunc[:, -1] = 1
    val_bootstrap = torch.zeros(1, 5, device=device)

    lr_continuous = compute_lambda_returns(rew, end, trunc, val_bootstrap, gamma=0.99, lambda_=0.0, continuous_reward=True)
    lr_atari = compute_lambda_returns(rew, end, trunc, val_bootstrap, gamma=0.99, lambda_=0.0, continuous_reward=False)

    # with lambda_=0 and val_bootstrap=0, lambda_returns == rew (or rew.sign())
    assert torch.allclose(lr_continuous, rew), f"continuous mode altered raw reward values: {lr_continuous}"
    assert torch.allclose(lr_atari, rew.sign()), f"Atari mode did not apply sign(): {lr_atari}"
    assert not torch.allclose(lr_continuous, rew.sign()), "continuous reward incorrectly collapsed to sign()"
    print(f"  raw rewards {rew.tolist()} -> continuous lambda_returns {lr_continuous.tolist()} (exact match, no sign())")
    print(f"  same rewards -> Atari lambda_returns {lr_atari.tolist()} (sign() applied)  OK")


def check_hydra_atari_regression() -> None:
    print("\n--- Full Hydra/Agent Atari regression ---")
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    from hydra import initialize, compose
    from hydra.utils import instantiate

    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="trainer", overrides=[])
        OmegaConf.resolve(cfg)

    from agent import Agent
    agent = Agent(instantiate(cfg.agent, num_actions=18))
    assert agent.actor_critic.continuous_action is False
    assert agent.actor_critic.actor_linear.out_features == 18
    print("  Agent(instantiate(cfg.agent, num_actions=18)) constructs OK, actor_critic stays discrete  OK")


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"Devices under test: {[str(d) for d in devices]}")

    for device in devices:
        check_atari_regression(device)
        check_continuous_policy(device, action_dim=4, low=-1.0, high=1.0)
        check_continuous_policy(device, action_dim=6, low=-1.0, high=1.0)
        check_continuous_policy(device, action_dim=4, low=-5.0, high=3.0)  # non-[-1,1] bounds
        check_continuous_reward_lambda_returns(device)

    check_hydra_atari_regression()
    print("\nAll Stage 6 checks passed.")


if __name__ == "__main__":
    main()
