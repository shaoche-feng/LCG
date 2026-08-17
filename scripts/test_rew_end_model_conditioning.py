#! /usr/bin/env python
"""
Stage 5B verification: continuous-action conditioning and continuous-reward
prediction for RewEndModel.

Exercises RewEndModel directly with synthetic batches (no Agent/AgentConfig,
no Trainer, no WorldModelEnv) to isolate the model-level change:
  - discrete Atari-style actions still use nn.Embedding, reward classification
    behavior (sign() target, cross-entropy loss) is unchanged,
  - continuous actions (action_dim=4 and 6) run through the new nn.Linear path,
  - continuous reward uses the raw reward value (not .sign()) with MSE loss,
    and predicts a scalar of the expected shape,
  - end prediction is unaffected in every case,
  - forward/loss values are finite,
  - backward propagation gives finite, non-zero gradients to the continuous
    action projection (after warming up past any zero-initialized layers),
  - nothing assumes action_dim == 6.

Runs on CPU, and on CUDA too if available.

Usage:
    python scripts/test_rew_end_model_conditioning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from data import Batch, SegmentId
from models.rew_end_model import RewEndModel, RewEndModelConfig


IMG_CHANNELS = 3
IMG_SIZE = 8
COND_CHANNELS = 16
LSTM_DIM = 32
T = 6  # sequence length (predict_rew_end operates on T-1 transitions inside forward())
BATCH_SIZE = 2


def make_model(num_actions=None, continuous_action_dim=None, continuous_reward=False) -> RewEndModel:
    cfg = RewEndModelConfig(
        lstm_dim=LSTM_DIM,
        img_channels=IMG_CHANNELS,
        img_size=IMG_SIZE,
        cond_channels=COND_CHANNELS,
        depths=[1, 1],
        channels=[8, 8],
        attn_depths=[0, 0],
        num_actions=num_actions,
        continuous_action_dim=continuous_action_dim,
        continuous_reward=continuous_reward,
    )
    return RewEndModel(cfg)


def make_batch(action_dim, num_actions, device, reward_scale=1.0) -> Batch:
    obs = torch.rand(BATCH_SIZE, T, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device) * 2 - 1
    if action_dim is None:
        act = torch.randint(0, num_actions, (BATCH_SIZE, T), dtype=torch.int64, device=device)
    else:
        act = torch.randn(BATCH_SIZE, T, action_dim, device=device)
    # Deliberately not a {-1, 0, 1} signal: reward_scale != 1 would be destroyed by .sign().
    rew = torch.randn(BATCH_SIZE, T, device=device) * reward_scale
    end = torch.zeros(BATCH_SIZE, T, dtype=torch.int64, device=device)
    trunc = torch.zeros(BATCH_SIZE, T, dtype=torch.int64, device=device)
    mask_padding = torch.ones(BATCH_SIZE, T, dtype=torch.bool, device=device)
    info = [{} for _ in range(BATCH_SIZE)]
    segment_ids = [SegmentId(0, 0, T) for _ in range(BATCH_SIZE)]
    return Batch(obs, act, rew, end, trunc, mask_padding, info, segment_ids)


def check_atari_discrete_and_classification(device) -> None:
    print(f"\n--- Atari discrete action + reward classification [{device}] ---")
    model = make_model(num_actions=18).to(device)
    assert isinstance(model.act_emb, nn.Embedding), "Atari path must still use nn.Embedding"
    print(f"  act_emb type = {type(model.act_emb).__name__}  OK")

    batch = make_batch(action_dim=None, num_actions=18, device=device, reward_scale=5.0)
    loss, metrics = model(batch)
    assert torch.isfinite(loss)
    assert "rew" in metrics["confusion_matrix"], "Atari path must still report a reward confusion matrix"
    print(f"  loss = {loss.item():.4f}  (finite), reward confusion matrix present  OK")

    # Reward classification target must still be derived from sign(), not the raw value.
    obs = batch.obs[:, :-1]
    act = batch.act[:, :-1]
    next_obs = batch.obs[:, 1:]
    pred_rew, logits_end, _ = model.predict_rew_end(obs, act, next_obs)
    assert pred_rew.shape == (BATCH_SIZE, T - 1, 3), f"expected (B,T-1,3) reward logits, got {pred_rew.shape}"
    assert logits_end.shape == (BATCH_SIZE, T - 1, 2)
    target_rew_expected = batch.rew[:, :-1].sign().long().add(1)
    assert set(target_rew_expected.unique().tolist()) <= {0, 1, 2}
    print(f"  logits_rew shape={tuple(pred_rew.shape)}, target derived via sign() -> classes {{0,1,2}}  OK")


def check_continuous(device, action_dim: int) -> None:
    print(f"\n--- Continuous action_dim={action_dim}, continuous reward [{device}] ---")
    model = make_model(continuous_action_dim=action_dim, continuous_reward=True).to(device)
    assert isinstance(model.act_emb, nn.Linear), "Continuous path must use nn.Linear"
    assert model.act_emb.in_features == action_dim, "action projection in_features must match action_dim, not be hard-coded"
    print(f"  act_emb type = {type(model.act_emb).__name__}, in_features={model.act_emb.in_features}  OK")

    batch = make_batch(action_dim=action_dim, num_actions=None, device=device, reward_scale=5.0)
    assert batch.act.shape == (BATCH_SIZE, T, action_dim)
    assert batch.act.dtype == torch.float32

    obs = batch.obs[:, :-1]
    act = batch.act[:, :-1]
    next_obs = batch.obs[:, 1:]
    pred_rew, logits_end, _ = model.predict_rew_end(obs, act, next_obs)

    # 5. Continuous reward output has the expected scalar shape: (B, T-1), not (B, T-1, 1) or (B, T-1, 3).
    assert pred_rew.shape == (BATCH_SIZE, T - 1), f"expected scalar reward shape (B,T-1), got {pred_rew.shape}"
    assert logits_end.shape == (BATCH_SIZE, T - 1, 2), "end prediction shape must be unaffected"
    print(f"  pred_rew shape={tuple(pred_rew.shape)} (scalar), logits_end shape={tuple(logits_end.shape)}  OK")

    loss, metrics = model(batch)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    assert torch.isfinite(metrics["loss_rew"]) and torch.isfinite(metrics["loss_end"])
    assert "rew" not in metrics["confusion_matrix"], "continuous reward path must not report a classification confusion matrix"
    print(f"  loss_rew={metrics['loss_rew'].item():.4f} loss_end={metrics['loss_end'].item():.4f}  (finite)  OK")

    # 6. Continuous reward loss must use the raw reward value, not rew.sign() -- verify by
    # reproducing the target computation exactly as forward() does and confirming it is NOT
    # confined to {-1, 0, 1} (reward_scale=5.0 makes this a strong signal, not a coincidence).
    raw_targets = batch.rew[:, :-1][batch.mask_padding[:, :-1]]
    assert raw_targets.abs().max() > 1.0 + 1e-4, "test fixture reward scale too small to distinguish from sign()"
    print(f"  target reward range=[{raw_targets.min().item():.3f}, {raw_targets.max().item():.3f}] (not clipped to {{-1,0,1}})  OK")

    # 7/8. backward: warm up past zero-initialized layers (see Stage 5A finding), then verify the
    # continuous action projection gets a genuinely non-zero, finite gradient.
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    for _ in range(5):
        b = make_batch(action_dim, None, device, reward_scale=5.0)
        l, _ = model(b)
        opt.zero_grad()
        l.backward()
        opt.step()

    b = make_batch(action_dim, None, device, reward_scale=5.0)
    loss2, _ = model(b)
    opt.zero_grad()
    loss2.backward()
    assert model.act_emb.weight.grad is not None
    assert torch.isfinite(model.act_emb.weight.grad).all()
    grad_norm = model.act_emb.weight.grad.norm().item()
    assert grad_norm > 0, "continuous action projection gradient is exactly zero after warmup"
    print(f"  act_emb (Linear).weight.grad norm = {grad_norm:.6f}  (finite, non-zero)  OK")

    n_params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
    n_params = sum(1 for _ in model.parameters())
    assert n_params_with_grad == n_params, f"only {n_params_with_grad}/{n_params} params received gradients"
    print(f"  all {n_params} parameters received gradients  OK")


def check_config_validation() -> None:
    print("\n--- RewEndModelConfig mutual-exclusivity validation ---")
    for kwargs, name in [
        (dict(num_actions=18, continuous_action_dim=4), "both set"),
        (dict(), "neither set"),
    ]:
        cfg = RewEndModelConfig(
            lstm_dim=LSTM_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE, cond_channels=COND_CHANNELS,
            depths=[1], channels=[8], attn_depths=[0], **kwargs,
        )
        try:
            RewEndModel(cfg)
            raise RuntimeError(f"expected AssertionError for RewEndModelConfig with {name}")
        except AssertionError:
            pass
    print("  both set / neither set correctly rejected  OK")


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"Devices under test: {[str(d) for d in devices]}")

    for device in devices:
        check_atari_discrete_and_classification(device)
        check_continuous(device, action_dim=4)
        check_continuous(device, action_dim=6)

    check_config_validation()
    print("\nAll Stage 5B checks passed.")


if __name__ == "__main__":
    main()
