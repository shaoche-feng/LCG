#! /usr/bin/env python
"""
Stage 5A verification: continuous-action conditioning for the diffusion world model.

Exercises Denoiser/InnerModel directly with synthetic batches (no Agent/AgentConfig,
no Trainer, no replay pipeline) to isolate the model-level change:
  - discrete Atari-style actions still run through the nn.Embedding path,
  - continuous actions (action_dim=4 and action_dim=6) run through the new nn.Linear path,
  - the action-conditioning tensor has the same shape regardless of action representation,
  - Denoiser.forward accepts (B, T, action_dim) continuous actions without shape errors,
  - forward/loss values are finite,
  - backward propagation works and the continuous projection gets finite gradients,
  - nothing assumes action_dim == 6.

Runs on CPU, and on CUDA too if available.

Usage:
    python scripts/test_diffusion_action_conditioning.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
import torch.nn as nn

from data import Batch, SegmentId
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig


IMG_CHANNELS = 3
IMG_SIZE = 8
NUM_STEPS_CONDITIONING = 4
COND_CHANNELS = 16  # must be divisible by NUM_STEPS_CONDITIONING
SEQ_LENGTH = 3  # number of autoregressive denoising steps in Denoiser.forward
TOTAL_T = NUM_STEPS_CONDITIONING + SEQ_LENGTH
BATCH_SIZE = 2


def make_denoiser(num_actions=None, continuous_action_dim=None) -> Denoiser:
    cfg = DenoiserConfig(
        inner_model=InnerModelConfig(
            img_channels=IMG_CHANNELS,
            num_steps_conditioning=NUM_STEPS_CONDITIONING,
            cond_channels=COND_CHANNELS,
            depths=[1, 1],
            channels=[8, 8],
            attn_depths=[0, 0],
            num_actions=num_actions,
            continuous_action_dim=continuous_action_dim,
        ),
        sigma_data=0.5,
        sigma_offset_noise=0.3,
    )
    denoiser = Denoiser(cfg)
    denoiser.setup_training(SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=2e-3, sigma_max=20))
    return denoiser


def make_batch(action_dim, num_actions, device) -> Batch:
    obs = torch.rand(BATCH_SIZE, TOTAL_T, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device) * 2 - 1
    if action_dim is None:
        act = torch.randint(0, num_actions, (BATCH_SIZE, TOTAL_T), dtype=torch.int64, device=device)
    else:
        act = torch.randn(BATCH_SIZE, TOTAL_T, action_dim, device=device)
    rew = torch.zeros(BATCH_SIZE, TOTAL_T, device=device)
    end = torch.zeros(BATCH_SIZE, TOTAL_T, dtype=torch.int64, device=device)
    trunc = torch.zeros(BATCH_SIZE, TOTAL_T, dtype=torch.int64, device=device)
    mask_padding = torch.ones(BATCH_SIZE, TOTAL_T, dtype=torch.bool, device=device)
    info = [{} for _ in range(BATCH_SIZE)]
    segment_ids = [SegmentId(0, 0, TOTAL_T) for _ in range(BATCH_SIZE)]
    return Batch(obs, act, rew, end, trunc, mask_padding, info, segment_ids)


def check_act_emb_shape_consistency(device) -> None:
    print(f"\n--- act_emb output shape consistency [{device}] ---")
    act_slice_discrete = torch.randint(0, 18, (BATCH_SIZE, NUM_STEPS_CONDITIONING), device=device)
    shapes = {}

    denoiser = make_denoiser(num_actions=18).to(device)
    out = denoiser.inner_model.act_emb(act_slice_discrete)
    shapes["discrete(num_actions=18)"] = tuple(out.shape)
    assert isinstance(denoiser.inner_model.act_emb[0], nn.Embedding)

    for action_dim in (4, 6):
        act_slice = torch.randn(BATCH_SIZE, NUM_STEPS_CONDITIONING, action_dim, device=device)
        denoiser_c = make_denoiser(continuous_action_dim=action_dim).to(device)
        out_c = denoiser_c.inner_model.act_emb(act_slice)
        shapes[f"continuous(action_dim={action_dim})"] = tuple(out_c.shape)
        assert isinstance(denoiser_c.inner_model.act_emb[0], nn.Linear)

    for name, shape in shapes.items():
        print(f"  {name}: act_emb output shape = {shape}")
    unique_shapes = set(shapes.values())
    assert len(unique_shapes) == 1, f"act_emb output shape differs across action representations: {shapes}"
    assert unique_shapes.pop() == (BATCH_SIZE, COND_CHANNELS)
    print(f"  -> identical downstream conditioning shape (B, cond_channels)=({BATCH_SIZE},{COND_CHANNELS})  OK")


def check_forward_backward(device, num_actions=None, continuous_action_dim=None) -> None:
    tag = f"num_actions={num_actions}" if continuous_action_dim is None else f"continuous_action_dim={continuous_action_dim}"
    print(f"\n--- forward/backward: {tag} [{device}] ---")

    denoiser = make_denoiser(num_actions=num_actions, continuous_action_dim=continuous_action_dim).to(device)
    act_proj = denoiser.inner_model.act_emb[0]

    # NOTE: InnerModel.conv_out.weight is zero-initialized (pre-existing EDM-style init trick,
    # unrelated to this change). That makes dL/dx == 0 through conv_out on the very first
    # backward pass, which trivially zeroes gradients for *every* upstream layer (act_emb,
    # cond_proj, unet, conv_in) regardless of action representation. A single-pass gradient
    # check would "pass" at exactly 0.0 even if the continuous path were disconnected, so we
    # warm up with a few optimizer steps first to move conv_out off zero, then verify the
    # action-conditioning layer receives a genuinely non-zero, finite gradient.
    opt = torch.optim.Adam(denoiser.parameters(), lr=1e-2)
    for _ in range(5):
        batch = make_batch(continuous_action_dim, num_actions, device)
        loss, _ = denoiser(batch)
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert denoiser.inner_model.conv_out.weight.abs().sum().item() > 0, "conv_out.weight still exactly zero after warmup"

    batch = make_batch(continuous_action_dim, num_actions, device)
    assert batch.act.shape[:2] == (BATCH_SIZE, TOTAL_T)
    if continuous_action_dim is not None:
        assert batch.act.shape == (BATCH_SIZE, TOTAL_T, continuous_action_dim)
        assert batch.act.dtype == torch.float32
    else:
        assert batch.act.shape == (BATCH_SIZE, TOTAL_T)
        assert batch.act.dtype == torch.int64

    loss, logs = denoiser(batch)
    assert torch.isfinite(loss), f"non-finite loss: {loss}"
    print(f"  loss = {loss.item():.4f}  (finite)  OK")
    assert "loss_denoising" in logs and torch.isfinite(logs["loss_denoising"])

    opt.zero_grad()
    loss.backward()

    assert act_proj.weight.grad is not None, "action-conditioning layer received no gradient"
    assert torch.isfinite(act_proj.weight.grad).all(), "action-conditioning layer gradient is not finite"
    grad_norm = act_proj.weight.grad.norm().item()
    assert grad_norm > 0, "action-conditioning layer gradient is exactly zero after warmup (path may be disconnected)"
    print(f"  {type(act_proj).__name__}.weight.grad norm = {grad_norm:.6f}  (finite, non-zero)  OK")

    # sanity: every trainable param in the model got *some* gradient (loss actually flows through)
    n_params_with_grad = sum(1 for p in denoiser.parameters() if p.grad is not None)
    n_params = sum(1 for _ in denoiser.parameters())
    assert n_params_with_grad == n_params, f"only {n_params_with_grad}/{n_params} params received gradients"
    print(f"  all {n_params} parameters received gradients  OK")


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"Devices under test: {[str(d) for d in devices]}")

    for device in devices:
        check_act_emb_shape_consistency(device)
        check_forward_backward(device, num_actions=18)
        check_forward_backward(device, continuous_action_dim=4)
        check_forward_backward(device, continuous_action_dim=6)

    # config-level validation: exactly one of num_actions / continuous_action_dim must be set
    from models.diffusion.inner_model import InnerModel

    bad_cfg_both = InnerModelConfig(
        img_channels=IMG_CHANNELS, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=COND_CHANNELS,
        depths=[1], channels=[8], attn_depths=[0], num_actions=18, continuous_action_dim=4,
    )
    bad_cfg_neither = InnerModelConfig(
        img_channels=IMG_CHANNELS, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=COND_CHANNELS,
        depths=[1], channels=[8], attn_depths=[0],
    )
    for bad_cfg, name in [(bad_cfg_both, "both set"), (bad_cfg_neither, "neither set")]:
        try:
            InnerModel(bad_cfg)
            raise RuntimeError(f"expected AssertionError for InnerModelConfig with {name}")
        except AssertionError:
            pass
    print("\nInnerModelConfig mutual-exclusivity validation (both set / neither set)  OK")

    print("\nAll Stage 5A checks passed.")


if __name__ == "__main__":
    main()
