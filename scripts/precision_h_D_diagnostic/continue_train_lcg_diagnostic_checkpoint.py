#! /usr/bin/env python
"""
Continue training the existing LCG diagnostic denoiser checkpoint
(scripts/train_lcg_diagnostic_checkpoint.py's output, saved at step=100) further, using the
SAME train/validation datasets already on disk (no fresh dm_control collection), until
validation EDM loss reasonably plateaus.

That earlier checkpoint was stopped purely on a gradient-flow-unlocking criterion (all 5
theta_S groups showing nonzero curvature mass for 2 consecutive checks, at step>=100) -- not
on validation-loss convergence. This script picks it up and continues ordinary Denoiser
training (same real EDM loss, same AdamW hyperparams, same data) to get a more representative
checkpoint for the full-dataset-vs-subset h_D diagnostic, without pursuing benchmark-grade
convergence.

Stopping rule: after each VAL_EVERY-step block, record val loss; stop once val loss's
relative improvement over the last PLATEAU_WINDOW steps drops below PLATEAU_REL_THRESHOLD
(and at least MIN_ADDITIONAL_STEPS have run), or MAX_ADDITIONAL_STEPS is reached.

Also re-checks, at every recorded step, that all 5 theta_S module groups remain
curvature-active (nonzero VJP-squared mass), reusing the exact unmodified
lcg.gauss_newton.compute_vjp / lcg.theta_s.selected_parameters primitives.

Usage:
    python scripts/continue_train_lcg_diagnostic_checkpoint.py
"""
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BatchSampler, Dataset, collate_segments_to_batch
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition, sample_valid_transitions
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from utils import configure_opt

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
VAL_DATASET_DIR = SCRATCH_BASE / "lcg_diag_val_dataset"
OLD_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
NEW_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser_converged.pt"

SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
NUM_STEPS_CONDITIONING = 4
NUM_AUTOREGRESSIVE_STEPS = 1
TRAIN_BATCH_SIZE = 16
OPTIMIZER_CFG = dict(lr=1e-4, weight_decay=1e-2, eps=1e-8)

VAL_EVERY = 100
MONITOR_B, MONITOR_STRATA = 8, 3
MIN_ADDITIONAL_STEPS = 300
PLATEAU_WINDOW = 500  # steps
PLATEAU_REL_THRESHOLD = 0.02
MAX_ADDITIONAL_STEPS = 3000


def theta_s_groups(denoiser: Denoiser):
    final_level = denoiser.inner_model.unet.u_blocks[-1]
    groups = [(f"resblock_{i}", list(rb.parameters())) for i, rb in enumerate(final_level.resblocks)]
    groups.append(("norm_out", list(denoiser.inner_model.norm_out.parameters())))
    groups.append(("conv_out", list(denoiser.inner_model.conv_out.parameters())))
    flat_check = [p for _, plist in groups for p in plist]
    real = selected_parameters(denoiser)
    assert len(flat_check) == len(real) and all(a is b for a, b in zip(flat_check, real))
    return groups


def group_slices(denoiser: Denoiser):
    offset = 0
    slices = []
    for name, plist in theta_s_groups(denoiser):
        n = sum(p.numel() for p in plist)
        slices.append((name, offset, offset + n))
        offset += n
    return slices


def quick_theta_s_report(denoiser, params, slices, dataset, B, num_strata, seed):
    torch.manual_seed(seed)
    device = denoiser.device
    segment_ids = sample_valid_transitions(dataset, B, NUM_STEPS_CONDITIONING, seed=seed)
    d_S = sum(p.numel() for p in params)
    group_sum = {name: 0.0 for name, _, _ in slices}
    all_zero_mask = torch.ones(d_S, dtype=torch.bool)

    was_training = denoiser.training
    denoiser.eval()
    for segment_id in segment_ids:
        obs, act, y = load_transition(dataset, segment_id, NUM_STEPS_CONDITIONING, device)
        for m in range(num_strata):
            sigma = sample_sigma_stratum(SIGMA_CFG, m, num_strata, 1, device)
            eps = torch.randn_like(y)
            y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act)
            v2 = (v * v).detach().cpu()
            all_zero_mask &= v2 == 0
            for name, start, stop in slices:
                group_sum[name] += v2[start:stop].sum().item()
    if was_training:
        denoiser.train()

    return group_sum, all_zero_mask.float().mean().item()


def make_train_loader(train_dataset: Dataset, batch_size: int):
    seq_length = NUM_STEPS_CONDITIONING + 1 + NUM_AUTOREGRESSIVE_STEPS
    bs = BatchSampler(train_dataset, rank=0, world_size=1, batch_size=batch_size, seq_length=seq_length, sample_weights=None)
    dl = DataLoader(dataset=train_dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    return iter(dl)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train", cache_in_ram=True)
    train_dataset.load_from_default_path()
    val_dataset = Dataset(VAL_DATASET_DIR, "lcg_diag_val", cache_in_ram=True)
    val_dataset.load_from_default_path()
    print(f"train dataset: {train_dataset.num_episodes} episodes, N={train_dataset.num_steps} steps", flush=True)
    print(f"val dataset:   {val_dataset.num_episodes} episodes, N={val_dataset.num_steps} steps", flush=True)

    ckpt = torch.load(OLD_CHECKPOINT_PATH, map_location=device, weights_only=False)
    action_dim = ckpt["action_dim"]
    start_step = ckpt["step"]
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.setup_training(SIGMA_CFG)
    print(f"Loaded existing diagnostic checkpoint (step={start_step}) from {OLD_CHECKPOINT_PATH}", flush=True)

    params = selected_parameters(denoiser)
    slices = group_slices(denoiser)
    optimizer = configure_opt(denoiser, **OPTIMIZER_CFG)

    train_iter = make_train_loader(train_dataset, TRAIN_BATCH_SIZE)
    val_seq_length = NUM_STEPS_CONDITIONING + 1 + NUM_AUTOREGRESSIVE_STEPS
    val_sampler = BatchSampler(val_dataset, 0, 1, TRAIN_BATCH_SIZE, val_seq_length, sample_weights=None)
    val_segments = [val_dataset[sid] for sid in val_sampler.sample()]
    val_batch = collate_segments_to_batch(val_segments).to(device)

    history = []  # (additional_step, val_loss)
    running_train_loss = None
    t_start = time.time()

    print(f"\n{'add_step':>9} {'total_step':>10} {'train_loss':>11} {'val_loss':>11} "
          f"{'frac_zero':>10} {'all_nonzero':>11} {'elapsed_s':>9}", flush=True)

    additional_step = 0
    stop_reason = None
    while additional_step <= MAX_ADDITIONAL_STEPS:
        if additional_step % VAL_EVERY == 0:
            denoiser.eval()
            with torch.no_grad():
                val_loss = denoiser(val_batch)[0].item()
            denoiser.train()
            group_sum, frac_zero = quick_theta_s_report(
                denoiser, params, slices, train_dataset, MONITOR_B, MONITOR_STRATA, seed=1234
            )
            all_nonzero = all(v > 0 for v in group_sum.values())
            history.append((additional_step, val_loss))
            train_loss_str = f"{running_train_loss:.5f}" if running_train_loss is not None else "n/a"
            print(f"{additional_step:>9} {start_step + additional_step:>10} {train_loss_str:>11} "
                  f"{val_loss:>11.5f} {frac_zero:>10.4f} {str(all_nonzero):>11} "
                  f"{time.time() - t_start:>9.1f}", flush=True)

            if not all_nonzero:
                print("  WARNING: at least one theta_S group has exactly-zero curvature mass!", flush=True)

            if additional_step >= MIN_ADDITIONAL_STEPS and additional_step >= PLATEAU_WINDOW:
                # find the val_loss recorded PLATEAU_WINDOW steps ago (nearest available check)
                target = additional_step - PLATEAU_WINDOW
                prev = min(history, key=lambda hs: abs(hs[0] - target))
                rel_improve = (prev[1] - val_loss) / (abs(prev[1]) + 1e-12)
                if rel_improve < PLATEAU_REL_THRESHOLD:
                    stop_reason = (
                        f"plateaued: val_loss improved only {rel_improve * 100:.2f}% "
                        f"over the last {additional_step - prev[0]} steps "
                        f"(threshold {PLATEAU_REL_THRESHOLD * 100:.1f}%)"
                    )
                    break

        if additional_step == MAX_ADDITIONAL_STEPS:
            stop_reason = f"reached MAX_ADDITIONAL_STEPS={MAX_ADDITIONAL_STEPS} without plateauing"
            break

        batch = next(train_iter).to(device)
        loss, metrics = denoiser(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        step_loss = metrics["loss_denoising"].item()
        running_train_loss = step_loss if running_train_loss is None else 0.98 * running_train_loss + 0.02 * step_loss
        additional_step += 1

    total_step = start_step + additional_step
    print(f"\nStopped after {additional_step} additional steps (total_step={total_step}). Reason: {stop_reason}", flush=True)

    # Final theta_S curvature-activity check, slightly larger budget than the monitoring passes.
    denoiser.eval()
    final_group_sum, final_frac_zero = quick_theta_s_report(
        denoiser, params, slices, train_dataset, B=20, num_strata=3, seed=4321
    )
    final_all_nonzero = all(v > 0 for v in final_group_sum.values())
    total_mass = sum(final_group_sum.values())
    print("\nFinal theta_S module-wise curvature-activity check (B=20, num_strata=3):", flush=True)
    for name, mass in final_group_sum.items():
        frac = mass / total_mass if total_mass > 0 else 0.0
        print(f"  {name:<12} mass={mass:.6g}  mass_fraction={frac:.4f}  active={mass > 0}", flush=True)
    print(f"  all groups curvature-active: {final_all_nonzero}", flush=True)
    print(f"  fraction of individual theta_S coordinates with exactly-zero VJP^2 (all draws): {final_frac_zero:.4f}", flush=True)

    torch.save(
        {"denoiser": denoiser.state_dict(), "step": total_step, "action_dim": action_dim,
         "continued_from_step": start_step, "stop_reason": stop_reason, "val_loss_history": history},
        NEW_CHECKPOINT_PATH,
    )
    print(f"\nSaved converged diagnostic checkpoint to {NEW_CHECKPOINT_PATH} (total_step={total_step}).", flush=True)
    print("\nFull val-loss trajectory (additional_step, val_loss):", flush=True)
    for s, v in history:
        print(f"  {s:>6}  {v:.6f}", flush=True)

    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
