#! /usr/bin/env python
"""
Train a genuinely matched N~5000 diagnostic denoiser checkpoint, using EXACTLY the same
architecture/optimizer/stopping-criterion recipe as the existing N=1540 diagnostic checkpoint
(recovered from git history: scripts/pre_implement_lcg_diagnostic/train_lcg_diagnostic_checkpoint.py
+ scripts/precision_h_D_diagnostic/continue_train_lcg_diagnostic_checkpoint.py, both deleted in
the LCG script-cleanup pass but preserved in git log) -- real Denoiser.forward EDM loss, real
AdamW via utils.configure_opt (same lr/weight_decay/eps), real BatchSampler/DataLoader replay
sampling, no shortcuts.

Two stages, combined into one script (the original split across two scripts was an artifact of
iterative interactive work, not a methodological requirement):

  Stage 1 (curvature-unlock): fresh denoiser (seed=0 init), trains until all 5 theta_S module
  groups (resblock_0/1/2, norm_out, conv_out) show nonzero VJP^2 curvature mass, stable for 2
  consecutive checks, min 100 / max 500 steps. The zero-initialized conv_out.weight /
  ResBlock.conv2.weight otherwise structurally gate the LCG VJP to exactly zero.

  Stage 2 (convergence): continues ordinary training from the Stage-1 checkpoint until
  validation EDM loss plateaus (relative improvement < 2% over the last 500 steps, min 300 /
  max 3000 additional steps).

Monitoring during both stages reuses the CURRENT production LCG primitives unchanged
(lcg.gauss_newton.compute_vjp, lcg.theta_s.selected_parameters, models.diffusion.denoiser.
sample_sigma_training_distribution) -- plain IID Monte Carlo, matching production exactly
(the original recipe's stratified sigma sampler, lcg.sigma_strata.sample_sigma_stratum, no
longer exists; this diagnostic-only monitoring loop uses the same IID estimator production
itself now uses, which is a strictly more faithful match to "current production" than
resurrecting the deleted stratified sampler would be).

Everything is saved under a durable repository-side directory
(docs/lcg_diagnostic/historical_precision_B_scaling/N5000/), NOT OS Temp or the Claude
scratchpad.

Usage:
    python scripts/historical_precision_diagnostic/train_checkpoint_N5000.py
"""
import sys
import time
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))

import hashlib
import json

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BatchSampler, Dataset, collate_segments_to_batch
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition, sample_uniform_historical_transitions
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.denoiser import sample_sigma_training_distribution
from models.diffusion.inner_model import InnerModelConfig
from utils import configure_opt

OUT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "historical_precision_B_scaling" / "N5000"
TRAIN_DATASET_DIR = OUT_DIR / "train_dataset"
VAL_DATASET_DIR = OUT_DIR / "val_dataset"
STAGE1_CHECKPOINT_PATH = OUT_DIR / "denoiser_stage1_unlock.pt"
CONVERGED_CHECKPOINT_PATH = OUT_DIR / "denoiser_converged.pt"
TRAINING_LOG_PATH = OUT_DIR / "training_log.json"

SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
NUM_STEPS_CONDITIONING = 4
NUM_AUTOREGRESSIVE_STEPS = 1
TRAIN_BATCH_SIZE = 16
OPTIMIZER_CFG = dict(lr=1e-4, weight_decay=1e-2, eps=1e-8)
MONITOR_M = 3  # production num_mc, matches the historical-precision estimator exactly

# Stage 1 (curvature-unlock)
STAGE1_MONITOR_B = 8
STAGE1_MAX_STEPS = 500
STAGE1_CHECK_STEPS = [0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
STAGE1_STABLE_CHECKS_AFTER_UNLOCK = 2
STAGE1_MIN_STEPS_BEFORE_STOP = 100

# Stage 2 (convergence)
STAGE2_VAL_EVERY = 100
STAGE2_MONITOR_B = 8
STAGE2_MIN_ADDITIONAL_STEPS = 300
STAGE2_PLATEAU_WINDOW = 500
STAGE2_PLATEAU_REL_THRESHOLD = 0.02
STAGE2_MAX_ADDITIONAL_STEPS = 3000


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


def quick_theta_s_report(denoiser, params, slices, dataset, B, num_mc, seed):
    """Diagnostic-only monitoring pass (not production LCG): reuses compute_vjp/
    sample_sigma_training_distribution unmodified, with a local generator (RNG isolation)."""
    device = denoiser.device
    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(seed)
    segment_ids = sample_uniform_historical_transitions(dataset, B, NUM_STEPS_CONDITIONING, seed=seed)
    d_S = sum(p.numel() for p in params)
    group_sum = {name: 0.0 for name, _, _ in slices}
    all_zero_mask = torch.ones(d_S, dtype=torch.bool)

    was_training = denoiser.training
    denoiser.eval()
    for segment_id in segment_ids:
        obs, act, y = load_transition(dataset, segment_id, NUM_STEPS_CONDITIONING, device)
        for _ in range(num_mc):
            sigma = sample_sigma_training_distribution(SIGMA_CFG, 1, device, generator=torch_gen)
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=torch_gen)
            y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, generator=torch_gen)
            v2 = (v * v).detach().cpu()
            all_zero_mask &= v2 == 0
            for name, start, stop in slices:
                group_sum[name] += v2[start:stop].sum().item()
    if was_training:
        denoiser.train()

    return group_sum, all_zero_mask.float().mean().item()


def build_fresh_denoiser(device: torch.device, action_dim: int, seed: int = 0) -> Denoiser:
    torch.manual_seed(seed)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    return Denoiser(cfg).to(device)


def make_train_loader(train_dataset: Dataset, batch_size: int):
    seq_length = NUM_STEPS_CONDITIONING + 1 + NUM_AUTOREGRESSIVE_STEPS
    bs = BatchSampler(train_dataset, rank=0, world_size=1, batch_size=batch_size, seq_length=seq_length, sample_weights=None)
    dl = DataLoader(dataset=train_dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    return iter(dl)


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def make_val_batch(val_dataset: Dataset, batch_size: int, device: torch.device):
    seq_length = NUM_STEPS_CONDITIONING + 1 + NUM_AUTOREGRESSIVE_STEPS
    val_sampler = BatchSampler(val_dataset, 0, 1, batch_size, seq_length, sample_weights=None)
    val_segments = [val_dataset[sid] for sid in val_sampler.sample()]
    return collate_segments_to_batch(val_segments).to(device)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    train_dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train_N5000", cache_in_ram=True)
    train_dataset.load_from_default_path()
    val_dataset = Dataset(VAL_DATASET_DIR, "lcg_diag_val_N5000", cache_in_ram=True)
    val_dataset.load_from_default_path()
    assert train_dataset.num_steps > 0 and val_dataset.num_steps > 0, "run build_replay_N5000.py first"
    print(f"train replay: {train_dataset.num_episodes} episodes, N={train_dataset.num_steps} steps", flush=True)
    print(f"val replay:   {val_dataset.num_episodes} episodes, N={val_dataset.num_steps} steps", flush=True)

    action_dim = int(train_dataset.load_episode(0).act.shape[-1])
    denoiser = build_fresh_denoiser(device, action_dim=action_dim, seed=0)
    denoiser.setup_training(SIGMA_CFG)
    params = selected_parameters(denoiser)
    slices = group_slices(denoiser)
    optimizer = configure_opt(denoiser, **OPTIMIZER_CFG)

    train_iter = make_train_loader(train_dataset, TRAIN_BATCH_SIZE)
    val_batch = make_val_batch(val_dataset, TRAIN_BATCH_SIZE, device)

    t_start = time.time()
    log = dict(
        dataset_train_N=train_dataset.num_steps, dataset_train_num_episodes=train_dataset.num_episodes,
        dataset_val_N=val_dataset.num_steps, action_dim=action_dim,
    )

    # ================================================================================
    # Stage 1: curvature-unlock
    # ================================================================================
    print("\n" + "=" * 88 + "\nSTAGE 1: curvature-unlock\n" + "=" * 88, flush=True)
    header = f"{'step':>5} {'train_loss':>11} {'val_loss':>11} {'frac_zero':>10} {'all_nonzero':>11} {'elapsed_s':>9}"
    print(header, flush=True)

    unlocked_streak = 0
    step = 0
    last_train_loss = None
    stage1_stop_step = None
    stage1_history = []

    while step <= STAGE1_MAX_STEPS:
        if step in STAGE1_CHECK_STEPS or step == STAGE1_MAX_STEPS:
            denoiser.eval()
            with torch.no_grad():
                val_loss = denoiser(val_batch)[0].item()
            denoiser.train()
            group_sum, frac_zero = quick_theta_s_report(denoiser, params, slices, train_dataset, STAGE1_MONITOR_B, MONITOR_M, seed=1234)
            all_nonzero = all(v > 0 for v in group_sum.values())
            stage1_history.append((step, val_loss))
            train_loss_str = f"{last_train_loss:.5f}" if last_train_loss is not None else "n/a"
            print(f"{step:>5} {train_loss_str:>11} {val_loss:>11.5f} {frac_zero:>10.4f} {str(all_nonzero):>11} "
                  f"{time.time() - t_start:>9.1f}", flush=True)

            unlocked_streak = unlocked_streak + 1 if all_nonzero else 0
            if all_nonzero and unlocked_streak >= STAGE1_STABLE_CHECKS_AFTER_UNLOCK and step >= STAGE1_MIN_STEPS_BEFORE_STOP:
                stage1_stop_step = step
                break

        if step == STAGE1_MAX_STEPS:
            break

        batch = next(train_iter).to(device)
        loss, metrics = denoiser(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_train_loss = metrics["loss_denoising"].item()
        step += 1

    if stage1_stop_step is None:
        print(f"\nStage 1 did NOT reach a stable all-groups-nonzero state within {STAGE1_MAX_STEPS} steps.", flush=True)
        stage1_stop_step = step
    else:
        print(f"\nStage 1: all 5 theta_S groups curvature-active, stable for {STAGE1_STABLE_CHECKS_AFTER_UNLOCK} "
              f"consecutive checks, at step {stage1_stop_step}.", flush=True)

    torch.save({"denoiser": denoiser.state_dict(), "step": stage1_stop_step, "action_dim": action_dim}, STAGE1_CHECKPOINT_PATH)
    print(f"Saved Stage-1 checkpoint to {STAGE1_CHECKPOINT_PATH} (step={stage1_stop_step}).", flush=True)
    log["stage1_stop_step"] = stage1_stop_step
    log["stage1_val_loss_history"] = stage1_history
    log["stage1_elapsed_s"] = time.time() - t_start

    # ================================================================================
    # Stage 2: convergence
    # ================================================================================
    print("\n" + "=" * 88 + "\nSTAGE 2: convergence\n" + "=" * 88, flush=True)
    start_step = stage1_stop_step
    print(f"{'add_step':>9} {'total_step':>10} {'train_loss':>11} {'val_loss':>11} "
          f"{'frac_zero':>10} {'all_nonzero':>11} {'elapsed_s':>9}", flush=True)

    stage2_history = []
    running_train_loss = None
    additional_step = 0
    stop_reason = None
    t_stage2 = time.time()

    while additional_step <= STAGE2_MAX_ADDITIONAL_STEPS:
        if additional_step % STAGE2_VAL_EVERY == 0:
            denoiser.eval()
            with torch.no_grad():
                val_loss = denoiser(val_batch)[0].item()
            denoiser.train()
            group_sum, frac_zero = quick_theta_s_report(denoiser, params, slices, train_dataset, STAGE2_MONITOR_B, MONITOR_M, seed=1234)
            all_nonzero = all(v > 0 for v in group_sum.values())
            stage2_history.append((additional_step, val_loss))
            train_loss_str = f"{running_train_loss:.5f}" if running_train_loss is not None else "n/a"
            print(f"{additional_step:>9} {start_step + additional_step:>10} {train_loss_str:>11} "
                  f"{val_loss:>11.5f} {frac_zero:>10.4f} {str(all_nonzero):>11} {time.time() - t_stage2:>9.1f}", flush=True)
            if not all_nonzero:
                print("  WARNING: at least one theta_S group has exactly-zero curvature mass!", flush=True)

            if additional_step >= STAGE2_MIN_ADDITIONAL_STEPS and additional_step >= STAGE2_PLATEAU_WINDOW:
                target = additional_step - STAGE2_PLATEAU_WINDOW
                prev = min(stage2_history, key=lambda hs: abs(hs[0] - target))
                rel_improve = (prev[1] - val_loss) / (abs(prev[1]) + 1e-12)
                if rel_improve < STAGE2_PLATEAU_REL_THRESHOLD:
                    stop_reason = (
                        f"plateaued: val_loss improved only {rel_improve * 100:.2f}% over the last "
                        f"{additional_step - prev[0]} steps (threshold {STAGE2_PLATEAU_REL_THRESHOLD * 100:.1f}%)"
                    )
                    break

        if additional_step == STAGE2_MAX_ADDITIONAL_STEPS:
            stop_reason = f"reached MAX_ADDITIONAL_STEPS={STAGE2_MAX_ADDITIONAL_STEPS} without plateauing"
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
    print(f"\nStage 2 stopped after {additional_step} additional steps (total_step={total_step}). Reason: {stop_reason}", flush=True)

    denoiser.eval()
    final_group_sum, final_frac_zero = quick_theta_s_report(denoiser, params, slices, train_dataset, B=20, num_mc=MONITOR_M, seed=4321)
    final_all_nonzero = all(v > 0 for v in final_group_sum.values())
    total_mass = sum(final_group_sum.values())
    print("\nFinal theta_S module-wise curvature-activity check (B=20, num_mc=3):", flush=True)
    for name, mass in final_group_sum.items():
        frac = mass / total_mass if total_mass > 0 else 0.0
        print(f"  {name:<12} mass={mass:.6g}  mass_fraction={frac:.4f}  active={mass > 0}", flush=True)
    print(f"  all groups curvature-active: {final_all_nonzero}", flush=True)
    print(f"  fraction of individual theta_S coordinates with exactly-zero VJP^2 (all draws): {final_frac_zero:.4f}", flush=True)

    final_train_loss = running_train_loss
    final_val_loss = stage2_history[-1][1] if stage2_history else None

    torch.save(
        {"denoiser": denoiser.state_dict(), "step": total_step, "action_dim": action_dim,
         "continued_from_step": start_step, "stop_reason": stop_reason, "val_loss_history": stage2_history},
        CONVERGED_CHECKPOINT_PATH,
    )
    checkpoint_sha256 = sha256_of(CONVERGED_CHECKPOINT_PATH)
    print(f"\nSaved converged checkpoint to {CONVERGED_CHECKPOINT_PATH} (total_step={total_step}).", flush=True)
    print(f"checkpoint sha256={checkpoint_sha256}", flush=True)

    log.update(dict(
        stage2_stop_reason=stop_reason, stage2_additional_steps=additional_step, total_step=total_step,
        stage2_val_loss_history=stage2_history, final_train_loss=final_train_loss, final_val_loss=final_val_loss,
        final_group_sum=final_group_sum, final_all_nonzero=final_all_nonzero, final_frac_zero=final_frac_zero,
        checkpoint_sha256=checkpoint_sha256, stage2_elapsed_s=time.time() - t_stage2,
        total_elapsed_s=time.time() - t_start,
    ))
    with open(TRAINING_LOG_PATH, "w") as f:
        json.dump(log, f, indent=2, default=str)
    print(f"\nWrote training log to {TRAINING_LOG_PATH}", flush=True)
    print("\nDone.", flush=True)


if __name__ == "__main__":
    main()
