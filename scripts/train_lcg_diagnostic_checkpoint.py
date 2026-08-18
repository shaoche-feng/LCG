#! /usr/bin/env python
"""
Produce a diagnostic DIAMOND denoiser checkpoint that is trained just far enough that the
zero-initialized conv_out.weight / ResBlock.conv2.weight no longer structurally gate the
LCG VJP to exactly zero for the rest of theta_S (see Stage 2.5's root-cause diagnosis).

Uses ONLY the normal DIAMOND denoiser training path: real Denoiser.forward EDM loss, real
AdamW via utils.configure_opt (same lr/weight_decay/eps as config/trainer.yaml), real
BatchSampler/DataLoader replay sampling. No custom loss, no shortcuts. Trains on real
cheetah/run dm_control replay data (the one real episode recovered from
outputs/2026-08-17/15-35-46, supplemented with freshly-collected random-policy episodes
from the same env config, exactly as in scripts/validate_lcg_stage2.py -- no trained
collection policy exists in this repo, so this is the best available "real replay data").

This is an engineering/diagnostic checkpoint, not a research result: the collection policy
is random, and the goal is solely to get real gradient signal flowing through theta_S so
Stage 2's diagnostics can be re-run meaningfully.

Periodically (small-budget, B=8/num_strata=3) inspects the 5 theta_S module groups
(resblock_0, resblock_1, resblock_2, norm_out, conv_out) until curvature demonstrably
propagates to all five, then continues a modest amount further, then saves the checkpoint
and re-runs the essential (modest-budget) Stage 2 diagnostics.

Usage:
    python scripts/train_lcg_diagnostic_checkpoint.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BatchSampler, Dataset, Episode, collate_segments_to_batch
from envs.dm_control_env import make_dm_control_env
from lcg.gauss_newton import compute_vjp
from lcg.precision import historical_precision, load_transition, sample_valid_transitions
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from utils import configure_opt

REPO_ROOT = Path(__file__).resolve().parent.parent
RUN_DIR = REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46"
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
VAL_DATASET_DIR = SCRATCH_BASE / "lcg_diag_val_dataset"
CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

NUM_STEPS_CONDITIONING = 4
NUM_AUTOREGRESSIVE_STEPS = 1  # matches config/trainer.yaml denoiser.training default
TRAIN_BATCH_SIZE = 16
TRAIN_TARGET_STEPS = 1500
VAL_TARGET_STEPS = 300
OPTIMIZER_CFG = dict(lr=1e-4, weight_decay=1e-2, eps=1e-8)  # matches config/trainer.yaml

MONITOR_B, MONITOR_STRATA = 8, 3
FULL_B = 40  # for the final/essential-diagnostics pass
MAX_TRAIN_STEPS = 500
CHECK_STEPS = [0, 1, 2, 3, 5, 10, 20, 30, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500]
STABLE_CHECKS_AFTER_UNLOCK = 2
MIN_STEPS_BEFORE_STOP = 100


# --------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------


def collect_episode(env, rng: np.random.Generator) -> Episode:
    obs_frames, act_list, rew_list, end_list, trunc_list = [], [], [], [], []
    raw_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    while not done:
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        next_raw_obs, rew, terminated, truncated, _ = env.step(action)
        obs_frames.append(raw_obs)
        act_list.append(action)
        rew_list.append(rew)
        end_list.append(int(terminated))
        trunc_list.append(int(truncated))
        raw_obs = next_raw_obs
        done = terminated or truncated
    obs = torch.from_numpy(np.stack(obs_frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    act = torch.from_numpy(np.stack(act_list)).float()
    rew = torch.tensor(rew_list, dtype=torch.float32)
    end = torch.tensor(end_list, dtype=torch.uint8)
    trunc = torch.tensor(trunc_list, dtype=torch.uint8)
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info={})


def build_dataset(directory: Path, name: str, target_num_steps: int, seed: int, include_stray: bool) -> Dataset:
    if directory.exists():
        shutil.rmtree(directory)
    dataset = Dataset(directory, name, cache_in_ram=True)
    if include_stray:
        stray_path = RUN_DIR / "dataset" / "train" / "000" / "00" / "0" / "0.pt"
        if stray_path.is_file():
            dataset.add_episode(Episode.load(stray_path))
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    return dataset


# --------------------------------------------------------------------------------------
# theta_S groups (duplicated small helper, consistent with diagnose_lcg_stage2_5.py)
# --------------------------------------------------------------------------------------


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


def zero_init_tensor_stats(denoiser: Denoiser):
    final_level = denoiser.inner_model.unet.u_blocks[-1]
    stats = {"conv_out.weight": denoiser.inner_model.conv_out.weight}
    for i, rb in enumerate(final_level.resblocks):
        stats[f"resblock_{i}.conv2.weight"] = rb.conv2.weight
    return {name: (p.norm().item(), p.abs().mean().item()) for name, p in stats.items()}


# --------------------------------------------------------------------------------------
# Fast monitoring pass: small B, per-group curvature mass + exact-zero fraction
# --------------------------------------------------------------------------------------


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

    total = sum(group_sum.values())
    frac_exact_zero = all_zero_mask.float().mean().item()
    return group_sum, total, frac_exact_zero


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------


def build_fresh_denoiser(device: torch.device, action_dim: int, seed: int = 0) -> Denoiser:
    torch.manual_seed(seed)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    return denoiser


def make_train_loader(train_dataset: Dataset, batch_size: int):
    seq_length = NUM_STEPS_CONDITIONING + 1 + NUM_AUTOREGRESSIVE_STEPS
    bs = BatchSampler(train_dataset, rank=0, world_size=1, batch_size=batch_size, seq_length=seq_length, sample_weights=None)
    dl = DataLoader(dataset=train_dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    return iter(dl)


# --------------------------------------------------------------------------------------
# Main training + monitoring loop
# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_diag_train", TRAIN_TARGET_STEPS, seed=0, include_stray=True)
    val_dataset = build_dataset(VAL_DATASET_DIR, "lcg_diag_val", VAL_TARGET_STEPS, seed=999, include_stray=False)
    print(f"train dataset: {train_dataset.num_episodes} episodes, N={train_dataset.num_steps} steps")
    print(f"val dataset:   {val_dataset.num_episodes} episodes, N={val_dataset.num_steps} steps")

    action_dim = int(train_dataset.load_episode(0).act.shape[-1])
    denoiser = build_fresh_denoiser(device, action_dim=action_dim).to(device)
    denoiser.setup_training(SIGMA_CFG)
    params = selected_parameters(denoiser)
    slices = group_slices(denoiser)
    optimizer = configure_opt(denoiser, **OPTIMIZER_CFG)

    train_iter = make_train_loader(train_dataset, TRAIN_BATCH_SIZE)
    val_seq_length = NUM_STEPS_CONDITIONING + 1 + NUM_AUTOREGRESSIVE_STEPS
    val_sampler = BatchSampler(val_dataset, 0, 1, TRAIN_BATCH_SIZE, val_seq_length, sample_weights=None)
    val_segments = [val_dataset[sid] for sid in val_sampler.sample()]
    val_batch = collate_segments_to_batch(val_segments).to(device)

    unlocked_streak = 0
    step = 0
    header = (
        f"{'step':>5} {'train_loss':>11} {'val_loss':>11} {'convout|w|':>11} {'convout_mabs':>12} "
        f"{'rb0conv2|w|':>11} {'rb1conv2|w|':>11} {'rb2conv2|w|':>11} {'frac_zero':>10} "
        f"{'rb0_mass':>9} {'rb1_mass':>9} {'rb2_mass':>9} {'normout_mass':>12} {'convout_mass':>12}"
    )

    print("\n" + header)
    stop_step = None

    while step <= MAX_TRAIN_STEPS:
        if step in CHECK_STEPS or step == MAX_TRAIN_STEPS:
            denoiser.eval()
            with torch.no_grad():
                val_loss = denoiser(val_batch)[0].item()
            denoiser.train()

            z = zero_init_tensor_stats(denoiser)
            group_sum, total, frac_zero = quick_theta_s_report(denoiser, params, slices, train_dataset, MONITOR_B, MONITOR_STRATA, seed=1234)
            mass_frac = {name: (group_sum[name] / total if total > 0 else 0.0) for name in group_sum}

            train_loss_str = f"{last_train_loss:.5f}" if step > 0 else "n/a"
            print(
                f"{step:>5} {train_loss_str:>11} {val_loss:>11.5f} "
                f"{z['conv_out.weight'][0]:>11.5g} {z['conv_out.weight'][1]:>12.5g} "
                f"{z['resblock_0.conv2.weight'][0]:>11.5g} {z['resblock_1.conv2.weight'][0]:>11.5g} "
                f"{z['resblock_2.conv2.weight'][0]:>11.5g} {frac_zero:>10.4f} "
                f"{mass_frac['resblock_0']:>9.4f} {mass_frac['resblock_1']:>9.4f} {mass_frac['resblock_2']:>9.4f} "
                f"{mass_frac['norm_out']:>12.4f} {mass_frac['conv_out']:>12.4f}"
            )

            all_groups_nonzero = all(group_sum[name] > 0 for name in group_sum)
            if all_groups_nonzero:
                unlocked_streak += 1
            else:
                unlocked_streak = 0

            if all_groups_nonzero and unlocked_streak >= STABLE_CHECKS_AFTER_UNLOCK and step >= MIN_STEPS_BEFORE_STOP:
                stop_step = step
                break

        if step == MAX_TRAIN_STEPS:
            break

        batch = next(train_iter).to(device)
        loss, metrics = denoiser(batch)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        last_train_loss = metrics["loss_denoising"].item()
        step += 1

    if stop_step is None:
        print(f"\nDid NOT reach a stable all-groups-nonzero state within {MAX_TRAIN_STEPS} steps.")
        stop_step = step
    else:
        print(f"\nAll 5 theta_S groups showed nonzero curvature mass, stable for "
              f"{STABLE_CHECKS_AFTER_UNLOCK} consecutive checks, at step {stop_step}.")

    torch.save({"denoiser": denoiser.state_dict(), "step": stop_step, "action_dim": action_dim}, CHECKPOINT_PATH)
    print(f"Saved diagnostic checkpoint to {CHECKPOINT_PATH} (step={stop_step}).")

    # ---------------------------------------------------------------------------------
    # Essential Stage 2 diagnostics re-run, modest budget
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("ESSENTIAL STAGE 2 DIAGNOSTICS ON TRAINED CHECKPOINT (modest budget)")
    print("=" * 88)
    denoiser.eval()
    N = train_dataset.num_steps
    B_ESSENTIAL = 20
    DAMPING = 1e-4
    BETA = 1.0

    def run_h(B, num_strata, seed):
        return historical_precision(
            denoiser, params, train_dataset, SIGMA_CFG, B=B, num_strata=num_strata,
            N=N, beta=BETA, damping=DAMPING, seed=seed,
        )

    # 1. basic h_D distribution
    h_main = run_h(B_ESSENTIAL, 3, seed=0)
    q50, q90, q99 = torch.quantile(h_main, torch.tensor([0.5, 0.9, 0.99], device=h_main.device)).tolist()
    frac_damped = (h_main <= DAMPING * 1.01).float().mean().item()
    print(f"\n[1] basic h_D (B={B_ESSENTIAL}, num_strata=3, N={N}):")
    print(f"    min={h_main.min().item():.6g} median={q50:.6g} mean={h_main.mean().item():.6g} "
          f"p90={q90:.6g} p99={q99:.6g} max={h_main.max().item():.6g}")
    print(f"    fraction at damping floor: {frac_damped:.4f}")

    # 2. module-wise curvature mass (reuse h_main, no extra VJPs)
    print(f"\n[2] module-wise curvature mass (from the same h_D above):")
    excess_total = (h_main - DAMPING).sum().item()
    for name, start, stop in group_slices(denoiser):
        excess = (h_main[start:stop] - DAMPING).sum().item()
        print(f"    {name:<12} params={stop - start:>7}  mass_fraction={excess / excess_total if excess_total else 0:.4f}")

    # 3. seed stability + batch-size scaling B vs 2B (share draws)
    def pearson_corr(a, b):
        a, b = a - a.mean(), b - b.mean()
        return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()

    seeds = [0, 1, 2]
    hs_B = [h_main] + [run_h(B_ESSENTIAL, 3, seed=s) for s in seeds[1:]]
    hs_2B = [run_h(2 * B_ESSENTIAL, 3, seed=100 + s) for s in seeds]

    corrs, rel_l2 = [], []
    for i in range(len(hs_B)):
        for j in range(i + 1, len(hs_B)):
            corrs.append(pearson_corr(hs_B[i], hs_B[j]))
            rel_l2.append(((hs_B[i] - hs_B[j]).norm() / hs_B[j].norm()).item())
    print(f"\n[3] seed stability (B={B_ESSENTIAL}, num_strata=3, {len(hs_B)} seeds):")
    print(f"    mean pairwise correlation: {np.mean(corrs):.4f}   mean pairwise rel_L2: {np.mean(rel_l2):.4f}")

    norm_B = torch.stack([(h - DAMPING).norm() for h in hs_B])
    norm_2B = torch.stack([(h - DAMPING).norm() for h in hs_2B])
    print(f"\n[4] batch-size scaling B={B_ESSENTIAL} vs 2B={2 * B_ESSENTIAL} (N/B-corrected):")
    print(f"    ||h-damping|| : B -> mean={norm_B.mean():.4g} std={norm_B.std():.4g} | "
          f"2B -> mean={norm_2B.mean():.4g} std={norm_2B.std():.4g}")
    print(f"    ratio (2B/B): {(norm_2B.mean() / norm_B.mean()).item():.4f} (expect ~1.0)")
    print(f"    coordinate correlation, one B draw vs one 2B draw: {pearson_corr(hs_B[0], hs_2B[0]):.4f}")

    # 5. num_strata=1 vs 3 (compute-matched: 3x B for num_strata=1)
    hs_strata1 = [run_h(3 * B_ESSENTIAL, 1, seed=200 + s) for s in seeds]
    corrs1, rel_l2_1 = [], []
    for i in range(len(hs_strata1)):
        for j in range(i + 1, len(hs_strata1)):
            corrs1.append(pearson_corr(hs_strata1[i], hs_strata1[j]))
            rel_l2_1.append(((hs_strata1[i] - hs_strata1[j]).norm() / hs_strata1[j].norm()).item())
    print(f"\n[5] stratification, compute-matched (num_strata=3,B={B_ESSENTIAL} vs num_strata=1,B={3 * B_ESSENTIAL}):")
    print(f"    num_strata=3: mean_corr={np.mean(corrs):.4f} mean_rel_l2={np.mean(rel_l2):.4f}")
    print(f"    num_strata=1: mean_corr={np.mean(corrs1):.4f} mean_rel_l2={np.mean(rel_l2_1):.4f}")

    print("\nDone.")


if __name__ == "__main__":
    main()
