#! /usr/bin/env python
"""
LCG historical-precision diagnostic: full-dataset h_D vs subset (B) h_D.

Does NOT modify src/lcg/*.py. Reuses lcg.gauss_newton.compute_vjp,
lcg.sigma_strata.sample_sigma_stratum, lcg.precision.load_transition, lcg.theta_s.* exactly
as validated in Stages 1-2, with the SAME formula as lcg.precision.historical_precision:

    h_D(B) = damping*1 + beta*(N/B) * sum_{i in B} g_i,   g_i = (1/num_strata) sum_m v_im^2

Design note (deterministic per-transition probe reuse):
lcg.precision.historical_precision and lcg.gauss_newton.compute_vjp draw sigma/eps/xi from
the GLOBAL torch RNG stream; only *which* transitions are selected is seeded (via
np.random.seed inside sample_valid_transitions). Calling historical_precision once on the
full dataset and once on a subset would NOT give the same transition matching (sigma,eps,xi)
draws in both calls, since the RNG stream position depends on call order/count -- this would
mix transition-subsampling error with sigma/eps/xi resampling noise, which is exactly what
this diagnostic is designed to avoid. Fix used here (diagnostic-script-only, no src/lcg/*.py
changes): reseed the global RNG via torch.manual_seed(seed) to a value derived ONLY from each
transition's own identity (episode_id, start, stratum_idx) immediately before calling the
same unmodified sample_sigma_stratum/compute_vjp functions for that (transition, stratum).
This guarantees bit-identical (sigma,eps,xi) whenever a transition appears in both the full
reference and any subset, purely from the transition's own identity -- independent of call
order or which other transitions are present in a given call.

Full-dataset reference construction: lcg.precision.sample_valid_transitions draws a stochastic
WITH-REPLACEMENT bootstrap sample from BatchSampler even when B=N (episode-length-weighted),
so it does NOT enumerate "every valid transition exactly once" -- calling it with B=N would
leave ~37% of the dataset's transitions completely absent from the "full" reference and
duplicate others, which is not what Algorithm 2's sum_{i=1}^N means. This script instead
deterministically enumerates every one of the N=dataset.num_steps valid transitions directly
(one per real, unpadded environment step across all episodes), matching the same validity
contract lcg.precision.load_transition already checks (target frame must be real/unpadded).

Efficiency note: because per-transition probes are identity-seeded (not call-order-seeded),
every subset's sum over its B members is EXACTLY reconstructible from the same per-transition
g_i values used in the full reference. So this script computes each of the N transitions'
g_i EXACTLY ONCE (a single pass, N*num_strata VJPs total) and accumulates it into the full
sum AND into every subset (B, seed) whose fixed index set contains that transition -- no
extra VJPs are spent on the subset sweep at all. Subset "runtime" is reported both as the
actual (near-zero, reused) wall time and as an estimated standalone cost extrapolated from
the measured per-VJP cost, labeled accordingly.

Usage:
    python scripts/diagnose_lcg_precision_full_vs_subset.py
"""
import sys
import time
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root (no ancestor has both src/ and scripts/)")
        p = p.parent
    return p


sys.path.insert(0, str(_find_repo_root(Path(__file__).parent) / "src"))

import hashlib
import numpy as np
import torch

from data import Dataset, SegmentId
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser_converged.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"

SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
NUM_STEPS_CONDITIONING = 4
NUM_STRATA = 3
DAMPING = 1e-4
BETA = 1.0
B_LIST = [20, 40, 80, 160, 320, 640]
NUM_SEEDS = 5
BASE_SALT = "lcg_precision_full_vs_subset_v1"
PROGRESS_EVERY = 100


# --------------------------------------------------------------------------------------
# Setup
# --------------------------------------------------------------------------------------


def load_converged_denoiser(device):
    ckpt = torch.load(CHECKPOINT_PATH, map_location=device, weights_only=False)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=ckpt["action_dim"],
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    print(f"Loaded converged diagnostic checkpoint (total_step={ckpt['step']}, "
          f"continued_from_step={ckpt.get('continued_from_step')}) from {CHECKPOINT_PATH}", flush=True)
    return denoiser


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


def enumerate_all_transitions(dataset: Dataset, num_steps_conditioning: int):
    """One SegmentId per real, unpadded environment step across the whole dataset --
    matches load_transition's validity contract exactly, giving exactly dataset.num_steps
    unique transitions, deterministically (no BatchSampler stochastic resampling)."""
    segment_ids = []
    for episode_id in range(dataset.num_episodes):
        length = int(dataset.lengths[episode_id])
        for t in range(length):
            start = t - num_steps_conditioning
            stop = t + 1
            segment_ids.append(SegmentId(episode_id, start, stop))
    return segment_ids


def transition_stratum_seed(episode_id: int, start: int, stratum_idx: int, base_salt: str) -> int:
    key = f"{base_salt}:{episode_id}:{start}:{stratum_idx}".encode()
    return int.from_bytes(hashlib.sha256(key).digest()[:4], "big")


def compute_g_i(denoiser, params, dataset, segment_id, num_strata, sigma_cfg, base_salt, device):
    obs, act, y = load_transition(dataset, segment_id, NUM_STEPS_CONDITIONING, device)
    d_S = sum(p.numel() for p in params)
    g = torch.zeros(d_S, device=device)
    for m in range(num_strata):
        seed = transition_stratum_seed(segment_id.episode_id, segment_id.start, m, base_salt)
        torch.manual_seed(seed)
        sigma = sample_sigma_stratum(sigma_cfg, m, num_strata, 1, device)
        eps = torch.randn_like(y)
        y_sigma = (y + sigma.view(-1, 1, 1, 1) * eps).detach()
        v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act)
        g = g + (v * v) / num_strata
    return g


# --------------------------------------------------------------------------------------
# Metrics
# --------------------------------------------------------------------------------------


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


def cosine_sim(a: torch.Tensor, b: torch.Tensor) -> float:
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


def _rank(x: torch.Tensor) -> torch.Tensor:
    """Argsort-based ranks (ties broken arbitrarily -- fine here since h_D is a continuous
    sum of squared VJPs and exact float32 ties are astronomically unlikely). Avoids
    scipy.stats.spearmanr, which triggers a Windows/conda libiomp5md.dll double-init crash
    (OMP Error #15) when scipy and torch are both loaded in this environment."""
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    return ranks


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    return pearson_corr(_rank(a), _rank(b))


def region_masks(h_full: torch.Tensor):
    d_S = h_full.numel()
    order = torch.argsort(h_full, descending=True)
    top10_n = max(1, int(round(0.10 * d_S)))
    top1_n = max(1, int(round(0.01 * d_S)))
    bottom50_n = max(1, int(round(0.50 * d_S)))
    masks = {
        "all": torch.ones(d_S, dtype=torch.bool),
        "top10pct": torch.zeros(d_S, dtype=torch.bool),
        "top1pct": torch.zeros(d_S, dtype=torch.bool),
        "bottom50pct": torch.zeros(d_S, dtype=torch.bool),
    }
    masks["top10pct"][order[:top10_n]] = True
    masks["top1pct"][order[:top1_n]] = True
    masks["bottom50pct"][order[-bottom50_n:]] = True
    return masks


def compare(h_sub: torch.Tensor, h_full: torch.Tensor, mask: torch.Tensor) -> dict:
    a, b = h_sub[mask], h_full[mask]
    rel_l2 = ((a - b).norm() / (b.norm() + 1e-30)).item()
    norm_ratio = (a.norm() / (b.norm() + 1e-30)).item()
    return dict(
        pearson=pearson_corr(a, b),
        spearman=spearman_corr(a, b),
        rel_l2=rel_l2,
        norm_ratio=norm_ratio,
        cosine=cosine_sim(a, b),
        n_coords=int(mask.sum().item()),
    )


def mean_std(values):
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std())


# --------------------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train", cache_in_ram=True)
    dataset.load_from_default_path()
    N = dataset.num_steps
    print(f"dataset: {dataset.num_episodes} episodes, N={N} steps", flush=True)

    denoiser = load_converged_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)
    slices = group_slices(denoiser)
    print(f"theta_S: d_S={d_S}", flush=True)

    all_segment_ids = enumerate_all_transitions(dataset, NUM_STEPS_CONDITIONING)
    assert len(all_segment_ids) == N, f"enumeration produced {len(all_segment_ids)} transitions, expected N={N}"
    print(f"Deterministically enumerated {len(all_segment_ids)} unique valid transitions "
          f"(one per real step; matches N exactly, no BatchSampler resampling).", flush=True)

    # ---- Fix the B/seed subset index sets up front (without replacement, from the N-list) ----
    subset_targets = []  # list of dicts: B, seed, indices(set), accumulator
    for B in B_LIST:
        for s in range(NUM_SEEDS):
            rng = np.random.default_rng(B * 100000 + s)
            idx = rng.choice(N, size=B, replace=False)
            subset_targets.append(dict(B=B, seed=s, indices=set(int(i) for i in idx),
                                        accum=torch.zeros(d_S, device=device)))
    print(f"Prepared {len(subset_targets)} subset replicates across B in {B_LIST} ({NUM_SEEDS} seeds each).", flush=True)

    # ---- Single pass over all N transitions: each g_i computed exactly once ----
    sum_full = torch.zeros(d_S, device=device)
    t0 = time.time()
    vjp_times = []
    for i, segment_id in enumerate(all_segment_ids):
        t_vjp0 = time.time()
        g_i = compute_g_i(denoiser, params, dataset, segment_id, NUM_STRATA, SIGMA_CFG, BASE_SALT, device)
        vjp_times.append((time.time() - t_vjp0) / NUM_STRATA)  # per-VJP, not per-transition

        sum_full += g_i
        for target in subset_targets:
            if i in target["indices"]:
                target["accum"] += g_i

        if (i + 1) % PROGRESS_EVERY == 0 or (i + 1) == N:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            eta = (N - (i + 1)) / rate if rate > 0 else float("nan")
            print(f"  [{i + 1:>5}/{N}] elapsed={elapsed:>7.1f}s  rate={rate:>6.2f} transitions/s  "
                  f"eta={eta:>7.1f}s", flush=True)

    total_time = time.time() - t0
    avg_vjp_ms = 1000.0 * float(np.mean(vjp_times))
    print(f"\nFull single pass complete: {N} transitions x {NUM_STRATA} strata = {N * NUM_STRATA} VJPs "
          f"in {total_time:.1f}s ({avg_vjp_ms:.2f} ms/VJP).", flush=True)

    h_D_full = DAMPING + BETA * sum_full
    for target in subset_targets:
        B = target["B"]
        target["h_D"] = DAMPING + BETA * (N / B) * target["accum"]
        target["estimated_standalone_runtime_s"] = (avg_vjp_ms / 1000.0) * B * NUM_STRATA

    # ---------------------------------------------------------------------------------
    # Part 1 sanity: basic validity of h_D_full
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 1: full-dataset reference h_D_full")
    print("=" * 88)
    finite = torch.isfinite(h_D_full).all().item()
    nonneg = (h_D_full >= 0).all().item()
    q50, q90, q99 = torch.quantile(h_D_full, torch.tensor([0.5, 0.9, 0.99], device=device)).tolist()
    print(f"  finite={finite}  nonneg={nonneg}")
    print(f"  min={h_D_full.min().item():.6g}  median={q50:.6g}  mean={h_D_full.mean().item():.6g}  "
          f"p90={q90:.6g}  p99={q99:.6g}  max={h_D_full.max().item():.6g}")
    print(f"  ||h_D_full||_2 = {h_D_full.norm().item():.6g}")
    assert finite and nonneg

    # ---------------------------------------------------------------------------------
    # Part 3: per-replicate comparison vs h_D_full (all coordinates)
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 3: subset-size sweep -- per-replicate comparison to h_D_full (all coordinates)")
    print("=" * 88)
    print(f"\n{'B':>5} {'seed':>4} {'pearson':>9} {'spearman':>9} {'rel_L2':>9} {'norm_ratio':>10} "
          f"{'cosine':>8} {'est_runtime_s':>13}")
    masks_full = region_masks(h_D_full)
    for target in subset_targets:
        m = compare(target["h_D"], h_D_full, masks_full["all"])
        target["metrics_all"] = m
        print(f"{target['B']:>5} {target['seed']:>4} {m['pearson']:>9.5f} {m['spearman']:>9.5f} "
              f"{m['rel_l2']:>9.4f} {m['norm_ratio']:>10.4f} {m['cosine']:>8.5f} "
              f"{target['estimated_standalone_runtime_s']:>13.2f}", flush=True)

    print(f"\n{'B':>5} {'pearson(mean±std)':>20} {'spearman(mean±std)':>20} {'rel_L2(mean±std)':>18} "
          f"{'norm_ratio(mean±std)':>22}")
    summary_by_B = {}
    for B in B_LIST:
        group = [t for t in subset_targets if t["B"] == B]
        p_m, p_s = mean_std([t["metrics_all"]["pearson"] for t in group])
        sp_m, sp_s = mean_std([t["metrics_all"]["spearman"] for t in group])
        l2_m, l2_s = mean_std([t["metrics_all"]["rel_l2"] for t in group])
        nr_m, nr_s = mean_std([t["metrics_all"]["norm_ratio"] for t in group])
        summary_by_B[B] = dict(pearson=(p_m, p_s), spearman=(sp_m, sp_s), rel_l2=(l2_m, l2_s), norm_ratio=(nr_m, nr_s))
        print(f"{B:>5} {p_m:>10.5f}±{p_s:<8.5f} {sp_m:>10.5f}±{sp_s:<8.5f} "
              f"{l2_m:>8.4f}±{l2_s:<8.4f} {nr_m:>10.4f}±{nr_s:<10.4f}")

    # ---------------------------------------------------------------------------------
    # Part 4: magnitude-region breakdown + module-wise curvature mass
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 4: agreement by magnitude region")
    print("=" * 88)
    for region_name, mask in masks_full.items():
        print(f"\n--- region: {region_name} (n_coords={int(mask.sum().item())}) ---")
        print(f"{'B':>5} {'pearson(mean±std)':>20} {'rel_L2(mean±std)':>18}")
        for B in B_LIST:
            group = [t for t in subset_targets if t["B"] == B]
            region_metrics = [compare(t["h_D"], h_D_full, mask) for t in group]
            p_m, p_s = mean_std([m["pearson"] for m in region_metrics])
            l2_m, l2_s = mean_std([m["rel_l2"] for m in region_metrics])
            print(f"{B:>5} {p_m:>10.5f}±{p_s:<8.5f} {l2_m:>8.4f}±{l2_s:<8.4f}")

    print("\n" + "=" * 88)
    print("PART 4b: module-wise curvature mass (full reference vs subset, mean over seeds)")
    print("=" * 88)
    excess_total_full = (h_D_full - DAMPING).sum().item()
    print(f"\n--- full reference (N={N}) ---")
    for name, start, stop in slices:
        excess = (h_D_full[start:stop] - DAMPING).sum().item()
        frac = excess / excess_total_full if excess_total_full else 0.0
        print(f"  {name:<12} params={stop - start:>7}  mass_fraction={frac:.4f}")

    for B in B_LIST:
        group = [t for t in subset_targets if t["B"] == B]
        print(f"\n--- subset B={B} (mean mass_fraction over {len(group)} seeds) ---")
        for name, start, stop in slices:
            fracs = []
            for t in group:
                excess_total_sub = (t["h_D"] - DAMPING).sum().item()
                excess = (t["h_D"][start:stop] - DAMPING).sum().item()
                fracs.append(excess / excess_total_sub if excess_total_sub else 0.0)
            f_m, f_s = mean_std(fracs)
            print(f"  {name:<12} params={stop - start:>7}  mass_fraction={f_m:.4f}±{f_s:.4f}")

    # ---------------------------------------------------------------------------------
    # Part 5: explicit answers
    # ---------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 5: explicit answers")
    print("=" * 88)
    b40 = summary_by_B.get(40)
    if b40:
        print(f"\n[Q1] Was the previous B=40 h_D estimate close to the full-data estimate?")
        print(f"     B=40 vs h_D_full: pearson={b40['pearson'][0]:.4f}±{b40['pearson'][1]:.4f}  "
              f"rel_L2={b40['rel_l2'][0]:.4f}±{b40['rel_l2'][1]:.4f}  "
              f"norm_ratio={b40['norm_ratio'][0]:.4f}±{b40['norm_ratio'][1]:.4f}")

    print(f"\n[Q2] Plateau point (Pearson/rel_L2 vs B):")
    for B in B_LIST:
        s = summary_by_B[B]
        print(f"     B={B:>4}: pearson={s['pearson'][0]:.4f}  rel_L2={s['rel_l2'][0]:.4f}")

    print(f"\n[Q3] Is B=40 sufficient for practical LCG use, or a major noise source? "
          f"(see Q1/Q2 numbers above and the seed-to-seed std at B=40)")

    print(f"\n[Q4] Residual variation across transition-subset seeds with probes held fixed "
          f"(std of rel_L2 and pearson across the {NUM_SEEDS} seeds, per B):")
    for B in B_LIST:
        s = summary_by_B[B]
        print(f"     B={B:>4}: std(pearson)={s['pearson'][1]:.5f}  std(rel_L2)={s['rel_l2'][1]:.5f}")

    print(f"\n[Q5] Does the N/B correction preserve scale? norm_ratio (||h_D(B)||/||h_D_full||) "
          f"should be close to 1.0 for all B if so:")
    for B in B_LIST:
        s = summary_by_B[B]
        print(f"     B={B:>4}: norm_ratio={s['norm_ratio'][0]:.4f}±{s['norm_ratio'][1]:.4f}")

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
