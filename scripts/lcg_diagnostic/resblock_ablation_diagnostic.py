#! /usr/bin/env python
"""
Final decoder-stage ResBlock ablation: which subset of {R1, R2, R3, norm_out, conv_out}
(the 3 ResBlocks inside denoiser.inner_model.unet.u_blocks[-1], plus the output head)
preserves the current full/default theta_S's LCG ranking, and at what M / runtime / memory
cost. DIAGNOSTIC ONLY -- does not modify src/lcg/*.py or any production default
(config/intrinsic_reward/lcg.yaml's theta_s/candidate_num_mc=12/precision_reference_size=320/
precision_num_mc=3 are all left untouched). Does not start Random-vs-LCG training.

Model structure (verified from the actual checkpoint before writing any config below, NOT
assumed -- see the printed "STRUCTURE VERIFICATION" section at the top of this script's
output): denoiser.inner_model.unet.u_blocks[-1] is a lcg-unrelated `models.blocks.ResBlocks`
module whose only children are `resblocks` (an nn.ModuleList of exactly 3 ResBlock instances,
prefixes unet.u_blocks.3.resblocks.{0,1,2}.), each with 10 parameter tensors and d_S=217,664;
nothing else lives under u_blocks[-1] outside resblocks.*.

Six theta_S configurations tested (the nested nested-suffix nested_prefix, plus 2 extra
individual points -- mandatory per spec):
    R1              = resblocks.0                              (d_S=217,664)
    R2              = resblocks.1                              (d_S=217,664)
    R3              = resblocks.2                              (d_S=217,664)
    R2+R3           = resblocks.1 + resblocks.2                 (d_S=435,328)
    3R              = resblocks.0 + resblocks.1 + resblocks.2   (d_S=652,992)
    full/default    = 3R + norm_out + conv_out                  (d_S=654,851, current production)

Methodology provenance (same as the prior theta_S x M diagnostic, and per explicit
instruction to reuse -- not redefine -- the established consensus_decile_reversal
methodology): pairwise-reversal / margin-decile / Spearman / Pearson / top-q-overlap
machinery, NUM_PAIRS=5000/seed=42, NUM_DECILES=10, 8 held-out diagnostic seeds, and the tie
policy (consensus ties excluded from the reversal denominator; estimator ties are NOT a
reversal) are the SAME functions used in
scripts/lcg_diagnostic/theta_s_M_sweep_consensus_diagnostic.py, which itself recovered them
verbatim from the deleted scripts/forward_JVP/simple_mc/consensus_decile_reversal_diagnostic.py
/ consensus_margin_reversal_analysis.py and scripts/backward_VJP/3-stratum/
diagnose_lcg_backward_variance_setup.py (commit 3a834d3 deletion, recovered via
`git show 3a834d3~1:<path>`). Consensus construction uses the same adaptation as that prior
script (independent M=100 bank on a seed disjoint from the 8 diagnostic seeds), for the same
reason: each diagnostic seed's master bank here is only max(M_LIST) samples, leaving no room
for a same-bank held-out split.

Usage:
    python scripts/lcg_diagnostic/resblock_ablation_diagnostic.py
"""
import itertools
import json
import sys
import time
from pathlib import Path
from typing import Dict, List, Tuple


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch
from torch import Tensor

from data import Dataset
from lcg.forward_jvp import JVPBank, make_jvp_bank, score_one_jvp_bank
from lcg.precision import historical_precision
from lcg.reward_normalization import RunningRMS, RunningRMSConfig
from lcg.theta_s import ThetaSConfig, frozen_named_parameters, selected_named_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

DOCS_CKPT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "checkpoint"
DENOISER_PATH = DOCS_CKPT_DIR / "lcg_diag_denoiser_converged.pt"
CANDIDATES_PATH = DOCS_CKPT_DIR / "frozen_candidates_480.pt"
TRAIN_DATASET_DIR = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad\lcg_diag_train_dataset"
)

import os
OUT_DIR = Path(os.environ["RESBLOCK_OUT_DIR"]) if os.environ.get("RESBLOCK_OUT_DIR") else (
    _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "resblock_ablation"
)
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_DIR / "results.json"
M_SWEEP_CSV = OUT_DIR / "M_sweep_summary.csv"
CROSS_THETA_CSV = OUT_DIR / "cross_theta_summary.csv"
PARETO_CSV = OUT_DIR / "pareto_summary.csv"

NUM_STEPS_CONDITIONING = 4
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

B_HIST = 320
M_H = 3
DAMPING = 1e-4
BETA = 1.0
HIST_SEED = 12345
CHUNK_SIZE = 16
PRODUCTION_M_R = 12

_FAST = os.environ.get("LCG_DIAG_FAST") == "1"  # correctness smoke-test mode only, NOT for reported results

NUM_DIAG_SEEDS = 3 if _FAST else 8
M_LIST = [1, 2, 3] if _FAST else [1, 2, 3, 4, 5, 6, 8, 10, 12, 16]
MASTER_BANK_SIZE = max(M_LIST)
CONSENSUS_NUM_SAMPLES = 6 if _FAST else 100
CONSENSUS_SEED_BASE = 91000  # distinct namespace from the prior theta_S x M diagnostic's 90000+

NUM_PAIRS = 200 if _FAST else 5000
PAIR_SEED = 42
NUM_DECILES = 10

PROFILE_WARMUP = 1
PROFILE_REPS = 3

RESBLOCK_PREFIX = "unet.u_blocks.3.resblocks.{}."
THETA_S_CONFIGS = {
    "R1": ThetaSConfig(include=(RESBLOCK_PREFIX.format(0) + "*",)),
    "R2": ThetaSConfig(include=(RESBLOCK_PREFIX.format(1) + "*",)),
    "R3": ThetaSConfig(include=(RESBLOCK_PREFIX.format(2) + "*",)),
    "R1+R2": ThetaSConfig(include=(RESBLOCK_PREFIX.format(0) + "*", RESBLOCK_PREFIX.format(1) + "*")),
    "R1+R3": ThetaSConfig(include=(RESBLOCK_PREFIX.format(0) + "*", RESBLOCK_PREFIX.format(2) + "*")),
    "R2+R3": ThetaSConfig(include=(RESBLOCK_PREFIX.format(1) + "*", RESBLOCK_PREFIX.format(2) + "*")),
    "3R": ThetaSConfig(include=tuple(RESBLOCK_PREFIX.format(i) + "*" for i in range(3))),
    "full": ThetaSConfig(include=("unet.u_blocks.3.*", "norm_out.*", "conv_out.*")),
}
THETA_ORDER = ["R1", "R2", "R3", "R1+R2", "R1+R3", "R2+R3", "3R", "full"]
# Fixed, immutable per-name consensus-seed offset -- NOT derived from THETA_ORDER.index(name).
# R1/R2/R3/R2+R3/3R/full's offsets (0-5) match their position in the ORIGINAL 6-config
# THETA_ORDER (["R1","R2","R3","R2+R3","3R","full"]) used by the first full run of this
# script, before R1+R2/R1+R3 were inserted into the middle of the list. Using list-index
# directly would have silently shifted R2+R3/3R/full onto different seeds than they were
# originally computed with the moment R1+R2/R1+R3 were added -- this mapping is what makes
# the consensus-vector backfill (see RESBLOCK_BACKFILL_CONSENSUS) bit-identical instead of a
# quietly different re-estimate. New configs always get new, never-reused offsets.
CONSENSUS_SEED_OFFSET = {"R1": 0, "R2": 1, "R3": 2, "R2+R3": 3, "3R": 4, "full": 5, "R1+R2": 6, "R1+R3": 7}
EXPECTED_D_S = {"R1": 217664, "R2": 217664, "R3": 217664, "R1+R2": 435328, "R1+R3": 435328,
                 "R2+R3": 435328, "3R": 652992, "full": 654851}
EXPECTED_N_TENSORS = {"R1": 10, "R2": 10, "R3": 10, "R1+R2": 20, "R1+R3": 20, "R2+R3": 20, "3R": 30, "full": 34}
ANCHOR_THETA = "full"

# Which configs THIS invocation should compute -- lets a follow-up run add only the newly
# introduced configs (e.g. R1+R2, R1+R3) without recomputing configs an earlier run of this
# same script already produced (their raw per-seed vectors/consensus/profiling are reused
# unchanged from results.json, not rerun). Comma-separated theta_S names, or "all" (default).
_ONLY = os.environ.get("RESBLOCK_ONLY", "all")
RUN_THETA_ORDER = THETA_ORDER if _ONLY == "all" else [n.strip() for n in _ONLY.split(",")]
_RENDER_ONLY = os.environ.get("RESBLOCK_RENDER_ONLY") == "1"  # skip all computation, just reformat results.json
_BACKFILL_CONSENSUS = os.environ.get("RESBLOCK_BACKFILL_CONSENSUS") == "1"  # add missing consensus_vector only


# ------------------------------------------------------------------------------------------
# Recovered verbatim (same as scripts/lcg_diagnostic/theta_s_M_sweep_consensus_diagnostic.py)
# ------------------------------------------------------------------------------------------
def pearson_corr(a: Tensor, b: Tensor) -> float:
    a, b = a.double(), b.double()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-30)).item()


def _rank(x: Tensor) -> Tensor:
    order = torch.argsort(x)
    ranks = torch.empty_like(order, dtype=torch.float64)
    ranks[order] = torch.arange(x.numel(), dtype=torch.float64, device=x.device)
    return ranks


def spearman_corr(a: Tensor, b: Tensor) -> float:
    return pearson_corr(_rank(a), _rank(b))


def top_q_overlap(a: Tensor, b: Tensor, q: float) -> float:
    n = a.numel()
    k = max(1, int(round(q * n)))
    top_a = set(torch.topk(a, k).indices.tolist())
    top_b = set(torch.topk(b, k).indices.tolist())
    return len(top_a & top_b) / len(top_a)


def cv(x: Tensor) -> float:
    mean = x.mean()
    std = x.std(unbiased=True)
    return (std / (mean.abs() + 1e-30)).item()


def sample_pairs(num_candidates: int, num_pairs: int, seed: int) -> List[Tuple[int, int]]:
    rng = np.random.default_rng(seed)
    all_pairs = list(itertools.combinations(range(num_candidates), 2))
    pair_idx = rng.choice(len(all_pairs), size=num_pairs, replace=False)
    return [all_pairs[i] for i in pair_idx]


def decile_assignment(consensus: Tensor, idx_i: Tensor, idx_j: Tensor, num_deciles: int = NUM_DECILES):
    raw_diff = (consensus[idx_i] - consensus[idx_j]).detach().cpu().numpy()
    delta = np.abs(raw_diff)
    consensus_sign = np.sign(raw_diff)
    order = np.argsort(delta)
    n = len(idx_i)
    assert n % num_deciles == 0
    per_decile = n // num_deciles
    decile_of = np.empty(n, dtype=int)
    ranges = []
    for d in range(num_deciles):
        idxs = order[d * per_decile:(d + 1) * per_decile]
        decile_of[idxs] = d
        ranges.append((float(delta[idxs].min()), float(delta[idxs].max())))
    valid_mask = consensus_sign != 0
    return decile_of, ranges, consensus_sign, valid_mask


def reversal_rate_by_decile(estimate_vectors: Tensor, consensus_sign, idx_i, idx_j, decile_of, valid_mask,
                             num_deciles: int = NUM_DECILES):
    reversed_mask = _reversed_mask(estimate_vectors, consensus_sign, idx_i, idx_j)
    overall = float(reversed_mask[:, valid_mask].mean())
    per_decile = {}
    for d in range(num_deciles):
        mask = (decile_of == d) & valid_mask
        per_decile[d] = float(reversed_mask[:, mask].mean()) if mask.sum() > 0 else float("nan")
    return overall, per_decile


def _reversed_mask(estimate_vectors: Tensor, consensus_sign, idx_i, idx_j) -> np.ndarray:
    diffs = (estimate_vectors[:, idx_i] - estimate_vectors[:, idx_j]).detach().cpu().numpy()
    seed_sign = np.sign(diffs)
    return (seed_sign != consensus_sign[None, :]) & (seed_sign != 0)


def per_seed_overall_reversal(estimate_vectors: Tensor, consensus_sign, idx_i, idx_j, valid_mask) -> np.ndarray:
    """Per-seed (not pooled) overall reversal rate -- shape (num_seeds,). Used for the
    reference format's Part 6 (per-seed mean/std/min/max), same tie policy as
    reversal_rate_by_decile, just not averaged across seeds before reporting spread."""
    reversed_mask = _reversed_mask(estimate_vectors, consensus_sign, idx_i, idx_j)
    return reversed_mask[:, valid_mask].mean(axis=1)


def percentiles(x: np.ndarray) -> Dict[str, float]:
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    names = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"]
    out = {name: float(np.percentile(x, q)) for name, q in zip(names, qs)}
    out["mean"] = float(x.mean())
    out["std"] = float(x.std(ddof=1))
    out["cv"] = float(x.std(ddof=1) / (abs(x.mean()) + 1e-30))
    return out


# ------------------------------------------------------------------------------------------
def load_denoiser(device: torch.device) -> Denoiser:
    ckpt = torch.load(DENOISER_PATH, map_location=device, weights_only=False)
    action_dim = ckpt["action_dim"]
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    print(f"loaded checkpoint (total_step={ckpt['step']}, action_dim={action_dim}) from {DENOISER_PATH}", flush=True)
    return denoiser


def load_candidates(device: torch.device):
    payload = torch.load(CANDIDATES_PATH, map_location=device, weights_only=True)
    return [(c[0].to(device), c[1].to(device), c[2].to(device)) for c in payload]


def load_dataset() -> Dataset:
    dataset = Dataset(TRAIN_DATASET_DIR, "resblock_ablation_ds", cache_in_ram=True)
    dataset.load_from_default_path()
    return dataset


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def verify_structure(denoiser: Denoiser) -> None:
    print("\n" + "=" * 88)
    print("STRUCTURE VERIFICATION (denoiser.inner_model.unet.u_blocks[-1])")
    print("=" * 88)
    final_stage = denoiser.inner_model.unet.u_blocks[-1]
    resblocks = final_stage.resblocks
    print(f"  final_stage type={type(final_stage).__name__}  children={[n for n, _ in final_stage.named_children()]}")
    print(f"  num_resblocks={len(resblocks)}")
    for i, rb in enumerate(resblocks):
        named = list(rb.named_parameters())
        d_S = sum(p.numel() for _, p in named)
        print(f"    ResBlock {i} (prefix unet.u_blocks.3.resblocks.{i}.): n_tensors={len(named)} d_S={d_S}")
    all_names = set(n for n, _ in final_stage.named_parameters())
    resblock_names = set()
    for i in range(len(resblocks)):
        for name, _ in resblocks[i].named_parameters():
            resblock_names.add(f"resblocks.{i}.{name}")
    outside = all_names - resblock_names
    print(f"  params outside resblocks.* in final stage: {sorted(outside)}")
    assert len(resblocks) == 3, f"expected 3 ResBlocks, got {len(resblocks)}"
    assert not outside, f"unexpected params outside resblocks.*: {outside}"


def profile_scoring(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, master_bank, candidates, device):
    results = {}
    num_candidates = len(candidates)
    for M in M_LIST:
        sliced = JVPBank(master_bank.sigmas[:M], master_bank.epsilons[:M], master_bank.epsilons_offset[:M], master_bank.etas[:M])
        for _ in range(PROFILE_WARMUP):
            _ = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, sliced, candidates, CHUNK_SIZE)
        sync(device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        times_s = []
        for _ in range(PROFILE_REPS):
            if device.type == "cuda":
                start_evt, end_evt = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
                start_evt.record()
                _ = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, sliced, candidates, CHUNK_SIZE)
                end_evt.record()
                torch.cuda.synchronize()
                times_s.append(start_evt.elapsed_time(end_evt) / 1000.0)
            else:
                t0 = time.time()
                _ = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, sliced, candidates, CHUNK_SIZE)
                times_s.append(time.time() - t0)
        median_s = float(np.median(times_s))
        peak_alloc = float(torch.cuda.max_memory_allocated(device)) / 1e6 if device.type == "cuda" else float("nan")
        peak_reserved = float(torch.cuda.max_memory_reserved(device)) / 1e6 if device.type == "cuda" else float("nan")
        results[M] = dict(
            total_scoring_seconds_median=median_s, ms_per_candidate=median_s / num_candidates * 1000.0,
            ms_per_candidate_per_probe=median_s / num_candidates / M * 1000.0,
            peak_alloc_mb=peak_alloc, peak_reserved_mb=peak_reserved, reps_s=times_s,
        )
        print(f"    profile M={M:>2}: median={median_s:.3f}s  ms/cand={results[M]['ms_per_candidate']:.3f}  "
              f"ms/(cand*probe)={results[M]['ms_per_candidate_per_probe']:.4f}  "
              f"peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB", flush=True)
    return results


def rms_distribution(raw_scores: np.ndarray):
    rms = RunningRMS(RunningRMSConfig())
    raw_t = torch.from_numpy(raw_scores)
    normalized_t = rms(raw_t)
    return dict(
        raw=percentiles(raw_scores), normalized=percentiles(normalized_t.numpy()),
        rms_s2=float(rms.s2.item()), rms_scale=float((rms.cfg.alpha / (rms.s2.sqrt() + rms.cfg.eps)).item()),
    )


def _k(d, key):
    """Dict lookup tolerant of JSON round-trips (int keys become strings)."""
    return d[key] if key in d else d[str(key)]


def print_theta_report(theta_name: str, res: dict) -> None:
    """Reference-style per-theta_S report, matching the structure of
    docs/lcg_diagnostic/forward_JVP/simple_mc_vs_3-stratum/
    consensus_decile_reversal_M5_8_12_16_20_24.txt (Parts 1/2/3-4-5/6/7/8 + Required Final
    Answers), populated with THIS diagnostic's own M_LIST/config set -- same underlying
    reversal/Spearman/Pearson/top-q machinery, only the presentation is restructured.
    Works identically whether res came from a fresh computation this run or was loaded
    from results.json (a prior run's or a render-only pass)."""
    m_sweep = res["m_sweep"]
    Ms = [row["M"] for row in m_sweep]
    by_M = {row["M"]: row for row in m_sweep}
    n_pairs_per_decile = NUM_PAIRS // NUM_DECILES

    print("\n" + "#" * 88)
    print(f"DETAILED REPORT: theta_S = {theta_name}  d_S={res['d_S']}  n_tensors={res['n_tensors']}  "
          f"include={res['include']}")
    print("#" * 88)
    print(f"  h_D construction (B={B_HIST}, M_h={M_H}): {res['h_D_construction_seconds']:.2f}s  "
          f"peak_alloc={res['h_D_peak_alloc_mb']:.1f}MB  peak_reserved={res['h_D_peak_reserved_mb']:.1f}MB")

    print("\n" + "=" * 88)
    print("PART 1 -- held-out consensus construction (independent bank, seed disjoint from the diagnostic seeds)")
    print("=" * 88)
    craw = res["consensus_rms"]["raw"]
    print(f"  consensus: min={craw['min']:.4f}  max={craw['max']:.4f}  mean={craw['mean']:.4f}  "
          f"cv={res['consensus_cv']:.4f}  (M={CONSENSUS_NUM_SAMPLES} independent probes, "
          f"finite=True nonnegative=True)")

    print("\n" + "=" * 88)
    print(f"PART 2 -- {NUM_PAIRS} candidate pairs (seed={PAIR_SEED}), consensus margin deciles")
    print("=" * 88)
    for d, (lo, hi) in enumerate(res["decile_ranges"]):
        print(f"  decile {d}: delta range=[{lo:.4g}, {hi:.4g}]  n_pairs={n_pairs_per_decile}")

    print("\n" + "=" * 88)
    print("PART 3/4/5 -- reversal probability by consensus-margin decile")
    print("=" * 88)
    print("  tie policy: exact consensus ties excluded (none expected); estimator-side ties are NOT a reversal.")
    header = f"{'decile':>6} {'n_pairs':>8}" + "".join(f" {'M=' + str(M):>8}" for M in Ms)
    print("\n" + header)
    for d in range(NUM_DECILES):
        row_str = f"{d:>6} {n_pairs_per_decile:>8}"
        for M in Ms:
            row_str += f" {by_M[M][f'decile{d}_reversal']:>8.4f}"
        print(row_str)
    print("\n  overall reversal rate P_rev(M) (all pairs x diagnostic seeds):")
    for M in Ms:
        print(f"    M={M:<3} P_rev={by_M[M]['overall_reversal']:.4f}")

    print("\n" + "=" * 88)
    print("PART 6 -- per-seed reversal-rate variability (mean +/- std across diagnostic seeds)")
    print("=" * 88)
    has_per_seed = all("per_seed_overall_reversal" in by_M[M] for M in Ms)
    if has_per_seed:
        print(f"\n{'M':>3} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}")
        for M in Ms:
            arr = np.array(by_M[M]["per_seed_overall_reversal"])
            print(f"{M:>3} {arr.mean():>8.4f} {arr.std():>8.4f} {arr.min():>8.4f} {arr.max():>8.4f}")
    else:
        print("  (not available for this config -- computed before per-seed reversal tracking was added to "
              "this script; see the pooled overall_reversal in Part 3/4/5 instead)")

    print("\n" + "=" * 88)
    print("PART 7 -- focus on clearly-separated pairs: deciles 7, 8, 9")
    print("=" * 88)
    print(f"\n{'M':>3} {'decile7':>9} {'decile8':>9} {'decile9':>9} {'overall':>9}")
    for M in Ms:
        r = by_M[M]
        print(f"{M:>3} {r['decile7_reversal']:>9.4f} {r['decile8_reversal']:>9.4f} {r['decile9_reversal']:>9.4f} "
              f"{r['overall_reversal']:>9.4f}")

    print("\n" + "=" * 88)
    print("PART 8 -- production scoring runtime/memory association (measured, not a diagnostic-only harness)")
    print("=" * 88)
    print(f"\n{'M':>3} {'runtime_s':>10} {'ms/cand':>10} {'peak_alloc_MB':>14} {'peak_reserv_MB':>15} "
          f"{'decile9_rev':>12} {'overall_rev':>12}")
    for M in Ms:
        prof = _k(res["profiling"], M)
        r = by_M[M]
        print(f"{M:>3} {prof['total_scoring_seconds_median']:>10.2f} {prof['ms_per_candidate']:>10.3f} "
              f"{prof['peak_alloc_mb']:>14.1f} {prof['peak_reserved_mb']:>15.1f} "
              f"{r['decile9_reversal']:>12.4f} {r['overall_reversal']:>12.4f}")
    print(f"\n  NOTE: measured via the actual production score_one_jvp_bank path (chunk_size={CHUNK_SIZE}), "
          f"CUDA events with synchronization, 1 warmup + {PROFILE_REPS} timed reps, median reported -- "
          f"not the diagnostic-only chunk_size=4 harness used in earlier stages of this project.")

    print("\n  supplementary rank-overlap metrics (kept alongside decile reversal, not a replacement for it):")
    print(f"\n{'M':>3} {'Spearman':>17} {'Pearson':>9} {'top10':>9} {'top20':>9}")
    for M in Ms:
        r = by_M[M]
        print(f"{M:>3} {r['spearman_mean']:>8.4f}+-{r['spearman_std']:>6.4f} {r['pearson_mean']:>9.4f} "
              f"{r['top10_mean']:>9.4f} {r['top20_mean']:>9.4f}")

    print("\n  raw reward distribution / RMS-normalized reward (production RunningRMS, single-call semantics):")
    print(f"\n{'M':>3} {'raw_mean':>10} {'raw_cv':>8} {'rms_min':>9} {'rms_median':>10} {'rms_max':>9}")
    for M in Ms:
        rd = _k(res["rms_by_M"], M)
        print(f"{M:>3} {rd['raw']['mean']:>10.4f} {rd['raw']['cv']:>8.4f} {rd['normalized']['min']:>9.4f} "
              f"{rd['normalized']['median']:>10.4f} {rd['normalized']['max']:>9.4f}")

    print("\n" + "=" * 88)
    print("REQUIRED FINAL ANSWERS")
    print("=" * 88)
    print("  1. Reversal probability vs M: see overall P_rev(M) table above (Part 3/4/5) -- should be "
          "monotonically decreasing as M grows, at every decile.")
    d0_last, d9_last = by_M[Ms[-1]]["decile0_reversal"], by_M[Ms[-1]]["decile9_reversal"]
    print(f"  2. Errors concentrated in low-margin pairs? decile0 at M={Ms[-1]}={d0_last:.4f} vs "
          f"decile9 at M={Ms[-1]}={d9_last:.4f}")
    for M in Ms:
        r = by_M[M]
        print(f"  M={M}: decile7/8/9={r['decile7_reversal']:.4f}/{r['decile8_reversal']:.4f}/{r['decile9_reversal']:.4f}  "
              f"overall={r['overall_reversal']:.4f}")
    smallest_reliable = next((M for M in Ms if by_M[M]["decile9_reversal"] < 0.02), None)
    print(f"  Smallest M with decile-9 reversal < 2%: {smallest_reliable if smallest_reliable is not None else 'none in tested range'}")
    print(f"  d_S={res['d_S']}  h_D_construction={res['h_D_construction_seconds']:.2f}s  "
          f"consensus_cv={res['consensus_cv']:.4f}  "
          f"RMS_normalized_range_at_max_M=[{_k(res['rms_by_M'], Ms[-1])['normalized']['min']:.4f}, "
          f"{_k(res['rms_by_M'], Ms[-1])['normalized']['max']:.4f}]")


def to_jsonable(obj):
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, (np.floating, np.integer)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, torch.Tensor):
        return obj.detach().cpu().tolist()
    return obj


def _load_existing_results():
    if not RESULTS_JSON.is_file():
        return {}, {}
    with open(RESULTS_JSON) as f:
        payload = json.load(f)
    all_results = payload.get("per_theta", {})
    consensus_by_theta = {}
    for name, res in all_results.items():
        if "consensus_vector" in res:
            consensus_by_theta[name] = torch.tensor(res["consensus_vector"])
    return all_results, consensus_by_theta


def main():
    num_candidates = 480  # fixed frozen candidate set size; used below even in render-only mode
    idx_i_t = torch.tensor([p[0] for p in sample_pairs(num_candidates, NUM_PAIRS, PAIR_SEED)])
    idx_j_t = torch.tensor([p[1] for p in sample_pairs(num_candidates, NUM_PAIRS, PAIR_SEED)])
    print(f"sampled {NUM_PAIRS} candidate pairs (seed={PAIR_SEED}) for margin-decile analysis", flush=True)

    if _RENDER_ONLY:
        print(f"RESBLOCK_RENDER_ONLY=1: skipping all GPU computation, reformatting {RESULTS_JSON} only", flush=True)
        all_results, consensus_by_theta = _load_existing_results()
        assert all_results, f"no existing results found at {RESULTS_JSON} to render"
        _finish(all_results, consensus_by_theta, idx_i_t, idx_j_t)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser = load_denoiser(device)
    verify_structure(denoiser)
    candidates = load_candidates(device)
    dataset = load_dataset()
    print(f"\ndataset: num_episodes={dataset.num_episodes} num_steps={dataset.num_steps}  candidates: n={len(candidates)}", flush=True)
    assert dataset.num_steps >= B_HIST
    assert len(candidates) == num_candidates

    for name in THETA_ORDER:
        theta_cfg = THETA_S_CONFIGS[name]
        named = selected_named_parameters(denoiser, theta_cfg)
        d_S = sum(p.numel() for p in named.values())
        assert len(named) == EXPECTED_N_TENSORS[name], f"{name}: expected {EXPECTED_N_TENSORS[name]} tensors, got {len(named)}"
        assert d_S == EXPECTED_D_S[name], f"{name}: expected d_S={EXPECTED_D_S[name]}, got {d_S}"
    print(f"\nverified all {len(THETA_ORDER)} theta_S configs match expected n_tensors/d_S: {EXPECTED_D_S}", flush=True)

    all_results, consensus_by_theta = _load_existing_results()
    if all_results:
        print(f"loaded {len(all_results)} previously-computed theta_S config(s) from {RESULTS_JSON}: "
              f"{sorted(all_results.keys())} -- will NOT be recomputed", flush=True)
    # default ("all"): only compute configs not already present, so a plain rerun never
    # redundantly redoes hours of finished work. An explicit RESBLOCK_ONLY=... always forces
    # recompute of exactly the named configs, even if already present (e.g. deliberate redo).
    if _ONLY == "all":
        run_order = [n for n in THETA_ORDER if n not in all_results]
    else:
        run_order = [n for n in THETA_ORDER if n in RUN_THETA_ORDER]
    print(f"this invocation will compute: {run_order}", flush=True)

    if _BACKFILL_CONSENSUS:
        to_backfill = [n for n in THETA_ORDER if n in all_results and "consensus_vector" not in all_results[n]]
        print(f"RESBLOCK_BACKFILL_CONSENSUS=1: backfilling consensus_vector for {to_backfill} "
              f"(h_D + independent consensus bank only, reusing the SAME seeds -- bit-identical "
              f"reproduction, not a new estimate; the existing M-sweep/profiling/rms for these "
              f"configs are left untouched)", flush=True)
        for theta_name in to_backfill:
            theta_idx = CONSENSUS_SEED_OFFSET[theta_name]
            theta_cfg = THETA_S_CONFIGS[theta_name]
            theta_s_named = selected_named_parameters(denoiser, theta_cfg)
            frozen_named = frozen_named_parameters(denoiser, theta_s_named)
            d_S = sum(p.numel() for p in theta_s_named.values())
            params = list(theta_s_named.values())
            h_D = historical_precision(
                denoiser, params, dataset, SIGMA_CFG, B=B_HIST, N=dataset.num_steps, num_mc=M_H,
                beta=BETA, damping=DAMPING, seed=HIST_SEED,
            )
            h_D_inv_sqrt = h_D.rsqrt()
            consensus_seed = CONSENSUS_SEED_BASE + theta_idx
            consensus_bank = make_jvp_bank(
                SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=CONSENSUS_NUM_SAMPLES, seed=consensus_seed
            )
            r_cons = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, consensus_bank, candidates, CHUNK_SIZE)
            recomputed_cv, stored_cv = cv(r_cons), all_results[theta_name]["consensus_cv"]
            assert abs(recomputed_cv - stored_cv) < 1e-4, (
                f"{theta_name}: recomputed consensus (cv={recomputed_cv:.6f}) does not match the "
                f"originally stored consensus_cv={stored_cv:.6f}) -- reproducibility check failed, "
                f"NOT saving this backfill."
            )
            all_results[theta_name]["consensus_vector"] = r_cons.detach().cpu().tolist()
            consensus_by_theta[theta_name] = r_cons.detach().clone()
            print(f"  backfilled {theta_name}: recomputed cv={recomputed_cv:.4f} (stored={stored_cv:.4f}, match OK)", flush=True)
            with open(RESULTS_JSON, "w") as f:
                json.dump(to_jsonable(dict(per_theta=all_results)), f, indent=2)

    for theta_name in run_order:
        theta_idx = CONSENSUS_SEED_OFFSET[theta_name]
        theta_cfg = THETA_S_CONFIGS[theta_name]
        print("\n" + "=" * 88)
        print(f"theta_S = {theta_name}  include={theta_cfg.include}")
        print("=" * 88)

        theta_s_named = selected_named_parameters(denoiser, theta_cfg)
        frozen_named = frozen_named_parameters(denoiser, theta_s_named)
        d_S = sum(p.numel() for p in theta_s_named.values())
        params = list(theta_s_named.values())

        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        sync(device)
        t0 = time.time()
        h_D = historical_precision(
            denoiser, params, dataset, SIGMA_CFG, B=B_HIST, N=dataset.num_steps, num_mc=M_H,
            beta=BETA, damping=DAMPING, seed=HIST_SEED,
        )
        sync(device)
        t_hD = time.time() - t0
        hD_peak_alloc = float(torch.cuda.max_memory_allocated(device)) / 1e6 if device.type == "cuda" else float("nan")
        hD_peak_reserved = float(torch.cuda.max_memory_reserved(device)) / 1e6 if device.type == "cuda" else float("nan")
        h_D_inv_sqrt = h_D.rsqrt()
        print(f"  h_D construction (B={B_HIST}, M_h={M_H}): {t_hD:.2f}s  peak_alloc={hD_peak_alloc:.1f}MB "
              f"peak_reserved={hD_peak_reserved:.1f}MB  h_D: finite={torch.isfinite(h_D).all().item()} "
              f"min={h_D.min().item():.4g} max={h_D.max().item():.4g}", flush=True)

        sync(device)
        t0 = time.time()
        master_banks = {
            seed: make_jvp_bank(SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=MASTER_BANK_SIZE, seed=seed)
            for seed in range(NUM_DIAG_SEEDS)
        }
        sync(device)
        t_bank = time.time() - t0
        print(f"  JVP/CRN master-bank construction (8 seeds x M={MASTER_BANK_SIZE}): {t_bank:.2f}s", flush=True)

        r_hat = {seed: {} for seed in range(NUM_DIAG_SEEDS)}
        for seed in range(NUM_DIAG_SEEDS):
            bank = master_banks[seed]
            for M in M_LIST:
                sliced = JVPBank(bank.sigmas[:M], bank.epsilons[:M], bank.epsilons_offset[:M], bank.etas[:M])
                r_hat[seed][M] = score_one_jvp_bank(
                    denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, sliced, candidates, CHUNK_SIZE
                )
        print(f"  nested M={M_LIST} sweep complete for {NUM_DIAG_SEEDS} diagnostic seeds", flush=True)

        consensus_seed = CONSENSUS_SEED_BASE + theta_idx
        consensus_bank = make_jvp_bank(
            SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=CONSENSUS_NUM_SAMPLES, seed=consensus_seed
        )
        r_cons = score_one_jvp_bank(
            denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, consensus_bank, candidates, CHUNK_SIZE
        )
        consensus_by_theta[theta_name] = r_cons.detach().clone()
        print(f"  independent consensus (seed={consensus_seed}, M={CONSENSUS_NUM_SAMPLES}): "
              f"min={r_cons.min().item():.4f} max={r_cons.max().item():.4f} mean={r_cons.mean().item():.4f}  "
              f"cv={cv(r_cons):.4f}", flush=True)

        decile_of, decile_ranges, consensus_sign, valid_mask = decile_assignment(r_cons, idx_i_t, idx_j_t)

        m_sweep_rows = []
        for M in M_LIST:
            vectors = torch.stack([r_hat[s][M] for s in range(NUM_DIAG_SEEDS)], dim=0)
            sp = np.array([spearman_corr(vectors[s], r_cons) for s in range(NUM_DIAG_SEEDS)])
            pe = np.array([pearson_corr(vectors[s], r_cons) for s in range(NUM_DIAG_SEEDS)])
            t10 = np.array([top_q_overlap(vectors[s], r_cons, 0.10) for s in range(NUM_DIAG_SEEDS)])
            t20 = np.array([top_q_overlap(vectors[s], r_cons, 0.20) for s in range(NUM_DIAG_SEEDS)])
            overall_rev, decile_rev = reversal_rate_by_decile(vectors, consensus_sign, idx_i_t, idx_j_t, decile_of, valid_mask)
            per_seed_rev = per_seed_overall_reversal(vectors, consensus_sign, idx_i_t, idx_j_t, valid_mask)
            row = dict(
                M=M, spearman_mean=sp.mean(), spearman_std=sp.std(), pearson_mean=pe.mean(), pearson_std=pe.std(),
                top10_mean=t10.mean(), top10_std=t10.std(), top20_mean=t20.mean(), top20_std=t20.std(),
                overall_reversal=overall_rev, per_seed_overall_reversal=per_seed_rev.tolist(),
                **{f"decile{d}_reversal": decile_rev[d] for d in range(NUM_DECILES)},
            )
            m_sweep_rows.append(row)
            print(f"    M={M:>2}: Spearman={sp.mean():.4f}+-{sp.std():.4f}  Pearson={pe.mean():.4f}  "
                  f"top10={t10.mean():.4f}  top20={t20.mean():.4f}  overall_rev={overall_rev:.4f}  "
                  f"D7={decile_rev[7]:.4f} D8={decile_rev[8]:.4f} D9={decile_rev[9]:.4f}", flush=True)

        print("  profiling (production score_one_jvp_bank path, chunk_size=16, CUDA events, 1 warmup + 3 reps):", flush=True)
        profile_results = profile_scoring(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, master_banks[0], candidates, device)

        rms_rows = {}
        for M in M_LIST:
            r_bar_M = torch.stack([r_hat[s][M] for s in range(NUM_DIAG_SEEDS)], dim=0).mean(dim=0).cpu().numpy()
            rms_rows[M] = rms_distribution(r_bar_M)

        consensus_rms = rms_distribution(r_cons.cpu().numpy())

        all_results[theta_name] = dict(
            include=list(theta_cfg.include), n_tensors=len(theta_s_named), d_S=d_S,
            h_D_construction_seconds=t_hD, h_D_peak_alloc_mb=hD_peak_alloc, h_D_peak_reserved_mb=hD_peak_reserved,
            bank_construction_seconds=t_bank, decile_ranges=decile_ranges,
            m_sweep=m_sweep_rows, profiling=profile_results, rms_by_M=rms_rows,
            consensus_rms=consensus_rms, consensus_cv=cv(r_cons),
            consensus_vector=r_cons.detach().cpu().tolist(),
            r_hat_mean_over_seeds={M: torch.stack([r_hat[s][M] for s in range(NUM_DIAG_SEEDS)], dim=0).mean(dim=0).cpu().tolist()
                                    for M in M_LIST},
        )
        # incremental save after every theta_S -- never lose completed configs to an
        # interruption again (this was the exact gap that made an earlier mid-run stop risky).
        # Raw per-theta data only (no cross-theta/pareto recompute here -- those need the
        # full config set and are printed/saved once, at the very end, by _finish()).
        with open(RESULTS_JSON, "w") as f:
            json.dump(to_jsonable(dict(per_theta=all_results)), f, indent=2)
        print(f"  [incremental save: {theta_name} written to {RESULTS_JSON}]", flush=True)

    _finish(all_results, consensus_by_theta, idx_i_t, idx_j_t)


def _finish(all_results, consensus_by_theta, idx_i_t, idx_j_t):
    # Normalize device: configs loaded from results.json come back as CPU tensors, while
    # ones computed live in this process (fresh or just-backfilled) are on whatever `device`
    # was -- mixing the two in a pairwise comparison crashes. These are tiny (480-length)
    # vectors, so CPU is fine for all of the (cheap) cross-theta post-processing below.
    consensus_by_theta = {k: v.cpu() for k, v in consensus_by_theta.items()}
    idx_i_t, idx_j_t = idx_i_t.cpu(), idx_j_t.cpu()

    available = [n for n in THETA_ORDER if n in all_results]
    print(f"\n(reporting on {len(available)}/{len(THETA_ORDER)} theta_S configs available: {available})", flush=True)

    print("\n" + "#" * 88)
    print("DETAILED PER-THETA_S REPORTS (reference consensus_decile_reversal format)")
    print("#" * 88)
    for theta_name in available:
        print_theta_report(theta_name, all_results[theta_name])

    # ------------------------------------------------------------------------------------
    # Cross-theta_S: full pairwise matrices (consensus vs consensus)
    # ------------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("CROSS-THETA_S PAIRWISE MATRICES (consensus vs consensus)")
    print("=" * 88)
    pairwise_rows = []
    for a, b in itertools.combinations(available, 2):
        ca, cb = consensus_by_theta[a], consensus_by_theta[b]
        sp, pe = spearman_corr(ca, cb), pearson_corr(ca, cb)
        t10, t20 = top_q_overlap(ca, cb, 0.10), top_q_overlap(ca, cb, 0.20)
        _, _, sign_ab, valid_ab = decile_assignment(ca, idx_i_t, idx_j_t)
        overall_rev_ab, _ = reversal_rate_by_decile(cb.unsqueeze(0), sign_ab, idx_i_t, idx_j_t,
                                                      np.zeros(NUM_PAIRS, dtype=int), valid_ab, num_deciles=1)
        concordance = 1.0 - overall_rev_ab
        for metric, val in [("spearman", sp), ("pearson", pe), ("top10", t10), ("top20", t20), ("concordance", concordance)]:
            pairwise_rows.append(dict(kind="pairwise", a=a, b=b, decile="", metric=metric, value=val))
        print(f"  {a} vs {b}: Spearman={sp:.4f} Pearson={pe:.4f} top10={t10:.4f} top20={t20:.4f} concordance={concordance:.4f}", flush=True)

    anchored_reversal, anchored_rows = {}, []
    if ANCHOR_THETA in consensus_by_theta:
        print(f"\n  margin-decile reversal, anchored on '{ANCHOR_THETA}' consensus:")
        anchor_cons = consensus_by_theta[ANCHOR_THETA]
        anchor_decile_of, anchor_ranges, anchor_sign, anchor_valid = decile_assignment(anchor_cons, idx_i_t, idx_j_t)
        for theta_name in available:
            if theta_name == ANCHOR_THETA:
                continue
            overall_rev, decile_rev = reversal_rate_by_decile(
                consensus_by_theta[theta_name].unsqueeze(0), anchor_sign, idx_i_t, idx_j_t, anchor_decile_of, anchor_valid
            )
            anchored_reversal[theta_name] = dict(overall=overall_rev, by_decile=decile_rev)
            for d in range(NUM_DECILES):
                anchored_rows.append(dict(kind="anchored_reversal", a=theta_name, b=ANCHOR_THETA, decile=d, metric="reversal", value=decile_rev[d]))
            anchored_rows.append(dict(kind="anchored_reversal", a=theta_name, b=ANCHOR_THETA, decile="overall", metric="reversal", value=overall_rev))
            print(f"    {theta_name} vs {ANCHOR_THETA}: overall={overall_rev:.4f}  "
                  f"D7={decile_rev[7]:.4f} D8={decile_rev[8]:.4f} D9={decile_rev[9]:.4f}  D0={decile_rev[0]:.4f}", flush=True)
    else:
        print(f"\n  ('{ANCHOR_THETA}' not yet available -- skipping full-anchored margin-decile reversal for now)")

    # ------------------------------------------------------------------------------------
    # Pareto summary (only meaningful once the anchor 'full' config is available)
    # ------------------------------------------------------------------------------------
    pareto_rows = []
    if ANCHOR_THETA in consensus_by_theta:
        print("\n" + "=" * 88)
        print("PARETO SUMMARY (theta_S x M)")
        print("=" * 88)
        full_cons = consensus_by_theta[ANCHOR_THETA]
        for theta_name in available:
            res = all_results[theta_name]
            for row in res["m_sweep"]:
                M = row["M"]
                prof = _k(res["profiling"], M)
                r_bar_M = torch.tensor(_k(res["r_hat_mean_over_seeds"], M), device=full_cons.device)
                sp_full = spearman_corr(r_bar_M, full_cons)
                top20_full = top_q_overlap(r_bar_M, full_cons, 0.20)
                overall_rev_full, decile_rev_full = reversal_rate_by_decile(
                    r_bar_M.unsqueeze(0), anchor_sign, idx_i_t, idx_j_t, anchor_decile_of, anchor_valid
                )
                rms_row = _k(res["rms_by_M"], M)
                rms_range = (rms_row["normalized"]["min"], rms_row["normalized"]["max"])
                raw_cv = rms_row["raw"]["cv"]
                pareto_rows.append(dict(
                    theta_s=theta_name, d_S=res["d_S"], M=M,
                    h_D_construction_s=res["h_D_construction_seconds"],
                    runtime_s=prof["total_scoring_seconds_median"], ms_per_candidate=prof["ms_per_candidate"],
                    peak_reserved_mb=prof["peak_reserved_mb"],
                    within_theta_spearman=row["spearman_mean"],
                    spearman_vs_full=sp_full, top20_vs_full=top20_full,
                    overall_reversal_vs_full=overall_rev_full,
                    d7_reversal_vs_full=decile_rev_full[7], d8_reversal_vs_full=decile_rev_full[8], d9_reversal_vs_full=decile_rev_full[9],
                    raw_reward_cv=raw_cv, rms_reward_min=rms_range[0], rms_reward_max=rms_range[1],
                ))
        for r in pareto_rows:
            print(f"  {r['theta_s']:<6} d_S={r['d_S']:>7} M={r['M']:>2}  runtime={r['runtime_s']:.2f}s  "
                  f"ms/cand={r['ms_per_candidate']:.3f}  within_sp={r['within_theta_spearman']:.4f}  "
                  f"sp_vs_full={r['spearman_vs_full']:.4f}  top20_vs_full={r['top20_vs_full']:.4f}  "
                  f"D7={r['d7_reversal_vs_full']:.4f} D8={r['d8_reversal_vs_full']:.4f} D9={r['d9_reversal_vs_full']:.4f}  "
                  f"raw_cv={r['raw_reward_cv']:.4f}", flush=True)

    # ------------------------------------------------------------------------------------
    # Save (idempotent -- called after every theta_S so nothing already computed can ever
    # be lost to an interruption again; always reflects the full 'available' set so far)
    # ------------------------------------------------------------------------------------
    payload = dict(
        config=dict(B_HIST=B_HIST, M_H=M_H, DAMPING=DAMPING, BETA=BETA, HIST_SEED=HIST_SEED, CHUNK_SIZE=CHUNK_SIZE,
                    PRODUCTION_M_R=PRODUCTION_M_R, NUM_DIAG_SEEDS=NUM_DIAG_SEEDS, M_LIST=M_LIST,
                    CONSENSUS_NUM_SAMPLES=CONSENSUS_NUM_SAMPLES, CONSENSUS_SEED_BASE=CONSENSUS_SEED_BASE,
                    NUM_PAIRS=NUM_PAIRS, PAIR_SEED=PAIR_SEED, NUM_DECILES=NUM_DECILES,
                    EXPECTED_D_S=EXPECTED_D_S, EXPECTED_N_TENSORS=EXPECTED_N_TENSORS),
        per_theta=all_results, cross_theta_pairwise=pairwise_rows, cross_theta_anchored_reversal=anchored_reversal,
        pareto_rows=pareto_rows,
    )
    with open(RESULTS_JSON, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)

    import csv
    with open(M_SWEEP_CSV, "w", newline="") as f:
        fieldnames = ["theta_s", "M", "spearman_mean", "spearman_std", "pearson_mean", "top10_mean", "top10_std",
                      "top20_mean", "top20_std", "overall_reversal"] + [f"decile{d}_reversal" for d in range(NUM_DECILES)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for theta_name in available:
            for row in all_results[theta_name]["m_sweep"]:
                writer.writerow({"theta_s": theta_name, **{k: row[k] for k in fieldnames if k != "theta_s"}})

    with open(CROSS_THETA_CSV, "w", newline="") as f:
        fieldnames = ["kind", "a", "b", "decile", "metric", "value"]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pairwise_rows)
        writer.writerows(anchored_rows)

    if pareto_rows:
        with open(PARETO_CSV, "w", newline="") as f:
            fieldnames = list(pareto_rows[0].keys())
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(pareto_rows)

    print(f"\nsaved results.json / M_sweep_summary.csv / cross_theta_summary.csv"
          f"{' / pareto_summary.csv' if pareto_rows else ''} to {OUT_DIR}", flush=True)
    print("\nDiagnostic complete." if not _RENDER_ONLY else "\nRender complete.")


if __name__ == "__main__":
    main()
