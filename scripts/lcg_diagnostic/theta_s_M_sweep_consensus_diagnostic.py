#! /usr/bin/env python
"""
theta_S x candidate-MC(M) sweep, held-out consensus-margin decile reversal diagnostic.

DIAGNOSTIC ONLY. Does not modify src/lcg/*.py or any production default. Does not change
candidate_num_mc (M_r=12) in config/intrinsic_reward/lcg.yaml. Does not start Random-vs-LCG
training. Uses the durable real checkpoint/candidate set under
docs/lcg_diagnostic/checkpoint/ (lcg_diag_denoiser_converged.pt, frozen_candidates_480.pt --
the same assets scripts/historical_precision_diagnostic/smoke_test.py uses) and the matched
N=1540 training dataset that converged checkpoint was trained on (still present under this
session's scratchpad at .../scratchpad/lcg_diag_train_dataset -- NOT re-archived under docs/,
which is a durability caveat called out in the final report, not silently ignored).

Methodology provenance (per explicit instruction: reuse the previous consensus_decile_reversal
definition exactly, do not silently redefine it): the pairwise-reversal / margin-decile /
Spearman / Pearson / top-q-overlap machinery below is recovered VERBATIM from
scripts/forward_JVP/simple_mc/consensus_decile_reversal_diagnostic.py and its sibling
consensus_margin_reversal_analysis.py, both deleted in commit 3a834d3 ("Remove obsolete LCG
diagnostic scripts") and recovered via `git show 3a834d3~1:<path>`; the correlation/overlap
primitives (pearson_corr, _rank, spearman_corr, top_q_overlap, cv) are recovered verbatim from
scripts/backward_VJP/3-stratum/diagnose_lcg_backward_variance_setup.py (same deletion commit).
NUM_DIAG_SEEDS=8, NUM_PAIRS=5000, PAIR_SEED=42, NUM_DECILES=10, and the tie-handling policy
(consensus ties excluded from the reversal denominator; estimator-side ties treated as NOT a
reversal) are copied unchanged from that recovered script.

One necessary, explicitly-flagged ADAPTATION of the consensus-construction *mechanism* (not
the reversal/correlation *metrics themselves*): the recovered script built its held-out
consensus by splitting each seed's own long (96-sample) draw sequence into an evaluation
range [0:24] and a held-out range [24:96] of the SAME per-seed stream. Here, per this task's
explicit spec, each of the 8 diagnostic seeds' master bank is only M=10 samples (nested
prefixes 1..10 ARE the thing being evaluated, leaving no room for a same-bank held-out split).
So the independent consensus reference is instead built from a SEPARATE, much-larger
(CONSENSUS_NUM_SAMPLES=100, i.e. 10x the max swept M) Full-CRN bank drawn from a seed far
outside the 8 diagnostic seeds' namespace (CONSENSUS_SEED_BASE=90000+), guaranteeing zero
probe sharing with any test bank. The reversal/Spearman/Pearson/top-k formulas applied against
this reference are byte-for-byte the recovered ones.

Usage:
    python scripts/lcg_diagnostic/theta_s_M_sweep_consensus_diagnostic.py
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

# ----------------------------------------------------------------------------------------
# Fixed assets / production settings (per instruction: do not change production math)
# ----------------------------------------------------------------------------------------
DOCS_CKPT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "checkpoint"
DENOISER_PATH = DOCS_CKPT_DIR / "lcg_diag_denoiser_converged.pt"
CANDIDATES_PATH = DOCS_CKPT_DIR / "frozen_candidates_480.pt"
# Ephemeral-scratchpad caveat (see module docstring): this is the dataset the converged
# checkpoint above was actually trained on; it is not archived under docs/lcg_diagnostic/.
TRAIN_DATASET_DIR = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad\lcg_diag_train_dataset"
)

OUT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "theta_s_M_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_JSON = OUT_DIR / "results.json"
M_SWEEP_CSV = OUT_DIR / "M_sweep_summary.csv"
PARETO_CSV = OUT_DIR / "pareto_summary.csv"

NUM_STEPS_CONDITIONING = 4
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

B_HIST = 320       # production precision_reference_size
M_H = 3            # production precision_num_mc
DAMPING = 1e-4     # production default
BETA = 1.0         # production default
HIST_SEED = 12345  # fixed, documented; same seed reused across theta_S for comparability
CHUNK_SIZE = 16    # production candidate_chunk_size
PRODUCTION_M_R = 12  # NOT changed; used only as a reference point in the final comparison

import os
_FAST = os.environ.get("LCG_DIAG_FAST") == "1"  # correctness smoke-test mode only, NOT for reported results

NUM_DIAG_SEEDS = 3 if _FAST else 8              # same held-out-seed count as the recovered diagnostic
M_LIST = list(range(1, 4)) if _FAST else list(range(1, 11))
CONSENSUS_NUM_SAMPLES = 6 if _FAST else 100     # 10x max M; independent bank, see module docstring
CONSENSUS_SEED_BASE = 90000     # far outside the 0..7 diagnostic-seed namespace

NUM_PAIRS = 200 if _FAST else 5000
PAIR_SEED = 42
NUM_DECILES = 10

PROFILE_WARMUP = 1
PROFILE_REPS = 3

THETA_S_CONFIGS = {
    "conv_out": ThetaSConfig(include=("conv_out.*",)),
    "norm_out_conv_out": ThetaSConfig(include=("norm_out.*", "conv_out.*")),
    "default": ThetaSConfig(include=("unet.u_blocks.3.*", "norm_out.*", "conv_out.*")),
}
EXPECTED_D_S = {"conv_out": 1731, "norm_out_conv_out": 1859, "default": 654851}
EXPECTED_N_TENSORS = {"conv_out": 2, "norm_out_conv_out": 4, "default": 34}
ANCHOR_THETA = "default"


# ------------------------------------------------------------------------------------------
# Recovered VERBATIM from diagnose_lcg_backward_variance_setup.py (deleted at 3a834d3,
# recovered via `git show 3a834d3~1:scripts/backward_VJP/3-stratum/
# diagnose_lcg_backward_variance_setup.py`). Do not alter these definitions.
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


# ------------------------------------------------------------------------------------------
# Recovered (structure verbatim) from consensus_decile_reversal_diagnostic.py (same deletion
# commit, recovered via `git show 3a834d3~1:scripts/forward_JVP/simple_mc/
# consensus_decile_reversal_diagnostic.py`): pair sampling, margin-decile construction, and
# the reversal tie-handling policy. Parameterized here (num_candidates, consensus tensor) so
# it can be reused per theta_S/per anchor without duplicating the logic.
# ------------------------------------------------------------------------------------------
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
    """estimate_vectors: (num_rows, num_candidates). Returns (overall_rate, {decile: rate})."""
    diffs = (estimate_vectors[:, idx_i] - estimate_vectors[:, idx_j]).detach().cpu().numpy()
    seed_sign = np.sign(diffs)
    reversed_mask = (seed_sign != consensus_sign[None, :]) & (seed_sign != 0)
    overall = float(reversed_mask[:, valid_mask].mean())
    per_decile = {}
    for d in range(num_deciles):
        mask = (decile_of == d) & valid_mask
        per_decile[d] = float(reversed_mask[:, mask].mean()) if mask.sum() > 0 else float("nan")
    return overall, per_decile


def percentiles(x: np.ndarray) -> Dict[str, float]:
    qs = [0, 1, 5, 25, 50, 75, 95, 99, 100]
    names = ["min", "p1", "p5", "p25", "median", "p75", "p95", "p99", "max"]
    out = {name: float(np.percentile(x, q)) for name, q in zip(names, qs)}
    out["mean"] = float(x.mean())
    out["std"] = float(x.std(ddof=1))
    return out


# ------------------------------------------------------------------------------------------
# Setup
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
    dataset = Dataset(TRAIN_DATASET_DIR, "theta_s_M_sweep_ds", cache_in_ram=True)
    dataset.load_from_default_path()
    return dataset


def sync(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize()


def profile_scoring(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, master_bank, candidates, device):
    """Per M: 1 warmup + PROFILE_REPS timed CUDA-event-synchronized reps; report median
    total scoring seconds, ms/candidate, ms/(candidate*probe), and peak allocated/reserved
    CUDA memory (max over the timed reps)."""
    results = {}
    num_candidates = len(candidates)
    for M in M_LIST:
        sliced = JVPBank(master_bank.sigmas[:M], master_bank.epsilons[:M], master_bank.epsilons_offset[:M], master_bank.etas[:M])

        # warmup (not timed)
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
            total_scoring_seconds_median=median_s,
            ms_per_candidate=median_s / num_candidates * 1000.0,
            ms_per_candidate_per_probe=median_s / num_candidates / M * 1000.0,
            peak_alloc_mb=peak_alloc,
            peak_reserved_mb=peak_reserved,
            reps_s=times_s,
        )
        print(f"    profile M={M:>2}: median={median_s:.3f}s  ms/cand={results[M]['ms_per_candidate']:.3f}  "
              f"ms/(cand*probe)={results[M]['ms_per_candidate_per_probe']:.4f}  "
              f"peak_alloc={peak_alloc:.1f}MB  peak_reserved={peak_reserved:.1f}MB", flush=True)
    return results


def rms_distribution(raw_scores: np.ndarray):
    rms = RunningRMS(RunningRMSConfig())  # production defaults: enabled=True, alpha=1.0, ema_decay=0.99, eps=1e-8
    raw_t = torch.from_numpy(raw_scores)
    normalized_t = rms(raw_t)
    return dict(
        raw=percentiles(raw_scores),
        normalized=percentiles(normalized_t.numpy()),
        rms_s2=float(rms.s2.item()),
        rms_scale=float((rms.cfg.alpha / (rms.s2.sqrt() + rms.cfg.eps)).item()),
    )


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


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser = load_denoiser(device)
    candidates = load_candidates(device)
    dataset = load_dataset()
    print(f"dataset: num_episodes={dataset.num_episodes} num_steps={dataset.num_steps}  "
          f"candidates: n={len(candidates)}", flush=True)
    assert dataset.num_steps >= B_HIST, f"dataset has {dataset.num_steps} transitions, need >= B_HIST={B_HIST}"

    idx_i_t = torch.tensor([p[0] for p in sample_pairs(len(candidates), NUM_PAIRS, PAIR_SEED)])
    idx_j_t = torch.tensor([p[1] for p in sample_pairs(len(candidates), NUM_PAIRS, PAIR_SEED)])
    print(f"sampled {NUM_PAIRS} candidate pairs (seed={PAIR_SEED}) for margin-decile analysis", flush=True)

    all_results = {}
    consensus_by_theta: Dict[str, Tensor] = {}
    hD_by_theta: Dict[str, Tensor] = {}

    theta_order = ["conv_out", "norm_out_conv_out", "default"]
    for theta_name in theta_order:
        theta_cfg = THETA_S_CONFIGS[theta_name]
        print("\n" + "=" * 88)
        print(f"theta_S = {theta_name}  include={theta_cfg.include}")
        print("=" * 88)

        theta_s_named = selected_named_parameters(denoiser, theta_cfg)
        frozen_named = frozen_named_parameters(denoiser, theta_s_named)
        d_S = sum(p.numel() for p in theta_s_named.values())
        assert len(theta_s_named) == EXPECTED_N_TENSORS[theta_name], (
            f"{theta_name}: expected {EXPECTED_N_TENSORS[theta_name]} tensors, got {len(theta_s_named)}"
        )
        assert d_S == EXPECTED_D_S[theta_name], f"{theta_name}: expected d_S={EXPECTED_D_S[theta_name]}, got {d_S}"
        print(f"  verified: n_tensors={len(theta_s_named)} d_S={d_S} (matches expected)", flush=True)

        params = list(theta_s_named.values())
        sync(device)
        t0 = time.time()
        h_D = historical_precision(
            denoiser, params, dataset, SIGMA_CFG, B=B_HIST, N=dataset.num_steps, num_mc=M_H,
            beta=BETA, damping=DAMPING, seed=HIST_SEED,
        )
        sync(device)
        t_hD = time.time() - t0
        h_D_inv_sqrt = h_D.rsqrt()
        hD_by_theta[theta_name] = h_D.detach().clone()
        print(f"  h_D construction (B={B_HIST}, M_h={M_H}): {t_hD:.2f}s  "
              f"h_D: finite={torch.isfinite(h_D).all().item()} min={h_D.min().item():.4g} max={h_D.max().item():.4g}",
              flush=True)

        sync(device)
        t0 = time.time()
        master_banks = {
            seed: make_jvp_bank(SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=10, seed=seed)
            for seed in range(NUM_DIAG_SEEDS)
        }
        sync(device)
        t_bank = time.time() - t0
        print(f"  JVP/CRN master-bank construction (8 seeds x M=10): {t_bank:.2f}s", flush=True)

        # ---- nested M=1..10 sweep, per diagnostic seed, via the UNMODIFIED production path
        r_hat = {seed: {} for seed in range(NUM_DIAG_SEEDS)}
        for seed in range(NUM_DIAG_SEEDS):
            bank = master_banks[seed]
            for M in M_LIST:
                sliced = JVPBank(bank.sigmas[:M], bank.epsilons[:M], bank.epsilons_offset[:M], bank.etas[:M])
                r_hat[seed][M] = score_one_jvp_bank(
                    denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, sliced, candidates, CHUNK_SIZE
                )
        print(f"  nested M=1..10 sweep complete for {NUM_DIAG_SEEDS} diagnostic seeds", flush=True)

        # ---- independent consensus (disjoint seed, much larger M)
        consensus_seed = CONSENSUS_SEED_BASE + theta_order.index(theta_name)
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

        # ---- per-M metrics vs own consensus (mean+-std across the 8 diagnostic seeds)
        m_sweep_rows = []
        for M in M_LIST:
            vectors = torch.stack([r_hat[s][M] for s in range(NUM_DIAG_SEEDS)], dim=0)  # (8,480)
            sp = np.array([spearman_corr(vectors[s], r_cons) for s in range(NUM_DIAG_SEEDS)])
            pe = np.array([pearson_corr(vectors[s], r_cons) for s in range(NUM_DIAG_SEEDS)])
            t10 = np.array([top_q_overlap(vectors[s], r_cons, 0.10) for s in range(NUM_DIAG_SEEDS)])
            t20 = np.array([top_q_overlap(vectors[s], r_cons, 0.20) for s in range(NUM_DIAG_SEEDS)])
            overall_rev, decile_rev = reversal_rate_by_decile(vectors, consensus_sign, idx_i_t, idx_j_t, decile_of, valid_mask)
            row = dict(
                M=M, spearman_mean=sp.mean(), spearman_std=sp.std(),
                pearson_mean=pe.mean(), pearson_std=pe.std(),
                top10_mean=t10.mean(), top10_std=t10.std(),
                top20_mean=t20.mean(), top20_std=t20.std(),
                overall_reversal=overall_rev,
                **{f"decile{d}_reversal": decile_rev[d] for d in range(NUM_DECILES)},
            )
            m_sweep_rows.append(row)
            print(f"    M={M:>2}: Spearman={sp.mean():.4f}+-{sp.std():.4f}  Pearson={pe.mean():.4f}  "
                  f"top10={t10.mean():.4f}+-{t10.std():.4f}  top20={t20.mean():.4f}+-{t20.std():.4f}  "
                  f"overall_rev={overall_rev:.4f}  decile9_rev={decile_rev[9]:.4f}", flush=True)

        # ---- profiling
        print("  profiling (production score_one_jvp_bank path, chunk_size=16, CUDA events, 1 warmup + 3 reps):", flush=True)
        profile_results = profile_scoring(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, master_banks[0], candidates, device)

        # ---- RMS-normalized reward distribution per M (mean-across-8-seeds r_hat_M)
        rms_rows = {}
        for M in M_LIST:
            r_bar_M = torch.stack([r_hat[s][M] for s in range(NUM_DIAG_SEEDS)], dim=0).mean(dim=0).cpu().numpy()
            rms_rows[M] = rms_distribution(r_bar_M)

        consensus_rms = rms_distribution(r_cons.cpu().numpy())

        all_results[theta_name] = dict(
            include=list(theta_cfg.include), n_tensors=len(theta_s_named), d_S=d_S,
            h_D_construction_seconds=t_hD, bank_construction_seconds=t_bank,
            decile_ranges=decile_ranges,
            m_sweep=m_sweep_rows, profiling=profile_results, rms_by_M=rms_rows,
            consensus_rms=consensus_rms, consensus_cv=cv(r_cons),
            r_hat_mean_over_seeds={M: torch.stack([r_hat[s][M] for s in range(NUM_DIAG_SEEDS)], dim=0).mean(dim=0).cpu().tolist()
                                    for M in M_LIST},
        )

    # ------------------------------------------------------------------------------------
    # Cross-theta_S comparison using HIGH-QUALITY CONSENSUS scores (isolates parameter-
    # subset effect from MC noise, per instruction)
    # ------------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("CROSS-THETA_S COMPARISON (consensus vs consensus)")
    print("=" * 88)
    pairwise = {"spearman": {}, "pearson": {}, "top10": {}, "top20": {}, "concordance": {}}
    for a, b in itertools.combinations(theta_order, 2):
        ca, cb = consensus_by_theta[a], consensus_by_theta[b]
        sp = spearman_corr(ca, cb)
        pe = pearson_corr(ca, cb)
        t10 = top_q_overlap(ca, cb, 0.10)
        t20 = top_q_overlap(ca, cb, 0.20)
        _, ranges_ab, sign_ab, valid_ab = decile_assignment(ca, idx_i_t, idx_j_t)
        overall_rev_ab, _ = reversal_rate_by_decile(cb.unsqueeze(0), sign_ab, idx_i_t, idx_j_t,
                                                      np.zeros(NUM_PAIRS, dtype=int), valid_ab, num_deciles=1)
        concordance = 1.0 - overall_rev_ab
        pairwise["spearman"][f"{a}_vs_{b}"] = sp
        pairwise["pearson"][f"{a}_vs_{b}"] = pe
        pairwise["top10"][f"{a}_vs_{b}"] = t10
        pairwise["top20"][f"{a}_vs_{b}"] = t20
        pairwise["concordance"][f"{a}_vs_{b}"] = concordance
        print(f"  {a} vs {b}: Spearman={sp:.4f}  Pearson={pe:.4f}  top10={t10:.4f}  top20={t20:.4f}  "
              f"concordance(1-rev)={concordance:.4f}  "
              f"[[note: literal Kendall-tau-b was not part of the recovered methodology; "
              f"'concordance' here is the same sign-agreement-over-5000-pairs mechanism as the "
              f"recovered decile-reversal diagnostic, reused unchanged, not a new metric]]", flush=True)

    # margin-decile reversal anchored on the DEFAULT theta_S consensus
    print(f"\n  margin-decile reversal, anchored on '{ANCHOR_THETA}' consensus:")
    anchor_cons = consensus_by_theta[ANCHOR_THETA]
    anchor_decile_of, anchor_ranges, anchor_sign, anchor_valid = decile_assignment(anchor_cons, idx_i_t, idx_j_t)
    anchored_reversal = {}
    for theta_name in theta_order:
        if theta_name == ANCHOR_THETA:
            continue
        overall_rev, decile_rev = reversal_rate_by_decile(
            consensus_by_theta[theta_name].unsqueeze(0), anchor_sign, idx_i_t, idx_j_t, anchor_decile_of, anchor_valid
        )
        anchored_reversal[theta_name] = dict(overall=overall_rev, by_decile=decile_rev)
        print(f"    {theta_name} vs {ANCHOR_THETA}: overall_reversal={overall_rev:.4f}  "
              f"decile9={decile_rev[9]:.4f}  decile0={decile_rev[0]:.4f}", flush=True)

    # ------------------------------------------------------------------------------------
    # Pareto summary: one row per (theta_S, M)
    # ------------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PARETO SUMMARY (theta_S x M)")
    print("=" * 88)
    pareto_rows = []
    default_cons = consensus_by_theta[ANCHOR_THETA]
    for theta_name in theta_order:
        res = all_results[theta_name]
        for row in res["m_sweep"]:
            M = row["M"]
            prof = res["profiling"][M]
            r_bar_M = torch.tensor(res["r_hat_mean_over_seeds"][M], device=default_cons.device)
            default_sp = spearman_corr(r_bar_M, default_cons)
            default_top20 = top_q_overlap(r_bar_M, default_cons, 0.20)
            rms_range = (res["rms_by_M"][M]["normalized"]["min"], res["rms_by_M"][M]["normalized"]["max"])
            pareto_rows.append(dict(
                theta_s=theta_name, d_S=res["d_S"], M=M,
                runtime_s=prof["total_scoring_seconds_median"], ms_per_candidate=prof["ms_per_candidate"],
                peak_reserved_mb=prof["peak_reserved_mb"],
                within_theta_spearman=row["spearman_mean"],
                default_theta_spearman=default_sp,
                top20_overlap_with_default=default_top20,
                high_margin_reversal_decile9=row["decile9_reversal"],
                rms_reward_min=rms_range[0], rms_reward_max=rms_range[1],
            ))
    for r in pareto_rows:
        print(f"  {r['theta_s']:<18} d_S={r['d_S']:>7} M={r['M']:>2}  runtime={r['runtime_s']:.2f}s  "
              f"ms/cand={r['ms_per_candidate']:.3f}  peak_reserved={r['peak_reserved_mb']:.0f}MB  "
              f"within_sp={r['within_theta_spearman']:.4f}  default_sp={r['default_theta_spearman']:.4f}  "
              f"top20_vs_default={r['top20_overlap_with_default']:.4f}  "
              f"decile9_rev={r['high_margin_reversal_decile9']:.4f}  "
              f"rms_range=[{r['rms_reward_min']:.3f},{r['rms_reward_max']:.3f}]", flush=True)

    # cheapest (theta_S, M) retaining most of default's ranking behavior -- reported, not applied
    THRESH_SPEARMAN = 0.95
    THRESH_TOP20 = 0.85
    candidates_ok = [r for r in pareto_rows if r["default_theta_spearman"] >= THRESH_SPEARMAN
                     and r["top20_overlap_with_default"] >= THRESH_TOP20]
    print(f"\n  candidates meeting default_theta_spearman>={THRESH_SPEARMAN} AND top20_vs_default>={THRESH_TOP20}: "
          f"{len(candidates_ok)}/{len(pareto_rows)}")
    if candidates_ok:
        cheapest = min(candidates_ok, key=lambda r: r["runtime_s"])
        print(f"  CHEAPEST (theta_S, M) meeting both thresholds (reported only, NOT auto-configured): "
              f"theta_S={cheapest['theta_s']} M={cheapest['M']} runtime={cheapest['runtime_s']:.2f}s "
              f"default_sp={cheapest['default_theta_spearman']:.4f} top20_vs_default={cheapest['top20_overlap_with_default']:.4f}")
    else:
        print("  no (theta_S, M) combination met both thresholds in the tested range.")

    # ------------------------------------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------------------------------------
    payload = dict(
        config=dict(B_HIST=B_HIST, M_H=M_H, DAMPING=DAMPING, BETA=BETA, HIST_SEED=HIST_SEED,
                    CHUNK_SIZE=CHUNK_SIZE, PRODUCTION_M_R=PRODUCTION_M_R, NUM_DIAG_SEEDS=NUM_DIAG_SEEDS,
                    M_LIST=M_LIST, CONSENSUS_NUM_SAMPLES=CONSENSUS_NUM_SAMPLES,
                    CONSENSUS_SEED_BASE=CONSENSUS_SEED_BASE, NUM_PAIRS=NUM_PAIRS, PAIR_SEED=PAIR_SEED,
                    NUM_DECILES=NUM_DECILES),
        per_theta=all_results,
        cross_theta_pairwise=pairwise,
        cross_theta_anchored_reversal=anchored_reversal,
        pareto_rows=pareto_rows,
    )
    with open(RESULTS_JSON, "w") as f:
        json.dump(to_jsonable(payload), f, indent=2)
    print(f"\nsaved raw results to {RESULTS_JSON}", flush=True)

    import csv
    with open(M_SWEEP_CSV, "w", newline="") as f:
        fieldnames = ["theta_s", "M", "spearman_mean", "spearman_std", "pearson_mean", "top10_mean", "top10_std",
                      "top20_mean", "top20_std", "overall_reversal"] + [f"decile{d}_reversal" for d in range(NUM_DECILES)]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for theta_name in theta_order:
            for row in all_results[theta_name]["m_sweep"]:
                writer.writerow({"theta_s": theta_name, **{k: row[k] for k in fieldnames if k != "theta_s"}})
    print(f"saved M-sweep summary CSV to {M_SWEEP_CSV}", flush=True)

    with open(PARETO_CSV, "w", newline="") as f:
        fieldnames = list(pareto_rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(pareto_rows)
    print(f"saved Pareto summary CSV to {PARETO_CSV}", flush=True)

    print("\nDiagnostic complete.")


if __name__ == "__main__":
    main()
