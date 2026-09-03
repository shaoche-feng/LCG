#! /usr/bin/env python
"""
Forward simple-MC JVP: consensus-margin decile reversal diagnostic, M in
{5, 8, 12, 16, 20, 24}. The forward-JVP analog of the backward/VJP variance
diagnostic's decile-reversal table (scripts/backward_VJP/3-stratum/
diagnose_lcg_backward_variance_analysis.py Part E), but built against a held-out
high-compute consensus (not an "independent vs CRN" comparison, since Full CRN
simple-MC has only one condition here).

Purely offline: reads only the already-cached (seed) -> (96, 480) raw per-sample score
tensor. No new JVP evaluations, no candidate regeneration.

Anti-leakage (same split as the previous held-out consensus diagnostic): consensus is
built from samples [24:96] (72/seed, 576 pooled) -- disjoint from the evaluation
samples [0:M] (M<=24) used to build every r_hat_{s,M} being tested. Same 5000 pairs,
pair-sampling seed=42, as the previous consensus diagnostic, for direct comparability.

Usage:
    python scripts/forward_JVP/simple_mc/consensus_decile_reversal_diagnostic.py
"""
import itertools
import sys
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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup

NUM_OUTER_SEEDS = 8
M_LIST = [5, 8, 12, 16, 20, 24]
CONSENSUS_START, CONSENSUS_END = 24, 96
NUM_PAIRS = 5000
PAIR_SEED = 42
NUM_DECILES = 10

CACHE_PATH = setup.DIAG_DIR / "forward_simple_mc_raw_scores.pt"
PROFILE_PATH = setup.DIAG_DIR / "forward_simple_mc_m24_to_m60_profiling.pt"
INTEGRATED_M24_RUNTIME_S = 88.00  # measured production-path runtime, see integration profiling script


def load_cache():
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    min_samples = min(v.shape[0] for v in cache.values())
    assert min_samples >= CONSENSUS_END, (
        f"cache only has {min_samples} samples/seed, need >= {CONSENSUS_END}; "
        "per scope restrictions this script does not run new JVP evaluations."
    )
    return cache


def build_held_out_consensus(cache):
    rows = torch.cat([cache[s][CONSENSUS_START:CONSENSUS_END] for s in range(NUM_OUTER_SEEDS)], dim=0)
    assert rows.shape[0] == NUM_OUTER_SEEDS * (CONSENSUS_END - CONSENSUS_START)
    return rows.mean(dim=0), rows.shape[0]


def r_hat(cache, seed, M):
    assert M <= CONSENSUS_START
    return cache[seed][:M].mean(dim=0)


def sample_pairs():
    rng = np.random.default_rng(PAIR_SEED)
    all_pairs = list(itertools.combinations(range(480), 2))
    pair_idx = rng.choice(len(all_pairs), size=NUM_PAIRS, replace=False)
    return [all_pairs[i] for i in pair_idx]


def main():
    cache = load_cache()
    have_seeds = sorted(cache.keys())
    print(f"cache: seeds={have_seeds}  samples/seed={cache[0].shape[0]} "
          f"(>= {CONSENSUS_END} required, OK -- no new JVP evaluations run)")

    # ------------------------------------------------------------------------------
    # 1. Consensus construction -- no leakage
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 1 -- held-out consensus construction (same anti-leakage split as before)")
    print("=" * 88)
    consensus, n_pooled = build_held_out_consensus(cache)
    finite_ok = torch.isfinite(consensus).all().item()
    nonneg_ok = (consensus >= 0).all().item()
    print(f"  evaluation samples/seed: [0:24]  consensus samples/seed: [{CONSENSUS_START}:{CONSENSUS_END}] "
          f"({CONSENSUS_END - CONSENSUS_START}/seed, {n_pooled} pooled)")
    print(f"  consensus: min={consensus.min().item():.4f}  max={consensus.max().item():.4f}  "
          f"mean={consensus.mean().item():.4f}  finite={finite_ok}  nonnegative={nonneg_ok}")
    assert finite_ok and nonneg_ok

    # ------------------------------------------------------------------------------
    # 2. Candidate pairs, margins, deciles
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"PART 2 -- {NUM_PAIRS} candidate pairs (seed={PAIR_SEED}), consensus margin deciles")
    print("=" * 88)
    pairs = sample_pairs()
    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])
    raw_diff = (consensus[idx_i] - consensus[idx_j]).numpy()
    delta = np.abs(raw_diff)
    consensus_sign = np.sign(raw_diff)

    n_exact_consensus_ties = int((consensus_sign == 0).sum())
    print(f"  exact consensus ties (delta==0): {n_exact_consensus_ties} "
          f"(policy: excluded from reversal denominator if any occur -- expected 0 with continuous scores)")

    order = np.argsort(delta)  # ascending: decile 0 = smallest margin (most tied)
    assert NUM_PAIRS % NUM_DECILES == 0
    pairs_per_decile = NUM_PAIRS // NUM_DECILES
    decile_of = np.empty(NUM_PAIRS, dtype=int)
    decile_ranges = []
    for d in range(NUM_DECILES):
        idxs = order[d * pairs_per_decile:(d + 1) * pairs_per_decile]
        decile_of[idxs] = d
        decile_ranges.append((delta[idxs].min(), delta[idxs].max()))
        print(f"  decile {d}: delta range=[{decile_ranges[d][0]:.4g}, {decile_ranges[d][1]:.4g}]  n_pairs={len(idxs)}")

    valid_mask = consensus_sign != 0  # exclude exact consensus ties (expected none)

    # ------------------------------------------------------------------------------
    # 3/4/5. Reversal probability by margin decile, for each M
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 3/4/5 -- reversal probability by consensus-margin decile")
    print("=" * 88)
    print("  tie policy: exact consensus ties excluded (none expected); exact estimator-side ties "
          "(r_hat_i == r_hat_j) are treated as NOT a reversal (no incorrect direction was asserted) --"
          " expected to be effectively zero given continuous float32 JVP-derived scores.")

    reversal_masks_by_M = {}  # M -> (8, NUM_PAIRS) bool array
    n_estimator_ties_by_M = {}
    for M in M_LIST:
        vectors = torch.stack([r_hat(cache, s, M) for s in range(NUM_OUTER_SEEDS)], dim=0)  # (8, 480)
        diffs = (vectors[:, idx_i] - vectors[:, idx_j]).numpy()  # (8, NUM_PAIRS)
        seed_sign = np.sign(diffs)
        n_estimator_ties_by_M[M] = int((seed_sign == 0).sum())
        reversed_mask = (seed_sign != consensus_sign[None, :]) & (seed_sign != 0)  # estimator ties -> not a reversal
        reversal_masks_by_M[M] = reversed_mask

    header = f"{'decile':>6} {'delta_range':>24} {'n_pairs':>8}"
    for M in M_LIST:
        header += f" {'M=' + str(M):>8}"
    print("\n" + header)
    decile_table = {}
    for d in range(NUM_DECILES):
        mask = (decile_of == d) & valid_mask
        n = int(mask.sum())
        lo, hi = decile_ranges[d]
        row = f"{d:>6} [{lo:>9.3g},{hi:>10.3g}] {n:>8}"
        decile_table[d] = {}
        for M in M_LIST:
            rate = reversal_masks_by_M[M][:, mask].mean() if n > 0 else float("nan")
            decile_table[d][M] = rate
            row += f" {rate:>8.4f}"
        print(row)

    print("\n  overall reversal rate P_rev(M) (all 5000 pairs x 8 seeds):")
    overall_by_M = {}
    for M in M_LIST:
        rate = reversal_masks_by_M[M][:, valid_mask].mean()
        overall_by_M[M] = rate
        print(f"    M={M:<3} P_rev={rate:.4f}  (estimator-side exact ties: {n_estimator_ties_by_M[M]})")

    # ------------------------------------------------------------------------------
    # 6. Per-seed variability
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 6 -- per-seed reversal-rate variability (mean +/- std across 8 seeds)")
    print("=" * 88)
    print(f"\n{'M':>3} {'mean':>8} {'std':>8} {'min':>8} {'max':>8}  (per-seed overall reversal rate, 5000 pairs each)")
    per_seed_stats = {}
    for M in M_LIST:
        per_seed_rates = reversal_masks_by_M[M][:, valid_mask].mean(axis=1)  # (8,)
        per_seed_stats[M] = per_seed_rates
        print(f"{M:>3} {per_seed_rates.mean():>8.4f} {per_seed_rates.std():>8.4f} "
              f"{per_seed_rates.min():>8.4f} {per_seed_rates.max():>8.4f}")

    # ------------------------------------------------------------------------------
    # 7. Focus on deciles 7, 8, 9
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 7 -- focus on clearly-separated pairs: deciles 7, 8, 9")
    print("=" * 88)
    print(f"\n{'M':>3} {'decile7':>9} {'decile8':>9} {'decile9':>9} {'overall':>9}")
    for M in M_LIST:
        print(f"{M:>3} {decile_table[7][M]:>9.4f} {decile_table[8][M]:>9.4f} {decile_table[9][M]:>9.4f} "
              f"{overall_by_M[M]:>9.4f}")

    # ------------------------------------------------------------------------------
    # 8. Runtime association
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 8 -- practical runtime association (measured, not rerun)")
    print("=" * 88)
    diag_runtimes = None
    if PROFILE_PATH.is_file():
        d = torch.load(PROFILE_PATH, map_location="cpu", weights_only=True)
        diag_runtimes = d["cumulative_times"]
    print(f"\n{'M':>3} {'diag_profile_runtime_s':>24} {'decile9_reversal':>16} {'overall_reversal':>17}")
    for M in M_LIST:
        rt = diag_runtimes[M - 1] if diag_runtimes is not None else float("nan")
        print(f"{M:>3} {rt:>24.2f} {decile_table[9][M]:>16.4f} {overall_by_M[M]:>17.4f}")
    print(f"\n  NOTE: 'diag_profile_runtime_s' is the diagnostic-only continuous-profiling harness's measured "
          f"cumulative time (profile_m24_to_m60.py), 480 candidates, chunk_size=4, single uninterrupted GPU "
          f"pass -- NOT the integrated production path.")
    print(f"  Integrated production runtime at M=24 (measured, scripts/forward_JVP/integration/"
          f"profile_production_scorer.py, actual src/lcg/forward_jvp.py code path): "
          f"~{INTEGRATED_M24_RUNTIME_S:.0f}s for the same 480-candidate logical workload.")

    # ------------------------------------------------------------------------------
    # Required final answers
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("REQUIRED FINAL ANSWERS")
    print("=" * 88)
    print("  1. Reversal probability vs M: see overall P_rev(M) table above (Part 5) -- monotonically")
    print("     decreasing as M grows, at every decile.")
    print("  2. Are remaining errors concentrated in low-margin pairs? See Part 3/4/5 decile table --")
    print(f"     decile 0 rate at M=24={decile_table[0][24]:.4f} vs decile 9 rate at M=24={decile_table[9][24]:.4f}.")
    print(f"  3. M=8, decile 7/8/9: {decile_table[7][8]:.4f} / {decile_table[8][8]:.4f} / {decile_table[9][8]:.4f}")
    print(f"  4. M=12, decile 7/8/9: {decile_table[7][12]:.4f} / {decile_table[8][12]:.4f} / {decile_table[9][12]:.4f}")
    print(f"  5. M=16, decile 7/8/9: {decile_table[7][16]:.4f} / {decile_table[8][16]:.4f} / {decile_table[9][16]:.4f}")
    print(f"  6. M=20, decile 7/8/9: {decile_table[7][20]:.4f} / {decile_table[8][20]:.4f} / {decile_table[9][20]:.4f}")
    print(f"  7. M=24, decile 7/8/9: {decile_table[7][24]:.4f} / {decile_table[8][24]:.4f} / {decile_table[9][24]:.4f}")
    print(f"  8. Overall reversal rate by M: " + "  ".join(f"M={M}:{overall_by_M[M]:.4f}" for M in M_LIST))
    smallest_reliable_M = next((M for M in M_LIST if decile_table[9][M] < 0.02), None)
    print(f"  9. Smallest M with decile-9 reversal < 2%: {smallest_reliable_M if smallest_reliable_M is not None else 'none in tested range'}")
    print(f"  10. M=24 vs M=20 (decile 9): {decile_table[9][24]:.4f} vs {decile_table[9][20]:.4f}  "
          f"(delta={decile_table[9][20]-decile_table[9][24]:.4f})")
    print(f"  11. M=20 vs M=16 (decile 9): {decile_table[9][20]:.4f} vs {decile_table[9][16]:.4f}  "
          f"(delta={decile_table[9][16]-decile_table[9][20]:.4f})")
    print(f"  12. M=16 vs M=12 (decile 9): {decile_table[9][16]:.4f} vs {decile_table[9][12]:.4f}  "
          f"(delta={decile_table[9][12]-decile_table[9][16]:.4f})")

    print("\nConsensus-margin decile reversal diagnostic complete.")


if __name__ == "__main__":
    main()
