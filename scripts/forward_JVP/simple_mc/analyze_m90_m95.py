#! /usr/bin/env python
"""
Forward simple-MC JVP: full analysis at the identified M_0.90 (=51) operating point, plus
M=96 (the highest tested checkpoint; 0.95 was not reached within this budget, max=0.9438
at M=96). CPU-only, reads only the cached score tensor.

Usage:
    python scripts/forward_JVP/simple_mc/analyze_m90_m95.py
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup

NUM_OUTER_SEEDS = 8
M_090 = 51
M_HIGH = 96  # highest tested checkpoint; 0.95 was not reached
M_REF = 24

FWD_CACHE_PATH = setup.DIAG_DIR / "forward_simple_mc_raw_scores.pt"
BWD_CACHE_PATH = setup.DIAG_DIR / "partBC_raw_scores.pt"


def r_M(cache, seed, M, offset=0):
    rows = cache[seed]
    return rows[offset : offset + M].mean(dim=0)


def r_K_bwd(cache, seed, condition, K):
    rows = cache[(seed, condition)]
    return rows[:K].mean(dim=0)


def rank_shift_stats(a, b):
    ra, rb = setup._rank(a), setup._rank(b)
    abs_shift = (ra - rb).abs().numpy()
    return float(np.median(abs_shift)), float(np.percentile(abs_shift, 90))


def full_metrics(vectors, label):
    sps, pes, t10s, t20s = [], [], [], []
    for s1, s2 in itertools.combinations(range(NUM_OUTER_SEEDS), 2):
        a, b = vectors[s1], vectors[s2]
        sps.append(setup.spearman_corr(a, b))
        pes.append(setup.pearson_corr(a, b))
        t10s.append(setup.top_q_overlap(a, b, 0.10))
        t20s.append(setup.top_q_overlap(a, b, 0.20))
    sps = np.array(sps)
    stacked = torch.stack(vectors, dim=0)
    ranks = torch.stack([setup._rank(v) for v in vectors], dim=0)
    rank_std = ranks.std(dim=0, unbiased=True).numpy()
    score_cv = setup.cv(stacked, dim=0).numpy()

    print(f"\n--- {label} ---")
    print(f"  Spearman: mean={sps.mean():.4f} std={sps.std():.4f} median={np.median(sps):.4f} "
          f"p10={np.percentile(sps,10):.4f} p90={np.percentile(sps,90):.4f} min={sps.min():.4f}")
    print(f"  Pearson: mean={np.mean(pes):.4f}  top10={np.mean(t10s):.4f}  top20={np.mean(t20s):.4f}")
    print(f"  per-candidate rank_std: mean={rank_std.mean():.4f}  score_cv: mean={score_cv.mean():.4f}")


def main():
    fwd_cache = torch.load(FWD_CACHE_PATH, map_location="cpu", weights_only=True)
    bwd_cache = torch.load(BWD_CACHE_PATH, map_location="cpu", weights_only=True)
    min_avail = min(v.shape[0] for v in fwd_cache.values())
    print(f"cache: min samples available = {min_avail}")

    print("\n" + "=" * 88)
    print("PART 4 -- full metrics at M_ref=24, M_0.90=51, M_high=96")
    print("=" * 88)
    v24 = [r_M(fwd_cache, s, M_REF) for s in range(NUM_OUTER_SEEDS)]
    v51 = [r_M(fwd_cache, s, M_090) for s in range(NUM_OUTER_SEEDS)]
    v96 = [r_M(fwd_cache, s, M_HIGH) for s in range(NUM_OUTER_SEEDS)]
    full_metrics(v24, "M=24 (reference)")
    full_metrics(v51, "M=51 (M_0.90)")
    full_metrics(v96, "M=96 (highest tested; 0.95 NOT reached, closest=0.9438)")

    print("\n" + "=" * 88)
    print("PART 5 -- cross-M convergence (nested, share samples)")
    print("=" * 88)
    for (mA, vA, nameA), (mB, vB, nameB) in [
        ((M_REF, v24, "M=24"), (M_090, v51, "M_0.90=51")),
        ((M_090, v51, "M_0.90=51"), (M_HIGH, v96, "M=96")),
    ]:
        sps, t10s, t20s, meds, p90s = [], [], [], [], []
        for s in range(NUM_OUTER_SEEDS):
            a, b = vA[s], vB[s]
            sps.append(setup.spearman_corr(a, b))
            t10s.append(setup.top_q_overlap(a, b, 0.10))
            t20s.append(setup.top_q_overlap(a, b, 0.20))
            med, p90 = rank_shift_stats(a, b)
            meds.append(med)
            p90s.append(p90)
        print(f"  {nameA} <-> {nameB}: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  "
              f"top20={np.mean(t20s):.4f}  median_rank_shift={np.mean(meds):.1f}  p90_rank_shift={np.mean(p90s):.1f}")

    print("\n  -- disjoint check: samples[0:48] vs samples[48:96] (no shared samples, both M=48) --")
    sps, t10s, t20s = [], [], []
    for s in range(NUM_OUTER_SEEDS):
        a = r_M(fwd_cache, s, 48, offset=0)
        b = r_M(fwd_cache, s, 48, offset=48)
        sps.append(setup.spearman_corr(a, b))
        t10s.append(setup.top_q_overlap(a, b, 0.10))
        t20s.append(setup.top_q_overlap(a, b, 0.20))
    print(f"  disjoint M=48 vs M=48: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  top20={np.mean(t20s):.4f}")

    print("\n" + "=" * 88)
    print("PART 6 -- agreement with backward 3-stratum Full-CRN VJP K=16")
    print("=" * 88)
    backward_ref = torch.stack([r_K_bwd(bwd_cache, s, "crn", 16) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
    print(f"\n  {'estimator':>16} {'Spearman':>9} {'Pearson':>9} {'top10':>8} {'top20':>8} "
          f"{'median_shift':>13} {'p90_shift':>10}")
    for name, v in [("M=24", v24), ("M_0.90=51", v51), ("M=96", v96)]:
        avg = torch.stack(v).mean(dim=0)
        sp = setup.spearman_corr(avg, backward_ref)
        pe = setup.pearson_corr(avg, backward_ref)
        t10 = setup.top_q_overlap(avg, backward_ref, 0.10)
        t20 = setup.top_q_overlap(avg, backward_ref, 0.20)
        med, p90 = rank_shift_stats(avg, backward_ref)
        print(f"  {name:>16} {sp:>9.4f} {pe:>9.4f} {t10:>8.4f} {t20:>8.4f} {med:>13.1f} {p90:>10.1f}")

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
