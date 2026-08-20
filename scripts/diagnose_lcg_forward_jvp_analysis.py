#! /usr/bin/env python
"""
LCG forward/JVP diagnostic -- Stage F4 analysis (cross-seed/cross-K consistency) and Stage
F5 (comparison against the cached backward VJP reference, 3-stratum Full-CRN K=16).

Reads only cached score tensors -- no GPU, no JVPs/VJPs, no candidate regeneration.

Usage:
    python scripts/diagnose_lcg_forward_jvp_analysis.py
"""
import itertools
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup

NUM_OUTER_SEEDS = 8
K_LIST = [1, 2, 4, 8, 16]
FORWARD_CACHE_PATH = setup.DIAG_DIR / "forward_jvp_raw_scores.pt"
BACKWARD_CACHE_PATH = setup.DIAG_DIR / "partBC_raw_scores.pt"


def r_K_fwd(cache, seed, K):
    rows = cache[seed]
    assert rows.shape[0] >= K
    return rows[:K].mean(dim=0)


def r_K_bwd(cache, seed, condition, K):
    rows = cache[(seed, condition)]
    assert rows.shape[0] >= K
    return rows[:K].mean(dim=0)


def rank_shift_stats(a, b):
    ra, rb = setup._rank(a), setup._rank(b)
    abs_shift = (ra - rb).abs().numpy()
    return float(np.median(abs_shift)), float(np.percentile(abs_shift, 90))


def pairwise_seed_metrics_fwd(cache, K):
    vectors = [r_K_fwd(cache, s, K) for s in range(NUM_OUTER_SEEDS)]
    spearmans, pearsons, top10s, top20s = [], [], [], []
    for s1, s2 in itertools.combinations(range(NUM_OUTER_SEEDS), 2):
        a, b = vectors[s1], vectors[s2]
        spearmans.append(setup.spearman_corr(a, b))
        pearsons.append(setup.pearson_corr(a, b))
        top10s.append(setup.top_q_overlap(a, b, 0.10))
        top20s.append(setup.top_q_overlap(a, b, 0.20))
    return dict(spearman=np.array(spearmans), pearson=np.array(pearsons),
                top10=np.array(top10s), top20=np.array(top20s), vectors=vectors)


def rank_std_and_cv(vectors):
    stacked = torch.stack(vectors, dim=0)
    ranks = torch.stack([setup._rank(v) for v in vectors], dim=0)
    return ranks.std(dim=0, unbiased=True).numpy(), setup.cv(stacked, dim=0).numpy()


def section_f4(cache):
    print("\n" + "=" * 88)
    print("STAGE F4 -- forward-JVP cross-seed ranking consistency, per K")
    print("=" * 88)
    results = {}
    for K in K_LIST:
        m = pairwise_seed_metrics_fwd(cache, K)
        rank_std, score_cv = rank_std_and_cv(m["vectors"])
        results[K] = dict(m=m, rank_std=rank_std, score_cv=score_cv)
        print(f"\n  K={K:>2}  (n_seed_pairs={len(m['spearman'])})")
        print(f"    Spearman: mean={m['spearman'].mean():.4f} std={m['spearman'].std():.4f} "
              f"median={np.median(m['spearman']):.4f} p10={np.percentile(m['spearman'],10):.4f} "
              f"p90={np.percentile(m['spearman'],90):.4f} min={m['spearman'].min():.4f}")
        print(f"    Pearson:  mean={m['pearson'].mean():.4f}")
        print(f"    top10_overlap: mean={m['top10'].mean():.4f}   top20_overlap: mean={m['top20'].mean():.4f}")
        print(f"    per-candidate rank_std across seeds: mean={rank_std.mean():.4f} median={np.median(rank_std):.4f}")
        print(f"    per-candidate score_cv across seeds: mean={score_cv.mean():.4f} median={np.median(score_cv):.4f}")

    print("\n" + "-" * 88)
    print("STAGE F4 -- cross-K ranking convergence (within seed)")
    print("-" * 88)
    k_pairs = [(1, 2), (1, 4), (2, 4), (4, 8), (8, 16)]
    for k1, k2 in k_pairs:
        sps, t10s, t20s, meds, p90s = [], [], [], [], []
        for s in range(NUM_OUTER_SEEDS):
            a = r_K_fwd(cache, s, k1)
            b = r_K_fwd(cache, s, k2)
            sps.append(setup.spearman_corr(a, b))
            t10s.append(setup.top_q_overlap(a, b, 0.10))
            t20s.append(setup.top_q_overlap(a, b, 0.20))
            med, p90 = rank_shift_stats(a, b)
            meds.append(med)
            p90s.append(p90)
        print(f"  K={k1:>2} <-> K={k2:>2}: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  "
              f"top20={np.mean(t20s):.4f}  median_rank_shift={np.mean(meds):.1f}  p90_rank_shift={np.mean(p90s):.1f}")

    return results


def section_f5(fwd_cache, bwd_cache):
    print("\n" + "=" * 88)
    print("STAGE F5 -- forward JVP vs backward VJP (3-stratum Full-CRN K=16 reference)")
    print("=" * 88)
    backward_ref = torch.stack([r_K_bwd(bwd_cache, s, "crn", 16) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
    print(f"\n  backward reference: seed-averaged 3-stratum Full-CRN K=16 "
          f"(previously measured cross-seed Spearman ~0.93). Treated as a stability reference, not ground truth.")

    print(f"\n  {'fwd_K':>6} {'Spearman_vs_bwdK16':>19} {'Pearson':>9} {'top10':>8} {'top20':>8} "
          f"{'median_shift':>13} {'p90_shift':>10}")
    for K in K_LIST:
        fwd_avg = torch.stack([r_K_fwd(fwd_cache, s, K) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        sp = setup.spearman_corr(fwd_avg, backward_ref)
        pe = setup.pearson_corr(fwd_avg, backward_ref)
        t10 = setup.top_q_overlap(fwd_avg, backward_ref, 0.10)
        t20 = setup.top_q_overlap(fwd_avg, backward_ref, 0.20)
        med, p90 = rank_shift_stats(fwd_avg, backward_ref)
        print(f"  {K:>6} {sp:>19.4f} {pe:>9.4f} {t10:>8.4f} {t20:>8.4f} {med:>13.1f} {p90:>10.1f}")

    print("\n  equal-K comparison: forward K vs backward (crn) K, cross-seed Spearman side by side")
    print(f"\n  {'K':>4} {'fwd_cross_seed_Sp':>18} {'bwd_cross_seed_Sp':>18} {'fwd_vs_bwd_avg_Sp':>18}")
    for K in K_LIST:
        fwd_m = pairwise_seed_metrics_fwd(fwd_cache, K)
        bwd_vectors = [r_K_bwd(bwd_cache, s, "crn", K) for s in range(NUM_OUTER_SEEDS)]
        bwd_spearmans = []
        for s1, s2 in itertools.combinations(range(NUM_OUTER_SEEDS), 2):
            bwd_spearmans.append(setup.spearman_corr(bwd_vectors[s1], bwd_vectors[s2]))
        fwd_avg_K = torch.stack([r_K_fwd(fwd_cache, s, K) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        bwd_avg_K = torch.stack(bwd_vectors).mean(dim=0)
        cross_sp = setup.spearman_corr(fwd_avg_K, bwd_avg_K)
        print(f"  {K:>4} {fwd_m['spearman'].mean():>18.4f} {np.mean(bwd_spearmans):>18.4f} {cross_sp:>18.4f}")


def main():
    fwd_cache = torch.load(FORWARD_CACHE_PATH, map_location="cpu", weights_only=True)
    bwd_cache = torch.load(BACKWARD_CACHE_PATH, map_location="cpu", weights_only=True)

    have_seeds = sorted(fwd_cache.keys())
    print(f"Loaded forward-JVP cache: seeds={have_seeds}")
    for s, v in sorted(fwd_cache.items()):
        print(f"  seed={s}: {v.shape[0]} banks cached")
    assert set(have_seeds) == set(range(NUM_OUTER_SEEDS))

    bwd_conditions = sorted(set(c for s, c in bwd_cache.keys()))
    print(f"\nLoaded backward (previous-diagnostic) cache: conditions={bwd_conditions}")
    assert "crn" in bwd_conditions

    section_f4(fwd_cache)
    section_f5(fwd_cache, bwd_cache)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
