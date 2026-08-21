#! /usr/bin/env python
"""
Forward-JVP sampling diagnostic -- simple MC vs 3-stratum analysis (Sections 8-11).

Reads only cached score tensors -- no GPU, no JVPs, no candidate regeneration. Reuses the
existing cached forward-stratified (forward_jvp_raw_scores.pt, K in {1,2,4,8,16} via
cumulative averaging of 16 3-strata banks -- each K costing 3K JVPs/candidate) and backward
3-stratum Full-CRN K=16 (partBC_raw_scores.pt, condition="crn") results, per the "do not
recompute" instruction.

Usage:
    python scripts/diagnose_lcg_forward_simple_mc_analysis.py
"""
import itertools
import sys
from pathlib import Path

def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root (no ancestor has both src/ and scripts/)")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "forward_JVP" / "3-stratum_CRN"))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup

NUM_OUTER_SEEDS = 8
M_LIST = [3, 6, 12, 24]
K_EQUIV = {3: 1, 6: 2, 12: 4, 24: 8}  # simple-MC M -> equal-cost stratified K (both = 3K JVPs/candidate... wait M JVPs = 3K JVPs -> K = M/3)

SIMPLE_MC_CACHE_PATH = setup.DIAG_DIR / "forward_simple_mc_raw_scores.pt"
STRATIFIED_CACHE_PATH = setup.DIAG_DIR / "forward_jvp_raw_scores.pt"
BACKWARD_CACHE_PATH = setup.DIAG_DIR / "partBC_raw_scores.pt"


def r_M_simple(cache, seed, M, offset=0):
    """mean of samples [offset : offset+M) for this seed (nested if offset=0, disjoint
    otherwise)."""
    rows = cache[seed]
    assert rows.shape[0] >= offset + M
    return rows[offset : offset + M].mean(dim=0)


def r_K_strat(cache, seed, K):
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


def pairwise_seed_metrics(vectors):
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


def section8(mc_cache, strat_cache):
    print("\n" + "=" * 88)
    print("SECTION 8 -- cross-seed consistency: simple MC vs stratified, equal JVP cost")
    print("=" * 88)
    mc_results, strat_results = {}, {}
    print(f"\n{'JVPs/cand':>10} {'M(simpleMC)':>12} {'K(stratified)':>14} "
          f"{'mc_Spearman':>12} {'strat_Spearman':>15} {'mc_top10':>9} {'strat_top10':>12} "
          f"{'mc_top20':>9} {'strat_top20':>12}")
    for M in M_LIST:
        K = K_EQUIV[M]
        mc_vectors = [r_M_simple(mc_cache, s, M) for s in range(NUM_OUTER_SEEDS)]
        strat_vectors = [r_K_strat(strat_cache, s, K) for s in range(NUM_OUTER_SEEDS)]
        mc_m = pairwise_seed_metrics(mc_vectors)
        strat_m = pairwise_seed_metrics(strat_vectors)
        mc_rank_std, mc_cv = rank_std_and_cv(mc_vectors)
        strat_rank_std, strat_cv = rank_std_and_cv(strat_vectors)
        mc_results[M] = dict(m=mc_m, rank_std=mc_rank_std, cv=mc_cv)
        strat_results[K] = dict(m=strat_m, rank_std=strat_rank_std, cv=strat_cv)

        jvps = 3 * K
        print(f"{jvps:>10} {M:>12} {K:>14} {mc_m['spearman'].mean():>12.4f} "
              f"{strat_m['spearman'].mean():>15.4f} {mc_m['top10'].mean():>9.4f} "
              f"{strat_m['top10'].mean():>12.4f} {mc_m['top20'].mean():>9.4f} {strat_m['top20'].mean():>12.4f}")

    print(f"\nFull detail per budget:")
    for M in M_LIST:
        K = K_EQUIV[M]
        print(f"\n  --- {3*K} JVPs/candidate: simple MC M={M} ---")
        m = mc_results[M]["m"]
        print(f"    Spearman: mean={m['spearman'].mean():.4f} std={m['spearman'].std():.4f} "
              f"median={np.median(m['spearman']):.4f} p10={np.percentile(m['spearman'],10):.4f} "
              f"p90={np.percentile(m['spearman'],90):.4f} min={m['spearman'].min():.4f}")
        print(f"    Pearson: mean={m['pearson'].mean():.4f}  top10={m['top10'].mean():.4f}  top20={m['top20'].mean():.4f}")
        print(f"    rank_std: mean={mc_results[M]['rank_std'].mean():.4f}  score_cv: mean={mc_results[M]['cv'].mean():.4f}")

        print(f"  --- {3*K} JVPs/candidate: stratified K={K} ---")
        m = strat_results[K]["m"]
        print(f"    Spearman: mean={m['spearman'].mean():.4f} std={m['spearman'].std():.4f} "
              f"median={np.median(m['spearman']):.4f} p10={np.percentile(m['spearman'],10):.4f} "
              f"p90={np.percentile(m['spearman'],90):.4f} min={m['spearman'].min():.4f}")
        print(f"    Pearson: mean={m['pearson'].mean():.4f}  top10={m['top10'].mean():.4f}  top20={m['top20'].mean():.4f}")
        print(f"    rank_std: mean={strat_results[K]['rank_std'].mean():.4f}  score_cv: mean={strat_results[K]['cv'].mean():.4f}")

    return mc_results, strat_results


def section9(mc_cache, strat_cache):
    print("\n" + "=" * 88)
    print("SECTION 9 -- cross-method ranking agreement (seed-averaged), equal JVP cost")
    print("=" * 88)
    print(f"\n{'JVPs/cand':>10} {'M':>4} {'K':>4} {'Spearman':>9} {'Pearson':>9} {'top10':>8} "
          f"{'top20':>8} {'median_shift':>13} {'p90_shift':>10}")
    for M in M_LIST:
        K = K_EQUIV[M]
        mc_avg = torch.stack([r_M_simple(mc_cache, s, M) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        strat_avg = torch.stack([r_K_strat(strat_cache, s, K) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        sp = setup.spearman_corr(mc_avg, strat_avg)
        pe = setup.pearson_corr(mc_avg, strat_avg)
        t10 = setup.top_q_overlap(mc_avg, strat_avg, 0.10)
        t20 = setup.top_q_overlap(mc_avg, strat_avg, 0.20)
        med, p90 = rank_shift_stats(mc_avg, strat_avg)
        print(f"{3*K:>10} {M:>4} {K:>4} {sp:>9.4f} {pe:>9.4f} {t10:>8.4f} {t20:>8.4f} {med:>13.1f} {p90:>10.1f}")


def section10(mc_cache):
    print("\n" + "=" * 88)
    print("SECTION 10 -- cross-M convergence for simple MC")
    print("=" * 88)
    print("\n  -- nested-budget agreement (cumulative estimators SHARE samples; expect some inflation) --")
    nested_pairs = [(3, 6), (6, 12), (12, 24)]
    for m1, m2 in nested_pairs:
        sps, t10s, t20s, meds, p90s = [], [], [], [], []
        for s in range(NUM_OUTER_SEEDS):
            a = r_M_simple(mc_cache, s, m1)
            b = r_M_simple(mc_cache, s, m2)
            sps.append(setup.spearman_corr(a, b))
            t10s.append(setup.top_q_overlap(a, b, 0.10))
            t20s.append(setup.top_q_overlap(a, b, 0.20))
            med, p90 = rank_shift_stats(a, b)
            meds.append(med)
            p90s.append(p90)
        print(f"  M={m1:>2} <-> M={m2:>2}: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  "
              f"top20={np.mean(t20s):.4f}  median_rank_shift={np.mean(meds):.1f}  p90_rank_shift={np.mean(p90s):.1f}")

    print("\n  -- disjoint check: M=6 using samples[0:6] vs M=6 using samples[6:12] (no shared samples) --")
    sps, t10s, t20s = [], [], []
    for s in range(NUM_OUTER_SEEDS):
        a = r_M_simple(mc_cache, s, 6, offset=0)
        b = r_M_simple(mc_cache, s, 6, offset=6)
        sps.append(setup.spearman_corr(a, b))
        t10s.append(setup.top_q_overlap(a, b, 0.10))
        t20s.append(setup.top_q_overlap(a, b, 0.20))
    print(f"  disjoint M=6 vs M=6: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  top20={np.mean(t20s):.4f}")
    print("  (this isolates genuine convergence from sample-sharing inflation)")


def section11(mc_cache, strat_cache, bwd_cache):
    print("\n" + "=" * 88)
    print("SECTION 11 -- agreement with backward 3-stratum Full-CRN VJP K=16 (stability reference, not ground truth)")
    print("=" * 88)
    backward_ref = torch.stack([r_K_bwd(bwd_cache, s, "crn", 16) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)

    print(f"\n  {'estimator':>20} {'Spearman':>9} {'Pearson':>9} {'top10':>8} {'top20':>8} "
          f"{'median_shift':>13} {'p90_shift':>10}")
    for M in M_LIST:
        mc_avg = torch.stack([r_M_simple(mc_cache, s, M) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        sp = setup.spearman_corr(mc_avg, backward_ref)
        pe = setup.pearson_corr(mc_avg, backward_ref)
        t10 = setup.top_q_overlap(mc_avg, backward_ref, 0.10)
        t20 = setup.top_q_overlap(mc_avg, backward_ref, 0.20)
        med, p90 = rank_shift_stats(mc_avg, backward_ref)
        print(f"  {'simpleMC_M' + str(M):>20} {sp:>9.4f} {pe:>9.4f} {t10:>8.4f} {t20:>8.4f} {med:>13.1f} {p90:>10.1f}")

    for K in [1, 2, 4, 8]:
        strat_avg = torch.stack([r_K_strat(strat_cache, s, K) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        sp = setup.spearman_corr(strat_avg, backward_ref)
        pe = setup.pearson_corr(strat_avg, backward_ref)
        t10 = setup.top_q_overlap(strat_avg, backward_ref, 0.10)
        t20 = setup.top_q_overlap(strat_avg, backward_ref, 0.20)
        med, p90 = rank_shift_stats(strat_avg, backward_ref)
        print(f"  {'stratified_K' + str(K):>20} {sp:>9.4f} {pe:>9.4f} {t10:>8.4f} {t20:>8.4f} {med:>13.1f} {p90:>10.1f}")


def main():
    mc_cache = torch.load(SIMPLE_MC_CACHE_PATH, map_location="cpu", weights_only=True)
    strat_cache = torch.load(STRATIFIED_CACHE_PATH, map_location="cpu", weights_only=True)
    bwd_cache = torch.load(BACKWARD_CACHE_PATH, map_location="cpu", weights_only=True)

    print(f"Loaded simple-MC cache: seeds={sorted(mc_cache.keys())}")
    for s, v in sorted(mc_cache.items()):
        print(f"  seed={s}: {v.shape[0]} samples cached")
    assert set(mc_cache.keys()) == set(range(NUM_OUTER_SEEDS))

    print(f"\nLoaded stratified forward cache: seeds={sorted(strat_cache.keys())}, "
          f"banks/seed={strat_cache[0].shape[0]}")
    assert set(strat_cache.keys()) == set(range(NUM_OUTER_SEEDS))
    assert strat_cache[0].shape[0] >= 8

    bwd_conditions = sorted(set(c for s, c in bwd_cache.keys()))
    print(f"\nLoaded backward cache: conditions={bwd_conditions}")
    assert "crn" in bwd_conditions

    mc_results, strat_results = section8(mc_cache, strat_cache)
    section9(mc_cache, strat_cache)
    section10(mc_cache)
    section11(mc_cache, strat_cache, bwd_cache)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
