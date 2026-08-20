#! /usr/bin/env python
"""
LCG backward/VJP variance diagnostic -- offline analysis for Parts B, C, D, E.

Reads only the cached (seed, condition) -> (num_banks, 480) raw per-bank score tensor
produced by diagnose_lcg_backward_variance_partBC.py. No GPU, no VJPs, no candidate
regeneration -- every K in {1,2,3,4,8,16[,32]} is reconstructed by averaging the first K
cached banks per seed, exactly the offline-cumulative-averaging optimization requested.

Usage:
    python scripts/diagnose_lcg_backward_variance_analysis.py
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
K_LIST = [1, 2, 3, 4, 8, 16]
NUM_PAIRS = 5000
PAIR_SEED = 42


def extend_k_list_if_available(cache, k_list):
    min_banks = min(v.shape[0] for v in cache.values())
    if min_banks >= 32 and 32 not in k_list:
        return k_list + [32]
    return k_list


def load_cache():
    cache = torch.load(setup.DIAG_DIR / "partBC_raw_scores.pt", map_location="cpu", weights_only=True)
    return cache


def r_K(cache, seed, condition, K):
    """(480,) mean of the first K cached per-bank scores for this (seed, condition)."""
    rows = cache[(seed, condition)]
    assert rows.shape[0] >= K, f"only {rows.shape[0]} banks cached for seed={seed} condition={condition}, need K={K}"
    return rows[:K].mean(dim=0)


def pairwise_seed_metrics(cache, condition, K):
    vectors = [r_K(cache, s, condition, K) for s in range(NUM_OUTER_SEEDS)]
    spearmans, pearsons, top10s, top20s = [], [], [], []
    for s1, s2 in itertools.combinations(range(NUM_OUTER_SEEDS), 2):
        a, b = vectors[s1], vectors[s2]
        spearmans.append(setup.spearman_corr(a, b))
        pearsons.append(setup.pearson_corr(a, b))
        top10s.append(setup.top_q_overlap(a, b, 0.10))
        top20s.append(setup.top_q_overlap(a, b, 0.20))
    return dict(spearman=np.array(spearmans), pearson=np.array(pearsons),
                top10=np.array(top10s), top20=np.array(top20s), vectors=vectors)


def rank_std_and_cv_across_seeds(vectors):
    stacked = torch.stack(vectors, dim=0)  # (8, 480)
    ranks = torch.stack([setup._rank(v) for v in vectors], dim=0)  # (8, 480)
    rank_std = ranks.std(dim=0, unbiased=True).numpy()  # (480,)
    score_cv = setup.cv(stacked, dim=0).numpy()  # (480,)
    return rank_std, score_cv


def summarize(values, name, indent="    "):
    arr = np.asarray(values, dtype=np.float64)
    print(f"{indent}{name:<24} mean={arr.mean():.4f}  median={np.median(arr):.4f}  "
          f"std={arr.std():.4f}  p10={np.percentile(arr, 10):.4f}  p90={np.percentile(arr, 90):.4f}  "
          f"min={arr.min():.4f}  max={arr.max():.4f}")


def part_bc(cache):
    print("\n" + "=" * 88)
    print("PART B/C -- cross-seed ranking consistency vs K, Independent vs CRN")
    print("=" * 88)
    results = {}
    for condition in ["independent", "crn"]:
        print(f"\n--- condition: {condition} ---")
        results[condition] = {}
        for K in K_LIST:
            m = pairwise_seed_metrics(cache, condition, K)
            rank_std, score_cv = rank_std_and_cv_across_seeds(m["vectors"])
            results[condition][K] = dict(m=m, rank_std=rank_std, score_cv=score_cv)
            print(f"\n  K={K:>2}  (n_seed_pairs={len(m['spearman'])})")
            print(f"    Spearman: mean={m['spearman'].mean():.4f} std={m['spearman'].std():.4f} "
                  f"median={np.median(m['spearman']):.4f} p10={np.percentile(m['spearman'],10):.4f} "
                  f"p90={np.percentile(m['spearman'],90):.4f} min={m['spearman'].min():.4f}")
            print(f"    Pearson:  mean={m['pearson'].mean():.4f}")
            print(f"    top10_overlap: mean={m['top10'].mean():.4f}   top20_overlap: mean={m['top20'].mean():.4f}")
            print(f"    per-candidate rank_std across seeds: mean={rank_std.mean():.4f} median={np.median(rank_std):.4f} "
                  f"p90={np.percentile(rank_std,90):.4f} max={rank_std.max():.4f}")
            print(f"    per-candidate score_cv across seeds: mean={score_cv.mean():.4f} median={np.median(score_cv):.4f} "
                  f"p90={np.percentile(score_cv,90):.4f} max={score_cv.max():.4f}")

    print("\n" + "-" * 88)
    print("PART C -- Independent vs CRN, side by side, at identical K")
    print("-" * 88)
    print(f"\n{'K':>3} {'Sp_ind':>8} {'Sp_crn':>8} {'d_Sp':>7}   {'t10_ind':>8} {'t10_crn':>8} {'d_t10':>7}   "
          f"{'t20_ind':>8} {'t20_crn':>8} {'d_t20':>7}")
    for K in K_LIST:
        si = results["independent"][K]["m"]["spearman"].mean()
        sc = results["crn"][K]["m"]["spearman"].mean()
        t10i = results["independent"][K]["m"]["top10"].mean()
        t10c = results["crn"][K]["m"]["top10"].mean()
        t20i = results["independent"][K]["m"]["top20"].mean()
        t20c = results["crn"][K]["m"]["top20"].mean()
        print(f"{K:>3} {si:>8.4f} {sc:>8.4f} {sc - si:>7.4f}   {t10i:>8.4f} {t10c:>8.4f} {t10c - t10i:>7.4f}   "
              f"{t20i:>8.4f} {t20c:>8.4f} {t20c - t20i:>7.4f}")

    return results


def part_d(cache, pairs):
    print("\n" + "=" * 88)
    print(f"PART D -- variance of pairwise candidate score differences ({len(pairs)} pairs)")
    print("=" * 88)
    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])

    ratio_stats_by_K = {}
    for K in K_LIST:
        var_ind_list = []
        var_crn_list = []
        for condition, store in [("independent", var_ind_list), ("crn", var_crn_list)]:
            vectors = torch.stack([r_K(cache, s, condition, K) for s in range(NUM_OUTER_SEEDS)], dim=0)  # (8,480)
            diffs = vectors[:, idx_i] - vectors[:, idx_j]  # (8, n_pairs)
            var = diffs.var(dim=0, unbiased=True).numpy()  # (n_pairs,)
            store.append(var)
        var_ind = var_ind_list[0]
        var_crn = var_crn_list[0]
        ratio = var_ind / np.clip(var_crn, 1e-30, None)
        ratio_stats_by_K[K] = ratio
        print(f"\n  K={K:>2}: Var_independent(diff) mean={var_ind.mean():.4g}   "
              f"Var_crn(diff) mean={var_crn.mean():.4g}")
        summarize(ratio, "variance-reduction ratio (ind/crn)")
        print(f"    fraction of pairs with ratio>1 (CRN reduces diff-variance): {(ratio > 1).mean():.4f}")

    return ratio_stats_by_K


def part_e(cache, pairs):
    print("\n" + "=" * 88)
    print("PART E -- high-compute consensus and score-margin reversal analysis")
    print("=" * 88)
    all_ind_rows = torch.cat([cache[(s, "independent")][:16] for s in range(NUM_OUTER_SEEDS)], dim=0)  # (128, 480)
    assert all_ind_rows.shape[0] == 128
    consensus = all_ind_rows.mean(dim=0)  # (480,)
    print(f"\n  consensus r_bar computed from {all_ind_rows.shape[0]} independent evaluations "
          f"(8 seeds x 16 banks). NOT ground truth -- a secondary diagnostic reference.")
    print(f"  consensus range: min={consensus.min().item():.6g} max={consensus.max().item():.6g}")

    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])
    delta = (consensus[idx_i] - consensus[idx_j]).abs().numpy()  # (n_pairs,)
    consensus_sign = np.sign((consensus[idx_i] - consensus[idx_j]).numpy())

    quantile_edges = np.quantile(delta, np.linspace(0, 1, 11))  # deciles
    bin_ids = np.clip(np.digitize(delta, quantile_edges[1:-1]), 0, 9)

    rev_k_list = [K for K in [1, 16, 32] if K in K_LIST]
    print(f"\n  reversal probability by consensus-margin decile (K={rev_k_list}, both conditions):")
    header = f"\n  {'decile':>6} {'delta_range':>22} {'n_pairs':>8}"
    for condition in ["independent", "crn"]:
        for K in rev_k_list:
            header += f" {condition[:3] + '_K' + str(K):>12}"
    print(header)

    rev_rates = {}
    for condition in ["independent", "crn"]:
        for K in rev_k_list:
            vectors = torch.stack([r_K(cache, s, condition, K) for s in range(NUM_OUTER_SEEDS)], dim=0)  # (8,480)
            diffs = vectors[:, idx_i] - vectors[:, idx_j]  # (8, n_pairs)
            seed_sign = torch.sign(diffs).numpy()  # (8, n_pairs)
            reversed_mask = seed_sign != consensus_sign[None, :]  # (8, n_pairs)
            rev_rates[(condition, K)] = reversed_mask

    for decile in range(10):
        mask = bin_ids == decile
        n = mask.sum()
        lo, hi = quantile_edges[decile], quantile_edges[decile + 1]
        row = f"  {decile:>6} [{lo:>9.3g},{hi:>9.3g}] {n:>8}"
        for condition in ["independent", "crn"]:
            for K in rev_k_list:
                rate = rev_rates[(condition, K)][:, mask].mean() if n > 0 else float("nan")
                row += f" {rate:>12.4f}"
        print(row)

    print("\n  overall reversal rate (all pairs, all seeds):")
    for condition in ["independent", "crn"]:
        for K in rev_k_list:
            rr = rev_rates[(condition, K)]
            print(f"    {condition:<12} K={K:<3} overall_reversal_rate={rr.mean():.4f}")

    print("\n  practical-K vs consensus (Spearman / top10 / top20), mean over the 8 seeds:")
    for condition in ["independent", "crn"]:
        print(f"\n  --- {condition} ---")
        for K in K_LIST:
            sps, t10s, t20s = [], [], []
            for s in range(NUM_OUTER_SEEDS):
                v = r_K(cache, s, condition, K)
                sps.append(setup.spearman_corr(v, consensus))
                t10s.append(setup.top_q_overlap(v, consensus, 0.10))
                t20s.append(setup.top_q_overlap(v, consensus, 0.20))
            print(f"    K={K:>2}: Spearman_vs_consensus mean={np.mean(sps):.4f}  "
                  f"top10_vs_consensus mean={np.mean(t10s):.4f}  top20_vs_consensus mean={np.mean(t20s):.4f}")

    return consensus, delta, rev_rates


def main():
    global K_LIST
    cache = load_cache()
    have_seeds = sorted(set(s for s, c in cache.keys()))
    have_conditions = sorted(set(c for s, c in cache.keys()))
    print(f"Loaded cache: seeds={have_seeds}  conditions={have_conditions}")
    for (s, c), v in sorted(cache.items()):
        print(f"  seed={s} condition={c}: {v.shape[0]} banks cached")
    assert set(have_seeds) == set(range(NUM_OUTER_SEEDS)), "missing outer seeds in cache"
    assert set(have_conditions) == {"independent", "crn"}, "missing condition in cache"

    K_LIST = extend_k_list_if_available(cache, K_LIST)
    print(f"K_LIST for this analysis run: {K_LIST}")

    rng = np.random.default_rng(PAIR_SEED)
    all_pairs = list(itertools.combinations(range(setup.NUM_CANDIDATES), 2))
    chosen = rng.choice(len(all_pairs), size=NUM_PAIRS, replace=False)
    pairs = [all_pairs[i] for i in chosen]
    print(f"\nSampled {len(pairs)} fixed candidate pairs (seed={PAIR_SEED}) for Parts D/E.")

    bc_results = part_bc(cache)
    d_results = part_d(cache, pairs)
    consensus, delta, rev_rates = part_e(cache, pairs)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
