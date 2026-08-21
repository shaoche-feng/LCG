#! /usr/bin/env python
"""
Simple-MC vs 3-stratum diagnostic -- offline analysis (Sections 6-12 of the spec).

Reads only the cached (seed, condition) -> (num_samples, 480) raw MC score tensor produced
by diagnose_lcg_simple_mc_compute.py, plus the PREVIOUS backward/VJP diagnostic's cached
(seed, condition) -> (num_banks, 480) stratified-CRN scores (reused unmodified, not
rerun, per Section 9). No GPU, no VJPs.

Reuses diagnose_lcg_backward_variance_analysis.py's r_K/pairwise_seed_metrics/
rank_std_and_cv_across_seeds/summarize/mean_std directly (they are generic over any
(seed,condition)->(rows,480) cache, regardless of what one "row" represents) so both
diagnostics are scored with identical methodology.

Usage:
    python scripts/diagnose_lcg_simple_mc_analysis.py
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

import numpy as np
import torch

import diagnose_lcg_backward_variance_analysis as prev_analysis
import diagnose_lcg_backward_variance_setup as setup
import diagnose_lcg_simple_mc_common as mc

NUM_OUTER_SEEDS = 8
M_LIST = [1, 2, 3, 6, 12]
NUM_PAIRS = 5000
PAIR_SEED = 42  # same seed as the previous diagnostic -> same fixed pair set

SIMPLE_MC_CACHE_PATH = setup.DIAG_DIR / "simple_mc_raw_scores.pt"
STRATIFIED_CACHE_PATH = setup.DIAG_DIR / "partBC_raw_scores.pt"

# Section 9: equal-VJP-cost matching between simple MC (1 VJP/sample) and the stratified
# CRN estimator (3 VJPs/bank).
EQUAL_COST_PAIRS = [(3, 1), (6, 2), (12, 4)]  # (simple_mc_M, stratified_K)


def extend_m_list_if_available(cache, m_list):
    min_samples = min(v.shape[0] for v in cache.values())
    if min_samples >= 24 and 24 not in m_list:
        return m_list + [24]
    return m_list


def rank_shift_stats(a: torch.Tensor, b: torch.Tensor):
    ra, rb = setup._rank(a), setup._rank(b)
    abs_shift = (ra - rb).abs().numpy()
    return float(np.median(abs_shift)), float(np.percentile(abs_shift, 90))


# --------------------------------------------------------------------------------------
# Section 6: cross-seed ranking consistency (delegates to the previous diagnostic's code)
# --------------------------------------------------------------------------------------


def section6(cache):
    print("\n" + "=" * 88)
    print("SECTION 6 -- cross-seed ranking consistency, per condition and M")
    print("=" * 88)
    results = {}
    for condition in mc.CONDITIONS:
        print(f"\n--- condition: {condition} ---")
        results[condition] = {}
        for M in M_LIST:
            m = prev_analysis.pairwise_seed_metrics(cache, condition, M)
            rank_std, score_cv = prev_analysis.rank_std_and_cv_across_seeds(m["vectors"])
            results[condition][M] = dict(m=m, rank_std=rank_std, score_cv=score_cv)
            print(f"\n  M={M:>2}  (n_seed_pairs={len(m['spearman'])})")
            print(f"    Spearman: mean={m['spearman'].mean():.4f} std={m['spearman'].std():.4f} "
                  f"median={np.median(m['spearman']):.4f} p10={np.percentile(m['spearman'],10):.4f} "
                  f"p90={np.percentile(m['spearman'],90):.4f} min={m['spearman'].min():.4f}")
            print(f"    Pearson:  mean={m['pearson'].mean():.4f}")
            print(f"    top10_overlap: mean={m['top10'].mean():.4f}   top20_overlap: mean={m['top20'].mean():.4f}")
            print(f"    per-candidate rank_std across seeds: mean={rank_std.mean():.4f} median={np.median(rank_std):.4f}")
            print(f"    per-candidate score_cv across seeds: mean={score_cv.mean():.4f} median={np.median(score_cv):.4f}")
    return results


# --------------------------------------------------------------------------------------
# Section 7: cross-M ranking convergence (within seed, within condition)
# --------------------------------------------------------------------------------------


def section7(cache):
    print("\n" + "=" * 88)
    print("SECTION 7 -- cross-M ranking convergence (within seed/condition)")
    print("=" * 88)
    m_pairs = [(1, 2), (1, 3), (1, 6), (1, 12), (2, 3), (3, 6), (6, 12)]
    m_pairs = [(a, b) for a, b in m_pairs if a in M_LIST and b in M_LIST]

    results = {}
    for condition in mc.CONDITIONS:
        print(f"\n--- condition: {condition} ---")
        results[condition] = {}
        for m1, m2 in m_pairs:
            sps, t10s, t20s, med_shifts, p90_shifts = [], [], [], [], []
            for s in range(NUM_OUTER_SEEDS):
                a = prev_analysis.r_K(cache, s, condition, m1)
                b = prev_analysis.r_K(cache, s, condition, m2)
                sps.append(setup.spearman_corr(a, b))
                t10s.append(setup.top_q_overlap(a, b, 0.10))
                t20s.append(setup.top_q_overlap(a, b, 0.20))
                med, p90 = rank_shift_stats(a, b)
                med_shifts.append(med)
                p90_shifts.append(p90)
            results[condition][(m1, m2)] = dict(spearman=np.mean(sps), top10=np.mean(t10s), top20=np.mean(t20s),
                                                 median_shift=np.mean(med_shifts), p90_shift=np.mean(p90_shifts))
            print(f"  M={m1:>2} <-> M={m2:>2}: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  "
                  f"top20={np.mean(t20s):.4f}  median_rank_shift={np.mean(med_shifts):.1f}  "
                  f"p90_rank_shift={np.mean(p90_shifts):.1f}")
    return results


# --------------------------------------------------------------------------------------
# Section 8: disjoint-sample cross-M check
# --------------------------------------------------------------------------------------


def section8(cache):
    print("\n" + "=" * 88)
    print("SECTION 8 -- disjoint-sample cross-M check (no shared samples between A and B)")
    print("=" * 88)
    disjoint_specs = [
        ("M1_vs_M3_disjoint", slice(0, 1), slice(1, 4)),
        ("M3_vs_M6_disjoint", slice(0, 3), slice(3, 9)),
        ("M6_vs_M6_splithalf", slice(0, 6), slice(6, 12)),
    ]
    results = {}
    for condition in mc.CONDITIONS:
        print(f"\n--- condition: {condition} ---")
        results[condition] = {}
        for name, sl_a, sl_b in disjoint_specs:
            sps, t10s, t20s = [], [], []
            for s in range(NUM_OUTER_SEEDS):
                rows = cache[(s, condition)]
                a = rows[sl_a].mean(dim=0)
                b = rows[sl_b].mean(dim=0)
                sps.append(setup.spearman_corr(a, b))
                t10s.append(setup.top_q_overlap(a, b, 0.10))
                t20s.append(setup.top_q_overlap(a, b, 0.20))
            results[condition][name] = dict(spearman=np.mean(sps), top10=np.mean(t10s), top20=np.mean(t20s))
            print(f"  {name:<22}: Spearman={np.mean(sps):.4f}  top10={np.mean(t10s):.4f}  top20={np.mean(t20s):.4f}")
    return results


# --------------------------------------------------------------------------------------
# Section 9: comparison against the existing 3-stratum estimator (cached, not rerun)
# --------------------------------------------------------------------------------------


def section9(mc_cache, strat_cache, section6_results):
    print("\n" + "=" * 88)
    print("SECTION 9 -- simple MC (full_crn) vs stratified CRN, equal VJP cost")
    print("=" * 88)
    results = {}
    for M, K in EQUAL_COST_PAIRS:
        print(f"\n--- {3 * K} VJPs/candidate: MC full_crn M={M}  vs  Stratified CRN K={K} ---")
        mc_m = section6_results["full_crn"][M]["m"]
        strat_m = prev_analysis.pairwise_seed_metrics(strat_cache, "crn", K)

        print(f"  cross-seed Spearman: MC={mc_m['spearman'].mean():.4f}   Stratified={strat_m['spearman'].mean():.4f}")
        print(f"  cross-seed top10:    MC={mc_m['top10'].mean():.4f}   Stratified={strat_m['top10'].mean():.4f}")
        print(f"  cross-seed top20:    MC={mc_m['top20'].mean():.4f}   Stratified={strat_m['top20'].mean():.4f}")

        # direct ranking agreement BETWEEN the two methods (seed-averaged point estimate each)
        mc_avg = torch.stack([prev_analysis.r_K(mc_cache, s, "full_crn", M) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        strat_avg = torch.stack([prev_analysis.r_K(strat_cache, s, "crn", K) for s in range(NUM_OUTER_SEEDS)]).mean(dim=0)
        sp = setup.spearman_corr(mc_avg, strat_avg)
        t10 = setup.top_q_overlap(mc_avg, strat_avg, 0.10)
        t20 = setup.top_q_overlap(mc_avg, strat_avg, 0.20)
        med_shift, p90_shift = rank_shift_stats(mc_avg, strat_avg)
        print(f"  BETWEEN-METHOD agreement (seed-averaged rankings): Spearman={sp:.4f}  top10={t10:.4f}  "
              f"top20={t20:.4f}  median_rank_shift={med_shift:.1f}  p90_rank_shift={p90_shift:.1f}")

        results[(M, K)] = dict(mc_spearman=mc_m["spearman"].mean(), strat_spearman=strat_m["spearman"].mean(),
                                between_method_spearman=sp, between_method_top10=t10, between_method_top20=t20)
    return results


# --------------------------------------------------------------------------------------
# Section 10: M=1 three-way comparison
# --------------------------------------------------------------------------------------


def section10(section6_results):
    print("\n" + "=" * 88)
    print("SECTION 10 -- M=1 three-way comparison (all cost exactly 1 VJP/candidate)")
    print("=" * 88)
    sp = {c: section6_results[c][1]["m"]["spearman"].mean() for c in mc.CONDITIONS}
    t10 = {c: section6_results[c][1]["m"]["top10"].mean() for c in mc.CONDITIONS}
    print(f"  {'condition':<14} {'Spearman':>10} {'top10':>8}")
    for c in mc.CONDITIONS:
        print(f"  {c:<14} {sp[c]:>10.4f} {t10[c]:>8.4f}")
    print(f"\n  xi-sharing benefit (independent -> xi_crn): d_Spearman={sp['xi_crn'] - sp['independent']:.4f}  "
          f"d_top10={t10['xi_crn'] - t10['independent']:.4f}")
    print(f"  additional sigma/eps-sharing benefit (xi_crn -> full_crn): d_Spearman={sp['full_crn'] - sp['xi_crn']:.4f}  "
          f"d_top10={t10['full_crn'] - t10['xi_crn']:.4f}")
    return dict(spearman=sp, top10=t10)


# --------------------------------------------------------------------------------------
# Section 11: pairwise candidate-difference variance
# --------------------------------------------------------------------------------------


def section11(cache, pairs):
    print("\n" + "=" * 88)
    print(f"SECTION 11 -- variance of pairwise candidate score differences ({len(pairs)} pairs)")
    print("=" * 88)
    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])

    var_by_condition_M = {}
    for M in [1, 3, 6, 12]:
        if M not in M_LIST:
            continue
        print(f"\n--- M={M} ---")
        vars_this_m = {}
        for condition in mc.CONDITIONS:
            vectors = torch.stack([prev_analysis.r_K(cache, s, condition, M) for s in range(NUM_OUTER_SEEDS)], dim=0)
            diffs = vectors[:, idx_i] - vectors[:, idx_j]
            var = diffs.var(dim=0, unbiased=True).numpy()
            vars_this_m[condition] = var
            print(f"  Var_{condition}(diff): mean={var.mean():.4g}  median={np.median(var):.4g}")
        ratio_ind_xi = vars_this_m["independent"] / np.clip(vars_this_m["xi_crn"], 1e-30, None)
        ratio_xi_full = vars_this_m["xi_crn"] / np.clip(vars_this_m["full_crn"], 1e-30, None)
        print(f"  ratio Var_independent/Var_xi_crn: mean={ratio_ind_xi.mean():.4f}  median={np.median(ratio_ind_xi):.4f}")
        print(f"  ratio Var_xi_crn/Var_full_crn:    mean={ratio_xi_full.mean():.4f}  median={np.median(ratio_xi_full):.4f}")
        var_by_condition_M[M] = vars_this_m
    return var_by_condition_M


# --------------------------------------------------------------------------------------
# Section 12: minimal margin-reversal analysis (tertiles, not deciles)
# --------------------------------------------------------------------------------------


def section12(cache, pairs):
    print("\n" + "=" * 88)
    print("SECTION 12 -- minimal margin-reversal analysis (secondary diagnostic, NOT ground truth)")
    print("=" * 88)
    all_ind_rows = torch.cat([cache[(s, "independent")] for s in range(NUM_OUTER_SEEDS)], dim=0)
    consensus = all_ind_rows.mean(dim=0)
    print(f"\n  consensus built from {all_ind_rows.shape[0]} pooled 'independent' evaluations "
          f"({NUM_OUTER_SEEDS} seeds x {all_ind_rows.shape[0] // NUM_OUTER_SEEDS} samples). NOT ground truth.")

    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])
    delta = (consensus[idx_i] - consensus[idx_j]).abs().numpy()
    consensus_sign = np.sign((consensus[idx_i] - consensus[idx_j]).numpy())

    tertile_edges = np.quantile(delta, [0.0, 1 / 3, 2 / 3, 1.0])
    bin_ids = np.clip(np.digitize(delta, tertile_edges[1:-1]), 0, 2)
    bin_names = ["low-margin", "mid-margin", "high-margin"]

    configs = [("independent", 1), ("xi_crn", 1), ("full_crn", 1), ("full_crn", 3)]
    print(f"\n  {'config':<18} {'low-margin':>11} {'mid-margin':>11} {'high-margin':>12} {'overall':>9}")
    results = {}
    for condition, M in configs:
        vectors = torch.stack([prev_analysis.r_K(cache, s, condition, M) for s in range(NUM_OUTER_SEEDS)], dim=0)
        diffs = vectors[:, idx_i] - vectors[:, idx_j]
        seed_sign = torch.sign(diffs).numpy()
        reversed_mask = seed_sign != consensus_sign[None, :]

        rates = []
        for b in range(3):
            mask = bin_ids == b
            rates.append(reversed_mask[:, mask].mean())
        overall = reversed_mask.mean()
        results[(condition, M)] = dict(rates=rates, overall=overall)
        label = f"{condition}_M{M}"
        print(f"  {label:<18} {rates[0]:>11.4f} {rates[1]:>11.4f} {rates[2]:>12.4f} {overall:>9.4f}")
    return results


# --------------------------------------------------------------------------------------


def main():
    mc_cache = torch.load(SIMPLE_MC_CACHE_PATH, map_location="cpu", weights_only=True)
    strat_cache = torch.load(STRATIFIED_CACHE_PATH, map_location="cpu", weights_only=True)

    have_seeds = sorted(set(s for s, c in mc_cache.keys()))
    have_conditions = sorted(set(c for s, c in mc_cache.keys()))
    print(f"Loaded simple-MC cache: seeds={have_seeds}  conditions={have_conditions}")
    for (s, c), v in sorted(mc_cache.items()):
        print(f"  seed={s} condition={c}: {v.shape[0]} samples cached")
    assert set(have_seeds) == set(range(NUM_OUTER_SEEDS))
    assert set(have_conditions) == set(mc.CONDITIONS)

    strat_seeds = sorted(set(s for s, c in strat_cache.keys()))
    strat_conditions = sorted(set(c for s, c in strat_cache.keys()))
    print(f"\nLoaded stratified (previous-diagnostic) cache: seeds={strat_seeds}  conditions={strat_conditions}")
    for (s, c), v in sorted(strat_cache.items()):
        if c == "crn":
            assert v.shape[0] >= 4, "need at least K=4 stratified CRN banks for Section 9's M=12 vs K=4 comparison"

    global M_LIST
    M_LIST = extend_m_list_if_available(mc_cache, M_LIST)
    print(f"\nM_LIST for this analysis run: {M_LIST}")

    rng = np.random.default_rng(PAIR_SEED)
    all_pairs = list(itertools.combinations(range(setup.NUM_CANDIDATES), 2))
    chosen = rng.choice(len(all_pairs), size=NUM_PAIRS, replace=False)
    pairs = [all_pairs[i] for i in chosen]
    print(f"Sampled {len(pairs)} fixed candidate pairs (seed={PAIR_SEED}, same seed as previous diagnostic).")

    s6 = section6(mc_cache)
    s7 = section7(mc_cache)
    s8 = section8(mc_cache)
    s9 = section9(mc_cache, strat_cache, s6)
    s10 = section10(s6)
    s11 = section11(mc_cache, pairs)
    s12 = section12(mc_cache, pairs)

    print("\nAnalysis complete.")


if __name__ == "__main__":
    main()
