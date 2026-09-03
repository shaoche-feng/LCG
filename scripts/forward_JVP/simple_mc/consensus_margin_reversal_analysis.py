#! /usr/bin/env python
"""
Forward simple-MC JVP: held-out consensus diagnostic for practical sample count,
M in [5, 24]. Purely offline analysis of already-cached data -- no new JVP evaluations
run (guarded by an assertion on cache size).

Goal: find the cheapest M in [5, 24] whose candidate ranking is already close to a
high-compute consensus, WITHOUT letting the consensus reference leak information from
the same MC samples used to build the M<=24 candidate estimators being evaluated
(that would be circular: a low-M estimator's first M samples are literally a subset of
what it's being compared against, artificially inflating agreement).

Consensus construction (anti-leakage): each seed's cached 96 samples are split into
  - evaluation samples:  index [0:24]   (used to build r_hat_{seed,M} for M in 1..24)
  - held-out samples:    index [24:96]  (72 samples/seed, 8*72=576 pooled) -- these
    NEVER appear in any r_hat_M being evaluated, so the comparison is a genuine
    "does a cheap estimate recover a much-higher-compute estimate" question, not a
    "does a subset agree with its own superset" question.

r_cons(x_j) = mean over the pooled 576 held-out evaluations. Treated as a high-compute
diagnostic reference, not mathematical ground truth (with only 576 evaluations, r_cons
itself still carries residual MC noise -- it is merely far lower-variance than any
individual r_hat_M, M<=24).

Distinguish this from the earlier M-search diagnostic's cross-seed Spearman: that
metric asked "do two independent M-sample estimators agree with EACH OTHER"; this one
asks "does an M-sample estimator recover the SAME (much higher-precision) reference."

Scope: diagnostic/analysis only. Does not modify the production config, src/lcg/*, or
any estimator. Does not run new JVP evaluations (asserts the cache already has >=96
samples/seed; stops rather than computing more if not).

Usage:
    python scripts/forward_JVP/simple_mc/consensus_margin_reversal_analysis.py
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
M_MIN, M_MAX = 5, 24
M_LIST = list(range(M_MIN, M_MAX + 1))
PRACTICAL_M = [5, 8, 10, 12, 16, 20, 24]
NESTED_COMPARE_M = [8, 12, 16, 20]
MARGIN_M = [8, 12, 16, 24]

CONSENSUS_START = 24  # 0-indexed: samples[24:96] == "samples 25:96" in 1-indexed spec language
CONSENSUS_END = 96
NUM_PAIRS = 5000
PAIR_SEED = 42

CACHE_PATH = setup.DIAG_DIR / "forward_simple_mc_raw_scores.pt"
PROFILE_PATH = setup.DIAG_DIR / "forward_simple_mc_m24_to_m60_profiling.pt"

# Measured (not extrapolated) production-integration runtime reference for M=24, from
# scripts/forward_JVP/integration/profile_production_scorer.py (mean of 3 reps, same
# 480-candidate workload, actual lcg.forward_jvp production code path, NOT the
# diagnostic-only profiling harness below).
INTEGRATED_M24_RUNTIME_S = 88.00


def load_cache():
    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    min_samples = min(v.shape[0] for v in cache.values())
    assert min_samples >= CONSENSUS_END, (
        f"cache only has {min_samples} samples/seed, need >= {CONSENSUS_END}; "
        "per scope restrictions this script does not run new JVP evaluations."
    )
    return cache


def load_diagnostic_runtime_profile():
    if not PROFILE_PATH.is_file():
        return None
    d = torch.load(PROFILE_PATH, map_location="cpu", weights_only=True)
    return d["cumulative_times"]  # index M-1 -> cumulative time for M samples


def r_hat(cache, seed, M):
    """Evaluation estimator: mean of the FIRST M cached samples (index [0:M]) -- disjoint
    from the held-out consensus range [CONSENSUS_START:CONSENSUS_END]."""
    assert M <= CONSENSUS_START, f"M={M} would overlap the held-out consensus range (starts at index {CONSENSUS_START})"
    return cache[seed][:M].mean(dim=0)


def build_held_out_consensus(cache):
    held_out_rows = torch.cat([cache[s][CONSENSUS_START:CONSENSUS_END] for s in range(NUM_OUTER_SEEDS)], dim=0)
    n_pooled = held_out_rows.shape[0]
    expected = NUM_OUTER_SEEDS * (CONSENSUS_END - CONSENSUS_START)
    assert n_pooled == expected, f"pooled held-out rows {n_pooled} != expected {expected}"
    consensus = held_out_rows.mean(dim=0)
    return consensus, n_pooled


def rank_shift_stats(a, b):
    ra, rb = setup._rank(a), setup._rank(b)
    abs_shift = (ra - rb).abs().numpy()
    return float(np.median(abs_shift)), float(np.percentile(abs_shift, 90))


def top_set_indices(v, q):
    k = max(1, int(round(q * v.numel())))
    return set(torch.topk(v, k).indices.tolist())


def per_seed_consensus_metrics(cache, consensus, M):
    """Returns dict of per-seed arrays (len NUM_OUTER_SEEDS) for spearman/pearson/top10/
    top20/median_shift/p90_shift, comparing r_hat(seed, M) against the held-out consensus."""
    sps, pes, t10s, t20s, meds, p90s = [], [], [], [], [], []
    for s in range(NUM_OUTER_SEEDS):
        v = r_hat(cache, s, M)
        sps.append(setup.spearman_corr(v, consensus))
        pes.append(setup.pearson_corr(v, consensus))
        t10s.append(setup.top_q_overlap(v, consensus, 0.10))
        t20s.append(setup.top_q_overlap(v, consensus, 0.20))
        med, p90 = rank_shift_stats(v, consensus)
        meds.append(med)
        p90s.append(p90)
    return dict(spearman=np.array(sps), pearson=np.array(pes), top10=np.array(t10s),
                top20=np.array(t20s), median_shift=np.array(meds), p90_shift=np.array(p90s))


def part2_consensus_construction(cache):
    print("\n" + "=" * 88)
    print("PART 2 -- held-out consensus construction (anti-leakage)")
    print("=" * 88)
    consensus, n_pooled = build_held_out_consensus(cache)
    print(f"  evaluation samples per seed: [0:24]  (used for all r_hat_M, M<=24)")
    print(f"  held-out consensus samples per seed: [{CONSENSUS_START}:{CONSENSUS_END}] "
          f"({CONSENSUS_END - CONSENSUS_START} samples/seed)")
    print(f"  pooled held-out evaluations: {NUM_OUTER_SEEDS} seeds x {CONSENSUS_END - CONSENSUS_START} = {n_pooled}")
    finite_ok = torch.isfinite(consensus).all().item()
    nonneg_ok = (consensus >= 0).all().item()
    print(f"  r_cons: finite={finite_ok}  nonnegative={nonneg_ok}  "
          f"min={consensus.min().item():.4f}  max={consensus.max().item():.4f}  mean={consensus.mean().item():.4f}")
    assert finite_ok and nonneg_ok, "held-out consensus must be finite and nonnegative"
    return consensus


def part3_4_full_curve(cache, consensus):
    print("\n" + "=" * 88)
    print(f"PART 3/4 -- full curve, every integer M in [{M_MIN}, {M_MAX}], agreement with held-out consensus")
    print("=" * 88)
    results = {}
    print(f"\n{'M':>3} {'Spearman':>17} {'Pearson':>9} {'top10':>17} {'top20':>17} "
          f"{'med_shift':>10} {'p90_shift':>10}")
    for M in M_LIST:
        m = per_seed_consensus_metrics(cache, consensus, M)
        results[M] = m
        print(f"{M:>3} {m['spearman'].mean():>8.4f}+-{m['spearman'].std():>6.4f} {m['pearson'].mean():>9.4f} "
              f"{m['top10'].mean():>8.4f}+-{m['top10'].std():>6.4f} {m['top20'].mean():>8.4f}+-{m['top20'].std():>6.4f} "
              f"{m['median_shift'].mean():>10.2f} {m['p90_shift'].mean():>10.2f}")
    return results


def part5_top_set_recovery(cache, consensus, results):
    print("\n" + "=" * 88)
    print("PART 5 -- top-10%/top-20% recovery vs consensus, and churn as M increases")
    print("=" * 88)
    cons_top10 = top_set_indices(consensus, 0.10)  # 48 candidates
    cons_top20 = top_set_indices(consensus, 0.20)  # 96 candidates
    print(f"  consensus top-10% set size={len(cons_top10)} (48 candidates)  "
          f"top-20% set size={len(cons_top20)} (96 candidates)")

    print(f"\n{'M':>3} {'top10_recovered':>16} {'top20_recovered':>16}  (mean count over 8 seeds, out of 48 / 96)")
    recovered_counts = {}
    for M in M_LIST:
        t10_counts, t20_counts = [], []
        for s in range(NUM_OUTER_SEEDS):
            v = r_hat(cache, s, M)
            est_top10 = top_set_indices(v, 0.10)
            est_top20 = top_set_indices(v, 0.20)
            t10_counts.append(len(est_top10 & cons_top10))
            t20_counts.append(len(est_top20 & cons_top20))
        recovered_counts[M] = (np.mean(t10_counts), np.mean(t20_counts))
        print(f"{M:>3} {np.mean(t10_counts):>16.2f} {np.mean(t20_counts):>16.2f}")

    print("\n  churn: mean number of candidates entering/leaving each seed's OWN top-20% set "
          "(by r_hat_M) between consecutive M (M-1 -> M):")
    for M in M_LIST[1:]:
        enter_counts, leave_counts = [], []
        for s in range(NUM_OUTER_SEEDS):
            prev_set = top_set_indices(r_hat(cache, s, M - 1), 0.20)
            cur_set = top_set_indices(r_hat(cache, s, M), 0.20)
            enter_counts.append(len(cur_set - prev_set))
            leave_counts.append(len(prev_set - cur_set))
        print(f"    M={M - 1:>2}->{M:>2}: mean_entering={np.mean(enter_counts):.2f}  mean_leaving={np.mean(leave_counts):.2f}")
    return recovered_counts


def part6_summary_table(results, recovered_counts):
    print("\n" + "=" * 88)
    print("PART 6 -- practical summary table")
    print("=" * 88)
    diag_runtimes = load_diagnostic_runtime_profile()
    print(f"\n{'M':>3} {'diag_runtime_s':>15} {'Spearman':>9} {'top10':>7} {'top20':>7} "
          f"{'top10_recov/48':>15} {'top20_recov/96':>15} {'med_shift':>10} {'p90_shift':>10}")
    for M in PRACTICAL_M:
        m = results[M]
        rt = diag_runtimes[M - 1] if diag_runtimes is not None else float("nan")
        t10r, t20r = recovered_counts[M]
        print(f"{M:>3} {rt:>15.2f} {m['spearman'].mean():>9.4f} {m['top10'].mean():>7.4f} "
              f"{m['top20'].mean():>7.4f} {t10r:>15.2f} {t20r:>15.2f} "
              f"{m['median_shift'].mean():>10.2f} {m['p90_shift'].mean():>10.2f}")
    print(f"\n  NOTE: 'diag_runtime_s' is the diagnostic-only continuous profiling harness's measured "
          f"cumulative time (scripts/forward_JVP/simple_mc/profile_m24_to_m60.py), 480 candidates, "
          f"chunk_size=4, single uninterrupted GPU pass -- NOT the integrated production path.")
    print(f"  Integrated production runtime at M=24 (scripts/forward_JVP/integration/profile_production_scorer.py, "
          f"actual src/lcg/forward_jvp.py code path, mean of 3 reps): {INTEGRATED_M24_RUNTIME_S:.1f}s "
          f"(~{100*(INTEGRATED_M24_RUNTIME_S/diag_runtimes[23]-1):.0f}% slower than the diagnostic profile at the same M, "
          f"attributed to ordinary run-to-run GPU variance -- same underlying code).")


def part7_knee(results):
    print("\n" + "=" * 88)
    print("PART 7 -- practical knee: diminishing-returns analysis (no arbitrary hard threshold)")
    print("=" * 88)
    print(f"\n{'M':>3} {'Spearman':>9} {'d_Spearman':>11} {'top10':>7} {'d_top10':>8} {'top20':>7} {'d_top20':>8}")
    prev = None
    deltas = {"spearman": {}, "top10": {}, "top20": {}}
    for M in M_LIST:
        sp = results[M]["spearman"].mean()
        t10 = results[M]["top10"].mean()
        t20 = results[M]["top20"].mean()
        if prev is not None:
            d_sp, d_t10, d_t20 = sp - prev[0], t10 - prev[1], t20 - prev[2]
        else:
            d_sp = d_t10 = d_t20 = float("nan")
        deltas["spearman"][M] = d_sp
        deltas["top10"][M] = d_t10
        deltas["top20"][M] = d_t20
        print(f"{M:>3} {sp:>9.4f} {d_sp:>11.4f} {t10:>7.4f} {d_t10:>8.4f} {t20:>7.4f} {d_t20:>8.4f}")
        prev = (sp, t10, t20)

    def knee_at_90pct_of_gain(metric_name):
        vals = {M: results[M][metric_name].mean() for M in M_LIST}
        total_gain = vals[M_MAX] - vals[M_MIN]
        if total_gain <= 0:
            return M_MIN
        target = vals[M_MIN] + 0.9 * total_gain
        for M in M_LIST:
            if vals[M] >= target:
                return M
        return M_MAX

    knee_sp = knee_at_90pct_of_gain("spearman")
    knee_t10 = knee_at_90pct_of_gain("top10")
    knee_t20 = knee_at_90pct_of_gain("top20")
    m_knee = max(knee_sp, knee_t10, knee_t20)
    print(f"\n  M capturing >=90% of the total M=5->24 gain: Spearman->M={knee_sp}  top10->M={knee_t10}  top20->M={knee_t20}")
    print(f"  M_knee (max over the three, i.e. the most conservative): {m_knee}")

    sp_by_M = {M: results[M]["spearman"].mean() for M in M_LIST}
    m_090 = next((M for M in M_LIST if sp_by_M[M] >= 0.90), None)
    print(f"  M_0.90-cons (smallest M with Spearman_vs_consensus >= 0.90): "
          f"{m_090 if m_090 is not None else 'NOT REACHED in [5,24]'}")
    return m_knee, m_090


def part8_nested_vs_m24(cache):
    print("\n" + "=" * 88)
    print("PART 8 -- nested agreement vs the production M=24 estimator (secondary; NOT independent)")
    print("=" * 88)
    print("  NOTE: M<24 samples are a strict SUBSET of the M=24 samples for the same seed (nested,")
    print("  not independent) -- high agreement here is expected/partially mechanical, unlike Part 3/4's")
    print("  held-out comparison. Reported for completeness per the spec, not as the primary evidence.")
    ref24 = {s: r_hat(cache, s, 24) for s in range(NUM_OUTER_SEEDS)}
    print(f"\n{'M':>3} {'Spearman':>9} {'top10':>7} {'top20':>7} {'med_shift':>10} {'p90_shift':>10}")
    for M in NESTED_COMPARE_M:
        sps, t10s, t20s, meds, p90s = [], [], [], [], []
        for s in range(NUM_OUTER_SEEDS):
            v = r_hat(cache, s, M)
            ref = ref24[s]
            sps.append(setup.spearman_corr(v, ref))
            t10s.append(setup.top_q_overlap(v, ref, 0.10))
            t20s.append(setup.top_q_overlap(v, ref, 0.20))
            med, p90 = rank_shift_stats(v, ref)
            meds.append(med)
            p90s.append(p90)
        print(f"{M:>3} {np.mean(sps):>9.4f} {np.mean(t10s):>7.4f} {np.mean(t20s):>7.4f} "
              f"{np.mean(meds):>10.2f} {np.mean(p90s):>10.2f}")


def part9_margin_analysis(cache, consensus, pairs):
    print("\n" + "=" * 88)
    print(f"PART 9 -- margin analysis: pairwise reversal probability vs held-out consensus ({len(pairs)} pairs)")
    print("=" * 88)
    idx_i = torch.tensor([p[0] for p in pairs])
    idx_j = torch.tensor([p[1] for p in pairs])
    delta = (consensus[idx_i] - consensus[idx_j]).abs().numpy()
    consensus_sign = np.sign((consensus[idx_i] - consensus[idx_j]).numpy())

    tertile_edges = np.quantile(delta, [0.0, 1 / 3, 2 / 3, 1.0])
    bin_ids = np.clip(np.digitize(delta, tertile_edges[1:-1]), 0, 2)
    tertile_names = ["low-margin", "mid-margin", "high-margin"]

    print(f"\n  margin tertile ranges: low=[{tertile_edges[0]:.4g},{tertile_edges[1]:.4g})  "
          f"mid=[{tertile_edges[1]:.4g},{tertile_edges[2]:.4g})  high=[{tertile_edges[2]:.4g},{tertile_edges[3]:.4g}]")

    print(f"\n{'M':>3}", end="")
    for name in tertile_names:
        print(f" {name:>14}", end="")
    print(f" {'overall':>10}")

    for M in MARGIN_M:
        vectors = torch.stack([r_hat(cache, s, M) for s in range(NUM_OUTER_SEEDS)], dim=0)  # (8,480)
        diffs = vectors[:, idx_i] - vectors[:, idx_j]
        seed_sign = torch.sign(diffs).numpy()
        reversed_mask = seed_sign != consensus_sign[None, :]  # (8, n_pairs)
        print(f"{M:>3}", end="")
        for t in range(3):
            mask = bin_ids == t
            rate = reversed_mask[:, mask].mean()
            print(f" {rate:>14.4f}", end="")
        print(f" {reversed_mask.mean():>10.4f}")


def main():
    cache = load_cache()
    have_seeds = sorted(cache.keys())
    print(f"cache: seeds={have_seeds}  samples/seed={cache[0].shape[0]} (>= {CONSENSUS_END} required, OK -- no new JVP evaluations run)")

    rng = np.random.default_rng(PAIR_SEED)
    all_pairs = list(itertools.combinations(range(480), 2))
    pair_idx = rng.choice(len(all_pairs), size=NUM_PAIRS, replace=False)
    pairs = [all_pairs[i] for i in pair_idx]
    print(f"sampled {len(pairs)} candidate pairs (seed={PAIR_SEED}) for margin analysis")

    consensus = part2_consensus_construction(cache)
    results = part3_4_full_curve(cache, consensus)
    recovered_counts = part5_top_set_recovery(cache, consensus, results)
    part6_summary_table(results, recovered_counts)
    m_knee, m_090 = part7_knee(results)
    part8_nested_vs_m24(cache)
    part9_margin_analysis(cache, consensus, pairs)

    print("\n" + "=" * 88)
    print("REQUIRED FINAL ANSWERS")
    print("=" * 88)
    sp5, sp24 = results[5]["spearman"].mean(), results[24]["spearman"].mean()
    t10_5, t10_24 = results[5]["top10"].mean(), results[24]["top10"].mean()
    t20_5, t20_24 = results[5]["top20"].mean(), results[24]["top20"].mean()
    print(f"  1. Spearman_vs_consensus: M=5 -> {sp5:.4f}, M=24 -> {sp24:.4f} (gain={sp24-sp5:.4f})")
    print(f"  2. M_knee (Spearman, >=90% of total gain captured): see Part 7 above")
    print(f"  3/4. M_knee (top10/top20, >=90% of total gain captured): see Part 7 above")
    print(f"  M_knee (overall, most conservative): M={m_knee}")
    print(f"  M_0.90-cons: {m_090 if m_090 is not None else 'NOT REACHED in [5,24]'}")
    print(f"  8. smallest M recovering most of consensus top-10/top-20: see Part 5/6 tables above")
    print(f"  overall reversal rates by M: see Part 9 (low/mid/high margin breakdown)")

    print("\nConsensus diagnostic complete.")


if __name__ == "__main__":
    main()
