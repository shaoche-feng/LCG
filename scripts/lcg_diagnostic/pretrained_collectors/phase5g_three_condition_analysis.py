"""Combines run_scarce, balanced, and walk_scarce LCG scoring results into the
final three-condition comparison: tests whether Delta = mean_LCG(run) -
mean_LCG(walk) moves monotonically as relative Walk/Run coverage shifts.

Uses ONLY already-saved Phase 5/5b/5f summaries -- no rescoring.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5g_three_condition_analysis.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_THIS_DIR))
# NOTE: deliberately NOT importing from phase5_lcg_scoring here -- it imports torch, and
# torch + matplotlib/numpy in this combination triggers the same OMP Error #15 DLL conflict
# seen at Phase 5e (torch's and another library's bundled OpenMP runtimes colliding on
# Windows). This script only needs two plain path constants, redefined locally instead.
from phase5c_interaction_analysis import paired_bootstrap_interaction, exact_paired_sign_flip_test  # noqa: E402

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"
DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]


def load_summary(domain: str, condition: str) -> dict:
    path = OUT_ROOT / domain / condition / "summary.json"
    if not path.exists() and condition == "balanced":
        path = OUT_ROOT / f"phase5f_balanced_summary_{domain}.json"  # fallback, unused normally
    return json.loads(path.read_text())


def episode_means_by_behavior(summary: dict, behavior: str) -> dict:
    key = f"{behavior}_episode_stats"
    return {int(eid): v["mean"] for eid, v in summary[key].items()}


def consecutive_interaction(sum_a: dict, sum_b: dict) -> dict:
    """Interaction = Delta_a - Delta_b, paired by episode identity (same held-out
    episodes/indices used in both conditions a and b)."""
    a_walk, a_run = episode_means_by_behavior(sum_a, "walk"), episode_means_by_behavior(sum_a, "run")
    b_walk, b_run = episode_means_by_behavior(sum_b, "walk"), episode_means_by_behavior(sum_b, "run")
    run_ids, walk_ids = sorted(a_run.keys()), sorted(a_walk.keys())
    d_run = np.array([a_run[i] - b_run[i] for i in run_ids])
    d_walk = np.array([a_walk[i] - b_walk[i] for i in walk_ids])
    boot = paired_bootstrap_interaction(d_run, d_walk, 10_000, 999)
    perm = exact_paired_sign_flip_test(d_run, d_walk)
    return {"interaction": boot["point_estimate"], "ci": [boot["ci_2.5"], boot["ci_97.5"]],
            "p_one_sided": perm["p_value_one_sided_interaction_gt_0"], "p_two_sided": perm["p_value_two_sided"]}


def main() -> None:
    summaries = {(d, c): load_summary(d, c) for d in DOMAINS for c in CONDITIONS}

    print("=== THREE-CONDITION DELTA TABLE ===")
    deltas = {}
    for d in DOMAINS:
        deltas[d] = {c: summaries[(d, c)]["delta_mean"] for c in CONDITIONS}
        print(f"{d}: run_scarce={deltas[d]['run_scarce']:+.4f}  balanced={deltas[d]['balanced']:+.4f}  "
              f"walk_scarce={deltas[d]['walk_scarce']:+.4f}")

    print("\n=== MONOTONIC ORDERING CHECK (point estimates) ===")
    ordering = {}
    for d in DOMAINS:
        rs, bal, ws = deltas[d]["run_scarce"], deltas[d]["balanced"], deltas[d]["walk_scarce"]
        monotonic = rs > bal > ws
        ordering[d] = monotonic
        print(f"{d}: {rs:+.4f} > {bal:+.4f} > {ws:+.4f}  ==  monotonic: {monotonic}")

    print("\n=== CONSECUTIVE-STEP INTERACTIONS (paired episode-level bootstrap + exact sign-flip) ===")
    step_interactions = {}
    for d in DOMAINS:
        rs_vs_bal = consecutive_interaction(summaries[(d, "run_scarce")], summaries[(d, "balanced")])
        bal_vs_ws = consecutive_interaction(summaries[(d, "balanced")], summaries[(d, "walk_scarce")])
        rs_vs_ws = consecutive_interaction(summaries[(d, "run_scarce")], summaries[(d, "walk_scarce")])
        step_interactions[d] = {"run_scarce_minus_balanced": rs_vs_bal, "balanced_minus_walk_scarce": bal_vs_ws,
                                 "run_scarce_minus_walk_scarce": rs_vs_ws}
        print(f"\n{d}:")
        print(f"  Delta_rs - Delta_bal = {rs_vs_bal['interaction']:+.4f}  CI={rs_vs_bal['ci']}  "
              f"p(1-sided >0)={rs_vs_bal['p_one_sided']:.4f}")
        print(f"  Delta_bal - Delta_ws = {bal_vs_ws['interaction']:+.4f}  CI={bal_vs_ws['ci']}  "
              f"p(1-sided >0)={bal_vs_ws['p_one_sided']:.4f}")
        print(f"  Delta_rs - Delta_ws  = {rs_vs_ws['interaction']:+.4f}  CI={rs_vs_ws['ci']}  "
              f"p(1-sided >0)={rs_vs_ws['p_one_sided']:.4f}  (full-range interaction, cf. Phase 5c)")

    print("\n=== BALANCED MODEL DETAIL ===")
    balanced_detail = {}
    for d in DOMAINS:
        s = summaries[(d, "balanced")]
        balanced_detail[d] = {
            "walk_stats": s["walk_stats"], "run_stats": s["run_stats"],
            "walk_episode_means": {k: v["mean"] for k, v in s["walk_episode_stats"].items()},
            "run_episode_means": {k: v["mean"] for k, v in s["run_episode_stats"].items()},
            "delta_mean": s["delta_mean"],
            "bootstrap": s["bootstrap"], "permutation_test": s["permutation_test"],
            "walk_denoising_loss_mean": s["walk_denoising_loss_mean"],
            "run_denoising_loss_mean": s["run_denoising_loss_mean"],
        }
        print(f"\n{d}/balanced:")
        print(f"  walk: {s['walk_stats']}")
        print(f"  run:  {s['run_stats']}")
        print(f"  walk episode means: {list(balanced_detail[d]['walk_episode_means'].values())}")
        print(f"  run episode means:  {list(balanced_detail[d]['run_episode_means'].values())}")
        print(f"  denoising loss: walk={s['walk_denoising_loss_mean']:.5f} run={s['run_denoising_loss_mean']:.5f}")

    # --- figure: one 3-point line per domain ---
    fig, ax = plt.subplots(figsize=(7, 5.5))
    colors = {"walker": "#2E86AB", "quadruped": "#C73E1D"}
    x = [0, 1, 2]
    labels = ["run_scarce", "balanced", "walk_scarce"]
    for d in DOMAINS:
        y = [deltas[d][c] for c in CONDITIONS]
        ax.plot(x, y, marker="o", markersize=9, linewidth=2.5, label=d, color=colors[d])
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x); ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, 2.3)
    ax.set_ylabel(r"$\Delta$ = mean LCG(run) $-$ mean LCG(walk)")
    ax.set_title("LCG response to relative Walk/Run coverage (3 conditions)")
    ax.legend(); ax.grid(alpha=0.3)
    fig.tight_layout()
    out_fig = OUT_ROOT / "three_condition_trend.png"
    fig.savefig(out_fig, dpi=150)
    print(f"\nSaved figure: {out_fig}")

    final = {"deltas": deltas, "monotonic_ordering": ordering, "step_interactions": step_interactions,
             "balanced_detail": balanced_detail}
    out_json = OUT_ROOT / "three_condition_summary.json"
    out_json.write_text(json.dumps(final, indent=2, default=str))
    print(f"Saved: {out_json}")


if __name__ == "__main__":
    main()
