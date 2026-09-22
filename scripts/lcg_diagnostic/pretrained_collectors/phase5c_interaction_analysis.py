"""Final analysis for the undersampling diagnostic: the scarcity x behavior
INTERACTION, using ONLY the already-saved Phase 5 / Phase 5b scores (no
retraining, no rescoring, no new candidate selection).

Interaction = Delta_run_scarce - Delta_walk_scarce, where
Delta_condition = mean_LCG(run) - mean_LCG(walk) under that condition's model.

Since the exact same held-out episodes/indices were scored under both
conditions, this pairs episode-level LCG means BY EPISODE ID across the two
models (not independently reshuffled), giving a paired bootstrap CI and an
exact paired sign-flip randomization test for the interaction.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5c_interaction_analysis.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import itertools
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
SCORING_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"

DOMAINS = ["walker", "quadruped"]
NUM_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 999  # same seed convention as Phase 5's own bootstrap


def load_episode_means(domain: str, condition: str) -> dict:
    summary = json.loads((SCORING_ROOT / domain / cond_dir(domain, condition) / "summary.json").read_text())
    walk_means = {int(eid): v["mean"] for eid, v in summary["walk_episode_stats"].items()}
    run_means = {int(eid): v["mean"] for eid, v in summary["run_episode_stats"].items()}
    return {
        "walk_means": walk_means, "run_means": run_means,
        "delta_mean": summary["delta_mean"],
        "walk_denoising_loss_mean": summary["walk_denoising_loss_mean"],
        "run_denoising_loss_mean": summary["run_denoising_loss_mean"],
    }


def paired_bootstrap_interaction(d_run: np.ndarray, d_walk: np.ndarray, num_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_run, n_walk = len(d_run), len(d_walk)
    draws = np.empty(num_resamples)
    for b in range(num_resamples):
        run_idx = rng.integers(0, n_run, size=n_run)
        walk_idx = rng.integers(0, n_walk, size=n_walk)
        draws[b] = d_run[run_idx].mean() - d_walk[walk_idx].mean()
    point = float(d_run.mean() - d_walk.mean())
    return {
        "point_estimate": point,
        "ci_2.5": float(np.percentile(draws, 2.5)),
        "ci_97.5": float(np.percentile(draws, 97.5)),
        "num_resamples": num_resamples, "seed": seed,
    }


def exact_paired_sign_flip_test(d_run: np.ndarray, d_walk: np.ndarray) -> dict:
    """Exact randomization test for paired episode-level differences: under the null
    (each d_i is exchangeable in sign), enumerate all 2^(n_run+n_walk) sign assignments.
    n_run=n_walk=5 here -> 2^10=1024, enumerated exactly (no Monte Carlo)."""
    n_run, n_walk = len(d_run), len(d_walk)
    observed = d_run.mean() - d_walk.mean()
    all_signs_run = list(itertools.product([1, -1], repeat=n_run))
    all_signs_walk = list(itertools.product([1, -1], repeat=n_walk))
    perm_stats = []
    for s_run in all_signs_run:
        flipped_run_mean = (np.array(s_run) * d_run).mean()
        for s_walk in all_signs_walk:
            flipped_walk_mean = (np.array(s_walk) * d_walk).mean()
            perm_stats.append(flipped_run_mean - flipped_walk_mean)
    perm_stats = np.array(perm_stats)
    p_one_sided = float((perm_stats >= observed).mean())
    p_two_sided = float((np.abs(perm_stats) >= abs(observed)).mean())
    return {
        "observed": float(observed), "num_sign_patterns": len(perm_stats),
        "p_value_one_sided_interaction_gt_0": p_one_sided, "p_value_two_sided": p_two_sided,
    }


def analyze_domain(domain: str) -> dict:
    rs = load_episode_means(domain, "run_scarce")
    ws = load_episode_means(domain, "walk_scarce")

    run_ids = sorted(rs["run_means"].keys())
    walk_ids = sorted(rs["walk_means"].keys())
    assert run_ids == sorted(ws["run_means"].keys()), "run episode IDs differ between conditions"
    assert walk_ids == sorted(ws["walk_means"].keys()), "walk episode IDs differ between conditions"

    d_run = np.array([rs["run_means"][i] - ws["run_means"][i] for i in run_ids])
    d_walk = np.array([rs["walk_means"][i] - ws["walk_means"][i] for i in walk_ids])

    delta_rs = rs["delta_mean"]
    delta_ws = ws["delta_mean"]
    interaction = delta_rs - delta_ws
    # sanity: interaction must equal mean(d_run) - mean(d_walk) algebraically
    assert abs(interaction - (d_run.mean() - d_walk.mean())) < 1e-3, "interaction algebra mismatch"

    boot = paired_bootstrap_interaction(d_run, d_walk, NUM_BOOTSTRAP, BOOTSTRAP_SEED)
    perm = exact_paired_sign_flip_test(d_run, d_walk)

    # episode-level overlap check (descriptive, corrects the earlier over-claim for walker)
    overlap_run_scarce = [r for r in rs["run_means"].values() if r < max(rs["walk_means"].values())] \
        if min(rs["run_means"].values()) < max(rs["walk_means"].values()) else []
    overlap_walk_scarce = [r for r in ws["run_means"].values() if r > min(ws["walk_means"].values())]

    result = {
        "domain": domain,
        "delta_run_scarce": delta_rs, "delta_walk_scarce": delta_ws,
        "interaction_point_estimate": interaction,
        "interaction_bootstrap": boot, "interaction_permutation": perm,
        "run_scarce_walk_episode_means": rs["walk_means"], "run_scarce_run_episode_means": rs["run_means"],
        "walk_scarce_walk_episode_means": ws["walk_means"], "walk_scarce_run_episode_means": ws["run_means"],
        "walk_scarce_overlap_note": (
            f"{len(overlap_walk_scarce)} of 5 run-episode means exceed the minimum walk-episode "
            f"mean under walk_scarce (partial, not full, separation)" if overlap_walk_scarce
            else "full separation under walk_scarce (no run episode exceeds any walk episode)"
        ),
        "secondary_denoising_loss": {
            "run_scarce_walk": rs["walk_denoising_loss_mean"], "run_scarce_run": rs["run_denoising_loss_mean"],
            "walk_scarce_walk": ws["walk_denoising_loss_mean"], "walk_scarce_run": ws["run_denoising_loss_mean"],
        },
    }
    return result


def make_figure(results: dict, out_path: Path, run_label: str = "run", walk_label: str = "walk") -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    x = [0, 1]
    labels = [f"{run_label}_scarce", f"{walk_label}_scarce"]
    colors = {"walker": "#2E86AB", "quadruped": "#C73E1D"}
    for domain, r in results.items():
        y = [r["delta_run_scarce"], r["delta_walk_scarce"]]
        ax.plot(x, y, marker="o", markersize=9, linewidth=2.5, label=domain, color=colors.get(domain))
        for xi, yi in zip(x, y):
            ax.annotate(f"{yi:+.2f}", (xi, yi), textcoords="offset points", xytext=(8, 6), fontsize=9)
    ax.axhline(0, color="black", linewidth=1, linestyle="--", alpha=0.7)
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_xlim(-0.3, 1.3)
    ax.set_ylabel(rf"$\Delta$ = mean LCG({run_label}) $-$ mean LCG({walk_label})")
    ax.set_title("LCG sign reversal under reversed scarcity")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    print(f"Saved figure: {out_path}")


def main() -> None:
    results = {domain: analyze_domain(domain) for domain in DOMAINS}

    print("\n=== FINAL INTERACTION TABLE ===")
    print(f"{'domain':<10} {'Delta_rs':>10} {'Delta_ws':>10} {'interaction':>12} "
          f"{'CI_low':>9} {'CI_high':>9} {'p(1-sided)':>11} {'p(2-sided)':>11}")
    for domain, r in results.items():
        b, p = r["interaction_bootstrap"], r["interaction_permutation"]
        print(f"{domain:<10} {r['delta_run_scarce']:>10.4f} {r['delta_walk_scarce']:>10.4f} "
              f"{r['interaction_point_estimate']:>12.4f} {b['ci_2.5']:>9.4f} {b['ci_97.5']:>9.4f} "
              f"{p['p_value_one_sided_interaction_gt_0']:>11.4f} {p['p_value_two_sided']:>11.4f}")

    print("\n=== WALKER WALK_SCARCE OVERLAP CORRECTION ===")
    print(results["walker"]["walk_scarce_overlap_note"])
    print("\n=== QUADRUPED WALK_SCARCE OVERLAP CHECK ===")
    print(results["quadruped"]["walk_scarce_overlap_note"])

    print("\n=== SECONDARY: held-out denoising loss (NOT used for LCG significance) ===")
    for domain, r in results.items():
        d = r["secondary_denoising_loss"]
        print(f"{domain}: run_scarce(walk={d['run_scarce_walk']:.5f}, run={d['run_scarce_run']:.5f})  "
              f"walk_scarce(walk={d['walk_scarce_walk']:.5f}, run={d['walk_scarce_run']:.5f})")

    out_dir = SCORING_ROOT
    (out_dir / "interaction_analysis.json").write_text(json.dumps(results, indent=2))
    make_figure(results, out_dir / "sign_reversal_figure.png")
    print(f"\nSaved: {out_dir / 'interaction_analysis.json'}")


if __name__ == "__main__":
    main()
