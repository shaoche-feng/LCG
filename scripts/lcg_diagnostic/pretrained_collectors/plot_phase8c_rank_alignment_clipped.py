"""Same as plot_phase8c_rank_alignment.py, but with the y-axis clipped per
panel to the 1st-97th percentile of the combined real-y/self-y scores across
all 3 seeds (with padding), so the extreme-outlier tail doesn't compress the
rest. Panel titles report how many points were clipped. Includes hopper.
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import spearmanr

ROOT = Path(r"C:\Users\jerry\Project_LCG\docs\lcg_undersample_diagnostic\phase8_imagined_rollout\phase8c_multiseed_selfy_check")
OUT_DIR = ROOT

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
WALK_COLORS = {"A": "#08306B", "B": "#4292C6", "C": "#9ECAE1"}
RUN_COLORS = {"A": "#7F0000", "B": "#D7301F", "C": "#FDBB84"}
LOW_PCT, HIGH_PCT = 1, 97


def load_rows(domain: str, seed: str, condition: str) -> list[dict]:
    path = ROOT / domain / seed / cond_dir(domain, condition) / "per_transition_scores.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    fig, axes = plt.subplots(len(DOMAINS), 3, figsize=(16, 4.5 * len(DOMAINS)), sharex=True)
    for i, domain in enumerate(DOMAINS):
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i, j]
            per_seed = {}
            all_vals = []
            for seed in SEEDS:
                rows = load_rows(domain, seed, condition)
                rows.sort(key=lambda r: float(r["real_y_lcg_score"]))
                behavior = np.array([r["behavior"] for r in rows])
                real_vals = np.array([float(r["real_y_lcg_score"]) for r in rows])
                self_vals = np.array([float(r["self_y_lcg_score"]) for r in rows])
                per_seed[seed] = (behavior, real_vals, self_vals)
                all_vals.append(real_vals)
                all_vals.append(self_vals)

            all_vals = np.concatenate(all_vals)
            lo, hi = np.percentile(all_vals, LOW_PCT), np.percentile(all_vals, HIGH_PCT)
            pad = (hi - lo) * 0.08
            ylo, yhi = lo - pad, hi + pad
            n_clipped = int(((all_vals < ylo) | (all_vals > yhi)).sum())

            for seed in SEEDS:
                behavior, real_vals, self_vals = per_seed[seed]
                ranks = np.arange(1, len(behavior) + 1)
                walk_mask, run_mask = behavior == "walk", behavior == "run"
                rho = spearmanr(real_vals, self_vals)[0]

                ax.scatter(ranks[walk_mask], self_vals[walk_mask], s=7, alpha=0.4, marker="o",
                           color=WALK_COLORS[seed], edgecolors="none",
                           label=f"Seed {seed} self-y* (Spearman={rho:.3f})")
                ax.scatter(ranks[run_mask], self_vals[run_mask], s=12, alpha=0.4, marker="^",
                           color=RUN_COLORS[seed], edgecolors="none")
                ax.scatter(ranks[walk_mask], real_vals[walk_mask], s=7, alpha=0.85, marker="o",
                           color=WALK_COLORS[seed], edgecolors="none", zorder=5)
                ax.scatter(ranks[run_mask], real_vals[run_mask], s=12, alpha=0.85, marker="^",
                           color=RUN_COLORS[seed], edgecolors="none", zorder=5)

            ax.set_ylim(ylo, yhi)
            ax.set_title(f"{domain}/{cond_dir(domain, condition)}  ({n_clipped} pts clipped)", fontsize=10.5)
            ax.grid(alpha=0.25)
            if i == len(DOMAINS) - 1:
                ax.set_xlabel("rank by that seed's own real-y LCG score (low to high)")
            if j == 0:
                ax.set_ylabel("LCG score")
            ax.legend(fontsize=6.5, loc="upper left", markerscale=1.8)

    fig.suptitle(f"Phase 8c: real-y vs self-y* alignment, outliers clipped "
                 f"(y-axis: {LOW_PCT}th-{HIGH_PCT}th pct + padding)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / "phase8c_rank_alignment_clipped.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
