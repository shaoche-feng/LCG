"""Same as plot_phase6_ens_categorical.py, but with the y-axis clipped per
panel to the 1st-97th percentile of the combined run/walk ENS_5 values (with
padding), so the extreme-outlier tail doesn't compress the main clouds.
Panel titles report how many points were clipped. Includes hopper.
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\jerry\Project_LCG\docs\lcg_undersample_diagnostic\phase6_full_ensemble")
OUT_DIR = ROOT

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
TARGET = "ens_S5"
WALK_COLOR = "#4292C6"
RUN_COLOR = "#D7301F"
RUN_X, WALK_X = 0.0, 1.0
JITTER = 0.16
RNG = np.random.default_rng(0)
LOW_PCT, HIGH_PCT = 1, 97


def load_rows(domain: str, condition: str) -> list[dict]:
    path = ROOT / domain / cond_dir(domain, condition) / "per_transition_scores.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    fig, axes = plt.subplots(len(DOMAINS), 3, figsize=(15, 4.5 * len(DOMAINS)))
    for row, domain in enumerate(DOMAINS):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            rows = load_rows(domain, condition)
            run_scores = np.array([float(r[TARGET]) for r in rows if r["behavior"] == "run"])
            walk_scores = np.array([float(r[TARGET]) for r in rows if r["behavior"] == "walk"])

            all_vals = np.concatenate([run_scores, walk_scores])
            lo, hi = np.percentile(all_vals, LOW_PCT), np.percentile(all_vals, HIGH_PCT)
            pad = (hi - lo) * 0.08
            ylo, yhi = lo - pad, hi + pad
            n_clipped = int(((all_vals < ylo) | (all_vals > yhi)).sum())

            x_run = RUN_X + RNG.uniform(-JITTER, JITTER, len(run_scores))
            x_walk = WALK_X + RNG.uniform(-JITTER, JITTER, len(walk_scores))
            ax.scatter(x_run, run_scores, s=8, alpha=0.4, marker="^", color=RUN_COLOR,
                       edgecolors="none", zorder=2)
            ax.scatter(x_walk, walk_scores, s=6, alpha=0.4, marker="o", color=WALK_COLOR,
                       edgecolors="none", zorder=2)
            ax.axhline(run_scores.mean(), xmin=0.08, xmax=0.48, color=RUN_COLOR, linewidth=1.6, zorder=3)
            ax.axhline(walk_scores.mean(), xmin=0.52, xmax=0.92, color=WALK_COLOR, linewidth=1.6, zorder=3)

            delta = run_scores.mean() - walk_scores.mean()
            ax.axvline(0.5, color="gray", linewidth=0.8, linestyle=":")
            ax.set_ylim(ylo, yhi)
            ax.set_xticks([RUN_X, WALK_X])
            ax.set_xticklabels([slot_dir(domain, "run"), slot_dir(domain, "walk")], fontsize=10, fontweight="bold")
            ax.set_xlim(-0.5, 1.5)
            ax.set_title(f"{domain}/{cond_dir(domain, condition)}  ({n_clipped} pts clipped)\n"
                         f"mean ENS_5 Delta ({slot_dir(domain, 'run')}-{slot_dir(domain, 'walk')}) = {delta:+.5f}", fontsize=10)
            ax.grid(alpha=0.2, axis="y")
            if col == 0:
                ax.set_ylabel("ENS_5 (ensemble disagreement)")

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=WALK_COLOR, markersize=7, label="walk  (hopper: stand)"),
        plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=RUN_COLOR, markersize=8, label="run  (hopper: hop)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2, fontsize=10, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Phase 6 full ensemble: categorical run-vs-walk separation of ENS_5 disagreement, outliers clipped\n"
                 f"(y-axis: {LOW_PCT}th-{HIGH_PCT}th pct + padding; ensemble members A/B/C)",
                 fontsize=12.5, y=1.03)
    fig.tight_layout()
    out_path = OUT_DIR / "ens_categorical_separation_clipped.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
