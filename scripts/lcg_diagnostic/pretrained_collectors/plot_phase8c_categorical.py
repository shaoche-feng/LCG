"""Same categorical run-vs-walk comparison as walk_run_categorical_separation.png,
applied to Phase 8c's multiseed SELF-Y check data only (the production-
faithful target -- real-y is already covered by the original multiseed
figure). One figure, N domains x 3 conditions, all 3 seeds shown together.
Torch-free -- reads the already-saved phase8c per_transition_scores.csv files.
Includes hopper (where "walk"="stand", "run"="hop") once its Phase 8c data exists.
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(r"C:\Users\jerry\Project_LCG\docs\lcg_undersample_diagnostic\phase8_imagined_rollout\phase8c_multiseed_selfy_check")
OUT_DIR = ROOT

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
TARGET = "self_y_lcg_score"
WALK_COLORS = {"A": "#08306B", "B": "#4292C6", "C": "#9ECAE1"}
RUN_COLORS = {"A": "#7F0000", "B": "#D7301F", "C": "#FDBB84"}
RUN_X = {"A": -0.35, "B": 0.0, "C": 0.35}
WALK_X = {"A": 0.65, "B": 1.0, "C": 1.35}
JITTER = 0.14
RNG = np.random.default_rng(0)


def load_rows(domain: str, seed: str, condition: str) -> list[dict]:
    path = ROOT / domain / seed / cond_dir(domain, condition) / "per_transition_scores.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    fig, axes = plt.subplots(len(DOMAINS), 3, figsize=(15, 4.5 * len(DOMAINS)))
    for row, domain in enumerate(DOMAINS):
        for col, condition in enumerate(CONDITIONS):
            ax = axes[row, col]
            run_means, walk_means = [], []
            per_seed = {}
            for seed in SEEDS:
                rows = load_rows(domain, seed, condition)
                run_scores = np.array([float(r[TARGET]) for r in rows if r["behavior"] == "run"])
                walk_scores = np.array([float(r[TARGET]) for r in rows if r["behavior"] == "walk"])
                per_seed[seed] = (run_scores, walk_scores)
                run_means.append(run_scores.mean())
                walk_means.append(walk_scores.mean())

            for seed in SEEDS:
                run_scores, walk_scores = per_seed[seed]
                x_run = RUN_X[seed] + RNG.uniform(-JITTER, JITTER, len(run_scores))
                x_walk = WALK_X[seed] + RNG.uniform(-JITTER, JITTER, len(walk_scores))
                ax.scatter(x_run, run_scores, s=8, alpha=0.4, marker="^", color=RUN_COLORS[seed],
                           edgecolors="none", zorder=2)
                ax.scatter(x_walk, walk_scores, s=6, alpha=0.4, marker="o", color=WALK_COLORS[seed],
                           edgecolors="none", zorder=2)
                ax.axhline(run_scores.mean(), color=RUN_COLORS[seed], linewidth=1.3, alpha=0.8, zorder=1)
                ax.axhline(walk_scores.mean(), color=WALK_COLORS[seed], linewidth=1.3, alpha=0.8,
                           linestyle="--", zorder=1)

            pooled_delta = np.mean(run_means) - np.mean(walk_means)
            ax.axvline(0.5, color="gray", linewidth=0.8, linestyle=":")
            ax.set_xticks([0.0, 1.0])
            ax.set_xticklabels([slot_dir(domain, "run"), slot_dir(domain, "walk")], fontsize=10, fontweight="bold")
            ax.set_xlim(-0.65, 1.65)
            ax.set_title(f"{domain}/{cond_dir(domain, condition)}\nmean Delta ({slot_dir(domain, 'run')}-{slot_dir(domain, 'walk')}), avg over seeds = {pooled_delta:+.2f}",
                         fontsize=10)
            ax.grid(alpha=0.2, axis="y")
            if col == 0:
                ax.set_ylabel("LCG score")

    handles = []
    for s in SEEDS:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=WALK_COLORS[s],
                                   markersize=7, label=f"Seed {s} walk / hopper: stand"))
        handles.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=RUN_COLORS[s],
                                   markersize=8, label=f"Seed {s} run / hopper: hop"))
    fig.legend(handles=handles, loc="upper center", ncol=3, fontsize=9, bbox_to_anchor=(0.5, 1.01))
    fig.suptitle("Categorical run-vs-walk separation under self-generated y* (production-faithful target)",
                 fontsize=13, y=1.03)
    fig.tight_layout()
    out_path = OUT_DIR / "categorical_separation_selfy.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
