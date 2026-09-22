"""Rank-alignment plot for Phase 8c: for each seed independently, sort that
seed's 1000 candidates by ITS OWN real-y LCG score (ascending), then plot
self-y* at those same rank positions -- same style as seed_rank_alignment.png
(circle=walk, triangle=run, color=seed shade within its behavior's hue
family), now showing real-y-vs-self-y agreement instead of seed-vs-seed
agreement, for all 3 seeds overlaid in each domain/condition panel.
Torch-free -- reads the already-saved phase8c per_transition_scores.csv files.
Includes hopper once its Phase 8c data exists.
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


def load_rows(domain: str, seed: str, condition: str) -> list[dict]:
    path = ROOT / domain / seed / cond_dir(domain, condition) / "per_transition_scores.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def main() -> None:
    fig, axes = plt.subplots(len(DOMAINS), 3, figsize=(16, 4.5 * len(DOMAINS)), sharex=True)
    for i, domain in enumerate(DOMAINS):
        for j, condition in enumerate(CONDITIONS):
            ax = axes[i, j]
            for seed in SEEDS:
                rows = load_rows(domain, seed, condition)
                rows.sort(key=lambda r: float(r["real_y_lcg_score"]))
                ranks = np.arange(1, len(rows) + 1)
                behavior = np.array([r["behavior"] for r in rows])
                real_vals = np.array([float(r["real_y_lcg_score"]) for r in rows])
                self_vals = np.array([float(r["self_y_lcg_score"]) for r in rows])
                walk_mask, run_mask = behavior == "walk", behavior == "run"
                rho = spearmanr(real_vals, self_vals)[0]

                ax.scatter(ranks[walk_mask], self_vals[walk_mask], s=5, alpha=0.35, marker="o",
                           color=WALK_COLORS[seed], edgecolors="none",
                           label=f"Seed {seed} self-y* (Spearman={rho:.3f})")
                ax.scatter(ranks[run_mask], self_vals[run_mask], s=9, alpha=0.35, marker="^",
                           color=RUN_COLORS[seed], edgecolors="none")
                ax.scatter(ranks[walk_mask], real_vals[walk_mask], s=5, alpha=0.85, marker="o",
                           color=WALK_COLORS[seed], edgecolors="none", zorder=5)
                ax.scatter(ranks[run_mask], real_vals[run_mask], s=9, alpha=0.85, marker="^",
                           color=RUN_COLORS[seed], edgecolors="none", zorder=5)

            ax.set_title(f"{domain}/{cond_dir(domain, condition)}", fontsize=11)
            ax.grid(alpha=0.25)
            if i == len(DOMAINS) - 1:
                ax.set_xlabel("rank by that seed's own real-y LCG score (low to high)")
            if j == 0:
                ax.set_ylabel("LCG score")
            ax.legend(fontsize=6.5, loc="upper left", markerscale=1.8)

    handles = []
    for s in SEEDS:
        handles.append(plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=WALK_COLORS[s],
                                   markersize=7, label=f"Seed {s} walk / hopper: stand"))
        handles.append(plt.Line2D([0], [0], marker="^", color="w", markerfacecolor=RUN_COLORS[s],
                                   markersize=8, label=f"Seed {s} run / hopper: hop"))
    fig.suptitle("Phase 8c: real-y vs self-y* rank alignment, per seed "
                 "(opaque=real-y reference, faint=self-y*)", fontsize=13)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out_path = OUT_DIR / "phase8c_rank_alignment.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
