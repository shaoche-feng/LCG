"""Filters the Phase 8c rank-alignment view down to only the candidates most
involved in a pairwise reversal between real-y and self-y*, per seed:
candidate k is "reversed" if that seed's real-y ranks k below k' but its
self-y* scores k above k' (or vice versa). Keeps only the top decile
(most-reversed 10%) per seed; real-y is kept in full as the reference.
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

ROOT = Path(r"C:\Users\jerry\Project_LCG\docs\lcg_undersample_diagnostic\phase8_imagined_rollout\phase8c_multiseed_selfy_check")
OUT_DIR = ROOT

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
WALK_COLORS = {"A": "#08306B", "B": "#4292C6", "C": "#9ECAE1"}
RUN_COLORS = {"A": "#7F0000", "B": "#D7301F", "C": "#FDBB84"}
KEEP_FRACTION = 0.10


def load_rows(domain: str, seed: str, condition: str) -> list[dict]:
    path = ROOT / domain / seed / cond_dir(domain, condition) / "per_transition_scores.csv"
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def reversal_counts(y: np.ndarray) -> np.ndarray:
    n = len(y)
    idx = np.arange(n)
    rank_sign = np.sign(idx[:, None] - idx[None, :])
    val_sign = np.sign(y[:, None] - y[None, :])
    discordant = (rank_sign * val_sign) < 0
    np.fill_diagonal(discordant, False)
    return discordant.sum(axis=1)


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
                n = len(rows)
                n_keep = max(1, int(round(n * KEEP_FRACTION)))

                ax.scatter(ranks[behavior == "walk"], real_vals[behavior == "walk"], s=5, alpha=0.7,
                           marker="o", color=WALK_COLORS[seed], edgecolors="none", zorder=5)
                ax.scatter(ranks[behavior == "run"], real_vals[behavior == "run"], s=9, alpha=0.7,
                           marker="^", color=RUN_COLORS[seed], edgecolors="none", zorder=5)

                rc = reversal_counts(self_vals)
                thresh = np.sort(rc)[-n_keep]
                mask = rc >= thresh
                walk_mask = mask & (behavior == "walk")
                run_mask = mask & (behavior == "run")
                ax.scatter(ranks[walk_mask], self_vals[walk_mask], s=20, alpha=0.75, marker="o",
                           color=WALK_COLORS[seed], edgecolors="none",
                           label=f"Seed {seed} self-y* (top {KEEP_FRACTION:.0%} reversed)")
                ax.scatter(ranks[run_mask], self_vals[run_mask], s=32, alpha=0.75, marker="^",
                           color=RUN_COLORS[seed], edgecolors="none")

            ax.set_title(f"{domain}/{cond_dir(domain, condition)}", fontsize=11)
            ax.grid(alpha=0.25)
            if i == len(DOMAINS) - 1:
                ax.set_xlabel("rank by that seed's own real-y LCG score (low to high)")
            if j == 0:
                ax.set_ylabel("LCG score")
            ax.legend(fontsize=6, loc="upper left", markerscale=1.5)

    fig.suptitle(f"Phase 8c: only the top {KEEP_FRACTION:.0%} most-reversed candidates per seed "
                 "(self-y* vs that seed's own real-y; everything concordant is filtered out)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.96])
    out_path = OUT_DIR / "phase8c_rank_reversals_filtered.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
