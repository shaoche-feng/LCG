"""Phase 7 supplementary figure: the same per-seed coverage-response trend
plot as figD_{domain}_focus_trends.png (produced by phase7_analysis.py for
only the 3 focus configs R1/R1+R2/full), extended to ALL 8 theta_S
configs AND all 6 candidate M values (5,6,7,8,9,12), one figure per
(domain, M). Torch-free -- loads the same saved q_values.npy files
phase7_analysis.py already used (post-hoc nested-prefix means), no rescoring.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase7d_all_trends.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase7_thetaS_M_robustness"

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
THETA_S_ORDER = ["R1", "R2", "R3", "R1+R2", "R1+R3", "R2+R3", "3R", "full"]
M_LIST = [5, 6, 7, 8, 9, 12]

EXPECTED_D_S = {"R1": 217664, "R2": 217664, "R3": 217664, "R1+R2": 435328, "R1+R3": 435328,
                 "R2+R3": 435328, "3R": 652992, "full": 654851}
COLORS = {"A": "#888888", "B": "#2E86AB", "C": "#C73E1D"}


def load_behavior_masks() -> dict:
    masks = {}
    for domain in DOMAINS:
        meta = json.loads((OUT_ROOT / domain / "candidates_meta.json").read_text())
        behavior = np.array([c["behavior"] for c in meta["candidates"]])
        masks[domain] = {"walk": behavior == "walk", "run": behavior == "run"}
    return masks


def load_all_q() -> dict:
    """Load every (domain, seed, condition, theta_S) q_values.npy once (small: 144 files,
    ~48KB each) so all 6 M values can be derived from cached arrays instead of re-reading
    from disk 6x."""
    q_by_key = {}
    for domain in DOMAINS:
        for s in SEEDS:
            for c in CONDITIONS:
                for t in THETA_S_ORDER:
                    q_path = OUT_ROOT / domain / s / c / t / "q_values.npy"
                    if q_path.exists():
                        q_by_key[(domain, s, c, t)] = np.load(q_path)
    return q_by_key


def main() -> None:
    masks = load_behavior_masks()
    q_by_key = load_all_q()

    for domain in DOMAINS:
        for M in M_LIST:
            fig, axes = plt.subplots(2, 4, figsize=(20, 9), sharey=False)
            axes = axes.flatten()
            for ax, t in zip(axes, THETA_S_ORDER):
                xs = [0, 1, 2]
                seed_vals = {}
                for s in SEEDS:
                    vals = []
                    ok = True
                    for c in CONDITIONS:
                        q = q_by_key.get((domain, s, c, t))
                        if q is None:
                            ok = False
                            break
                        score = q[:, :M].mean(axis=1)
                        d = score[masks[domain]["run"]].mean() - score[masks[domain]["walk"]].mean()
                        vals.append(d)
                    if ok:
                        seed_vals[s] = vals
                        ax.plot(xs, vals, marker="o", linewidth=1.5, alpha=0.85, color=COLORS[s], label=f"Seed {s}")
                if seed_vals:
                    mean_y = [np.mean([seed_vals[s][i] for s in seed_vals]) for i in range(3)]
                    ax.plot(xs, mean_y, marker="s", markersize=9, linewidth=3, color="black", label="mean", zorder=5)
                ax.axhline(0, color="gray", linewidth=1, linestyle="--")
                ax.set_xticks(xs)
                ax.set_xticklabels(["run_scarce", "balanced", "walk_scarce"], rotation=15)
                ax.set_title(f"{t}  (d_S={EXPECTED_D_S[t]:,})")
                ax.legend(fontsize=7.5)
                ax.grid(alpha=0.3)
            axes[0].set_ylabel(f"$\\Delta$ = mean score(run) $-$ mean score(walk), M={M}")
            axes[4].set_ylabel(f"$\\Delta$ = mean score(run) $-$ mean score(walk), M={M}")
            fig.suptitle(f"{domain}: coverage response across ALL 8 theta_S configs (M={M})", fontsize=14)
            fig.tight_layout(rect=[0, 0, 1, 0.95])
            out_path = OUT_ROOT / f"figD_all_{domain}_M{M}_all_theta_trends.png"
            fig.savefig(out_path, dpi=150)
            plt.close(fig)
            print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
