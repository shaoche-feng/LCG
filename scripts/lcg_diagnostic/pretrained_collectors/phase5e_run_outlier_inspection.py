"""Renders the raw context+target frames (no model inference needed -- LCG scores
the REAL ground-truth y directly via forward-mode JVPs, it never samples/denoises
an image) for the highest-LCG-scored RUN-behavior transitions under
walker/run_scarce, split into the transition_index==0 boundary cluster (mirrors
the same episode-start artifact found on the ensemble-disagreement side in
phase6e_outlier_inspection.py) and the next tier once that cluster is excluded,
plus a median-score comparison. Purpose: see what content actually earns a high
LCG score once the boundary cluster is set aside.

Torch-free except for loading the Dataset (CPU, no GPU/model needed).

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5e_run_outlier_inspection.py
"""
from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from phase5_lcg_scoring import SOURCE_POOLS  # noqa: E402
from phase6_pilot_ensemble import load_transition_5d  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402

DOMAIN = "walker"
CONDITION = "run_scarce"
BEHAVIOR = "run"
N_COND = 4  # num_steps_conditioning, reused from phase5/6 (walker)
TOP_N = 5

_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
LCG_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"
OUT_DIR = LCG_ROOT / DOMAIN / CONDITION / "outlier_inspection"


def to_img(t: torch.Tensor) -> np.ndarray:
    arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
    return np.clip((arr + 1) / 2, 0, 1)


def render_group(tag: str, rows: list[dict], pool: Dataset) -> None:
    fig, axes = plt.subplots(len(rows), N_COND + 1, figsize=(3 * (N_COND + 1), 3 * len(rows)))
    if len(rows) == 1:
        axes = axes[None, :]
    for i, r in enumerate(rows):
        eid, t = int(r["episode_id"]), int(r["transition_index"])
        seg = SegmentId(eid, t + 1 - (N_COND + 1), t + 1)
        obs5d, act, y = load_transition_5d(pool, seg, N_COND, "cpu")
        for k in range(N_COND):
            axes[i, k].imshow(to_img(obs5d[0, k]))
            axes[i, k].set_title(f"ctx t-{N_COND - k}" if i == 0 else "", fontsize=9)
            axes[i, k].axis("off")
        axes[i, N_COND].imshow(to_img(y))
        axes[i, N_COND].set_title("target y" if i == 0 else "", fontsize=9, fontweight="bold")
        axes[i, N_COND].axis("off")
        axes[i, 0].text(-0.15, 0.5, f"ep{eid} t{t}\nLCG={float(r['lcg_score']):.1f}",
                         transform=axes[i, 0].transAxes, fontsize=9, ha="right", va="center")

    fig.suptitle(f"{DOMAIN}/{CONDITION} run: {tag}", fontsize=13)
    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / f"run_{tag}.png"
    fig.savefig(out_path, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")


def main() -> None:
    csv_path = LCG_ROOT / DOMAIN / CONDITION / "per_transition_scores.csv"
    rows = list(csv.DictReader(open(csv_path, newline="")))
    run_rows = [r for r in rows if r["behavior"] == BEHAVIOR]

    zero_rows = sorted([r for r in run_rows if int(r["transition_index"]) == 0],
                        key=lambda r: -float(r["lcg_score"]))[:TOP_N]
    nonzero_rows = sorted([r for r in run_rows if int(r["transition_index"]) != 0],
                           key=lambda r: -float(r["lcg_score"]))[:TOP_N]
    median_rows = sorted(run_rows, key=lambda r: float(r["lcg_score"]))
    median_row = [median_rows[len(median_rows) // 2]]

    print("t=0 boundary cluster (top LCG):")
    for r in zero_rows:
        print(f"  ep{r['episode_id']} t{r['transition_index']} lcg={float(r['lcg_score']):.2f}")
    print("Next tier excluding t=0:")
    for r in nonzero_rows:
        print(f"  ep{r['episode_id']} t{r['transition_index']} lcg={float(r['lcg_score']):.2f}")
    print("Median:")
    for r in median_row:
        print(f"  ep{r['episode_id']} t{r['transition_index']} lcg={float(r['lcg_score']):.2f}")

    pool = Dataset(SOURCE_POOLS / DOMAIN / BEHAVIOR / "dataset", name=f"{DOMAIN}_{BEHAVIOR}_pool", cache_in_ram=True)
    pool.load_from_default_path()

    render_group("t0_boundary_cluster", zero_rows, pool)
    render_group("nonzero_top_tier", nonzero_rows, pool)
    render_group("median", median_row, pool)


if __name__ == "__main__":
    main()
