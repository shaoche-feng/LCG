"""Section D of the Phase 6 pilot: stability of ENS_S vs ENS_5 across S=1..4,
computed from phase6_pilot_ensemble.py's saved pilot_summary.json. Deliberately
torch-free (see phase6_pilot_ensemble.py's docstring for why scipy and torch
cannot share a process on this machine).

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase6b_pilot_stability.py
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy.stats import pearsonr, spearmanr

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase6_pilot_ensemble"


def main() -> None:
    summary = json.loads((OUT_ROOT / "pilot_summary.json").read_text())
    ens_matrix = np.array([r["ens"] for r in summary["per_candidate"]])  # (20, 5)
    num_samples = ens_matrix.shape[1]

    print("=== D. STABILITY AGAINST S=5 ===")
    stability = {}
    for S in range(1, num_samples):  # S=1..4 vs S=5
        a, b = ens_matrix[:, S - 1], ens_matrix[:, num_samples - 1]
        pear = float(pearsonr(a, b)[0])
        spear = float(spearmanr(a, b)[0])
        rel_err = float(np.mean(np.abs(a - b) / np.maximum(np.abs(b), 1e-8)))
        stability[S] = {"pearson_vs_S5": pear, "spearman_vs_S5": spear, "mean_abs_relative_diff": rel_err}
        print(f"S={S} vs S=5: Pearson={pear:.4f} Spearman={spear:.4f} mean_abs_rel_diff={rel_err:.4f}")

    summary["stability_vs_S5"] = stability
    (OUT_ROOT / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nUpdated: {OUT_ROOT / 'pilot_summary.json'} (added stability_vs_S5)")


if __name__ == "__main__":
    main()
