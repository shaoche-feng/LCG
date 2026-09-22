"""Flattens phase6_full_ensemble.py's already-saved per-candidate data (the
`per_candidate` list embedded in each domain/condition's ens_summary.json)
into a standalone per_transition_scores.csv in the same folder, matching the
per-transition-CSV convention used by phases 8/9. Torch-free -- reads JSON
already produced on disk, writes nothing new numerically.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase6d_export_csv.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
import json
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase6_full_ensemble"

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
NUM_SAMPLES = 5


def export_one(domain: str, condition: str) -> Path:
    src = OUT_ROOT / domain / cond_dir(domain, condition) / "ens_summary.json"
    data = json.loads(src.read_text())
    per_candidate = data["per_candidate"]

    fieldnames = ["domain", "condition", "behavior", "episode_id", "transition_index",
                  *[f"ens_S{s + 1}" for s in range(NUM_SAMPLES)],
                  "within_var_A", "within_var_B", "within_var_C", "within_sample_variance",
                  "error_A", "error_B", "error_C", "error_ensemble", "lcg_score"]

    out_path = OUT_ROOT / domain / cond_dir(domain, condition) / "per_transition_scores.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in per_candidate:
            row = {
                "domain": domain, "condition": condition,
                "behavior": r["behavior"], "episode_id": r["episode_id"],
                "transition_index": r["transition_index"],
                "within_sample_variance": r["within_sample_variance"],
                "error_ensemble": r["error_ensemble"], "lcg_score": r["lcg_score"],
            }
            for s in range(NUM_SAMPLES):
                row[f"ens_S{s + 1}"] = r["ens"][s]
            for label in ("A", "B", "C"):
                row[f"within_var_{label}"] = r["within_var_by_model"][label]
                row[f"error_{label}"] = r["error_by_model"][label]
            writer.writerow(row)

    print(f"Saved: {out_path} ({len(per_candidate)} rows)")
    return out_path


def main() -> None:
    for domain in DOMAINS:
        for condition in CONDITIONS:
            export_one(domain, condition)


if __name__ == "__main__":
    main()
