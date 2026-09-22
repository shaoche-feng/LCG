"""Extends Phase 5 (phase5_lcg_scoring.py) to hopper: core LCG scoring on
held-out stand("walk" slot)/hop("run" slot) transitions, Seed A, all 3
conditions. Reuses process_domain() unmodified except for probe_task="stand"
(hopper has no "walk" task -- see collect_source_pools.py's
DOMAIN_BEHAVIOR_TASK). Kept as a separate driver so it never touches the
existing walker/quadruped phase5_lcg_scoring/ output.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5_lcg_scoring_hopper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_lcg_scoring import process_domain, OUT_ROOT  # noqa: E402

DOMAIN = "hopper"
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for condition in CONDITIONS:
        all_results[condition] = process_domain(DOMAIN, condition, probe_task="stand")
    (OUT_ROOT / f"{DOMAIN}_phase5_summary.json").write_text(json.dumps(all_results, indent=2))

    print("\n\n=== HOPPER PHASE 5 FINAL SUMMARY ===")
    for condition, r in all_results.items():
        print(f"{condition}: delta_mean={r['delta_mean']:.6f} ratio={r['ratio_mean']:.4f} "
              f"run>walk direction={r['delta_mean'] > 0} "
              f"perm_p_run_gt_walk={r['permutation_test']['p_value_one_sided_run_gt_walk']:.4f} "
              f"perm_p_walk_gt_run={r['permutation_test']['p_value_one_sided_walk_gt_run']:.4f}")


if __name__ == "__main__":
    main()
