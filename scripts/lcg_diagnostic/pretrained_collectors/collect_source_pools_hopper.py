"""Extends Phase 2 (collect_source_pools.py) to a third domain: hopper. Kept as a
separate driver (rather than adding "hopper" to collect_source_pools.py's own
DOMAINS/main()) so this can never accidentally re-collect or wipe the existing
walker/quadruped pools -- it only imports and reuses collect_pool() as-is.

hopper has no walk/run tasks in dm_control (only stand/hop) -- see
DOMAIN_BEHAVIOR_TASK in collect_source_pools.py for the alias ("stand" -> our
"walk" slot, "hop" -> our "run" slot) that keeps every downstream script's
"walk"/"run" naming unmodified.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/collect_source_pools_hopper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from collect_source_pools import collect_pool, OUT_ROOT, SOURCES  # noqa: E402

DOMAIN = "hopper"


def main() -> None:
    all_manifests = {}
    for source_name in SOURCES:
        m = collect_pool(DOMAIN, source_name)
        all_manifests[f"{DOMAIN}/{source_name}"] = m

    with open(OUT_ROOT / f"{DOMAIN}_phase2_summary.json", "w") as f:
        json.dump(all_manifests, f, indent=2)

    print(f"\n\n=== {DOMAIN.upper()} PHASE 2 SUMMARY ===")
    for key, m in all_manifests.items():
        v = m["forward_velocity_stats"]
        v_str = f"fwd_vel_mean={v['mean']:.4f}" if v else "fwd_vel=n/a"
        print(f"{key:20s} episodes={m['num_episodes']:3d} transitions={m['num_transitions']:5d} "
              f"roundtrip={m['dataset_roundtrip_passed']} finite={m['all_values_finite']} {v_str}")


if __name__ == "__main__":
    main()
