"""Extends Phase 3 (construct_phase3_datasets.py) to hopper: freezes held-out
stand/hop episodes and builds the 3 mixtures (run_scarce/balanced/walk_scarce --
"run"/"walk" here mean hop/stand respectively, per collect_source_pools.py's
DOMAIN_BEHAVIOR_TASK alias). Kept as a separate driver so it can never touch the
existing walker/quadruped eval/mixture directories -- reuses every function from
construct_phase3_datasets.py unmodified.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/construct_phase3_datasets_hopper.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))

from construct_phase3_datasets import (  # noqa: E402
    build_eval_sets, build_mixture, validate_dataset, leakage_check,
    MIXTURES, EXPECTED_COUNTS, ACTION_DIM, EVAL_ROOT, MIXTURE_ROOT, PROBE_ROOT,
)

DOMAIN = "hopper"


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    MIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    eval_manifests = build_eval_sets(DOMAIN)
    mixture_manifests = {c: build_mixture(DOMAIN, c) for c in MIXTURES}
    leakage = leakage_check(DOMAIN, mixture_manifests, eval_manifests)

    print(f"\n\n=== {DOMAIN.upper()} VALIDATION: 2 eval datasets + 3 mixtures ===")
    validation_results = {}
    for source in ("walk", "run"):
        key = f"eval/{DOMAIN}/{source}"
        r = validate_dataset(EVAL_ROOT / DOMAIN / slot_dir(DOMAIN, source) / "dataset", 2, 1000, ACTION_DIM[DOMAIN])
        validation_results[key] = r
        print(f"  [{('PASS' if r['roundtrip_passed'] else 'FAIL')}] {key}: {r}")
    for condition in MIXTURES:
        key = f"mixture/{DOMAIN}/{condition}"
        r = validate_dataset(MIXTURE_ROOT / DOMAIN / cond_dir(DOMAIN, condition) / "dataset", 10, 5000, ACTION_DIM[DOMAIN])
        validation_results[key] = r
        expected = EXPECTED_COUNTS[condition]
        m = mixture_manifests[condition]
        counts_ok = (m["random"]["num_transitions"] == expected["random"]
                     and m["walk"]["num_transitions"] == expected["walk"]
                     and m["run"]["num_transitions"] == expected["run"])
        r["per_source_counts_ok"] = counts_ok
        print(f"  [{('PASS' if r['roundtrip_passed'] and counts_ok else 'FAIL')}] {key}: {r}")

    all_ok = all(r["roundtrip_passed"] for r in validation_results.values())
    all_ok = all_ok and all(r.get("per_source_counts_ok", True) for r in validation_results.values())
    all_ok = all_ok and leakage["leakage_free"]

    summary = {
        "eval_manifests": eval_manifests, "mixture_manifests": mixture_manifests,
        "leakage_check": leakage, "validation_results": validation_results, "all_ok": all_ok,
    }
    (PROBE_ROOT / f"{DOMAIN}_phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n\n{DOMAIN.upper()} PHASE 3 OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    assert all_ok, f"{DOMAIN} Phase 3 validation failed -- see {DOMAIN}_phase3_summary.json"


if __name__ == "__main__":
    main()
