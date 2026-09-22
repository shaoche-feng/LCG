"""Extends multiseed LCG scoring (phase5_multiseed_scoring.py) to hopper:
Seeds B(43)/C(44), all 3 conditions, reusing process_domain() unmodified
except for probe_task="stand" (hopper has no "walk" task). Resumable: skips
any (seed, condition) already in scoring_results.json.

Run from the LCG/ project root (after phase5_lcg_scoring_hopper.py, which
scores Seed A):
    python scripts/lcg_diagnostic/pretrained_collectors/phase5_multiseed_scoring_hopper.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_lcg_scoring import process_domain, OUT_ROOT, MODELS_ROOT  # noqa: E402

DOMAIN = "hopper"
SEEDS = {43: "B", 44: "C"}
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
MULTISEED_MODELS_ROOT = MODELS_ROOT / "multiseed"
MULTISEED_SCORING_ROOT = OUT_ROOT / "multiseed"


def main() -> None:
    results_path = MULTISEED_SCORING_ROOT / f"{DOMAIN}_scoring_results.json"
    MULTISEED_SCORING_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for seed, label in SEEDS.items():
        for condition in CONDITIONS:
            key = f"seed{seed}/{DOMAIN}/{condition}"
            if key in all_results:
                print(f">>> SKIPPING {key} (already scored)")
                continue
            checkpoint_path = (MULTISEED_MODELS_ROOT / f"seed{seed}" / DOMAIN / cond_dir(DOMAIN, condition) /
                                "checkpoints" / "agent_versions" / "agent_epoch_00001.pt")
            out_dir = MULTISEED_SCORING_ROOT / f"seed{seed}" / DOMAIN / cond_dir(DOMAIN, condition)
            print(f">>> SCORING {key}", flush=True)
            result = process_domain(DOMAIN, condition, checkpoint_path=checkpoint_path, out_dir=out_dir,
                                     probe_task="stand")
            result["seed"] = seed
            result["seed_label"] = label
            all_results[key] = result
            results_path.write_text(json.dumps(all_results, indent=2))
            print(f">>> DONE SCORING {key}", flush=True)

    print(f"ALL_{len(SEEDS) * len(CONDITIONS)}_HOPPER_MULTISEED_SCORING_DONE")


if __name__ == "__main__":
    main()
