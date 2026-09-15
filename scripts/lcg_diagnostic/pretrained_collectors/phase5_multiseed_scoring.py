"""LCG scoring for all 12 Seed B/C models, using the EXACT SAME held-out
episodes/transition indices/candidate bank seed/h_D seed/theta_S/production
LCG config as Seed A (phase5_lcg_scoring.py's module-level constants,
imported unmodified). Only the checkpoint path (and output directory) vary
per seed; the training-dataset path for h_D construction is the SAME
mixture dataset as Seed A for that domain/condition (mixture composition
does not vary by model seed).

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5_multiseed_scoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_lcg_scoring import process_domain, OUT_ROOT, MODELS_ROOT  # noqa: E402

SEEDS = {43: "B", 44: "C"}
DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
MULTISEED_MODELS_ROOT = MODELS_ROOT / "multiseed"
MULTISEED_SCORING_ROOT = OUT_ROOT / "multiseed"


def main() -> None:
    results_path = MULTISEED_SCORING_ROOT / "scoring_results.json"
    MULTISEED_SCORING_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for seed, label in SEEDS.items():
        for domain in DOMAINS:
            for condition in CONDITIONS:
                key = f"seed{seed}/{domain}/{condition}"
                if key in all_results:
                    print(f">>> SKIPPING {key} (already scored)")
                    continue
                checkpoint_path = (MULTISEED_MODELS_ROOT / f"seed{seed}" / domain / condition /
                                    "checkpoints" / "agent_versions" / "agent_epoch_00001.pt")
                out_dir = MULTISEED_SCORING_ROOT / f"seed{seed}" / domain / condition
                print(f">>> SCORING {key}", flush=True)
                result = process_domain(domain, condition, checkpoint_path=checkpoint_path, out_dir=out_dir)
                result["seed"] = seed
                result["seed_label"] = label
                all_results[key] = result
                results_path.write_text(json.dumps(all_results, indent=2))
                print(f">>> DONE SCORING {key}", flush=True)

    print("ALL_12_MULTISEED_SCORING_DONE")


if __name__ == "__main__":
    main()
