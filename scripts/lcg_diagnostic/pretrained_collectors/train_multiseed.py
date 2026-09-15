"""Trains the 12 new models for Seeds B (43) and C (44): 2 domains x 3
conditions x 2 seeds. Reuses train_one() from train_denoiser_only.py
unmodified in procedure -- only seed/theta0_path/run_dir are parametrized.
Seed A's existing 6 models are never touched.

Output layout: models/multiseed/seed{43,44}/{domain}/{condition}/

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/train_multiseed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_denoiser_only import train_one, MODELS_ROOT  # noqa: E402

SEEDS = {43: "B", 44: "C"}
DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
MULTISEED_ROOT = MODELS_ROOT / "multiseed"


def main() -> None:
    results_path = MULTISEED_ROOT / "training_results.json"
    MULTISEED_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for seed, label in SEEDS.items():
        for domain in DOMAINS:
            theta0_path = MODELS_ROOT / f"theta_0_{domain}_seed{seed}.pt"
            for condition in CONDITIONS:
                key = f"seed{seed}/{domain}/{condition}"
                if key in all_results:
                    print(f">>> SKIPPING {key} (already trained)")
                    continue
                run_dir = MULTISEED_ROOT / f"seed{seed}" / domain / condition
                print(f">>> STARTING {key} (Seed {label}={seed})", flush=True)
                result = train_one(domain, condition, seed=seed, theta0_path=theta0_path, run_dir=run_dir)
                result["seed_label"] = label
                all_results[key] = result
                results_path.write_text(json.dumps(all_results, indent=2))
                print(f">>> DONE {key}", flush=True)

    print("ALL_12_MULTISEED_MODELS_DONE")


if __name__ == "__main__":
    main()
