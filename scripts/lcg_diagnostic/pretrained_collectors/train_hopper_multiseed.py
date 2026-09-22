"""Trains all 9 hopper world models: 3 mixture conditions (run_scarce/balanced/
walk_scarce -- "run"/"walk" here mean hop/stand, per collect_source_pools.py's
DOMAIN_BEHAVIOR_TASK alias) x Seeds A(42)/B(43)/C(44). Reuses train_one() from
train_denoiser_only.py unmodified in procedure (probe_task="stand" is the only
hopper-specific parameter, since hopper has no "walk" task). Resumable: skips
any (seed, condition) already present in training_results.json.

Output layout, matching the existing walker/quadruped convention exactly:
    Seed A: models/hopper/{condition}/...
    Seed B/C: models/multiseed/seed{43,44}/hopper/{condition}/...

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/train_hopper_multiseed.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from train_denoiser_only import train_one, MODELS_ROOT, TRAINING_SEED  # noqa: E402

DOMAIN = "hopper"
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = {TRAINING_SEED: "A", 43: "B", 44: "C"}  # 42: "A" (Seed A's existing fixed seed)
MULTISEED_ROOT = MODELS_ROOT / "multiseed"


def run_dir_and_theta0(seed: int, condition: str) -> tuple[Path, Path]:
    if SEEDS[seed] == "A":
        return MODELS_ROOT / DOMAIN / cond_dir(DOMAIN, condition), MODELS_ROOT / f"theta_0_{DOMAIN}.pt"
    return (MULTISEED_ROOT / f"seed{seed}" / DOMAIN / cond_dir(DOMAIN, condition),
            MODELS_ROOT / f"theta_0_{DOMAIN}_seed{seed}.pt")


def main() -> None:
    results_path = MODELS_ROOT / f"{DOMAIN}_training_results.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for seed, label in SEEDS.items():
        for condition in CONDITIONS:
            key = f"seed{seed}({label})/{DOMAIN}/{condition}"
            if key in all_results:
                print(f">>> SKIPPING {key} (already trained)")
                continue
            run_dir, theta0_path = run_dir_and_theta0(seed, condition)
            print(f">>> STARTING {key}", flush=True)
            result = train_one(DOMAIN, condition, seed=seed, theta0_path=theta0_path, run_dir=run_dir,
                                probe_task="stand")
            result["seed_label"] = label
            all_results[key] = result
            results_path.write_text(json.dumps(all_results, indent=2))
            print(f">>> DONE {key}", flush=True)

    print("\n\n=== HOPPER TRAINING SUMMARY (9 runs) ===")
    for key, r in all_results.items():
        print(f"{key:35s} steps={r['denoiser_optimizer_steps_logged']:5d} "
              f"init_loss={r['initial_loss']:.4f} final_loss={r['final_loss']:.4f} "
              f"wall_clock={r['wall_clock_seconds']:.1f}s "
              f"ckpt_ok={r['checkpoint_validation']['checkpoint_reload_passed']}")

    print("ALL_9_HOPPER_MODELS_DONE")


if __name__ == "__main__":
    main()
