"""Creates the 3 theta_0 initializations for hopper (Seed A=0, B=43, C=44),
reusing create_theta0.py's build_theta0() unmodified except for probe_task="stand"
(hopper has no "walk" task -- see collect_source_pools.py's DOMAIN_BEHAVIOR_TASK).
Mirrors create_theta0.py + create_theta0_multiseed.py's exact procedure/seeds.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/create_theta0_hopper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_theta0 import build_theta0, OUT_DIR, THETA0_SEED  # noqa: E402

DOMAIN = "hopper"
SEEDS = {"A": THETA0_SEED, "B": 43, "C": 44}


def main() -> None:
    results = {}
    fingerprints = {}
    for label, seed in SEEDS.items():
        out_path = OUT_DIR / (f"theta_0_{DOMAIN}.pt" if label == "A" else f"theta_0_{DOMAIN}_seed{seed}.pt")
        r = build_theta0(DOMAIN, seed=seed, out_path=out_path, probe_task="stand")
        results[f"{DOMAIN}_{label}"] = r
        fingerprints[label] = r["denoiser_fingerprint_sha256"]

    print("\n=== FINGERPRINT UNIQUENESS CHECK (must all differ) ===")
    unique = len(set(fingerprints.values())) == len(fingerprints)
    print(f"{DOMAIN}: A={fingerprints['A'][:12]}... B={fingerprints['B'][:12]}... "
          f"C={fingerprints['C'][:12]}...  all_unique={unique}")
    assert unique, "theta_0 fingerprints collided across seeds -- something is wrong"

    (OUT_DIR / f"theta0_manifest_{DOMAIN}.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_DIR / f'theta0_manifest_{DOMAIN}.json'}")


if __name__ == "__main__":
    main()
