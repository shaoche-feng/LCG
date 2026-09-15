"""Creates theta_0 initializations for Seed B (43) and Seed C (44), paired
per-domain exactly like Seed A (0): reuses create_theta0.py's build_theta0()
unmodified (just parametrized), so the construction procedure is identical
to Seed A's, only the torch seed differs.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/create_theta0_multiseed.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from create_theta0 import build_theta0, fingerprint, OUT_DIR, DOMAINS  # noqa: E402

NEW_SEEDS = {"B": 43, "C": 44}


def main() -> None:
    existing = json.loads((OUT_DIR / "theta0_manifest.json").read_text())
    all_fingerprints = {domain: {"A": existing[domain]["denoiser_fingerprint_sha256"]} for domain in DOMAINS}

    results = {}
    for label, seed in NEW_SEEDS.items():
        for domain in DOMAINS:
            out_path = OUT_DIR / f"theta_0_{domain}_seed{seed}.pt"
            r = build_theta0(domain, seed=seed, out_path=out_path)
            results[f"{domain}_seed{seed}"] = r
            all_fingerprints[domain][label] = r["denoiser_fingerprint_sha256"]

    print("\n=== FINGERPRINT UNIQUENESS CHECK (must all differ within each domain) ===")
    all_ok = True
    for domain in DOMAINS:
        fps = all_fingerprints[domain]
        unique = len(set(fps.values())) == len(fps)
        all_ok = all_ok and unique
        print(f"{domain}: A={fps['A'][:12]}... B={fps['B'][:12]}... C={fps['C'][:12]}...  all_unique={unique}")
    assert all_ok, "theta_0 fingerprints collided across seeds -- something is wrong"

    (OUT_DIR / "theta0_manifest_multiseed.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved: {OUT_DIR / 'theta0_manifest_multiseed.json'}")


if __name__ == "__main__":
    main()
