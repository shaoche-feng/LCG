"""LCG scoring for the balanced models, using the EXACT SAME held-out
episodes/transition indices/candidate bank seeds/h_D seeds/theta_S/production
LCG config as Phase 5 (run_scarce) and Phase 5b (walk_scarce) -- see
phase5_lcg_scoring.py's module docstring and constants. Only the trained
checkpoint/training dataset differ (condition="balanced").

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5f_balanced_scoring.py
"""
from __future__ import annotations

import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_lcg_scoring import process_domain, DOMAINS, OUT_ROOT  # noqa: E402


def main() -> None:
    results = {}
    for domain in DOMAINS:
        results[domain] = process_domain(domain, "balanced")
    (OUT_ROOT / "phase5f_balanced_summary.json").write_text(json.dumps(results, indent=2))
    print("\n\n=== BALANCED SCORING SUMMARY ===")
    for domain, r in results.items():
        print(f"{domain}: delta_mean={r['delta_mean']:+.4f} ratio={r['ratio_mean']:.4f} "
              f"perm_p_two_sided={r['permutation_test']['p_value_two_sided']:.4f}")


if __name__ == "__main__":
    main()
