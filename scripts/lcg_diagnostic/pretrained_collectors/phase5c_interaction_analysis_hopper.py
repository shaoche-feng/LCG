"""Extends Phase 5c (phase5c_interaction_analysis.py) to hopper: the paired
bootstrap + exact sign-flip test for the scarcity x behavior interaction
(Delta_run_scarce - Delta_walk_scarce). Reuses analyze_domain()/make_figure()
unmodified -- purely reads Phase 5's already-saved summary.json files, no
torch/GPU needed. Kept as a separate driver so it never touches the existing
walker/quadruped interaction_analysis.json / sign_reversal_figure.png.

Run from the LCG/ project root (after phase5_lcg_scoring_hopper.py):
    python scripts/lcg_diagnostic/pretrained_collectors/phase5c_interaction_analysis_hopper.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5c_interaction_analysis import analyze_domain, make_figure, SCORING_ROOT  # noqa: E402
from hopper_naming import slot_dir  # noqa: E402

DOMAIN = "hopper"


def main() -> None:
    result = analyze_domain(DOMAIN)

    print("\n=== HOPPER INTERACTION TABLE ===")
    print(f"{'domain':<10} {'Delta_rs':>10} {'Delta_ws':>10} {'interaction':>12} "
          f"{'CI_low':>9} {'CI_high':>9} {'p(1-sided)':>11} {'p(2-sided)':>11}")
    b, p = result["interaction_bootstrap"], result["interaction_permutation"]
    print(f"{DOMAIN:<10} {result['delta_run_scarce']:>10.4f} {result['delta_walk_scarce']:>10.4f} "
          f"{result['interaction_point_estimate']:>12.4f} {b['ci_2.5']:>9.4f} {b['ci_97.5']:>9.4f} "
          f"{p['p_value_one_sided_interaction_gt_0']:>11.4f} {p['p_value_two_sided']:>11.4f}")

    print(f"\n=== {DOMAIN.upper()} WALK_SCARCE OVERLAP CHECK ===")
    print(result["walk_scarce_overlap_note"])

    print("\n=== SECONDARY: held-out denoising loss (NOT used for LCG significance) ===")
    d = result["secondary_denoising_loss"]
    print(f"{DOMAIN}: run_scarce(walk={d['run_scarce_walk']:.5f}, run={d['run_scarce_run']:.5f})  "
          f"walk_scarce(walk={d['walk_scarce_walk']:.5f}, run={d['walk_scarce_run']:.5f})")

    make_figure({DOMAIN: result}, SCORING_ROOT / f"{DOMAIN}_sign_reversal_figure.png",
                run_label=slot_dir(DOMAIN, "run"), walk_label=slot_dir(DOMAIN, "walk"))
    (SCORING_ROOT / f"{DOMAIN}_interaction_analysis.json").write_text(json.dumps({DOMAIN: result}, indent=2, default=str))
    print(f"\nSaved: {SCORING_ROOT / f'{DOMAIN}_interaction_analysis.json'}")


if __name__ == "__main__":
    main()
