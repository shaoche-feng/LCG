"""Reversed-scarcity control: LCG scoring for the walk_scarce models, using
the EXACT SAME held-out episodes/transition indices/candidate bank seeds/h_D
seeds/theta_S/production LCG config as Phase 5's run_scarce scoring (see
phase5_lcg_scoring.py's module docstring and constants -- nothing about the
evaluation protocol changes here, only which trained checkpoint is scored).

Also produces the final side-by-side run_scarce vs walk_scarce comparison
(the sign-reversal test) by combining this run's results with Phase 5's
already-saved phase5_summary.json.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5b_walkscarce_scoring.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_lcg_scoring import process_domain, DOMAINS, OUT_ROOT  # noqa: E402


def main() -> None:
    walkscarce_results = {}
    for domain in DOMAINS:
        walkscarce_results[domain] = process_domain(domain, "walk_scarce")
    (OUT_ROOT / "phase5b_walkscarce_summary.json").write_text(json.dumps(walkscarce_results, indent=2))

    runscarce_summary = json.loads((OUT_ROOT / "phase5_summary.json").read_text())

    print("\n\n=== REVERSED-SCARCITY COMPARISON (sign-reversal test) ===")
    comparison = {}
    for domain in DOMAINS:
        rs = runscarce_summary[domain]
        ws = walkscarce_results[domain]
        delta_runscarce = rs["delta_mean"]
        delta_walkscarce = ws["delta_mean"]
        sign_reversed = (delta_runscarce > 0) and (delta_walkscarce < 0)
        comparison[domain] = {
            "delta_runscarce_run_minus_walk": delta_runscarce,
            "delta_walkscarce_run_minus_walk": delta_walkscarce,
            "runscarce_direction_correct (>0)": delta_runscarce > 0,
            "walkscarce_direction_correct (<0)": delta_walkscarce < 0,
            "sign_reversal_observed": sign_reversed,
            "runscarce_perm_p_one_sided_run_gt_walk": rs["permutation_test"]["p_value_one_sided_run_gt_walk"],
            "runscarce_perm_p_two_sided": rs["permutation_test"]["p_value_two_sided"],
            "walkscarce_perm_p_one_sided_walk_gt_run": ws["permutation_test"]["p_value_one_sided_walk_gt_run"],
            "walkscarce_perm_p_two_sided": ws["permutation_test"]["p_value_two_sided"],
            "runscarce_denoising_loss_walk": rs["walk_denoising_loss_mean"],
            "runscarce_denoising_loss_run": rs["run_denoising_loss_mean"],
            "walkscarce_denoising_loss_walk": ws["walk_denoising_loss_mean"],
            "walkscarce_denoising_loss_run": ws["run_denoising_loss_mean"],
        }
        print(f"\n{domain}:")
        print(f"  Delta_runscarce  (run-walk) = {delta_runscarce:+.4f}  (expect >0)")
        print(f"  Delta_walkscarce (run-walk) = {delta_walkscarce:+.4f}  (expect <0)")
        print(f"  SIGN REVERSAL OBSERVED: {sign_reversed}")

    (OUT_ROOT / "reversed_scarcity_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\nSaved: {OUT_ROOT / 'reversed_scarcity_comparison.json'}")


if __name__ == "__main__":
    main()
