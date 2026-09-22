"""Extends Phase 6 (phase6_full_ensemble.py) to hopper: full A/B/C ensemble
disagreement diagnostic on the same 500 stand("walk" slot) + 500 hop("run"
slot) held-out candidates used by Phase 5, all 3 conditions. Reuses
build_domain_candidates/build_domain_noise/process_domain_condition/
load_lcg_scores unmodified except for probe_task="stand" threaded through
(hopper has no "walk" task). Kept as a separate driver so it never touches
the existing walker/quadruped phase6_full_ensemble/ output.

Run from the LCG/ project root (after phase5_lcg_scoring_hopper.py, since
this joins against its per_transition_scores.csv):
    python scripts/lcg_diagnostic/pretrained_collectors/phase6_full_ensemble_hopper.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from phase5_lcg_scoring import load_agent  # noqa: E402
from phase6_full_ensemble import (  # noqa: E402
    checkpoint_path, load_lcg_scores, build_domain_candidates, build_domain_noise,
    process_domain_condition, OUT_ROOT,
)

DOMAIN = "hopper"
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    runtime_log = {}

    print(f"\n{'#' * 70}\nDOMAIN: {DOMAIN}\n{'#' * 70}", flush=True)
    bootstrap_ckpt = checkpoint_path(DOMAIN, "run_scarce", "A")
    bootstrap_agent, _, _ = load_agent(DOMAIN, bootstrap_ckpt, probe_task="stand")
    n_cond = bootstrap_agent.denoiser.cfg.inner_model.num_steps_conditioning
    device = bootstrap_agent.denoiser.device
    del bootstrap_agent

    import torch
    torch.cuda.empty_cache()

    candidates = build_domain_candidates(DOMAIN, n_cond, device)
    noises = build_domain_noise(len(candidates), device)
    print(f"[{DOMAIN}] built {len(candidates)} candidates (500 walk + 500 run), n_cond={n_cond}, "
          f"noise shared across models A/B/C and across {CONDITIONS}")

    for condition in CONDITIONS:
        lcg_scores = load_lcg_scores(DOMAIN, condition)
        out_dir = OUT_ROOT / DOMAIN / cond_dir(DOMAIN, condition)
        r = process_domain_condition(DOMAIN, condition, candidates, noises, lcg_scores, out_dir, probe_task="stand")
        runtime_log[f"{DOMAIN}/{condition}"] = r

    (OUT_ROOT / f"{DOMAIN}_runtime_log.json").write_text(json.dumps(runtime_log, indent=2))
    print(f"\nALL_3_{DOMAIN.upper()}_ENSEMBLES_DONE")


if __name__ == "__main__":
    main()
