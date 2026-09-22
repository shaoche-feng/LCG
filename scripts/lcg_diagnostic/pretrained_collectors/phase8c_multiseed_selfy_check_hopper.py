"""Extends Phase 8c (phase8c_multiseed_selfy_check.py) to hopper:
production-faithful self-generated-y* check, all 3 conditions x seeds A/B/C
(9 evaluations). Reuses build_domain_candidates/process_seed unmodified
except for probe_task="stand" threaded through (hopper has no "walk" task).
Resumable: skips any (seed, condition) already in phase8c_summary.json.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase8c_multiseed_selfy_check_hopper.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))

from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset  # noqa: E402
from phase5_lcg_scoring import load_agent, MIXTURES  # noqa: E402
from phase6_full_ensemble import checkpoint_path  # noqa: E402
from phase8c_multiseed_selfy_check import build_domain_candidates, process_seed, OUT_ROOT  # noqa: E402

DOMAIN = "hopper"
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUT_ROOT / f"{DOMAIN}_phase8c_summary.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    bootstrap_ckpt = checkpoint_path(DOMAIN, "run_scarce", "A")
    bootstrap_agent, _, _ = load_agent(DOMAIN, bootstrap_ckpt, probe_task="stand")
    n_cond = bootstrap_agent.denoiser.cfg.inner_model.num_steps_conditioning
    device = bootstrap_agent.denoiser.device
    del bootstrap_agent
    torch.cuda.empty_cache()

    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    diffusion_cfg = instantiate(cfg.world_model_env.diffusion_sampler)
    assert diffusion_cfg.s_churn == 0.0
    sigmas = build_sigmas(diffusion_cfg.num_steps_denoising, diffusion_cfg.sigma_min,
                           diffusion_cfg.sigma_max, diffusion_cfg.rho, device)

    candidates = build_domain_candidates(DOMAIN, n_cond, device)
    print(f"\n[{DOMAIN}] built {len(candidates)} held-out candidates, n_cond={n_cond}", flush=True)

    for condition in CONDITIONS:
        train_dataset = Dataset(MIXTURES / DOMAIN / cond_dir(DOMAIN, condition) / "dataset", name=f"{DOMAIN}_{condition}_train", cache_in_ram=True)
        train_dataset.load_from_default_path()
        assert train_dataset.num_steps == 5000

        for seed in SEEDS:
            key = f"{DOMAIN}/{seed}/{condition}"
            if key in all_results:
                print(f">>> SKIPPING {key} (already done)", flush=True)
                continue
            print(f"\n{'=' * 70}\n{key}\n{'=' * 70}", flush=True)
            all_results[key] = process_seed(DOMAIN, condition, seed, candidates, train_dataset, sigmas,
                                             diffusion_cfg.order, device, probe_task="stand")
            results_path.write_text(json.dumps(all_results, indent=2, default=str))

    print("\n" + "=" * 100)
    print(f"{DOMAIN.upper()} PHASE 8c SUMMARY: real-y Delta vs self-y Delta, all seed/conditions")
    print("=" * 100)
    for key, r in all_results.items():
        print(f"{key:<28} real_y={r['real_y']['delta_mean']:+9.4f} (sig={r['real_y']['significant']})   "
              f"self_y={r['self_y']['delta_mean']:+9.4f} (sig={r['self_y']['significant']})   "
              f"ratio={r['delta_ratio_selfy_over_realy']:+.3f}   same_sign={r['same_sign']}")

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
