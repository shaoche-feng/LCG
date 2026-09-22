"""Extends Phase 9 (phase9_jvp_bank_stability.py) to hopper: JVP-bank-seed x M
stability sweep, Seed A only, all 3 conditions x 4 theta_S x 3 bank seeds x 8
M values. Duplicates phase9's main() loop body (rather than importing/calling
it) scoped to domain="hopper" only, since phase9's main() is a single large
function not already split into a per-domain helper -- editing it in place
would risk the already-completed walker/quadruped run. Every reusable piece
(build_domain_candidates, flatten_obs, load_phase8c_reference, summarize,
score_one_jvp_bank_with_probes, constants) is imported and reused verbatim;
only load_agent calls add probe_task="stand" (hopper has no "walk" task).

Writes into the SAME OUT_ROOT as phase9_jvp_bank_stability.py (results
naturally namespace by domain in both the directory tree and the JSON keys,
so this cannot collide with the existing walker/quadruped output) but its own
hopper_phase9_summary.json results file, to avoid any risk of two processes
racing on the same results file if ever run concurrently.

Run from the LCG/ project root (after phase8c_multiseed_selfy_check_hopper.py,
since the reproduction gate needs its Seed-A output):
    python scripts/lcg_diagnostic/pretrained_collectors/phase9_jvp_bank_stability_hopper.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import csv
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))

from lcg.forward_jvp import make_jvp_bank  # noqa: E402
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402
    load_agent, PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC,
    CANDIDATE_CHUNK_SIZE, PRECISION_SEED, MODELS_ROOT, MIXTURES,
)
from phase6_pilot_ensemble import sample_with_given_noise  # noqa: E402
from phase7_thetaS_M_scoring import score_one_jvp_bank_with_probes  # noqa: E402
from resblock_ablation_diagnostic import THETA_S_CONFIGS, EXPECTED_D_S, EXPECTED_N_TENSORS  # noqa: E402
from phase9_jvp_bank_stability import (  # noqa: E402
    build_domain_candidates, flatten_obs, load_phase8c_reference, summarize,
    BANK_SEEDS, THETA_S_NAMES, M_LIST, Y_SAMPLE_SEED, REPRO_RTOL, REPRO_ATOL, OUT_ROOT,
)

DOMAIN = "hopper"
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUT_ROOT / f"{DOMAIN}_phase9_summary.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    bootstrap_ckpt = MODELS_ROOT / DOMAIN / cond_dir(DOMAIN, "run_scarce") / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
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
    behaviors = np.array([c["behavior"] for c in candidates])
    episode_ids = np.array([c["episode_id"] for c in candidates])
    transition_indices = np.array([c["transition_index"] for c in candidates])
    print(f"\n[{DOMAIN}] built {len(candidates)} held-out candidates, n_cond={n_cond}", flush=True)

    for condition in CONDITIONS:
        print(f"\n{'=' * 70}\n{DOMAIN}/A/{condition}\n{'=' * 70}", flush=True)
        checkpoint_path = MODELS_ROOT / DOMAIN / cond_dir(DOMAIN, condition) / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
        agent, sigma_cfg, _ = load_agent(DOMAIN, checkpoint_path, probe_task="stand")
        denoiser = agent.denoiser

        t0 = time.time()
        gen = torch.Generator(device=device)
        gen.manual_seed(Y_SAMPLE_SEED)
        with torch.no_grad():
            selfy_flat = []
            for c in candidates:
                noise = torch.randn(1, 3, 64, 64, device=device, generator=gen)
                y_star = sample_with_given_noise(denoiser, sigmas, diffusion_cfg.order, c["obs5d"], c["act"], noise)
                selfy_flat.append((flatten_obs(c["obs5d"]), c["act"], y_star.float()))
        print(f"  self-y* generated in {time.time() - t0:.1f}s (shared across all theta_S/bank combos)", flush=True)

        train_dataset = Dataset(MIXTURES / DOMAIN / cond_dir(DOMAIN, condition) / "dataset", name=f"{DOMAIN}_{condition}_train", cache_in_ram=True)
        train_dataset.load_from_default_path()
        assert train_dataset.num_steps == 5000

        for theta_name in THETA_S_NAMES:
            theta_cfg = THETA_S_CONFIGS[theta_name]
            theta_s_named = selected_named_parameters(denoiser, theta_cfg)
            frozen_named = frozen_named_parameters(denoiser, theta_s_named)
            d_S = sum(p.numel() for p in theta_s_named.values())
            assert d_S == EXPECTED_D_S[theta_name], f"STOP: {theta_name} d_S={d_S} != {EXPECTED_D_S[theta_name]}"
            assert len(theta_s_named) == EXPECTED_N_TENSORS[theta_name]

            t0 = time.time()
            h_D = historical_precision(
                denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
                B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
                beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
            )
            assert_setup_valid(theta_s_named, h_D, d_S=d_S)
            h_D_inv_sqrt = h_D.rsqrt()
            t_hD = time.time() - t0

            y_shape = torch.Size([1, 3, 64, 64])
            phase8c_ref = None
            if theta_name == "full":
                phase8c_ref = load_phase8c_reference(DOMAIN, condition)

            for bank_seed in BANK_SEEDS:
                key = f"{DOMAIN}/{condition}/{theta_name}/bank{bank_seed}"
                if key in all_results:
                    print(f"  SKIPPING {key} (already done)", flush=True)
                    continue

                bank = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=CANDIDATE_NUM_MC, seed=bank_seed)
                t0 = time.time()
                q_values = score_one_jvp_bank_with_probes(
                    denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, selfy_flat, CANDIDATE_CHUNK_SIZE
                ).cpu().numpy()
                t_score = time.time() - t0
                assert q_values.shape == (len(candidates), max(M_LIST))

                scores_m12 = q_values[:, :12].mean(axis=1)
                if theta_name == "full" and bank_seed == 456:
                    mismatches = 0
                    for beh, eid, tidx, score in zip(behaviors, episode_ids, transition_indices, scores_m12):
                        ref = phase8c_ref[(beh, int(eid), int(tidx))]
                        if not np.isclose(score, ref, rtol=REPRO_RTOL, atol=REPRO_ATOL):
                            mismatches += 1
                    assert mismatches == 0, (
                        f"STOP: reproduction gate failed for {DOMAIN}/{condition}/full/bank456/M12: "
                        f"{mismatches}/{len(scores_m12)} transitions disagree with Phase 8c's saved "
                        f"self-y scores beyond rtol={REPRO_RTOL}/atol={REPRO_ATOL}"
                    )
                    print(f"  reproduction gate OK: full/bank456/M12 matches Phase 8c self-y scores "
                          f"(0/{len(scores_m12)} mismatches)", flush=True)

                m_results = {}
                per_m_scores = {}
                for M in M_LIST:
                    scores_M = q_values[:, :M].mean(axis=1)
                    per_m_scores[M] = scores_M
                    result = summarize(scores_M, behaviors, episode_ids)
                    m_results[str(M)] = result

                print(f"  {theta_name}/bank{bank_seed}: (hD={t_hD:.1f}s score={t_score:.1f}s) "
                      + ", ".join(f"M{M}: delta={m_results[str(M)]['delta_mean']:+.3f}"
                                  f"({'sig' if m_results[str(M)]['significant'] else 'ns'})" for M in M_LIST),
                      flush=True)

                out_dir = OUT_ROOT / DOMAIN / cond_dir(DOMAIN, condition) / theta_name / f"bank{bank_seed}"
                out_dir.mkdir(parents=True, exist_ok=True)
                m_fieldnames = [f"self_y_lcg_score_M{M}" for M in M_LIST]
                with open(out_dir / "per_transition_scores.csv", "w", newline="") as f:
                    writer = csv.DictWriter(f, fieldnames=["domain", "condition", "theta_s", "bank_seed",
                                                            "behavior", "episode_id", "transition_index",
                                                            *m_fieldnames])
                    writer.writeheader()
                    for i, (beh, eid, tidx) in enumerate(zip(behaviors, episode_ids, transition_indices)):
                        row = {"domain": DOMAIN, "condition": condition, "theta_s": theta_name,
                               "bank_seed": bank_seed, "behavior": beh, "episode_id": int(eid),
                               "transition_index": int(tidx)}
                        for M in M_LIST:
                            row[f"self_y_lcg_score_M{M}"] = float(per_m_scores[M][i])
                        writer.writerow(row)

                all_results[key] = {"h_D_seconds": t_hD, "scoring_seconds": t_score, "by_M": m_results}
                results_path.write_text(json.dumps(all_results, indent=2, default=str))

        del agent
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print(f"{DOMAIN.upper()} PHASE 9 SUMMARY: self-y Delta stability across 3 JVP banks, per theta_S, at M=12")
    print("=" * 100)
    for condition in CONDITIONS:
        for theta_name in THETA_S_NAMES:
            keys = [f"{DOMAIN}/{condition}/{theta_name}/bank{s}" for s in BANK_SEEDS]
            if all(k in all_results for k in keys):
                deltas = [all_results[k]["by_M"]["12"]["delta_mean"] for k in keys]
                print(f"{DOMAIN}/{condition}/{theta_name:<6} M12 deltas={[f'{d:+.3f}' for d in deltas]}  "
                      f"mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
