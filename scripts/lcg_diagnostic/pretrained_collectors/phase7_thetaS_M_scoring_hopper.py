"""Extends Phase 7 (phase7_thetaS_M_scoring.py) to hopper: theta_S x M
robustness sweep for all 9 existing hopper models (3 conditions x seeds
A/B/C) x all 8 theta_S subsets. Reuses score_one_jvp_bank_with_probes /
build_domain_candidates / load_lcg_reference unmodified except for
probe_task="stand" threaded through load_agent (hopper has no "walk" task).

EXPECTED_D_S/EXPECTED_N_TENSORS are reused UNCHANGED from
resblock_ablation_diagnostic.py -- verified this session that theta_S="full"
gives IDENTICAL d_S=654851 for hopper as for walker/quadruped (confirmed via
Phase 5's printed theta_s_dim), since these are resblock/U-Net parameter
counts that don't depend on continuous_action_dim. Same reproduction gate as
the original (checks theta_S="full"/M=12 against Phase 5/multiseed's already-
saved LCG scores) -- if this fails for any (seed, condition), STOP and
investigate rather than silently accept a mismatch.

Run from the LCG/ project root (after phase5_lcg_scoring_hopper.py AND
phase5_multiseed_scoring_hopper.py, since the reproduction gate needs both):
    python scripts/lcg_diagnostic/pretrained_collectors/phase7_thetaS_M_scoring_hopper.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))  # for resblock_ablation_diagnostic.py's THETA_S_CONFIGS

from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from lcg.forward_jvp import make_jvp_bank  # noqa: E402
from data import Dataset  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402
    load_agent, PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_CHUNK_SIZE,
    PRECISION_SEED, HELD_OUT_EPISODE_IDS, N_TRANS_PER_EPISODE, MIXTURES,
)
from phase6_full_ensemble import checkpoint_path  # noqa: E402
from phase7_thetaS_M_scoring import (  # noqa: E402 -- reuse the per-probe scorer + candidate builder unmodified
    score_one_jvp_bank_with_probes, build_domain_candidates, load_lcg_reference,
)
from resblock_ablation_diagnostic import THETA_S_CONFIGS, EXPECTED_D_S, EXPECTED_N_TENSORS  # noqa: E402

DOMAIN = "hopper"
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEED_LABELS = ["A", "B", "C"]
THETA_S_ORDER = ["full", "R1", "R2", "R3", "R1+R2", "R1+R3", "R2+R3", "3R"]
M_MAX = 12
BANK_SEED = 456

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase7_thetaS_M_robustness"

REPRO_RTOL = 1e-4
REPRO_ATOL = 1e-3


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    repro_gate_results = []

    print(f"theta_S configs to evaluate: {THETA_S_ORDER}")
    print(f"expected d_S: {EXPECTED_D_S}")

    print(f"\n{'#' * 70}\nDOMAIN: {DOMAIN}\n{'#' * 70}", flush=True)
    bootstrap_ckpt = checkpoint_path(DOMAIN, "run_scarce", "A")
    bootstrap_agent, sigma_cfg, _ = load_agent(DOMAIN, bootstrap_ckpt, probe_task="stand")
    n_cond = bootstrap_agent.denoiser.cfg.inner_model.num_steps_conditioning
    device = bootstrap_agent.denoiser.device
    del bootstrap_agent
    torch.cuda.empty_cache()

    candidates, cand_meta = build_domain_candidates(DOMAIN, n_cond, device)
    (OUT_ROOT / DOMAIN).mkdir(parents=True, exist_ok=True)
    cand_meta_path = OUT_ROOT / DOMAIN / "candidates_meta.json"
    if not cand_meta_path.exists():
        cand_meta_path.write_text(json.dumps({
            "domain": DOMAIN, "n_candidates": len(candidates), "held_out_episode_ids": HELD_OUT_EPISODE_IDS,
            "n_trans_per_episode": N_TRANS_PER_EPISODE, "candidates": cand_meta,
        }, indent=2))
    print(f"[{DOMAIN}] built {len(candidates)} candidates, n_cond={n_cond}", flush=True)

    y_shape = torch.Size([1, 3, 64, 64])
    banks = {}
    for theta_name in THETA_S_ORDER:
        d_S = EXPECTED_D_S[theta_name]
        banks[theta_name] = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=M_MAX, seed=BANK_SEED)
    print(f"[{DOMAIN}] built {len(banks)} theta_S banks (M_max={M_MAX}, seed={BANK_SEED})", flush=True)

    for condition in CONDITIONS:
        train_dataset = Dataset(MIXTURES / DOMAIN / cond_dir(DOMAIN, condition) / "dataset",
                                 name=f"{DOMAIN}_{condition}_train", cache_in_ram=True)
        train_dataset.load_from_default_path()
        assert train_dataset.num_steps == 5000

        for seed_label in SEED_LABELS:
            ckpt = checkpoint_path(DOMAIN, condition, seed_label)
            assert ckpt.exists(), f"missing checkpoint: {ckpt}"
            agent, _, _ = load_agent(DOMAIN, ckpt, probe_task="stand")
            denoiser = agent.denoiser
            lcg_ref = None

            for theta_name in THETA_S_ORDER:
                out_dir = OUT_ROOT / DOMAIN / seed_label / cond_dir(DOMAIN, condition) / theta_name
                q_path = out_dir / "q_values.npy"
                meta_path = out_dir / "metadata.json"
                if q_path.exists() and meta_path.exists():
                    print(f"  SKIP {DOMAIN}/{seed_label}/{condition}/{theta_name} (already done)", flush=True)
                    continue

                theta_cfg = THETA_S_CONFIGS[theta_name]
                theta_s_named = selected_named_parameters(denoiser, theta_cfg)
                frozen_named = frozen_named_parameters(denoiser, theta_s_named)
                d_S = sum(p.numel() for p in theta_s_named.values())
                assert d_S == EXPECTED_D_S[theta_name], (
                    f"STOP: {DOMAIN}/{seed_label}/{condition}/{theta_name} d_S={d_S} != "
                    f"expected {EXPECTED_D_S[theta_name]} -- hopper's architecture may not match "
                    f"walker/quadruped's after all; do not proceed without investigating.")
                assert len(theta_s_named) == EXPECTED_N_TENSORS[theta_name]

                t0 = time.time()
                h_D = historical_precision(
                    denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
                    B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
                    beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
                )
                assert_setup_valid(theta_s_named, h_D, d_S=d_S)
                h_D_inv_sqrt = h_D.rsqrt()
                assert torch.isfinite(h_D_inv_sqrt).all()
                t_hD = time.time() - t0

                t0 = time.time()
                q_values = score_one_jvp_bank_with_probes(
                    denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, banks[theta_name], candidates, CANDIDATE_CHUNK_SIZE
                )
                t_score = time.time() - t0
                assert torch.isfinite(q_values).all()
                q_np = q_values.cpu().numpy().astype(np.float32)
                assert q_np.shape == (len(candidates), M_MAX)

                repro = None
                if theta_name == "full":
                    if lcg_ref is None:
                        lcg_ref = load_lcg_reference(DOMAIN, seed_label, condition)
                    reconstructed_m12 = q_np.mean(axis=1)
                    existing = np.array([lcg_ref[(m["behavior"], m["episode_id"], m["transition_index"])] for m in cand_meta])
                    abs_err = np.abs(reconstructed_m12 - existing)
                    rel_err = abs_err / np.maximum(np.abs(existing), 1e-8)
                    allclose = bool(np.allclose(reconstructed_m12, existing, rtol=REPRO_RTOL, atol=REPRO_ATOL))
                    pear = float(np.corrcoef(reconstructed_m12, existing)[0, 1])
                    repro = {"max_abs_error": float(abs_err.max()), "mean_abs_error": float(abs_err.mean()),
                             "max_rel_error": float(rel_err.max()), "pearson": pear, "allclose": allclose}
                    status = "PASS" if allclose else "FAIL"
                    print(f"  REPRO GATE [{DOMAIN}/{seed_label}/{condition}] full/M12: {status}  "
                          f"max_abs_err={repro['max_abs_error']:.6g}  pearson={repro['pearson']:.8f}", flush=True)
                    if not allclose:
                        raise RuntimeError(f"STOP: reproduction gate FAILED for {DOMAIN}/{seed_label}/{condition} "
                                            f"full/M12 -- {repro}")
                    repro_gate_results.append({"domain": DOMAIN, "seed": seed_label, "condition": condition, **repro})

                out_dir.mkdir(parents=True, exist_ok=True)
                tmp_path = out_dir / "q_values.tmp.npy"
                np.save(tmp_path, q_np)
                tmp_path.replace(q_path)
                meta_path.write_text(json.dumps({
                    "domain": DOMAIN, "seed": seed_label, "condition": condition,
                    "checkpoint_path": str(ckpt), "theta_s_name": theta_name, "d_S": d_S,
                    "n_tensors": len(theta_s_named), "bank_seed": BANK_SEED, "candidate_num_mc": M_MAX,
                    "h_D_construction_seconds": t_hD, "candidate_scoring_seconds": t_score,
                    "reproduction_gate": repro,
                }, indent=2))
                print(f"  DONE {DOMAIN}/{seed_label}/{condition}/{theta_name}: "
                      f"d_S={d_S} h_D={t_hD:.1f}s score={t_score:.1f}s", flush=True)

            del agent
            torch.cuda.empty_cache()

    (OUT_ROOT / f"{DOMAIN}_reproduction_gate_summary.json").write_text(json.dumps(repro_gate_results, indent=2))
    print(f"\nSaved: {OUT_ROOT / f'{DOMAIN}_reproduction_gate_summary.json'}")
    print(f"\n{DOMAIN.upper()}_PHASE7_SCORING_DONE")


if __name__ == "__main__":
    main()
