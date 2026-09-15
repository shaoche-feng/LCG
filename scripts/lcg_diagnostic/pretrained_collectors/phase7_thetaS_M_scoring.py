"""Phase 7 -- theta_S x M Robustness of the Undersampling Diagnostic.

For all 18 EXISTING trained world models (2 domains x 3 coverage conditions x
3 model-training seeds -- Seed A already trained/scored by Phase 5/5-multiseed,
Seeds B/C by train_multiseed.py) and all 8 previously-defined theta_S
parameter subsets (R1, R2, R3, R1+R2, R1+R3, R2+R3, 3R, full -- reused
UNCHANGED from scripts/lcg_diagnostic/resblock_ablation_diagnostic.py, not
redefined), this script:

  1. computes that model's own h_D (production historical_precision,
     B=320, num_mc=3, beta=1, damping=1e-4, seed=123) for that theta_S, and
  2. runs ONE candidate-scoring pass (candidate_num_mc=12, chunk_size=16,
     Full-CRN, shared Walk+Run bank) over the SAME 1000 held-out candidates
     (500 Walk + 500 Run, episodes {7..11} x 100 evenly-spaced transitions)
     used throughout Phase 5/6, storing every per-probe q_m(x) instead of
     only their running mean.

No world model is retrained. No production code is modified: the per-probe
scorer below is a byte-for-byte reuse of jvp_through_F/unflatten_to_dict/
apply_noise_from_samples/make_jvp_bank (all unmodified production functions),
copied inline from phase5d_rms_diagnostic.py's score_one_jvp_bank_with_probes
rather than imported from that module, because that module also imports
scipy.stats which conflicts with torch's OpenMP runtime in the same process
on this machine (same issue documented in phase6_pilot_ensemble.py).

M=5,6,7,8,9,12 estimators are derived POST-HOC as nested prefix means of the
same 12 stored probes (never independently resampled) -- done in the
torch-free follow-up analysis script, not here.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase7_thetaS_M_scoring.py
"""
from __future__ import annotations

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

from lcg.forward_jvp import JVPBank, make_jvp_bank, jvp_through_F, unflatten_to_dict  # noqa: E402
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from models.diffusion.denoiser import apply_noise_from_samples  # noqa: E402
from data import Dataset  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402 -- reuse Phase 5's exact setup, no reimplementation
    load_agent, evenly_spaced_indices, build_candidates,
    PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC, CANDIDATE_CHUNK_SIZE,
    PRECISION_SEED, HELD_OUT_EPISODE_IDS, N_TRANS_PER_EPISODE, EPISODE_LEN,
    MODELS_ROOT, MIXTURES, SOURCE_POOLS,
)
from phase6_full_ensemble import checkpoint_path  # noqa: E402 -- reuse A/B/C checkpoint-path resolution
from resblock_ablation_diagnostic import THETA_S_CONFIGS, EXPECTED_D_S, EXPECTED_N_TENSORS  # noqa: E402

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEED_LABELS = ["A", "B", "C"]
# "full" scored first (not last) so the reproduction gate against already-saved Phase 5/
# multiseed LCG scores is checked almost immediately, rather than after ~7/8 of each model's
# theta_S sweep -- pure scoring-order choice, does not affect saved file layout or any report.
THETA_S_ORDER = ["full", "R1", "R2", "R3", "R1+R2", "R1+R3", "R2+R3", "3R"]
M_MAX = 12
BANK_SEED = 456  # same documented seed used for every (domain, theta_S) bank -- reused from Phase 5

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase7_thetaS_M_robustness"
LCG_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase5_lcg_scoring"

REPRO_RTOL = 1e-4
REPRO_ATOL = 1e-3


def score_one_jvp_bank_with_probes(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, candidates, chunk_size):
    """Byte-for-byte copy of phase5d_rms_diagnostic.py's function of the same name (see that
    file's docstring for the scipy/torch OMP-conflict reason it is copied here rather than
    imported). Mirrors lcg.forward_jvp.score_one_jvp_bank EXACTLY, storing every per-probe
    q_m(x) = 2*||J_F H_D^{-1/2} eta_m||^2 instead of only its running mean."""
    device = h_D_inv_sqrt.device
    d_S = sum(p.numel() for p in theta_s_named.values())
    assert h_D_inv_sqrt.numel() == d_S

    num_entries = bank.num_samples
    num_candidates = len(candidates)
    q_values = torch.zeros(num_candidates, num_entries, device=device)

    chunks = []
    for start in range(0, num_candidates, chunk_size):
        chunk = candidates[start:start + chunk_size]
        obs_batch = torch.cat([c[0] for c in chunk], dim=0)
        act_batch = torch.cat([c[1] for c in chunk], dim=0)
        y_batch = torch.cat([c[2] for c in chunk], dim=0)
        chunks.append((start, obs_batch, act_batch, y_batch, len(chunk)))

    for m in range(num_entries):
        sigma, eps, eps_offset, eta = bank.sigmas[m], bank.epsilons[m], bank.epsilons_offset[m], bank.etas[m]
        with torch.no_grad():
            z_flat = h_D_inv_sqrt * eta
        tangent_named = unflatten_to_dict(z_flat, theta_s_named)

        for start, obs_batch, act_batch, y_batch, B in chunks:
            y_sigma_batch = apply_noise_from_samples(y_batch, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            _, jvp_out = jvp_through_F(
                denoiser, theta_s_named, frozen_named, tangent_named, y_sigma_batch, sigma, obs_batch, act_batch
            )
            contribution = 2.0 * jvp_out.reshape(B, -1).square().sum(dim=1)
            q_values[start:start + B, m] = contribution

    return q_values.detach()


def build_domain_candidates(domain: str, num_steps_conditioning: int, device):
    """Reproduces phase5_lcg_scoring.process_domain's candidate construction exactly (same
    held-out episodes/indices, same walk-then-run concatenation order, same flattened
    obs/act/y format for JVP scoring)."""
    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    walk_pool = Dataset(SOURCE_POOLS / domain / "walk" / "dataset", name=f"{domain}_walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / domain / "run" / "dataset", name=f"{domain}_run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()

    walk_candidates, walk_meta = build_candidates(walk_pool, HELD_OUT_EPISODE_IDS, indices, num_steps_conditioning, device)
    run_candidates, run_meta = build_candidates(run_pool, HELD_OUT_EPISODE_IDS, indices, num_steps_conditioning, device)
    assert len(walk_candidates) == 500 and len(run_candidates) == 500
    all_candidates = walk_candidates + run_candidates
    all_meta = [{"behavior": "walk", **m} for m in walk_meta] + [{"behavior": "run", **m} for m in run_meta]
    return all_candidates, all_meta


def load_lcg_reference(domain: str, seed_label: str, condition: str) -> dict:
    """Loads Seed A/B/C's already-saved per-candidate LCG scores (full theta_S, M=12,
    production score_one_jvp_bank) for the reproduction gate."""
    if seed_label == "A":
        path = LCG_ROOT / domain / condition / "per_transition_scores.csv"
    else:
        seed_num = {"B": 43, "C": 44}[seed_label]
        path = LCG_ROOT / "multiseed" / f"seed{seed_num}" / domain / condition / "per_transition_scores.csv"
    import csv
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["behavior"], int(row["episode_id"]), int(row["transition_index"]))
            out[key] = float(row["lcg_score"])
    return out


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    repro_gate_results = []

    print(f"theta_S configs to evaluate: {THETA_S_ORDER}")
    print(f"expected d_S: {EXPECTED_D_S}")
    print(f"expected n_tensors: {EXPECTED_N_TENSORS}")

    for domain in DOMAINS:
        print(f"\n{'#' * 70}\nDOMAIN: {domain}\n{'#' * 70}", flush=True)
        bootstrap_ckpt = checkpoint_path(domain, "run_scarce", "A")
        bootstrap_agent, sigma_cfg, _ = load_agent(domain, bootstrap_ckpt)
        n_cond = bootstrap_agent.denoiser.cfg.inner_model.num_steps_conditioning
        device = bootstrap_agent.denoiser.device
        del bootstrap_agent
        torch.cuda.empty_cache()

        candidates, cand_meta = build_domain_candidates(domain, n_cond, device)
        (OUT_ROOT / domain).mkdir(parents=True, exist_ok=True)
        cand_meta_path = OUT_ROOT / domain / "candidates_meta.json"
        if not cand_meta_path.exists():
            cand_meta_path.write_text(json.dumps({
                "domain": domain, "n_candidates": len(candidates), "held_out_episode_ids": HELD_OUT_EPISODE_IDS,
                "n_trans_per_episode": N_TRANS_PER_EPISODE, "episode_len": EPISODE_LEN,
                "candidate_order": "walk (eid-major, t-minor, ascending) then run (same)",
                "candidates": cand_meta,
            }, indent=2))
        print(f"[{domain}] built {len(candidates)} candidates, n_cond={n_cond}", flush=True)

        y_shape = torch.Size([1, 3, 64, 64])
        banks = {}
        for theta_name in THETA_S_ORDER:
            d_S = EXPECTED_D_S[theta_name]
            banks[theta_name] = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=M_MAX, seed=BANK_SEED)
        print(f"[{domain}] built {len(banks)} theta_S banks (M_max={M_MAX}, seed={BANK_SEED})", flush=True)

        for condition in CONDITIONS:
            train_dataset = Dataset(MIXTURES / domain / condition / "dataset",
                                     name=f"{domain}_{condition}_train", cache_in_ram=True)
            train_dataset.load_from_default_path()
            assert train_dataset.num_steps == 5000

            for seed_label in SEED_LABELS:
                ckpt = checkpoint_path(domain, condition, seed_label)
                assert ckpt.exists(), f"missing checkpoint: {ckpt}"
                agent, _, _ = load_agent(domain, ckpt)
                denoiser = agent.denoiser
                lcg_ref = None  # lazily loaded only if/when theta_name == "full"

                for theta_name in THETA_S_ORDER:
                    out_dir = OUT_ROOT / domain / seed_label / condition / theta_name
                    q_path = out_dir / "q_values.npy"
                    meta_path = out_dir / "metadata.json"
                    if q_path.exists() and meta_path.exists():
                        print(f"  SKIP {domain}/{seed_label}/{condition}/{theta_name} (already done)", flush=True)
                        continue

                    theta_cfg = THETA_S_CONFIGS[theta_name]
                    theta_s_named = selected_named_parameters(denoiser, theta_cfg)
                    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
                    d_S = sum(p.numel() for p in theta_s_named.values())
                    assert d_S == EXPECTED_D_S[theta_name], (
                        f"STOP: {domain}/{seed_label}/{condition}/{theta_name} d_S={d_S} != "
                        f"expected {EXPECTED_D_S[theta_name]}")
                    assert len(theta_s_named) == EXPECTED_N_TENSORS[theta_name], (
                        f"STOP: {domain}/{seed_label}/{condition}/{theta_name} n_tensors={len(theta_s_named)} != "
                        f"expected {EXPECTED_N_TENSORS[theta_name]}")

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
                            lcg_ref = load_lcg_reference(domain, seed_label, condition)
                        reconstructed_m12 = q_np.mean(axis=1)
                        existing = np.array([lcg_ref[(m["behavior"], m["episode_id"], m["transition_index"])] for m in cand_meta])
                        abs_err = np.abs(reconstructed_m12 - existing)
                        rel_err = abs_err / np.maximum(np.abs(existing), 1e-8)
                        allclose = bool(np.allclose(reconstructed_m12, existing, rtol=REPRO_RTOL, atol=REPRO_ATOL))
                        pear = float(np.corrcoef(reconstructed_m12, existing)[0, 1])
                        repro = {
                            "max_abs_error": float(abs_err.max()), "mean_abs_error": float(abs_err.mean()),
                            "max_rel_error": float(rel_err.max()), "pearson": pear, "allclose": allclose,
                        }
                        status = "PASS" if allclose else "FAIL"
                        print(f"  REPRO GATE [{domain}/{seed_label}/{condition}] full/M12: {status}  "
                              f"max_abs_err={repro['max_abs_error']:.6g}  mean_abs_err={repro['mean_abs_error']:.6g}  "
                              f"max_rel_err={repro['max_rel_error']:.6g}  pearson={repro['pearson']:.8f}", flush=True)
                        if not allclose:
                            raise RuntimeError(
                                f"STOP: reproduction gate FAILED for {domain}/{seed_label}/{condition} full/M12 -- "
                                f"new per-probe scorer does not reproduce existing saved LCG scores. "
                                f"{repro}")
                        repro_gate_results.append({"domain": domain, "seed": seed_label, "condition": condition, **repro})

                    out_dir.mkdir(parents=True, exist_ok=True)
                    # np.save() auto-appends ".npy" if the given name doesn't already end with
                    # it, so the temp name must itself end in ".npy" or the later replace()
                    # looks for a filename that was never actually written.
                    tmp_path = out_dir / "q_values.tmp.npy"
                    np.save(tmp_path, q_np)
                    tmp_path.replace(q_path)
                    meta_path.write_text(json.dumps({
                        "domain": domain, "seed": seed_label, "condition": condition,
                        "checkpoint_path": str(ckpt), "theta_s_name": theta_name,
                        "theta_s_parameter_names": sorted(theta_s_named.keys()), "d_S": d_S,
                        "n_tensors": len(theta_s_named),
                        "precision_seed": PRECISION_SEED, "precision_reference_size": PRECISION_REFERENCE_SIZE,
                        "precision_num_mc": PRECISION_NUM_MC, "beta": BETA, "damping": DAMPING,
                        "bank_seed": BANK_SEED, "candidate_num_mc": M_MAX, "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
                        "held_out_episode_ids": HELD_OUT_EPISODE_IDS, "n_trans_per_episode": N_TRANS_PER_EPISODE,
                        "h_D_construction_seconds": t_hD, "candidate_scoring_seconds": t_score,
                        "reproduction_gate": repro,
                    }, indent=2))
                    print(f"  DONE {domain}/{seed_label}/{condition}/{theta_name}: "
                          f"d_S={d_S} h_D={t_hD:.1f}s score={t_score:.1f}s", flush=True)

                del agent
                torch.cuda.empty_cache()

        del candidates, banks
        torch.cuda.empty_cache()

    (OUT_ROOT / "reproduction_gate_summary.json").write_text(json.dumps(repro_gate_results, indent=2))
    print(f"\nSaved: {OUT_ROOT / 'reproduction_gate_summary.json'}")
    print("\nPHASE7_SCORING_DONE")


if __name__ == "__main__":
    main()
