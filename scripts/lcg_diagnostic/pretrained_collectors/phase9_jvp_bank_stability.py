"""Phase 9 -- JVP-bank AND probe-count (M) stability of LCG under the
self-generated-y* (production-faithful) setting: every prior phase (5-8)
reused the exact same candidate JVP bank seed (BANK_SEED=456) and M=12
throughout. This phase asks whether the undersampling result depends on
which random bank (which specific probe directions/sigma/eps draws) happened
to get used, AND how many of its probes are used, by independently
redrawing the bank 2 more times and, for each bank, deriving M=5..12 as
nested-prefix means of the same 12 stored per-probe values (same trick as
Phase 7 -- never independently resampled per M).

Scope: Seed A only, 4 theta_S subsets (R1, R1+R2, 3R, full/default -- reused
UNCHANGED from resblock_ablation_diagnostic.py), both domains x all 3
conditions, 3 bank seeds each (456 = original/reference, plus two new
documented seeds 654321 and 987654), M in {5,...,12}.

Efficiency note: self-generated y* depends only on the trained denoiser (via
the production DiffusionSampler), never on theta_S or the candidate bank --
so it is generated ONCE per domain/condition and reused across all 4 theta_S
x 3 bank_seed combinations for that domain/condition. h_D depends on
theta_S + model but not on the bank, so it is likewise computed once per
theta_S and reused across the 3 bank seeds. Only ONE M=12 scoring pass is
run per (theta_S, bank_seed) -- all 8 M values come free from the same pass.

No retraining, no production code changes -- score_one_jvp_bank_with_probes
mirrors production score_one_jvp_bank exactly (see phase7_thetaS_M_scoring.py
/ phase5d_rms_diagnostic.py for the byte-for-byte-reused pieces), storing
per-probe q_m instead of only their running mean.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase9_jvp_bank_stability.py
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
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_THIS_DIR.parent))  # for resblock_ablation_diagnostic.py's THETA_S_CONFIGS

from lcg.forward_jvp import make_jvp_bank  # noqa: E402
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402
    load_agent, evenly_spaced_indices, HELD_OUT_EPISODE_IDS, EPISODE_LEN, N_TRANS_PER_EPISODE,
    PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC, CANDIDATE_CHUNK_SIZE,
    PRECISION_SEED, MODELS_ROOT, MIXTURES, SOURCE_POOLS,
    describe, episode_level_means, bootstrap_ci, exact_permutation_test,
)
from phase6_pilot_ensemble import load_transition_5d, sample_with_given_noise  # noqa: E402
from resblock_ablation_diagnostic import THETA_S_CONFIGS, EXPECTED_D_S, EXPECTED_N_TENSORS  # noqa: E402
from phase7_thetaS_M_scoring import score_one_jvp_bank_with_probes  # noqa: E402 -- torch-only, no scipy

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
THETA_S_NAMES = ["R1", "R1+R2", "3R", "full"]
BANK_SEEDS = [456, 654321, 987654]  # 456 = original/reference seed reused throughout Phases 5-8
M_LIST = [5, 6, 7, 8, 9, 10, 11, 12]
Y_SAMPLE_SEED = 24681  # same seed used throughout Phase 8, for direct comparability
NUM_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 999
REPRO_RTOL = 1e-4
REPRO_ATOL = 1e-3

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase9_jvp_bank_stability"
PHASE8C_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase8_imagined_rollout" / "phase8c_multiseed_selfy_check"


def build_domain_candidates(domain: str, n_cond: int, device):
    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    walk_pool = Dataset(SOURCE_POOLS / domain / slot_dir(domain, "walk") / "dataset", name=f"{domain}_walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / domain / slot_dir(domain, "run") / "dataset", name=f"{domain}_run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()
    pools = {"walk": walk_pool, "run": run_pool}

    candidates = []
    for behavior in ("walk", "run"):
        for eid in HELD_OUT_EPISODE_IDS:
            for t in indices:
                t = int(t)
                seg = SegmentId(int(eid), t + 1 - (n_cond + 1), t + 1)
                obs5d, act, y_real = load_transition_5d(pools[behavior], seg, n_cond, device)
                candidates.append({"behavior": behavior, "episode_id": int(eid), "transition_index": t,
                                    "obs5d": obs5d, "act": act})
    assert len(candidates) == 1000
    return candidates


def flatten_obs(obs5d: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = obs5d.shape
    return obs5d.reshape(b, t * c, h, w)


def load_phase8c_reference(domain: str, condition: str) -> dict:
    """Phase 8c's already-saved Seed-A self-y scores (full theta_S, M=12,
    bank_seed=456) -- used as a reproduction-gate reference for that one
    specific (theta_S=full, bank_seed=456, M=12) cell."""
    path = PHASE8C_ROOT / domain / "A" / cond_dir(domain, condition) / "per_transition_scores.csv"
    out = {}
    with open(path, newline="") as f:
        for row in csv.DictReader(f):
            key = (row["behavior"], int(row["episode_id"]), int(row["transition_index"]))
            out[key] = float(row["self_y_lcg_score"])
    return out


def summarize(scores: np.ndarray, behaviors: np.ndarray, episode_ids: np.ndarray) -> dict:
    walk_scores, run_scores = scores[behaviors == "walk"], scores[behaviors == "run"]
    walk_ep = episode_level_means(walk_scores, episode_ids[behaviors == "walk"])
    run_ep = episode_level_means(run_scores, episode_ids[behaviors == "run"])
    walk_ep_means = np.array([v["mean"] for v in walk_ep.values()])
    run_ep_means = np.array([v["mean"] for v in run_ep.values()])
    boot = bootstrap_ci(walk_ep_means, run_ep_means, NUM_BOOTSTRAP, BOOTSTRAP_SEED)
    perm = exact_permutation_test(walk_ep_means, run_ep_means)
    walk_desc, run_desc = describe(walk_scores), describe(run_scores)
    delta = run_desc["mean"] - walk_desc["mean"]
    significant = not (boot["ci_2.5"] <= 0 <= boot["ci_97.5"])
    return {"walk_stats": walk_desc, "run_stats": run_desc, "delta_mean": delta,
            "bootstrap": boot, "permutation_test": perm, "significant": significant}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUT_ROOT / "phase9_summary.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for domain in DOMAINS:
        bootstrap_ckpt = MODELS_ROOT / domain / cond_dir(domain, "run_scarce") / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
        bootstrap_agent, _, _ = load_agent(domain, bootstrap_ckpt)
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

        candidates = build_domain_candidates(domain, n_cond, device)
        behaviors = np.array([c["behavior"] for c in candidates])
        episode_ids = np.array([c["episode_id"] for c in candidates])
        transition_indices = np.array([c["transition_index"] for c in candidates])
        print(f"\n[{domain}] built {len(candidates)} held-out candidates, n_cond={n_cond}", flush=True)

        for condition in CONDITIONS:
            print(f"\n{'=' * 70}\n{domain}/A/{condition}\n{'=' * 70}", flush=True)
            checkpoint_path = MODELS_ROOT / domain / cond_dir(domain, condition) / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
            agent, sigma_cfg, _ = load_agent(domain, checkpoint_path)
            denoiser = agent.denoiser

            # --- self-y* generated ONCE per domain/condition (theta_S/bank-independent) ---
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

            train_dataset = Dataset(MIXTURES / domain / cond_dir(domain, condition) / "dataset", name=f"{domain}_{condition}_train", cache_in_ram=True)
            train_dataset.load_from_default_path()
            assert train_dataset.num_steps == 5000

            for theta_name in THETA_S_NAMES:
                theta_cfg = THETA_S_CONFIGS[theta_name]
                theta_s_named = selected_named_parameters(denoiser, theta_cfg)
                frozen_named = frozen_named_parameters(denoiser, theta_s_named)
                d_S = sum(p.numel() for p in theta_s_named.values())
                assert d_S == EXPECTED_D_S[theta_name], f"STOP: {theta_name} d_S={d_S} != {EXPECTED_D_S[theta_name]}"
                assert len(theta_s_named) == EXPECTED_N_TENSORS[theta_name]

                # --- h_D computed ONCE per theta_S (bank-independent) ---
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
                    phase8c_ref = load_phase8c_reference(domain, condition)

                for bank_seed in BANK_SEEDS:
                    key = f"{domain}/{condition}/{theta_name}/bank{bank_seed}"
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

                    # M=12 mean, all probes -- this is the value score_one_jvp_bank would have
                    # returned directly; used below for the reproduction-gate check.
                    scores_m12 = q_values[:, :12].mean(axis=1)
                    if theta_name == "full" and bank_seed == 456:
                        mismatches = 0
                        for beh, eid, tidx, score in zip(behaviors, episode_ids, transition_indices, scores_m12):
                            ref = phase8c_ref[(beh, int(eid), int(tidx))]
                            if not np.isclose(score, ref, rtol=REPRO_RTOL, atol=REPRO_ATOL):
                                mismatches += 1
                        assert mismatches == 0, (
                            f"STOP: reproduction gate failed for {domain}/{condition}/full/bank456/M12: "
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

                    out_dir = OUT_ROOT / domain / cond_dir(domain, condition) / theta_name / f"bank{bank_seed}"
                    out_dir.mkdir(parents=True, exist_ok=True)
                    m_fieldnames = [f"self_y_lcg_score_M{M}" for M in M_LIST]
                    with open(out_dir / "per_transition_scores.csv", "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["domain", "condition", "theta_s", "bank_seed",
                                                                "behavior", "episode_id", "transition_index",
                                                                *m_fieldnames])
                        writer.writeheader()
                        for i, (beh, eid, tidx) in enumerate(zip(behaviors, episode_ids, transition_indices)):
                            row = {"domain": domain, "condition": condition, "theta_s": theta_name,
                                   "bank_seed": bank_seed, "behavior": beh, "episode_id": int(eid),
                                   "transition_index": int(tidx)}
                            for M in M_LIST:
                                row[f"self_y_lcg_score_M{M}"] = float(per_m_scores[M][i])
                            writer.writerow(row)

                    all_results[key] = {"h_D_seconds": t_hD, "scoring_seconds": t_score, "by_M": m_results}
                    results_path.write_text(json.dumps(all_results, indent=2, default=str))

            del agent
            torch.cuda.empty_cache()

        del candidates
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("PHASE 9 SUMMARY: self-y Delta stability across 3 independent JVP banks, per theta_S, at M=12")
    print("=" * 100)
    for domain in DOMAINS:
        for condition in CONDITIONS:
            for theta_name in THETA_S_NAMES:
                keys = [f"{domain}/{condition}/{theta_name}/bank{s}" for s in BANK_SEEDS]
                if all(k in all_results for k in keys):
                    deltas = [all_results[k]["by_M"]["12"]["delta_mean"] for k in keys]
                    print(f"{domain}/{condition}/{theta_name:<6} M12 deltas={[f'{d:+.3f}' for d in deltas]}  "
                          f"mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")

    print("\n" + "=" * 100)
    print("PHASE 9 SUMMARY: self-y Delta stability across M in {5..12}, per theta_S (bank456 only)")
    print("=" * 100)
    for domain in DOMAINS:
        for condition in CONDITIONS:
            for theta_name in THETA_S_NAMES:
                key = f"{domain}/{condition}/{theta_name}/bank456"
                if key in all_results:
                    deltas = [all_results[key]["by_M"][str(M)]["delta_mean"] for M in M_LIST]
                    print(f"{domain}/{condition}/{theta_name:<6} bank456 deltas={[f'{d:+.3f}' for d in deltas]}  "
                          f"mean={np.mean(deltas):+.4f}  std={np.std(deltas):.4f}")

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
