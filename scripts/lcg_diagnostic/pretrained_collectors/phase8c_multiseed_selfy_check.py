"""Phase 8c -- multi-seed extension of Phase 8b: does the real-y-vs-self-y*
agreement (and the walker/walk_scarce-specific dampening found for Seed A)
hold across all 3 independently-trained model seeds, or was it Seed-A
specific?

Same methodology as phase8b_full_selfy_check.py (production DiffusionSampler
y*, real conditioning context, identical h_D/theta_S/bank/held-out candidates
to Phase 5), now covering all 2 domains x 3 conditions x 3 seeds = 18
evaluations (vs Phase 8b's 6, Seed A only). Reuses phase6_full_ensemble.py's
checkpoint_path() for A/B/C resolution -- no retraining, no production code
changes. One self-y sampling seed per evaluation (Y_SAMPLE_SEED, same as
Phase 8b) since Phase 8a already established seed-to-seed stability of the
self-y result for a fixed model.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase8c_multiseed_selfy_check.py
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

from lcg.forward_jvp import make_jvp_bank, score_one_jvp_bank  # noqa: E402
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402
    load_agent, evenly_spaced_indices, HELD_OUT_EPISODE_IDS, EPISODE_LEN, N_TRANS_PER_EPISODE,
    PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC, CANDIDATE_CHUNK_SIZE,
    PRECISION_SEED, BANK_SEED, THETA_S, MODELS_ROOT, MIXTURES, SOURCE_POOLS,
    describe, episode_level_means, bootstrap_ci, exact_permutation_test,
)
from phase6_pilot_ensemble import load_transition_5d, sample_with_given_noise  # noqa: E402
from phase6_full_ensemble import checkpoint_path  # noqa: E402 -- reuse A/B/C path resolution

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
SEEDS = ["A", "B", "C"]
Y_SAMPLE_SEED = 24681
NUM_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 999

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase8_imagined_rollout" / "phase8c_multiseed_selfy_check"


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
                                    "obs5d": obs5d, "act": act, "y_real": y_real})
    assert len(candidates) == 1000
    return candidates


def flatten_obs(obs5d: torch.Tensor) -> torch.Tensor:
    b, t, c, h, w = obs5d.shape
    return obs5d.reshape(b, t * c, h, w)


def summarize(scores: np.ndarray, behaviors: np.ndarray, episode_ids: np.ndarray, tag: str) -> dict:
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
    print(f"    [{tag}] delta={delta:+.4f} boot_CI=[{boot['ci_2.5']:+.4f},{boot['ci_97.5']:+.4f}] "
          f"significant={significant} perm_p_run_gt_walk={perm['p_value_one_sided_run_gt_walk']:.4f}", flush=True)
    return {"walk_stats": walk_desc, "run_stats": run_desc, "delta_mean": delta,
            "bootstrap": boot, "permutation_test": perm, "significant": significant}


def process_seed(domain: str, condition: str, seed: str, candidates: list, train_dataset: Dataset,
                  sigmas, order: float, device, probe_task: str = "walk") -> dict:
    print(f"\n  --- seed {seed} ---", flush=True)
    ckpt = checkpoint_path(domain, condition, seed)
    assert ckpt.exists(), f"missing checkpoint: {ckpt}"
    agent, sigma_cfg, _ = load_agent(domain, ckpt, probe_task=probe_task)
    denoiser = agent.denoiser

    theta_s_named = selected_named_parameters(denoiser, THETA_S)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())

    t0 = time.time()
    h_D = historical_precision(
        denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
        B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
        beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
    )
    assert_setup_valid(theta_s_named, h_D, d_S=d_S)
    h_D_inv_sqrt = h_D.rsqrt()
    print(f"    h_D built in {time.time() - t0:.1f}s", flush=True)

    y_shape = torch.Size([1, denoiser.cfg.inner_model.img_channels, 64, 64])
    bank = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=CANDIDATE_NUM_MC, seed=BANK_SEED)

    behaviors = np.array([c["behavior"] for c in candidates])
    episode_ids = np.array([c["episode_id"] for c in candidates])
    transition_indices = np.array([c["transition_index"] for c in candidates])

    real_candidates = [(flatten_obs(c["obs5d"]), c["act"], c["y_real"]) for c in candidates]
    t0 = time.time()
    real_scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, real_candidates, CANDIDATE_CHUNK_SIZE).cpu().numpy()
    print(f"    real-y scoring done in {time.time() - t0:.1f}s", flush=True)
    real_result = summarize(real_scores, behaviors, episode_ids, "real_y")

    gen = torch.Generator(device=device)
    gen.manual_seed(Y_SAMPLE_SEED)
    with torch.no_grad():
        t0 = time.time()
        selfy_candidates = []
        for c in candidates:
            noise = torch.randn(1, 3, 64, 64, device=device, generator=gen)
            y_star = sample_with_given_noise(denoiser, sigmas, order, c["obs5d"], c["act"], noise)
            selfy_candidates.append((flatten_obs(c["obs5d"]), c["act"], y_star.float()))
        t_sample = time.time() - t0
        t0 = time.time()
        selfy_scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, selfy_candidates, CANDIDATE_CHUNK_SIZE).cpu().numpy()
        t_score = time.time() - t0
    print(f"    y*-seed={Y_SAMPLE_SEED}: sampled in {t_sample:.1f}s, scored in {t_score:.1f}s", flush=True)
    selfy_result = summarize(selfy_scores, behaviors, episode_ids, "self_y")

    out_dir = OUT_ROOT / domain / seed / cond_dir(domain, condition)
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "per_transition_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["domain", "seed", "behavior", "episode_id", "transition_index",
                                                "real_y_lcg_score", "self_y_lcg_score"])
        writer.writeheader()
        for beh, eid, tidx, r_score, s_score in zip(behaviors, episode_ids, transition_indices, real_scores, selfy_scores):
            writer.writerow({"domain": domain, "seed": seed, "behavior": beh, "episode_id": int(eid),
                              "transition_index": int(tidx), "real_y_lcg_score": float(r_score),
                              "self_y_lcg_score": float(s_score)})

    del agent
    torch.cuda.empty_cache()
    return {"real_y": real_result, "self_y": selfy_result,
            "delta_ratio_selfy_over_realy": (selfy_result["delta_mean"] / real_result["delta_mean"]
                                              if real_result["delta_mean"] != 0 else float("nan")),
            "same_sign": (real_result["delta_mean"] > 0) == (selfy_result["delta_mean"] > 0)}


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    results_path = OUT_ROOT / "phase8c_summary.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    for domain in DOMAINS:
        bootstrap_ckpt = checkpoint_path(domain, "run_scarce", "A")
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
        print(f"\n[{domain}] built {len(candidates)} held-out candidates, n_cond={n_cond}", flush=True)

        for condition in CONDITIONS:
            train_dataset = Dataset(MIXTURES / domain / cond_dir(domain, condition) / "dataset", name=f"{domain}_{condition}_train", cache_in_ram=True)
            train_dataset.load_from_default_path()
            assert train_dataset.num_steps == 5000

            for seed in SEEDS:
                key = f"{domain}/{seed}/{condition}"
                if key in all_results:
                    print(f">>> SKIPPING {key} (already done)", flush=True)
                    continue
                print(f"\n{'=' * 70}\n{key}\n{'=' * 70}", flush=True)
                all_results[key] = process_seed(domain, condition, seed, candidates, train_dataset, sigmas,
                                                 diffusion_cfg.order, device, probe_task="walk")
                results_path.write_text(json.dumps(all_results, indent=2, default=str))

        del candidates
        torch.cuda.empty_cache()

    print("\n" + "=" * 100)
    print("PHASE 8c SUMMARY: real-y Delta vs self-y Delta, all domain/seed/conditions")
    print("=" * 100)
    for key, r in all_results.items():
        print(f"{key:<28} real_y={r['real_y']['delta_mean']:+9.4f} (sig={r['real_y']['significant']})   "
              f"self_y={r['self_y']['delta_mean']:+9.4f} (sig={r['self_y']['significant']})   "
              f"ratio={r['delta_ratio_selfy_over_realy']:+.3f}   same_sign={r['same_sign']}")

    print(f"\nSaved: {results_path}")


if __name__ == "__main__":
    main()
