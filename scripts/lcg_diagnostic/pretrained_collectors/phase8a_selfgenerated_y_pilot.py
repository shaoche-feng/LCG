"""Phase 8a -- pilot: does the walk-vs-run LCG separation survive when the
candidate target y is the model's OWN diffusion-sampled prediction (as
production actually does during imagined rollouts -- see
src/lcg/intrinsic_reward.py + src/envs/world_model_env.py, where `y` is
literally `self.sampler.sample(...)`'s output, never a ground-truth frame),
instead of the real next observation every prior diagnostic phase (5/5b-f/
multiseed/6/7) has used?

This isolates ONE variable relative to Phase 5's original scoring: the `y`
fed into the candidate. Conditioning context (obs, act), theta_S, h_D config,
candidate bank (M=12), held-out episodes/transitions, and chunk_size are all
byte-identical to phase5_lcg_scoring.py. No world model retraining, no
production code changes -- y* is generated via the exact production
DiffusionSampler math (reusing phase6_pilot_ensemble.py's validated
sample_with_given_noise wrapper), and scored via the exact production,
unmodified score_one_jvp_bank.

Pilot scope: walker domain, Seed A, both run_scarce (established Delta=+16.49
under real y) and walk_scarce (established Delta=-1.00 under real y) -- the
two extremes of the already-documented effect. 3 independent y*-sampling
seeds per condition, to separate "changed because self-generated y differs
structurally from real y" from "changed because of ordinary sampling noise".

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase8a_selfgenerated_y_pilot.py
"""
from __future__ import annotations

import json
import sys
import time
from itertools import combinations
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

from lcg.forward_jvp import make_jvp_bank, score_one_jvp_bank  # noqa: E402 -- production, unmodified
from lcg.theta_s import selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, assert_setup_valid  # noqa: E402
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402

from phase5_lcg_scoring import (  # noqa: E402 -- reuse Phase 5's exact setup, no reimplementation
    load_agent, evenly_spaced_indices, HELD_OUT_EPISODE_IDS, EPISODE_LEN, N_TRANS_PER_EPISODE,
    PRECISION_REFERENCE_SIZE, PRECISION_NUM_MC, BETA, DAMPING, CANDIDATE_NUM_MC, CANDIDATE_CHUNK_SIZE,
    PRECISION_SEED, BANK_SEED, THETA_S, MODELS_ROOT, MIXTURES, SOURCE_POOLS,
    describe, episode_level_means, bootstrap_ci, exact_permutation_test,
)
from phase6_pilot_ensemble import load_transition_5d, sample_with_given_noise  # noqa: E402

DOMAIN = "walker"
CONDITIONS = ["run_scarce", "walk_scarce"]
Y_SAMPLE_SEEDS = [24681, 24682, 24683]  # 3 independent y*-sampling seeds; distinct namespace from prior seeds
NUM_BOOTSTRAP = 10_000
BOOTSTRAP_SEED = 999

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase8_imagined_rollout" / "phase8a_selfgenerated_y"


def build_domain_candidates(domain: str, n_cond: int, device):
    """Same held-out candidates as Phase 5 (500 walk + 500 run, episodes
    {7..11}, 100 evenly-spaced transitions/episode), but keeping the
    UNFLATTENED obs5d (needed to drive the diffusion sampler for y*) alongside
    the real y (kept only for reporting/comparison, never used for scoring
    here) and metadata."""
    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    walk_pool = Dataset(SOURCE_POOLS / domain / "walk" / "dataset", name=f"{domain}_walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / domain / "run" / "dataset", name=f"{domain}_run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()
    pools = {"walk": walk_pool, "run": run_pool}

    candidates = []
    for behavior in ("walk", "run"):  # matches phase5_lcg_scoring's walk-then-run order
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


def process_condition(condition: str, candidates: list, sigmas, order: float, device) -> dict:
    print(f"\n{'=' * 70}\n{DOMAIN}/{condition}\n{'=' * 70}", flush=True)
    checkpoint_path = MODELS_ROOT / DOMAIN / condition / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
    agent, sigma_cfg, _ = load_agent(DOMAIN, checkpoint_path)
    denoiser = agent.denoiser

    theta_s_named = selected_named_parameters(denoiser, THETA_S)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())

    train_dataset = Dataset(MIXTURES / DOMAIN / condition / "dataset", name=f"{DOMAIN}_{condition}_train", cache_in_ram=True)
    train_dataset.load_from_default_path()
    assert train_dataset.num_steps == 5000

    t0 = time.time()
    h_D = historical_precision(
        denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
        B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
        beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
    )
    assert_setup_valid(theta_s_named, h_D, d_S=d_S)
    h_D_inv_sqrt = h_D.rsqrt()
    print(f"  h_D built in {time.time() - t0:.1f}s", flush=True)

    y_shape = torch.Size([1, denoiser.cfg.inner_model.img_channels, 64, 64])
    bank = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=CANDIDATE_NUM_MC, seed=BANK_SEED)

    # --- real-y reference score (reproduces Phase 5 exactly, for the side-by-side comparison) ---
    real_candidates = [(flatten_obs(c["obs5d"]), c["act"], c["y_real"]) for c in candidates]
    t0 = time.time()
    real_scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, real_candidates, CANDIDATE_CHUNK_SIZE)
    real_scores = real_scores.cpu().numpy()
    print(f"  real-y scoring done in {time.time() - t0:.1f}s", flush=True)

    behaviors = np.array([c["behavior"] for c in candidates])
    episode_ids = np.array([c["episode_id"] for c in candidates])

    def summarize(scores: np.ndarray, tag: str) -> dict:
        walk_scores, run_scores = scores[behaviors == "walk"], scores[behaviors == "run"]
        walk_ep = episode_level_means(walk_scores, episode_ids[behaviors == "walk"])
        run_ep = episode_level_means(run_scores, episode_ids[behaviors == "run"])
        walk_ep_means = np.array([v["mean"] for v in walk_ep.values()])
        run_ep_means = np.array([v["mean"] for v in run_ep.values()])
        boot = bootstrap_ci(walk_ep_means, run_ep_means, NUM_BOOTSTRAP, BOOTSTRAP_SEED)
        perm = exact_permutation_test(walk_ep_means, run_ep_means)
        walk_desc, run_desc = describe(walk_scores), describe(run_scores)
        delta = run_desc["mean"] - walk_desc["mean"]
        print(f"  [{tag}] walk_mean={walk_desc['mean']:.4f} run_mean={run_desc['mean']:.4f} "
              f"delta={delta:+.4f} boot_CI=[{boot['ci_2.5']:+.4f},{boot['ci_97.5']:+.4f}] "
              f"perm_p_run_gt_walk={perm['p_value_one_sided_run_gt_walk']:.4f}", flush=True)
        return {"walk_stats": walk_desc, "run_stats": run_desc, "delta_mean": delta,
                "bootstrap": boot, "permutation_test": perm}

    result = {"real_y": summarize(real_scores, "real_y (Phase 5 reproduction)")}

    # --- self-generated y* scoring, repeated for 3 independent noise seeds ---
    selfy_results = {}
    for seed in Y_SAMPLE_SEEDS:
        gen = torch.Generator(device=device)
        gen.manual_seed(seed)
        with torch.no_grad():
            t0 = time.time()
            selfy_candidates = []
            for c in candidates:
                noise = torch.randn(1, 3, 64, 64, device=device, generator=gen)
                y_star = sample_with_given_noise(denoiser, sigmas, order, c["obs5d"], c["act"], noise)
                selfy_candidates.append((flatten_obs(c["obs5d"]), c["act"], y_star.float()))
            t_sample = time.time() - t0

            t0 = time.time()
            selfy_scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, selfy_candidates, CANDIDATE_CHUNK_SIZE)
            selfy_scores = selfy_scores.cpu().numpy()
            t_score = time.time() - t0
        print(f"  y*-seed={seed}: sampled 1000 y* in {t_sample:.1f}s, scored in {t_score:.1f}s", flush=True)
        selfy_results[seed] = summarize(selfy_scores, f"self_y (seed={seed})")

    result["self_y_by_seed"] = selfy_results
    result["self_y_delta_mean_across_seeds"] = {
        "mean": float(np.mean([r["delta_mean"] for r in selfy_results.values()])),
        "std": float(np.std([r["delta_mean"] for r in selfy_results.values()])),
        "values": [r["delta_mean"] for r in selfy_results.values()],
    }

    del agent
    torch.cuda.empty_cache()
    return result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    bootstrap_ckpt = MODELS_ROOT / DOMAIN / "run_scarce" / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
    bootstrap_agent, _, _ = load_agent(DOMAIN, bootstrap_ckpt)
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
    print(f"sampler: num_steps={diffusion_cfg.num_steps_denoising} order={diffusion_cfg.order} "
          f"s_churn={diffusion_cfg.s_churn} (fully deterministic given noise)")

    candidates = build_domain_candidates(DOMAIN, n_cond, device)
    print(f"built {len(candidates)} held-out candidates (500 walk + 500 run), n_cond={n_cond}")

    all_results = {}
    for condition in CONDITIONS:
        all_results[condition] = process_condition(condition, candidates, sigmas, diffusion_cfg.order, device)

    print("\n" + "=" * 70)
    print("PHASE 8a SUMMARY: real-y Delta vs self-y Delta (mean +/- std across 3 y*-seeds)")
    print("=" * 70)
    for condition in CONDITIONS:
        r = all_results[condition]
        print(f"{DOMAIN}/{condition}: real_y_delta={r['real_y']['delta_mean']:+.4f}  "
              f"self_y_delta={r['self_y_delta_mean_across_seeds']['mean']:+.4f} "
              f"+/- {r['self_y_delta_mean_across_seeds']['std']:.4f}  "
              f"same_sign={ (r['real_y']['delta_mean'] > 0) == (r['self_y_delta_mean_across_seeds']['mean'] > 0) }")

    (OUT_ROOT / "phase8a_summary.json").write_text(json.dumps(all_results, indent=2, default=str))
    print(f"\nSaved: {OUT_ROOT / 'phase8a_summary.json'}")


if __name__ == "__main__":
    main()
