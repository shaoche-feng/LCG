"""Pilot: ensemble disagreement between the 3 walker/run_scarce model seeds
(A/B/C), using each model's own mean of S full-diffusion-sampler predictions.

Uses the PRODUCTION diffusion sampler math exactly (models/diffusion/
diffusion_sampler.py, UNMODIFIED) via a local wrapper that only changes one
thing: accepting a pre-supplied initial noise tensor instead of drawing it
internally from the global RNG, so the same noise can be shared across
models A/B/C. This is mathematically identical to `DiffusionSampler.sample()`
whenever cfg.s_churn == 0 (verified true for the production config below,
so sigma_hat == sigma always and gamma-branch churn noise never fires --
the ONLY randomness in the whole trajectory is the initial x).

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase6_pilot_ensemble.py
"""
from __future__ import annotations

import json
import time
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from hydra.utils import instantiate
# NOTE: deliberately NOT importing scipy here -- torch + scipy.stats in the same process
# triggers an OMP Error #15 DLL conflict on this Windows machine (same issue hit at Phase
# 5e/5g). The scipy-dependent stability table (Pearson/Spearman) is computed in a separate,
# torch-free follow-up script (phase6b_pilot_stability.py) from this script's saved JSON.

_THIS_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(_THIS_DIR))
from phase5_lcg_scoring import (  # noqa: E402
    load_agent, evenly_spaced_indices, HELD_OUT_EPISODE_IDS, EPISODE_LEN, N_TRANS_PER_EPISODE,
    SOURCE_POOLS, MODELS_ROOT,
)
from models.diffusion.diffusion_sampler import build_sigmas  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402

DOMAIN = "walker"
CONDITION = "run_scarce"
SEED_CHECKPOINTS = {
    "A": MODELS_ROOT / DOMAIN / CONDITION / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt",
    "B": MODELS_ROOT / "multiseed" / "seed43" / DOMAIN / CONDITION / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt",
    "C": MODELS_ROOT / "multiseed" / "seed44" / DOMAIN / CONDITION / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt",
}
NUM_SAMPLES = 5
NOISE_SEED = 13579  # documented fixed seed for the per-candidate shared initial-noise draws
CANDIDATE_POSITIONS = [10, 60]  # indices into the 100 evenly-spaced Phase-5 positions, per episode (2/episode x 5 = 10/behavior)

_PROJECT_ROOT = _THIS_DIR.parent.parent.parent.parent
OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase6_pilot_ensemble"


def load_transition_5d(dataset: Dataset, segment_id: SegmentId, n: int, device):
    """Unlike lcg.precision.load_transition (which pre-flattens obs to (1,n*c,h,w) for the
    Candidate/JVP-scoring format), DiffusionSampler.sample() expects UNFLATTENED (b,t,c,h,w)
    obs and reshapes internally -- so this is a separate loader, not a reuse."""
    segment = dataset[segment_id]
    obs_5d = segment.obs[:n].unsqueeze(0).to(device)
    act = segment.act[:n].unsqueeze(0).to(device)
    y = segment.obs[n].unsqueeze(0).to(device)
    return obs_5d, act, y


def sample_with_given_noise(denoiser, sigmas: torch.Tensor, order: int, prev_obs_5d: torch.Tensor,
                             prev_act: torch.Tensor, initial_noise: torch.Tensor) -> torch.Tensor:
    """Mirrors DiffusionSampler.sample()'s body exactly for the gamma=0 case (verified true
    for the production config: s_churn=0.0 -> gamma_=0 always -> sigma_hat=sigma, no churn
    noise). Only behavioral difference: takes `initial_noise` as a parameter instead of
    drawing `torch.randn(...)` internally, so it can be shared across models."""
    b, t, c, h, w = prev_obs_5d.shape
    prev_obs = prev_obs_5d.reshape(b, t * c, h, w)
    x = initial_noise.clone()
    for sigma, next_sigma in zip(sigmas[:-1], sigmas[1:]):
        denoised = denoiser.denoise(x, sigma, prev_obs, prev_act)
        d = (x - denoised) / sigma
        dt = next_sigma - sigma
        if order == 1 or next_sigma == 0:
            x = x + d * dt
        else:
            x_2 = x + d * dt
            denoised_2 = denoiser.denoise(x_2, next_sigma, prev_obs, prev_act)
            d_2 = (x_2 - denoised_2) / next_sigma
            d_prime = (d + d_2) / 2
            x = x + d_prime * dt
    return x


def build_candidate_list():
    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    candidates = []
    for behavior in ("walk", "run"):
        for eid in HELD_OUT_EPISODE_IDS:
            for pos in CANDIDATE_POSITIONS:
                t = int(indices[pos])
                candidates.append({"behavior": behavior, "episode_id": eid, "transition_index": t})
    assert len(candidates) == 20
    assert sum(1 for c in candidates if c["behavior"] == "walk") == 10
    assert sum(1 for c in candidates if c["behavior"] == "run") == 10
    return candidates


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    # === Section 7: load models + sampler verification ===
    agents = {}
    sigma_cfg_ref = None
    diffusion_cfg = None
    for label, ckpt in SEED_CHECKPOINTS.items():
        assert ckpt.exists(), f"missing checkpoint for seed {label}: {ckpt}"
        agent, sigma_cfg, action_dim = load_agent(DOMAIN, ckpt)
        agents[label] = agent
        if sigma_cfg_ref is None:
            sigma_cfg_ref = sigma_cfg

    # Build the diffusion-sampler config exactly as production WorldModelEnv does
    from hydra import compose, initialize_config_dir
    with initialize_config_dir(version_base="1.3", config_dir=str(_THIS_DIR.parent.parent.parent / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    diffusion_cfg = instantiate(cfg.world_model_env.diffusion_sampler)
    device = agents["A"].denoiser.device
    sigmas = build_sigmas(diffusion_cfg.num_steps_denoising, diffusion_cfg.sigma_min,
                           diffusion_cfg.sigma_max, diffusion_cfg.rho, device)

    print("=== A. SAMPLER VERIFICATION ===")
    print(f"sampler class: models.diffusion.diffusion_sampler.DiffusionSampler (production, unmodified)")
    print(f"num_steps_denoising: {diffusion_cfg.num_steps_denoising}")
    print(f"sigma schedule (rho={diffusion_cfg.rho}, min={diffusion_cfg.sigma_min}, max={diffusion_cfg.sigma_max}): "
          f"{sigmas.cpu().tolist()}")
    print(f"order: {diffusion_cfg.order} ({'Euler' if diffusion_cfg.order == 1 else 'Heun'})")
    print(f"s_churn: {diffusion_cfg.s_churn}  -> gamma always 0 given s_churn=0, so trajectory is "
          f"FULLY DETERMINISTIC given the initial noise x (no other randomness enters)")
    assert diffusion_cfg.s_churn == 0.0, "pilot assumes production s_churn=0 (see docstring); found nonzero"

    n_cond = agents["A"].denoiser.cfg.inner_model.num_steps_conditioning
    walk_pool = Dataset(SOURCE_POOLS / DOMAIN / "walk" / "dataset", name="walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / DOMAIN / "run" / "dataset", name="run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()
    pools = {"walk": walk_pool, "run": run_pool}

    candidates = build_candidate_list()
    for c in candidates:
        obs5d, act, y = load_transition_5d(pools[c["behavior"]], SegmentId(c["episode_id"], c["transition_index"] + 1 - (n_cond + 1), c["transition_index"] + 1), n_cond, device)
        c["obs5d"], c["act"], c["y"] = obs5d, act, y

    print(f"output shape (one sample): {tuple(candidates[0]['obs5d'].shape)} obs, "
          f"prediction shape will be (1,3,64,64)")

    # --- determinism checks ---
    gen = torch.Generator(device=device)
    gen.manual_seed(NOISE_SEED)
    test_noise_1 = torch.randn(1, 3, 64, 64, device=device, generator=gen)
    test_noise_2 = torch.randn(1, 3, 64, 64, device=device, generator=gen)
    c0 = candidates[0]
    out_A_1a = sample_with_given_noise(agents["A"].denoiser, sigmas, diffusion_cfg.order, c0["obs5d"], c0["act"], test_noise_1)
    out_A_1b = sample_with_given_noise(agents["A"].denoiser, sigmas, diffusion_cfg.order, c0["obs5d"], c0["act"], test_noise_1)
    out_B_1a = sample_with_given_noise(agents["B"].denoiser, sigmas, diffusion_cfg.order, c0["obs5d"], c0["act"], test_noise_1)
    same_model_same_noise = torch.equal(out_A_1a, out_A_1b)
    diff_model_same_noise_differs = not torch.equal(out_A_1a, out_B_1a)
    print(f"determinism check: same model + same noise -> identical output: {same_model_same_noise}")
    print(f"determinism check: different model + same noise -> different output: {diff_model_same_noise_differs} "
          f"(max abs diff={torch.abs(out_A_1a.float() - out_B_1a.float()).max().item():.4f})")
    assert same_model_same_noise, "STOPPING: exact noise control could not be established (same model+noise gave different output)"

    print(f"output numerical range check (out_A_1a): min={out_A_1a.min().item():.4f} max={out_A_1a.max().item():.4f} "
          f"(expected [-1,1], byte-quantized)")

    # === Sections 1-5: full pilot over 20 candidates ===
    P = 3 * 64 * 64
    all_results = []
    t_single_sample = []
    for ci, c in enumerate(candidates):
        gen = torch.Generator(device=device)
        gen.manual_seed(NOISE_SEED + ci)  # distinct noise per candidate, shared across models A/B/C
        noises = [torch.randn(1, 3, 64, 64, device=device, generator=gen) for _ in range(NUM_SAMPLES)]

        y_hat = {label: [] for label in ("A", "B", "C")}
        for label, agent in agents.items():
            for s in range(NUM_SAMPLES):
                t0 = time.time()
                y = sample_with_given_noise(agent.denoiser, sigmas, diffusion_cfg.order, c["obs5d"], c["act"], noises[s])
                t_single_sample.append(time.time() - t0)
                y_hat[label].append(y.float())

        cum_means = {label: [] for label in ("A", "B", "C")}
        for label in ("A", "B", "C"):
            stack = torch.stack(y_hat[label], dim=0)  # (5,1,3,64,64)
            cumsum = torch.cumsum(stack, dim=0)
            counts = torch.arange(1, NUM_SAMPLES + 1, device=device, dtype=torch.float32).view(-1, 1, 1, 1, 1)
            cum_means[label] = cumsum / counts  # (5,1,3,64,64) -- cum_means[label][S-1] = mu_label^(S)

        ens_per_S = []
        for S in range(NUM_SAMPLES):
            mus = [cum_means[label][S] for label in ("A", "B", "C")]
            mu_bar = sum(mus) / 3
            ens = sum(((mu - mu_bar) ** 2).sum().item() for mu in mus) / (3 * P)
            ens_per_S.append(ens)

        within_var_k = {}
        for label in ("A", "B", "C"):
            mu5 = cum_means[label][NUM_SAMPLES - 1]
            per_sample_sq = [((y_hat[label][s] - mu5) ** 2).sum().item() / P for s in range(NUM_SAMPLES)]
            within_var_k[label] = float(np.mean(per_sample_sq))
        within_sample_variance = float(np.mean(list(within_var_k.values())))

        all_results.append({
            **{k: v for k, v in c.items() if k not in ("obs5d", "act", "y")},
            "ens": ens_per_S, "within_sample_variance": within_sample_variance,
            "within_var_by_model": within_var_k,
        })
        print(f"  candidate {ci+1}/20 ({c['behavior']} ep{c['episode_id']} t{c['transition_index']}): "
              f"ENS_1..5={[round(e,4) for e in ens_per_S]}  within_var={within_sample_variance:.4f}")

    mean_t_per_sample = float(np.mean(t_single_sample))
    print(f"\nmean ms per full diffusion sample: {mean_t_per_sample*1000:.2f} ms")

    # Section 8 (stability of ENS_S vs ENS_5, needs scipy) is computed separately in
    # phase6b_pilot_stability.py, torch-free, from this script's saved per-candidate JSON.
    ens_matrix = np.array([r["ens"] for r in all_results])  # (20, 5)

    # === Section 9: Walk vs Run pilot signal ===
    print("\n=== C. WALK VS RUN PILOT SIGNAL ===")
    behaviors = np.array([r["behavior"] for r in all_results])
    walk_run_table = {}
    for S in range(1, NUM_SAMPLES + 1):
        vals = ens_matrix[:, S - 1]
        mean_walk = float(vals[behaviors == "walk"].mean())
        mean_run = float(vals[behaviors == "run"].mean())
        delta = mean_run - mean_walk
        walk_run_table[S] = {"mean_ENS_walk": mean_walk, "mean_ENS_run": mean_run, "delta_ENS": delta}
        print(f"S={S}: mean_ENS(walk)={mean_walk:.4f}  mean_ENS(run)={mean_run:.4f}  Delta_ENS={delta:+.4f}")

    # === Section 5 summary: within vs between ===
    mean_within = float(np.mean([r["within_sample_variance"] for r in all_results]))
    mean_between_S5 = float(ens_matrix[:, -1].mean())
    print(f"\n=== E. WITHIN-MODEL VS BETWEEN-MODEL (at S=5) ===")
    print(f"mean within-model sampling variance: {mean_within:.4f}")
    print(f"mean between-model disagreement (ENS_5): {mean_between_S5:.4f}")
    print(f"ratio between/within: {mean_between_S5/mean_within:.4f}")

    # === Visualization: 3 representative candidates ===
    idx_walk = next(i for i, r in enumerate(all_results) if r["behavior"] == "walk")
    idx_run = next(i for i, r in enumerate(all_results) if r["behavior"] == "run")
    idx_high = int(np.argmax(ens_matrix[:, -1]))
    viz_indices = {"walk_example": idx_walk, "run_example": idx_run, "high_disagreement": idx_high}

    def to_img(t: torch.Tensor) -> np.ndarray:
        arr = t.squeeze(0).permute(1, 2, 0).cpu().numpy()
        return np.clip((arr + 1) / 2, 0, 1)

    for tag, ci in viz_indices.items():
        c = candidates[ci]
        gen = torch.Generator(device=device)
        gen.manual_seed(NOISE_SEED + ci)
        noises = [torch.randn(1, 3, 64, 64, device=device, generator=gen) for _ in range(NUM_SAMPLES)]
        y_hat = {label: [] for label in ("A", "B", "C")}
        for label, agent in agents.items():
            for s in range(NUM_SAMPLES):
                y_hat[label].append(sample_with_given_noise(agent.denoiser, sigmas, diffusion_cfg.order,
                                                              c["obs5d"], c["act"], noises[s]).float())
        mus = {label: torch.stack(y_hat[label]).mean(dim=0) for label in ("A", "B", "C")}
        mu_bar = sum(mus.values()) / 3
        var_map = sum((mus[label] - mu_bar) ** 2 for label in ("A", "B", "C")).squeeze(0).mean(dim=0).cpu().numpy() / 3

        fig, axes = plt.subplots(3, 6, figsize=(15, 8))
        for row, label in enumerate(("A", "B", "C")):
            for s in range(NUM_SAMPLES):
                axes[row, s].imshow(to_img(y_hat[label][s]))
                axes[row, s].set_title(f"{label}{s+1}", fontsize=9)
                axes[row, s].axis("off")
            axes[row, 5].imshow(to_img(mus[label]))
            axes[row, 5].set_title(f"mu_{label}", fontsize=9, fontweight="bold")
            axes[row, 5].axis("off")
        fig.suptitle(f"{tag}: {c['behavior']} ep{c['episode_id']} t{c['transition_index']} "
                     f"(ENS_5={ens_matrix[ci,-1]:.4f})")
        fig.tight_layout()
        fig.savefig(OUT_ROOT / f"grid_{tag}.png", dpi=130)
        plt.close(fig)

        fig2, axes2 = plt.subplots(1, 3, figsize=(10, 3.5))
        axes2[0].imshow(to_img(c["y"])); axes2[0].set_title("ground truth"); axes2[0].axis("off")
        axes2[1].imshow(to_img(mu_bar)); axes2[1].set_title("ensemble mean mu_bar"); axes2[1].axis("off")
        im = axes2[2].imshow(var_map, cmap="inferno"); axes2[2].set_title("between-model variance"); axes2[2].axis("off")
        fig2.colorbar(im, ax=axes2[2], fraction=0.046)
        fig2.suptitle(f"{tag}: GT / ensemble mean / disagreement heatmap")
        fig2.tight_layout()
        fig2.savefig(OUT_ROOT / f"detail_{tag}.png", dpi=130)
        plt.close(fig2)
        print(f"Saved grid_{tag}.png and detail_{tag}.png")

    # === Runtime extrapolation ===
    ms_per_sample = mean_t_per_sample * 1000
    ms_per_candidate_S1 = ms_per_sample * 3  # 3 models x 1 sample
    ms_per_candidate_S5 = ms_per_sample * 3 * 5  # 3 models x 5 samples
    full_experiment_candidates = 6 * 200
    full_experiment_seconds = full_experiment_candidates * (ms_per_candidate_S5 / 1000)
    print(f"\n=== F. RUNTIME ===")
    print(f"ms per full diffusion sample (3 denoising steps): {ms_per_sample:.2f} ms")
    print(f"ms per candidate at S=1 (3 models x 1 sample): {ms_per_candidate_S1:.2f} ms")
    print(f"ms per candidate at S=5 (3 models x 5 samples): {ms_per_candidate_S5:.2f} ms")
    print(f"ESTIMATED full experiment (6 ensembles x 200 candidates x S=5 x K=3): "
          f"{full_experiment_seconds:.1f}s = {full_experiment_seconds/60:.1f} min = {full_experiment_seconds/3600:.2f} hours")

    summary = {
        "sampler_verification": {
            "num_steps_denoising": diffusion_cfg.num_steps_denoising,
            "sigma_schedule": sigmas.cpu().tolist(), "order": diffusion_cfg.order, "s_churn": diffusion_cfg.s_churn,
            "same_model_same_noise_identical": same_model_same_noise,
            "diff_model_same_noise_differs": diff_model_same_noise_differs,
        },
        "per_candidate": all_results,
        "walk_vs_run_by_S": walk_run_table,
        "mean_within_sample_variance": mean_within,
        "mean_between_model_disagreement_S5": mean_between_S5,
        "runtime": {
            "ms_per_full_diffusion_sample": ms_per_sample,
            "ms_per_candidate_S1": ms_per_candidate_S1,
            "ms_per_candidate_S5": ms_per_candidate_S5,
            "estimated_full_experiment_hours": full_experiment_seconds / 3600,
        },
    }
    (OUT_ROOT / "pilot_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSaved: {OUT_ROOT / 'pilot_summary.json'}")


if __name__ == "__main__":
    main()
