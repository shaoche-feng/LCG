"""Phase 5: LCG scoring on held-out Walk vs Run transitions, for the two
trained run_scarce denoisers (walker, quadruped). Uses the PRODUCTION LCG
implementation (src/lcg/*) completely unmodified -- theta_s.py,
precision.py::historical_precision, forward_jvp.py::make_jvp_bank /
score_one_jvp_bank -- called directly rather than through the RL/actor-critic
intrinsic-reward hook, since candidates here are real held-out transitions,
not imagined rollout steps.

Held-out episode note: Phase 3 froze only 2 episodes/behavior as the official
`eval/` datasets. Phase 5 needs 5 episodes/behavior (500 candidates), which
doesn't fit in that 2-episode freeze. Training used random={0,1}, walk={0-6},
run={0} (see mixtures/*/run_scarce/manifest.json), so episodes {7,8,9,10,11}
of BOTH the walk and run source pools are guaranteed untouched by training for
both behaviors -- this script reads those 5 episodes directly from the
Phase-2 source pools (source_pools/{domain}/{walk,run}/dataset), which is a
strict superset of Phase 3's frozen {10,11} and never overlaps training. No
trajectories are altered or regenerated -- only existing episodes are read.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase5_lcg_scoring.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import sys
from itertools import combinations
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))

OmegaConf.register_new_resolver("eval", eval)

from agent import Agent, get_action_space_kwargs  # noqa: E402
from dm_control_env import DMControlEnv  # noqa: E402
from data import Dataset, SegmentId  # noqa: E402
from lcg.theta_s import ThetaSConfig, selected_named_parameters, frozen_named_parameters  # noqa: E402
from lcg.precision import historical_precision, load_transition, assert_setup_valid  # noqa: E402
from lcg.forward_jvp import make_jvp_bank, score_one_jvp_bank  # noqa: E402
from models.diffusion.denoiser import apply_noise_from_samples, sample_sigma_training_distribution  # noqa: E402

PROBE_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic"
MODELS_ROOT = PROBE_ROOT / "models"
SOURCE_POOLS = PROBE_ROOT / "source_pools"
MIXTURES = PROBE_ROOT / "mixtures"
OUT_ROOT = PROBE_ROOT / "phase5_lcg_scoring"

DOMAINS = ["walker", "quadruped"]
HELD_OUT_EPISODE_IDS = [7, 8, 9, 10, 11]  # disjoint from training: random={0,1}, walk={0-6}, run={0}
N_TRANS_PER_EPISODE = 100
EPISODE_LEN = 500

# Production LCG settings -- current defaults, unchanged (config/intrinsic_reward/lcg.yaml)
PRECISION_REFERENCE_SIZE = 320
PRECISION_NUM_MC = 3
BETA = 1.0
DAMPING = 1e-4
CANDIDATE_NUM_MC = 12
CANDIDATE_CHUNK_SIZE = 16
THETA_S = ThetaSConfig(include=("unet.u_blocks.3.*", "norm_out.*", "conv_out.*"), exclude=())

# Documented fixed seeds for this diagnostic (separate namespaces, no overlap with training seed 42
# or theta0 seed 0)
PRECISION_SEED = 123          # historical_precision's internal RNG (segment sampling + backward VJP noise)
BANK_SEED = 456               # shared candidate JVP bank (sigma/eps/eps_offset/eta) -- ONE per domain, used for both Walk and Run
DENOISING_LOSS_SEED = 789     # secondary diagnostic only, independent of the above
BOOTSTRAP_SEED = 999
NUM_BOOTSTRAP = 10_000


def load_agent(domain: str, checkpoint_path: Path, probe_task: str = "walk"):
    """probe_task only selects which dm_control task to instantiate for reading the domain's
    action-space bounds (identical across every task in a domain) -- default "walk" is
    unchanged for walker/quadruped; hopper (no "walk" task) passes probe_task="stand"."""
    probe = DMControlEnv(domain_name=domain, task_name=probe_task, size=64, camera_id=0, action_repeat=2)
    fake_env = SimpleNamespace(
        is_discrete=False, action_dim=probe.action_dim,
        action_low=torch.as_tensor(probe.action_low), action_high=torch.as_tensor(probe.action_high),
    )
    action_kwargs = get_action_space_kwargs(fake_env)
    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.load(checkpoint_path, load_denoiser=True, load_rew_end_model=False, load_actor_critic=False)
    agent.denoiser.eval()
    sigma_cfg = instantiate(cfg.denoiser.sigma_distribution)
    return agent, sigma_cfg, action_kwargs["continuous_action_dim"]


def evenly_spaced_indices(episode_len: int, n: int) -> np.ndarray:
    idx = np.round(np.linspace(0, episode_len - 1, n)).astype(int)
    assert len(set(idx.tolist())) == n, f"evenly-spaced index generation produced duplicates: {idx}"
    return idx


def build_candidates(source_dataset: Dataset, episode_ids, indices, num_steps_conditioning: int, device):
    seq_length = num_steps_conditioning + 1
    candidates, meta = [], []
    for eid in episode_ids:
        episode = source_dataset.load_episode(eid)
        env_seed = episode.info.get("env_seed")
        for t in indices:
            seg = SegmentId(eid, int(t) + 1 - seq_length, int(t) + 1)
            obs, act, y = load_transition(source_dataset, seg, num_steps_conditioning, device)
            candidates.append((obs, act, y))
            meta.append({"episode_id": int(eid), "env_seed": env_seed, "transition_index": int(t)})
    return candidates, meta


def denoising_loss(denoiser, obs, act, y, sigma_cfg, gen: torch.Generator) -> float:
    sigma = sample_sigma_training_distribution(sigma_cfg, 1, denoiser.device, generator=gen)
    eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=gen)
    eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=y.device, generator=gen)
    y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
    with torch.no_grad():
        cs = denoiser.compute_conditioners(sigma)
        model_output = denoiser.compute_model_output(y_sigma, obs, act, cs)
        target = (y - cs.c_skip * y_sigma) / cs.c_out
        loss = F.mse_loss(model_output, target)
    return float(loss.item())


def describe(x: np.ndarray) -> dict:
    return {
        "n": len(x), "mean": float(np.mean(x)), "std": float(np.std(x)), "median": float(np.median(x)),
        "p10": float(np.percentile(x, 10)), "p25": float(np.percentile(x, 25)),
        "p75": float(np.percentile(x, 75)), "p90": float(np.percentile(x, 90)),
        "min": float(np.min(x)), "max": float(np.max(x)),
    }


def episode_level_means(scores: np.ndarray, episode_ids: np.ndarray) -> dict:
    out = {}
    for eid in sorted(set(episode_ids.tolist())):
        mask = episode_ids == eid
        out[int(eid)] = {"mean": float(scores[mask].mean()), "median": float(np.median(scores[mask])), "n": int(mask.sum())}
    return out


def bootstrap_ci(walk_ep_means: np.ndarray, run_ep_means: np.ndarray, num_resamples: int, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    n_w, n_r = len(walk_ep_means), len(run_ep_means)
    diffs = np.empty(num_resamples)
    for i in range(num_resamples):
        w = rng.choice(walk_ep_means, size=n_w, replace=True)
        r = rng.choice(run_ep_means, size=n_r, replace=True)
        diffs[i] = r.mean() - w.mean()
    return {
        "point_estimate": float(run_ep_means.mean() - walk_ep_means.mean()),
        "ci_2.5": float(np.percentile(diffs, 2.5)),
        "ci_97.5": float(np.percentile(diffs, 97.5)),
        "num_resamples": num_resamples, "seed": seed,
    }


def exact_permutation_test(walk_ep_means: np.ndarray, run_ep_means: np.ndarray) -> dict:
    combined = np.concatenate([walk_ep_means, run_ep_means])
    n_w, n_r = len(walk_ep_means), len(run_ep_means)
    n_total = n_w + n_r
    observed = run_ep_means.mean() - walk_ep_means.mean()
    all_diffs = []
    for run_idx in combinations(range(n_total), n_r):
        run_idx = set(run_idx)
        walk_idx = [i for i in range(n_total) if i not in run_idx]
        run_vals = combined[list(run_idx)]
        walk_vals = combined[walk_idx]
        all_diffs.append(run_vals.mean() - walk_vals.mean())
    all_diffs = np.array(all_diffs)
    # One-sided p-value direction is fixed by the PRIMARY hypothesis for whichever condition is
    # being scored: run_scarce expects run>walk (upper tail), walk_scarce expects walk>run
    # (lower tail). Two-sided is direction-agnostic and reported alongside regardless.
    p_upper = float((all_diffs >= observed).mean())  # P(diff >= observed) -- run_scarce's H1 direction
    p_lower = float((all_diffs <= observed).mean())  # P(diff <= observed) -- walk_scarce's H1 direction
    p_two_sided = float((np.abs(all_diffs) >= abs(observed)).mean())
    return {
        "observed_diff": float(observed), "num_permutations": len(all_diffs),
        "p_value_one_sided_run_gt_walk": p_upper, "p_value_one_sided_walk_gt_run": p_lower,
        "p_value_two_sided": p_two_sided,
    }


def process_domain(domain: str, condition: str, checkpoint_path: Path = None, out_dir: Path = None,
                    probe_task: str = "walk") -> dict:
    """checkpoint_path/out_dir default to Seed A's exact original paths when not passed, so
    existing call sites (and Seed A's saved results) are unaffected. Multi-seed callers pass
    an explicit seed-specific checkpoint_path and out_dir; the training dataset path
    (MIXTURES/domain/condition/dataset) is NOT parametrized by seed -- mixture composition
    is identical across all model-training seeds by design, only the checkpoint differs.
    probe_task only selects which dm_control task load_agent() instantiates to read the
    domain's action-space bounds -- default "walk" is unchanged for walker/quadruped;
    hopper (no "walk" task) passes probe_task="stand"."""
    print(f"\n{'=' * 70}\n{domain}/{condition}\n{'=' * 70}")
    checkpoint_path = checkpoint_path if checkpoint_path is not None else (
        MODELS_ROOT / domain / cond_dir(domain, condition) / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
    )
    assert checkpoint_path.exists(), f"missing checkpoint: {checkpoint_path}"

    agent, sigma_cfg, action_dim = load_agent(domain, checkpoint_path, probe_task=probe_task)
    denoiser = agent.denoiser
    device = denoiser.device
    num_steps_conditioning = denoiser.cfg.inner_model.num_steps_conditioning

    theta_s_named = selected_named_parameters(denoiser, THETA_S)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    print(f"theta_S: {len(theta_s_named)} tensors, d_S={d_S}")

    # --- h_D from this model's own 5000-transition training mixture ---
    train_dataset = Dataset(MIXTURES / domain / cond_dir(domain, condition) / "dataset", name=f"{domain}_{condition}_train", cache_in_ram=True)
    train_dataset.load_from_default_path()
    assert train_dataset.num_steps == 5000, f"expected 5000 training transitions, got {train_dataset.num_steps}"

    h_D = historical_precision(
        denoiser, list(theta_s_named.values()), train_dataset, sigma_cfg,
        B=PRECISION_REFERENCE_SIZE, N=train_dataset.num_steps, num_mc=PRECISION_NUM_MC,
        beta=BETA, damping=DAMPING, seed=PRECISION_SEED,
    )
    assert_setup_valid(theta_s_named, h_D, d_S=d_S)
    h_D_inv_sqrt = h_D.rsqrt()
    assert torch.isfinite(h_D_inv_sqrt).all(), "h_D_inv_sqrt contains non-finite values"
    print(f"h_D: finite={torch.isfinite(h_D).all().item()} all_positive={(h_D > 0).all().item()} "
          f"sum={h_D.sum().item():.6f}")

    # --- shared candidate JVP bank (ONE per domain -> Full CRN shared across Walk and Run) ---
    y_shape = torch.Size([1, denoiser.cfg.inner_model.img_channels, 64, 64])
    bank = make_jvp_bank(sigma_cfg, y_shape, d_S, device, num_samples=CANDIDATE_NUM_MC, seed=BANK_SEED)

    # --- held-out candidates: episodes {7,8,9,10,11} from the source pools, 100 evenly-spaced transitions each ---
    indices = evenly_spaced_indices(EPISODE_LEN, N_TRANS_PER_EPISODE)
    walk_pool = Dataset(SOURCE_POOLS / domain / slot_dir(domain, "walk") / "dataset", name=f"{domain}_walk_pool", cache_in_ram=True)
    walk_pool.load_from_default_path()
    run_pool = Dataset(SOURCE_POOLS / domain / slot_dir(domain, "run") / "dataset", name=f"{domain}_run_pool", cache_in_ram=True)
    run_pool.load_from_default_path()

    walk_candidates, walk_meta = build_candidates(walk_pool, HELD_OUT_EPISODE_IDS, indices, num_steps_conditioning, device)
    run_candidates, run_meta = build_candidates(run_pool, HELD_OUT_EPISODE_IDS, indices, num_steps_conditioning, device)
    assert len(walk_candidates) == 500 and len(run_candidates) == 500
    assert len(set(m["episode_id"] for m in walk_meta)) == 5
    assert len(set(m["episode_id"] for m in run_meta)) == 5

    # leakage re-check: held-out episode ids/seeds vs training episode ids/seeds (per manifest)
    mix_manifest = json.loads((MIXTURES / domain / cond_dir(domain, condition) / "manifest.json").read_text())
    train_walk_ids = set(mix_manifest["walk"]["episode_ids"])
    train_run_ids = set(mix_manifest["run"]["episode_ids"])
    train_seeds = set(mix_manifest["walk"]["env_seeds"]) | set(mix_manifest["run"]["env_seeds"]) | set(mix_manifest["random"]["env_seeds"])
    held_out_seeds = {m["env_seed"] for m in walk_meta} | {m["env_seed"] for m in run_meta}
    assert set(HELD_OUT_EPISODE_IDS).isdisjoint(train_walk_ids), "walk held-out overlaps training episode ids"
    assert set(HELD_OUT_EPISODE_IDS).isdisjoint(train_run_ids), "run held-out overlaps training episode ids"
    assert held_out_seeds.isdisjoint(train_seeds), "held-out env seeds overlap training env seeds"
    print(f"leakage check: held_out_ids={HELD_OUT_EPISODE_IDS} vs train_walk_ids={sorted(train_walk_ids)} "
          f"train_run_ids={sorted(train_run_ids)} -> disjoint OK; seed overlap={held_out_seeds & train_seeds}")

    # --- score: concatenate Walk+Run, ONE score_one_jvp_bank call -> shared CRN by construction ---
    all_candidates = walk_candidates + run_candidates
    scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, all_candidates, CANDIDATE_CHUNK_SIZE)
    scores = scores.cpu().numpy()
    assert np.isfinite(scores).all(), "non-finite LCG scores"
    walk_scores, run_scores = scores[:500], scores[500:]

    # --- secondary: denoising loss on the same transitions, independent seeded generator ---
    gen = torch.Generator(device=device)
    gen.manual_seed(DENOISING_LOSS_SEED)
    walk_losses = np.array([denoising_loss(denoiser, *c, sigma_cfg, gen) for c in walk_candidates])
    run_losses = np.array([denoising_loss(denoiser, *c, sigma_cfg, gen) for c in run_candidates])

    # --- per-transition rows ---
    rows = []
    for meta, score, loss in zip(walk_meta, walk_scores, walk_losses):
        rows.append({"domain": domain, "behavior": "walk", **meta, "lcg_score": float(score), "denoising_loss": float(loss)})
    for meta, score, loss in zip(run_meta, run_scores, run_losses):
        rows.append({"domain": domain, "behavior": "run", **meta, "lcg_score": float(score), "denoising_loss": float(loss)})

    walk_ep_ids = np.array([m["episode_id"] for m in walk_meta])
    run_ep_ids = np.array([m["episode_id"] for m in run_meta])
    walk_ep_stats = episode_level_means(walk_scores, walk_ep_ids)
    run_ep_stats = episode_level_means(run_scores, run_ep_ids)
    walk_ep_means = np.array([v["mean"] for v in walk_ep_stats.values()])
    run_ep_means = np.array([v["mean"] for v in run_ep_stats.values()])

    boot = bootstrap_ci(walk_ep_means, run_ep_means, NUM_BOOTSTRAP, BOOTSTRAP_SEED)
    perm = exact_permutation_test(walk_ep_means, run_ep_means)

    walk_desc = describe(walk_scores)
    run_desc = describe(run_scores)
    delta_mean = run_desc["mean"] - walk_desc["mean"]
    ratio_mean = run_desc["mean"] / walk_desc["mean"]
    delta_median = run_desc["median"] - walk_desc["median"]

    print(f"walk: {walk_desc}")
    print(f"run:  {run_desc}")
    print(f"delta_mean={delta_mean:.6f} ratio_mean={ratio_mean:.4f} delta_median={delta_median:.6f}")
    print(f"bootstrap 95% CI for (run-walk): [{boot['ci_2.5']:.6f}, {boot['ci_97.5']:.6f}]")
    print(f"exact permutation p (one-sided run>walk)={perm['p_value_one_sided_run_gt_walk']:.4f} "
          f"p (one-sided walk>run)={perm['p_value_one_sided_walk_gt_run']:.4f} "
          f"p (two-sided)={perm['p_value_two_sided']:.4f}")
    print(f"walk episode means: {walk_ep_means.tolist()}")
    print(f"run episode means:  {run_ep_means.tolist()}")
    print(f"walk denoising loss mean={walk_losses.mean():.6f}  run denoising loss mean={run_losses.mean():.6f}")

    domain_dir = out_dir if out_dir is not None else (OUT_ROOT / domain / cond_dir(domain, condition))
    domain_dir.mkdir(parents=True, exist_ok=True)
    import csv
    with open(domain_dir / "per_transition_scores.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    result = {
        "domain": domain,
        "condition": condition,
        "checkpoint_path": str(checkpoint_path),
        "theta_s_dim": d_S,
        "theta_s_tensor_count": len(theta_s_named),
        "theta_s_parameter_names": sorted(theta_s_named.keys()),
        "h_D_sum": float(h_D.sum().item()),
        "h_D_all_finite": bool(torch.isfinite(h_D).all()),
        "h_D_all_positive": bool((h_D > 0).all()),
        "h_D_inv_sqrt_all_finite": bool(torch.isfinite(h_D_inv_sqrt).all()),
        "lcg_config": {
            "precision_reference_size": PRECISION_REFERENCE_SIZE, "precision_num_mc": PRECISION_NUM_MC,
            "beta": BETA, "damping": DAMPING, "candidate_num_mc": CANDIDATE_NUM_MC,
            "candidate_chunk_size": CANDIDATE_CHUNK_SIZE,
        },
        "seeds": {"precision_seed": PRECISION_SEED, "bank_seed": BANK_SEED,
                   "denoising_loss_seed": DENOISING_LOSS_SEED, "bootstrap_seed": BOOTSTRAP_SEED},
        "held_out_episode_ids": HELD_OUT_EPISODE_IDS,
        "training_dataset_provenance": mix_manifest,
        "walk_stats": walk_desc, "run_stats": run_desc,
        "delta_mean": delta_mean, "ratio_mean": ratio_mean, "delta_median": delta_median,
        "walk_episode_stats": walk_ep_stats, "run_episode_stats": run_ep_stats,
        "bootstrap": boot, "permutation_test": perm,
        "walk_denoising_loss_mean": float(walk_losses.mean()), "run_denoising_loss_mean": float(run_losses.mean()),
        "n_walk_candidates": len(walk_candidates), "n_run_candidates": len(run_candidates),
        "n_walk_unique_episodes": len(set(walk_ep_ids.tolist())), "n_run_unique_episodes": len(set(run_ep_ids.tolist())),
    }
    (domain_dir / "summary.json").write_text(json.dumps(result, indent=2))
    return result


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_results = {}
    for domain in DOMAINS:
        all_results[domain] = process_domain(domain, "run_scarce")
    (OUT_ROOT / "phase5_summary.json").write_text(json.dumps(all_results, indent=2))

    print("\n\n=== PHASE 5 FINAL SUMMARY ===")
    for domain, r in all_results.items():
        print(f"{domain}: delta_mean={r['delta_mean']:.6f} ratio={r['ratio_mean']:.4f} "
              f"run>walk direction correct={r['delta_mean'] > 0} "
              f"perm_p={r['permutation_test']['p_value_one_sided_run_gt_walk']:.4f}")


if __name__ == "__main__":
    main()
