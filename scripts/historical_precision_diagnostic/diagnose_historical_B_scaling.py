#! /usr/bin/env python
"""
Historical precision B-scaling diagnostic: as replay size N grows, does the required
historical subset B also need to grow? Stage A only (N~1540, the sole trustworthy
matched checkpoint/replay snapshot found in this repository -- see the Stage B/C audit
in the accompanying report). Diagnostic only -- does not modify production
historical_precision(), LCGConfig, or any estimator.

Key compute optimization: per-transition contributions g_i (production M_h=3, backward
VJP) are computed ONCE for every transition in the dataset, then every B is derived by
nested-subset summation of the cached g_i -- no VJP is ever recomputed per B.

Usage:
    python scripts/historical_precision_diagnostic/diagnose_historical_B_scaling.py
"""
import hashlib
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))

import numpy as np
import torch

from data import Dataset, Episode
from envs.dm_control_env import make_dm_control_env
from lcg.forward_jvp import frozen_named_parameters, make_jvp_bank, score_one_jvp_bank, selected_named_parameters
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition, sample_uniform_historical_transitions
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.denoiser import apply_noise_from_samples, sample_sigma_training_distribution
from models.diffusion.inner_model import InnerModelConfig

SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DENOISER_PATH = SCRATCH_BASE / "lcg_diag_denoiser_converged.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
CANDIDATES_PATH = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "checkpoint" / "frozen_candidates_480.pt"

OUT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "historical_precision_B_scaling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
NUM_STEPS_CONDITIONING = 4
TRAIN_TARGET_STEPS = 1500
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
DAMPING = 1e-4
BETA = 1.0
M_H = 3  # production historical-precision simple-MC budget

B_GRID = [40, 80, 160, 320, 640, 1280]
NUM_SUBSET_SEEDS = 8
CONTRIBUTION_SEED = 12345  # one fixed seed for the g_i computation (isolates B-effects from MC noise)

CANDIDATE_SUBSET_SIZE = 160
CANDIDATE_M = 12
CANDIDATE_CHUNK_SIZE = 16
CANDIDATE_BANK_SEED = 424242


def collect_episode(env, rng):
    obs_frames, act_list, rew_list, end_list, trunc_list = [], [], [], [], []
    raw_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    while not done:
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        next_raw_obs, rew, terminated, truncated, _ = env.step(action)
        obs_frames.append(raw_obs)
        act_list.append(action)
        rew_list.append(rew)
        end_list.append(int(terminated))
        trunc_list.append(int(truncated))
        raw_obs = next_raw_obs
        done = terminated or truncated
    obs = torch.from_numpy(np.stack(obs_frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    act = torch.from_numpy(np.stack(act_list)).float()
    rew = torch.tensor(rew_list, dtype=torch.float32)
    end = torch.tensor(end_list, dtype=torch.uint8)
    trunc = torch.tensor(trunc_list, dtype=torch.uint8)
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info={})


def rebuild_or_load_train_dataset():
    if TRAIN_DATASET_DIR.is_dir() and any(TRAIN_DATASET_DIR.rglob("*.pt")):
        dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train", cache_in_ram=True)
        dataset.load_from_default_path()
        if dataset.num_steps > 0:
            print(f"loaded existing train dataset: {dataset.num_episodes} episodes, N={dataset.num_steps}", flush=True)
            return dataset
    print("train dataset missing/empty -- rebuilding with the original recipe (seed=0)...", flush=True)
    if TRAIN_DATASET_DIR.exists():
        shutil.rmtree(TRAIN_DATASET_DIR)
    dataset = Dataset(TRAIN_DATASET_DIR, "lcg_diag_train", cache_in_ram=True)
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(0)
    while dataset.num_steps < TRAIN_TARGET_STEPS:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    print(f"rebuilt train dataset: {dataset.num_episodes} episodes, N={dataset.num_steps}", flush=True)
    return dataset


def load_denoiser(device):
    ckpt = torch.load(DENOISER_PATH, map_location=device, weights_only=False)
    action_dim = ckpt["action_dim"]
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    return denoiser, action_dim, ckpt["step"]


def sha256_of(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pearson(a, b):
    a, b = a.double(), b.double()
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman(a, b):
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson(ra, rb)


def top_q_overlap(a, b, q):
    k = max(1, int(round(q * a.numel())))
    sa = set(torch.topk(a, k).indices.tolist())
    sb = set(torch.topk(b, k).indices.tolist())
    return len(sa & sb) / k


def h_D_comparison(h_test, h_ref):
    inv_test, inv_ref = h_test.rsqrt(), h_ref.rsqrt()
    return dict(
        raw_pearson=pearson(h_test, h_ref), raw_spearman=spearman(h_test, h_ref),
        raw_rel_l2=((h_test - h_ref).norm() / h_ref.norm()).item(),
        raw_norm_ratio=(h_test.norm() / h_ref.norm()).item(),
        inv_sqrt_pearson=pearson(inv_test, inv_ref), inv_sqrt_spearman=spearman(inv_test, inv_ref),
        inv_sqrt_rel_l2=((inv_test - inv_ref).norm() / inv_ref.norm()).item(),
    )


def candidate_metrics(scores_test, scores_ref):
    return dict(
        spearman=spearman(scores_test, scores_ref), pearson=pearson(scores_test, scores_ref),
        top10=top_q_overlap(scores_test, scores_ref, 0.10), top20=top_q_overlap(scores_test, scores_ref, 0.20),
        rms_ratio=(scores_test.pow(2).mean().sqrt() / scores_ref.pow(2).mean().sqrt()).item(),
    )


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim, ckpt_step = load_denoiser(device)
    denoiser_sha256 = sha256_of(DENOISER_PATH)
    print(f"checkpoint: {DENOISER_PATH} (total_step={ckpt_step}) sha256={denoiser_sha256}", flush=True)

    dataset = rebuild_or_load_train_dataset()
    N = dataset.num_steps
    theta_s_named = selected_named_parameters(denoiser)
    params = list(theta_s_named.values())
    d_S = sum(p.numel() for p in params)
    print(f"N={N}  d_S={d_S}", flush=True)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True).stdout.strip()

    manifest = dict(
        git_commit=git_commit, checkpoint_path=str(DENOISER_PATH), checkpoint_sha256=denoiser_sha256,
        checkpoint_total_step=ckpt_step, dataset_path=str(TRAIN_DATASET_DIR), dataset_N=N,
        dataset_num_episodes=dataset.num_episodes,
        dataset_generation_provenance="dm_control cheetah/run, seed=0, same recipe as the original "
                                       "train_lcg_diagnostic_checkpoint.py (now deleted, evidence in docs/)",
        theta_s_dim=d_S, M_h=M_H, beta=BETA, damping=DAMPING,
        sigma_cfg=dict(loc=SIGMA_CFG.loc, scale=SIGMA_CFG.scale, sigma_min=SIGMA_CFG.sigma_min, sigma_max=SIGMA_CFG.sigma_max),
        sigma_offset_noise=denoiser.cfg.sigma_offset_noise, contribution_seed=CONTRIBUTION_SEED,
        B_grid=B_GRID, num_subset_seeds=NUM_SUBSET_SEEDS,
        candidate_subset_size=CANDIDATE_SUBSET_SIZE, candidate_M=CANDIDATE_M,
        candidate_chunk_size=CANDIDATE_CHUNK_SIZE, candidate_bank_seed=CANDIDATE_BANK_SEED, device=str(device),
    )

    # ------------------------------------------------------------------------------
    # Compute per-transition contributions g_i ONCE for every transition in the dataset
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nComputing per-transition contributions g_i (production M_h=3, once per transition)\n" + "=" * 88, flush=True)
    t0 = time.perf_counter()
    # enumerate ALL N transitions deterministically (global index -> (episode,t), same
    # mapping sample_uniform_historical_transitions uses) via a full without-replacement draw of size N
    all_segment_ids = sample_uniform_historical_transitions(dataset, N, NUM_STEPS_CONDITIONING, seed=0, replace=False)
    # order doesn't matter for correctness (each contributes independently); use this order consistently

    torch_gen = torch.Generator(device=device)
    torch_gen.manual_seed(CONTRIBUTION_SEED)

    g_all = torch.zeros(N, d_S, device=device)
    for i, segment_id in enumerate(all_segment_ids):
        obs, act, y = load_transition(dataset, segment_id, NUM_STEPS_CONDITIONING, device)
        g_i = torch.zeros(d_S, device=device)
        for _ in range(M_H):
            sigma = sample_sigma_training_distribution(SIGMA_CFG, 1, device, generator=torch_gen)
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=torch_gen)
            eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=device, generator=torch_gen)
            y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, generator=torch_gen)
            g_i = g_i + (v * v) / M_H
        g_all[i] = g_i
        if (i + 1) % 400 == 0 or (i + 1) == N:
            print(f"  contributions: {i + 1}/{N}  elapsed={time.perf_counter() - t0:.1f}s", flush=True)

    t_contributions = time.perf_counter() - t0
    manifest["t_contributions_s"] = t_contributions
    print(f"\ncontribution computation: {t_contributions:.1f}s total, {1000 * t_contributions / (N * M_H):.2f} ms/VJP", flush=True)

    # ------------------------------------------------------------------------------
    # Full-replay reference h_D^full = damping*1 + beta * sum_i g_i  (N/B=1 since B=N)
    # ------------------------------------------------------------------------------
    h_D_full = DAMPING * torch.ones(d_S, device=device) + BETA * g_all.sum(dim=0)
    torch.save(dict(h_D_full=h_D_full.cpu(), g_all_sha256=hashlib.sha256(g_all.cpu().numpy().tobytes()).hexdigest()),
               OUT_DIR / "reference_h_D_full_N1540.pt")
    print(f"h_D_full: shape={tuple(h_D_full.shape)} finite={torch.isfinite(h_D_full).all().item()} "
          f"positive={(h_D_full > 0).all().item()} min={h_D_full.min().item():.4g} max={h_D_full.max().item():.4g}", flush=True)

    # ------------------------------------------------------------------------------
    # Nested B subsets, 8 independent permutation seeds, pure subset summation (cheap)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nNested B-subset construction (cheap: pure summation of cached g_i)\n" + "=" * 88, flush=True)
    t0 = time.perf_counter()
    valid_B = [B for B in B_GRID if B <= N]
    skipped_B = [B for B in B_GRID if B > N]
    if skipped_B:
        print(f"  skipping B>N: {skipped_B} (N={N})")

    h_D_by_B_seed = {B: [] for B in valid_B}
    for seed_idx in range(NUM_SUBSET_SEEDS):
        rng = np.random.default_rng(seed_idx + 500_000)  # local, does not touch global numpy state
        perm = rng.permutation(N)
        for B in valid_B:
            idx = perm[:B]
            g_sum = g_all[idx].sum(dim=0)
            h_D_B = DAMPING * torch.ones(d_S, device=device) + BETA * (N / B) * g_sum
            h_D_by_B_seed[B].append(h_D_B)
    t_subsets = time.perf_counter() - t0
    manifest["t_subset_construction_s"] = t_subsets
    print(f"subset construction (all {len(valid_B)} B values x {NUM_SUBSET_SEEDS} seeds): {t_subsets:.3f}s", flush=True)

    print(f"\n{'B':>5} {'B/N':>7} {'raw_pearson':>12} {'raw_spearman':>13} {'raw_rel_l2':>11} "
          f"{'inv_sqrt_pearson':>17} {'inv_sqrt_spearman':>18} {'inv_sqrt_rel_l2':>16}")
    h_D_metrics_summary = {}
    for B in valid_B:
        metrics_list = [h_D_comparison(h, h_D_full) for h in h_D_by_B_seed[B]]
        summary = {k: (float(np.mean([m[k] for m in metrics_list])), float(np.std([m[k] for m in metrics_list]))) for k in metrics_list[0]}
        h_D_metrics_summary[B] = summary
        print(f"{B:>5} {B / N:>7.4f} {summary['raw_pearson'][0]:>7.4f}+-{summary['raw_pearson'][1]:<4.4f} "
              f"{summary['raw_spearman'][0]:>8.4f}+-{summary['raw_spearman'][1]:<4.4f} "
              f"{summary['raw_rel_l2'][0]:>6.4f}+-{summary['raw_rel_l2'][1]:<4.4f} "
              f"{summary['inv_sqrt_pearson'][0]:>12.4f}+-{summary['inv_sqrt_pearson'][1]:<4.4f} "
              f"{summary['inv_sqrt_spearman'][0]:>13.4f}+-{summary['inv_sqrt_spearman'][1]:<4.4f} "
              f"{summary['inv_sqrt_rel_l2'][0]:>11.4f}+-{summary['inv_sqrt_rel_l2'][1]:<4.4f}")

    # ------------------------------------------------------------------------------
    # Downstream candidate-score comparison: ONE fixed candidate CRN bank shared across
    # h_D_full and every h_D^(B)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nDownstream candidate-score comparison\n" + "=" * 88, flush=True)
    t0 = time.perf_counter()
    candidates_all = torch.load(CANDIDATES_PATH, map_location=device, weights_only=True)
    candidates = [(c[0].to(device), c[1].to(device), c[2].to(device)) for c in candidates_all[:CANDIDATE_SUBSET_SIZE]]
    print(f"  using {len(candidates)}/{len(candidates_all)} candidates (first {CANDIDATE_SUBSET_SIZE}, deterministic)")

    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    bank = make_jvp_bank(SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=CANDIDATE_M, seed=CANDIDATE_BANK_SEED)
    print(f"  fixed candidate CRN bank built once (M={CANDIDATE_M}, seed={CANDIDATE_BANK_SEED}), reused for every h_D")

    def score_with(h_D_variant):
        return score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_variant,
                                   h_D_variant.rsqrt(), bank, candidates, CANDIDATE_CHUNK_SIZE)

    scores_full = score_with(h_D_full)
    print(f"  reference (h_D_full) scores: finite={torch.isfinite(scores_full).all().item()} "
          f"nonneg={(scores_full >= 0).all().item()} mean={scores_full.mean().item():.3f}")

    print(f"\n{'B':>5} {'spearman':>16} {'pearson':>16} {'top10':>16} {'top20':>16} {'rms_ratio':>16}")
    candidate_metrics_summary = {}
    for B in valid_B:
        per_seed_metrics = [candidate_metrics(score_with(h), scores_full) for h in h_D_by_B_seed[B]]
        summary = {k: (float(np.mean([m[k] for m in per_seed_metrics])), float(np.std([m[k] for m in per_seed_metrics]))) for k in per_seed_metrics[0]}
        candidate_metrics_summary[B] = summary
        print(f"{B:>5} " + " ".join(f"{summary[k][0]:>7.4f}+-{summary[k][1]:<6.4f}" for k in ["spearman", "pearson", "top10", "top20", "rms_ratio"]))

    t_candidate_scoring = time.perf_counter() - t0
    manifest["t_candidate_scoring_s"] = t_candidate_scoring
    print(f"\ncandidate-scoring stage: {t_candidate_scoring:.1f}s ({len(valid_B) * NUM_SUBSET_SEEDS + 1} scoring calls)", flush=True)
    if device.type == "cuda":
        manifest["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"peak CUDA memory: {manifest['peak_cuda_memory_mb']:.1f}MB")

    # ------------------------------------------------------------------------------
    # save
    # ------------------------------------------------------------------------------
    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)
    results_summary = dict(
        h_D_metrics_by_B={str(B): {k: list(v) for k, v in s.items()} for B, s in h_D_metrics_summary.items()},
        candidate_metrics_by_B={str(B): {k: list(v) for k, v in s.items()} for B, s in candidate_metrics_summary.items()},
    )
    with open(OUT_DIR / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("\nStage A diagnostic complete.", flush=True)


if __name__ == "__main__":
    main()
