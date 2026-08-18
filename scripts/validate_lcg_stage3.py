#! /usr/bin/env python
"""
LCG Stage 3 validation: offline candidate precision contribution / Algorithm 3
(lcg.candidate_score), using the trained diagnostic checkpoint from Stage 2.6 (where all
theta_S coordinates have nonzero gradient -- see scripts/train_lcg_diagnostic_checkpoint.py).

Does NOT modify src/lcg/precision.py, gauss_newton.py, sigma_strata.py, theta_s.py, or
WorldModelEnv/Trainer. Uses lcg.gauss_newton.compute_vjp (Stage 1) and
lcg.precision.historical_precision (Stage 2) exactly as validated.

Candidate generation: conditioning windows x* are drawn from a held-out static replay set
(a fresh random-policy cheetah/run collection, seed distinct from both the checkpoint's
training set and its validation set); y* is a genuine model-imagined next observation drawn
via DIAMOND's own DiffusionSampler (not the ground-truth next frame). Algorithm 3 then
independently corrupts that y* as y_sigma = y* + sigma*eps, exactly as in Stage 1/2.

Runs diagnostics 1-6 plus cost, as specified.

Usage:
    python scripts/validate_lcg_stage3.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from data import BatchSampler, Dataset, Episode
from envs.dm_control_env import make_dm_control_env
from lcg.candidate_score import candidate_score, candidate_score_per_stratum
from lcg.precision import historical_precision, load_transition, sample_valid_transitions
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DIAG_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"  # same recipe as Stage 2.6, for h_D
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage3_heldout_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)

NUM_STEPS_CONDITIONING = 4
NUM_CANDIDATES = 20
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0
NUM_SEEDS = 5


# --------------------------------------------------------------------------------------
# Setup: reload the Stage-2.6 trained diagnostic checkpoint, rebuild its exact train
# dataset (for h_D) and a fresh held-out set (for candidates).
# --------------------------------------------------------------------------------------


def collect_episode(env, rng: np.random.Generator) -> Episode:
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


def build_dataset(directory: Path, name: str, target_num_steps: int, seed: int, include_stray: bool) -> Dataset:
    if directory.exists():
        shutil.rmtree(directory)
    dataset = Dataset(directory, name, cache_in_ram=True)
    if include_stray:
        stray_path = REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46" / "dataset" / "train" / "000" / "00" / "0" / "0.pt"
        if stray_path.is_file():
            dataset.add_episode(Episode.load(stray_path))
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    return dataset


def load_diagnostic_denoiser(device: torch.device) -> Denoiser:
    ckpt = torch.load(DIAG_CHECKPOINT_PATH, map_location=device, weights_only=False)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=ckpt["action_dim"],
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    print(f"Loaded Stage-2.6 diagnostic checkpoint (trained step={ckpt['step']}, action_dim={ckpt['action_dim']}).")
    return denoiser


def theta_s_groups(denoiser: Denoiser):
    final_level = denoiser.inner_model.unet.u_blocks[-1]
    groups = [(f"resblock_{i}", list(rb.parameters())) for i, rb in enumerate(final_level.resblocks)]
    groups.append(("norm_out", list(denoiser.inner_model.norm_out.parameters())))
    groups.append(("conv_out", list(denoiser.inner_model.conv_out.parameters())))
    flat_check = [p for _, plist in groups for p in plist]
    real = selected_parameters(denoiser)
    assert len(flat_check) == len(real) and all(a is b for a, b in zip(flat_check, real))
    return groups


def group_slices(denoiser: Denoiser):
    offset = 0
    slices = []
    for name, plist in theta_s_groups(denoiser):
        n = sum(p.numel() for p in plist)
        slices.append((name, offset, offset + n))
        offset += n
    return slices


# --------------------------------------------------------------------------------------
# Candidate generation: real x* conditioning window + model-imagined y* via the real
# DiffusionSampler (not ground-truth next observation).
# --------------------------------------------------------------------------------------


def generate_candidates(denoiser, dataset, num_candidates, seed):
    sampler = DiffusionSampler(denoiser, SAMPLER_CFG)
    device = denoiser.device
    segment_ids = sample_valid_transitions(dataset, num_candidates, NUM_STEPS_CONDITIONING, seed=seed)
    candidates = []
    for segment_id in segment_ids:
        segment = dataset[segment_id]
        n = NUM_STEPS_CONDITIONING
        obs_window = segment.obs[:n].unsqueeze(0).to(device)  # (1, n, C, H, W) for the sampler
        act_window = segment.act[:n].unsqueeze(0).to(device)  # (1, n, action_dim)
        with torch.no_grad():
            y_star, _ = sampler.sample(obs_window, act_window)  # (1, C, H, W), model-imagined
        x_obs_flat = obs_window.reshape(1, -1, obs_window.shape[-2], obs_window.shape[-1])
        candidates.append((x_obs_flat, act_window, y_star.detach()))
    return candidates


# --------------------------------------------------------------------------------------
# Small stats helpers
# --------------------------------------------------------------------------------------


def pearson_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman_corr(a: torch.Tensor, b: torch.Tensor) -> float:
    rank_a = torch.argsort(torch.argsort(a)).float()
    rank_b = torch.argsort(torch.argsort(b)).float()
    return pearson_corr(rank_a, rank_b)


def topk_mass_fraction(vec: torch.Tensor, fracs=(0.001, 0.01, 0.10)):
    d = vec.numel()
    sorted_vec, _ = torch.sort(vec, descending=True)
    cum = torch.cumsum(sorted_vec, dim=0)
    total = cum[-1].item()
    out = {}
    for f in fracs:
        k = max(1, int(round(d * f)))
        out[f] = (cum[k - 1].item() / total) if total > 0 else float("nan")
    return out


# --------------------------------------------------------------------------------------
# Diagnostic 1: direct algebra check
# --------------------------------------------------------------------------------------


def diagnostic_1(one_candidate_v_sq_mean: torch.Tensor, one_candidate_h: torch.Tensor):
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 1: direct algebra check  sum_k v_k^2/h_k == v^T diag(h)^-1 v")
    print("=" * 88)

    torch.manual_seed(0)
    d = 37
    h = torch.rand(d) * 5 + 0.1
    v = torch.randn(d) * 3
    lhs = (v.square() / h).sum()
    rhs = v @ torch.diag(1.0 / h) @ v
    rel_err = ((lhs - rhs).abs() / rhs.abs()).item()
    print(f"  [synthetic, d={d}] lhs={lhs.item():.10g}  rhs={rhs.item():.10g}  rel_err={rel_err:.3e}")
    assert rel_err < 1e-5

    idx = torch.randperm(one_candidate_h.numel())[:2000]
    h_sub, v2_sub = one_candidate_h[idx], one_candidate_v_sq_mean[idx]
    v_sub = v2_sub.sqrt()  # a valid v with the same v^2 (sign is irrelevant to v^2/h)
    lhs2 = (v2_sub / h_sub).sum()
    rhs2 = v_sub @ torch.diag(1.0 / h_sub) @ v_sub
    rel_err2 = ((lhs2 - rhs2).abs() / rhs2.abs()).item()
    print(f"  [real subset, d=2000 of d_S={one_candidate_h.numel()}] lhs={lhs2.item():.10g}  "
          f"rhs={rhs2.item():.10g}  rel_err={rel_err2:.3e}")
    assert rel_err2 < 1e-4


# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_diagnostic_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage3_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage3_heldout", 400, seed=555, include_stray=False)
    print(f"train (for h_D): {train_dataset.num_episodes} episodes, N={train_dataset.num_steps} steps")
    print(f"held-out (for candidates): {heldout_dataset.num_episodes} episodes, N={heldout_dataset.num_steps} steps")

    # ---- frozen h_D (Algorithm 2, unmodified) ----
    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,)
    assert torch.all(h_D > 0)
    print(f"h_D computed and frozen: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, "
          f"max={h_D.max().item():.4g}, all>0: {bool(torch.all(h_D > 0))}")

    before_state = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    # ---- fixed candidate set: real x*, model-imagined y* ----
    candidates = generate_candidates(denoiser, heldout_dataset, NUM_CANDIDATES, seed=42)
    print(f"Generated {len(candidates)} candidates: real held-out x*, DiffusionSampler-imagined y*.")

    # ---- Diagnostic 2 + collect data reused by 1, 3, 6 ----
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 2: positivity and numerical validity")
    print("=" * 88)
    scores, all_diag = [], []
    t0 = time.perf_counter()
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    for x_obs, x_act, y_star in candidates:
        score, diag = candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=3, return_diagnostics=True)
        scores.append(score)
        all_diag.append(diag)
    t_main = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None

    for k, v in denoiser.state_dict().items():
        assert torch.equal(before_state[k], v), f"denoiser parameter {k} changed!"
    print("  [integrity] denoiser parameters unchanged before vs after: OK")

    scores_t = torch.tensor(scores)
    finite = torch.isfinite(scores_t).all().item()
    nonneg = (scores_t >= 0).all().item()
    v2h_finite = all(torch.isfinite(d["v_squared"] / h_D.cpu().unsqueeze(0)).all().item() for d in all_diag)
    print(f"  finite: {finite}  nonnegative: {nonneg}  v^2/h_D finite everywhere: {v2h_finite}")
    q = torch.quantile(scores_t, torch.tensor([0.5, 0.9, 0.99]))
    print(f"  score: min={scores_t.min().item():.6g}  median={q[0].item():.6g}  mean={scores_t.mean().item():.6g}  "
          f"p90={q[1].item():.6g}  p99={q[2].item():.6g}  max={scores_t.max().item():.6g}")
    assert finite and nonneg and v2h_finite

    # ---- Diagnostic 1 (uses one real candidate's data, computed above) ----
    v_sq_mean0 = all_diag[0]["v_squared"].mean(dim=0)
    diagnostic_1(v_sq_mean0, h_D.cpu())

    # ---- Diagnostic 3: contribution concentration ----
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 3: contribution concentration")
    print("=" * 88)
    slices = group_slices(denoiser)
    h_D_cpu = h_D.cpu()
    group_frac_per_candidate = {name: [] for name, _, _ in slices}
    topk_per_candidate = {0.001: [], 0.01: [], 0.10: []}
    for diag in all_diag:
        per_coord = diag["v_squared"].mean(dim=0) / h_D_cpu  # (d_S,), sums to this candidate's score
        total = per_coord.sum().item()
        for name, start, stop in slices:
            group_frac_per_candidate[name].append(per_coord[start:stop].sum().item() / total)
        tk = topk_mass_fraction(per_coord)
        for f in topk_per_candidate:
            topk_per_candidate[f].append(tk[f])

    print("  module-wise mean score fraction across candidates:")
    for name, _, _ in slices:
        vals = torch.tensor(group_frac_per_candidate[name])
        print(f"    {name:<12} mean={vals.mean().item():.4f}  std={vals.std().item():.4f}")

    print("\n  top-k coordinate concentration (mean fraction of total score across candidates):")
    for f in (0.001, 0.01, 0.10):
        vals = torch.tensor(topk_per_candidate[f])
        print(f"    top {f * 100:>5.1f}% of coords: mean={vals.mean().item():.4f}  std={vals.std().item():.4f}")

    # ---- Diagnostic 4: MC / ranking stability across seeds ----
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 4: Monte Carlo / ranking stability across seeds")
    print("=" * 88)
    seeds = list(range(NUM_SEEDS))

    def scores_for_seed(num_strata, seed, cand_list):
        torch.manual_seed(seed)
        return torch.tensor([
            candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=num_strata)
            for x_obs, x_act, y_star in cand_list
        ])

    scores_k3 = [scores_for_seed(3, s, candidates) for s in seeds]
    pear, spear = [], []
    for i in range(len(scores_k3)):
        for j in range(i + 1, len(scores_k3)):
            pear.append(pearson_corr(scores_k3[i], scores_k3[j]))
            spear.append(spearman_corr(scores_k3[i], scores_k3[j]))
    stacked = torch.stack(scores_k3)  # (num_seeds, num_candidates)
    rel_var = (stacked.std(dim=0) / stacked.mean(dim=0).clamp_min(1e-12))
    print(f"  num_strata=3, {len(seeds)} seeds, {len(candidates)} candidates:")
    print(f"    mean Pearson correlation (pairwise across seeds): {np.mean(pear):.4f}")
    print(f"    mean Spearman rank correlation (pairwise across seeds): {np.mean(spear):.4f}")
    print(f"    relative variation (std/mean) per candidate: mean={rel_var.mean().item():.4f}  "
          f"median={rel_var.median().item():.4f}  max={rel_var.max().item():.4f}")

    # ---- Diagnostic 5: K=1 vs K=3 ----
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 5: num_strata=1 vs num_strata=3")
    print("=" * 88)
    t0 = time.perf_counter()
    scores_k1 = [scores_for_seed(1, s, candidates) for s in seeds]
    t_k1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    _ = [scores_for_seed(3, s, candidates) for s in seeds]  # re-time K=3 under identical conditions
    t_k3 = time.perf_counter() - t0

    pear1, spear1 = [], []
    for i in range(len(scores_k1)):
        for j in range(i + 1, len(scores_k1)):
            pear1.append(pearson_corr(scores_k1[i], scores_k1[j]))
            spear1.append(spearman_corr(scores_k1[i], scores_k1[j]))
    stacked1 = torch.stack(scores_k1)
    rel_var1 = (stacked1.std(dim=0) / stacked1.mean(dim=0).clamp_min(1e-12))

    cross_pear = pearson_corr(scores_k1[0], scores_k3[0])
    cross_spear = spearman_corr(scores_k1[0], scores_k3[0])
    print(f"  K=1: mean Pearson={np.mean(pear1):.4f}  mean Spearman={np.mean(spear1):.4f}  "
          f"rel_var mean={rel_var1.mean().item():.4f}")
    print(f"  K=3: mean Pearson={np.mean(pear):.4f}  mean Spearman={np.mean(spear):.4f}  "
          f"rel_var mean={rel_var.mean().item():.4f}")
    print(f"  K=1 vs K=3 (one seed each) score correlation: Pearson={cross_pear:.4f}  Spearman={cross_spear:.4f}")
    print(f"  runtime for {len(seeds)} seeds x {len(candidates)} candidates: K=1 -> {t_k1:.2f}s  K=3 -> {t_k3:.2f}s  "
          f"(ratio K3/K1={t_k3 / t_k1:.2f})")

    # ---- Diagnostic 6: effect of historical precision ----
    print("\n" + "=" * 88)
    print("DIAGNOSTIC 6: effect of historical precision (h_D-weighted vs raw GN magnitude)")
    print("=" * 88)
    raw_scores = torch.tensor([d["v_squared"].mean(dim=0).sum().item() for d in all_diag])
    weighted_scores = scores_t
    print(f"  raw sum(v^2) vs h_D-weighted sum(v^2/h_D) across {len(candidates)} candidates:")
    print(f"    Pearson correlation: {pearson_corr(raw_scores, weighted_scores):.4f}")
    print(f"    Spearman rank correlation: {spearman_corr(raw_scores, weighted_scores):.4f}")

    # ---- Cost ----
    print("\n" + "=" * 88)
    print("COST")
    print("=" * 88)
    print(f"  ms/candidate (K=3, main pass): {t_main / len(candidates) * 1000:.2f} ms")
    print(f"  VJPs/candidate: 3 (num_strata)")
    if peak_mem_mb is not None:
        print(f"  peak GPU memory (main pass): {peak_mem_mb:.1f} MB")
    else:
        print("  peak GPU memory: n/a (CPU)")
    print(f"  K=1 vs K=3 cost ratio (from diagnostic 5 timing): {t_k3 / t_k1:.2f} (expect ~3.0)")
    print(f"  d_S: {d_S}")

    print("\nStage 3 validation complete.")


if __name__ == "__main__":
    main()
