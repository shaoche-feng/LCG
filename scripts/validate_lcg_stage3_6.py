#! /usr/bin/env python
"""
LCG Stage 3.6: small CRN (common random numbers) protocol validation, following Stage 3.5's
finding that CRN at M=3 recovers most of the ranking signal that independent-per-candidate
sampling destroys at the same VJP budget.

Does NOT change the LCG objective or lcg.candidate_score's estimator -- CRN only changes
which (sigma, eps, xi) draws different candidates receive, built entirely from
lcg.gauss_newton.compute_vjp (Stage 1, unmodified) and lcg.candidate_score (Stage 3,
unmodified, used as-is for the independent-sampling and reference conditions). No changes
to WorldModelEnv/ActorCritic/Trainer.

Question this script answers: not "does a fixed CRN bank reproduce itself" (trivially yes),
but "do several *independently generated* fixed 3-probe CRN banks each approximate the same
higher-budget reference ranking" -- i.e. is CRN's benefit a real, reproducible reduction in
ranking noise, or a fluke of one particular bank.

Sections:
  1. Fixed pool of 40 candidates (real held-out x*, DiffusionSampler-imagined y*), frozen h_D.
  2. High-budget reference (M_ref=64, independent per-candidate sampling, as in Stage 3.5).
  3. 8 independent-sampling M=3 trials vs 8 independently-generated CRN M=3 banks, each
     compared individually against the reference: Pearson, Spearman, top-10%/top-k overlap,
     runtime. Also cross-trial/cross-bank agreement.
  4. Using the exact same CRN-bank probes, raw sum(v^2) vs LCG sum(v^2/h_D): does h_D change
     the ranking under matched (noise-free-of-that-confound) conditions.
  5. A written assessment (no new code) of freezing one CRN bank for an entire actor-critic
     inner training phase, resampled only on world-model/h_D refresh -- grounded in the
     measured per-bank-vs-reference and cross-bank numbers above.

Usage:
    python scripts/validate_lcg_stage3_6.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch

from data import Dataset, Episode
from envs.dm_control_env import make_dm_control_env
from lcg.candidate_score import candidate_score
from lcg.gauss_newton import compute_vjp
from lcg.precision import historical_precision, sample_valid_transitions
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DIAG_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage3_6_heldout_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)

NUM_STEPS_CONDITIONING = 4
NUM_CANDIDATES = 40
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0

M = 3
M_REF = 64
NUM_INDEP_TRIALS = 8
NUM_CRN_BANKS = 8
TOPK_FRAC_A = 0.10
TOPK_FRAC_B = 0.20


# --------------------------------------------------------------------------------------
# Setup (identical recipe to scripts/validate_lcg_stage3.py / diagnose_lcg_stage3_5.py)
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


def generate_candidates(denoiser, dataset, num_candidates, seed):
    sampler = DiffusionSampler(denoiser, SAMPLER_CFG)
    device = denoiser.device
    segment_ids = sample_valid_transitions(dataset, num_candidates, NUM_STEPS_CONDITIONING, seed=seed)
    candidates = []
    for segment_id in segment_ids:
        segment = dataset[segment_id]
        n = NUM_STEPS_CONDITIONING
        obs_window = segment.obs[:n].unsqueeze(0).to(device)
        act_window = segment.act[:n].unsqueeze(0).to(device)
        with torch.no_grad():
            y_star, _ = sampler.sample(obs_window, act_window)
        x_obs_flat = obs_window.reshape(1, -1, obs_window.shape[-2], obs_window.shape[-1])
        candidates.append((x_obs_flat, act_window, y_star.detach()))
    return candidates


# --------------------------------------------------------------------------------------
# CRN bank machinery -- built only from compute_vjp / sample_sigma_stratum (unmodified)
# --------------------------------------------------------------------------------------


def make_crn_bank(y_shape, device, seed, num_strata=M):
    torch.manual_seed(seed)
    bank = []
    for m in range(num_strata):
        sigma = sample_sigma_stratum(SIGMA_CFG, m, num_strata, 1, device)
        eps = torch.randn(y_shape, device=device)
        xi = torch.randn(y_shape, device=device)
        bank.append((sigma, eps, xi))
    return bank


def score_with_bank(denoiser, params, h_D, bank, x_obs, x_act, y_star):
    """Returns (weighted_score, raw_score) computed from the SAME v^2 draws, so their
    difference isolates h_D's effect. weighted_score matches candidate_score()'s formula
    exactly (sum over coords, mean over strata); raw_score uses the same structure without
    the h_D division.
    """
    v_sq_per_stratum = []
    for sigma, eps, xi in bank:
        y_sigma = (y_star + sigma.view(-1, 1, 1, 1) * eps).detach()
        v, _ = compute_vjp(denoiser, params, y_sigma, sigma, x_obs, x_act, xi=xi)
        v_sq_per_stratum.append(v * v)
    v_sq = torch.stack(v_sq_per_stratum)  # (num_strata, d_S)
    weighted = (v_sq / h_D.unsqueeze(0)).sum(dim=1).mean().item()
    raw = v_sq.sum(dim=1).mean().item()
    return weighted, raw


# --------------------------------------------------------------------------------------
# Stats helpers
# --------------------------------------------------------------------------------------


def pearson_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson_corr(ra, rb)


def topk_overlap(scores, ref_scores, frac):
    n = len(scores)
    k = max(1, int(round(n * frac)))
    top = set(torch.topk(torch.as_tensor(scores), k).indices.tolist())
    top_ref = set(torch.topk(torch.as_tensor(ref_scores), k).indices.tolist())
    return len(top & top_ref) / k


def summarize(label, vs_ref_pearson, vs_ref_spearman, vs_ref_topA, vs_ref_topB, runtimes):
    p = np.array(vs_ref_pearson)
    s = np.array(vs_ref_spearman)
    ka = np.array(vs_ref_topA)
    kb = np.array(vs_ref_topB)
    print(f"\n  [{label}] n={len(p)}")
    print(f"    Pearson vs ref:   mean={p.mean():.4f}  std={p.std():.4f}  min={p.min():.4f}  max={p.max():.4f}")
    print(f"    Spearman vs ref:  mean={s.mean():.4f}  std={s.std():.4f}  min={s.min():.4f}  max={s.max():.4f}")
    print(f"    top-{TOPK_FRAC_A * 100:.0f}% overlap vs ref: mean={ka.mean():.4f}  std={ka.std():.4f}")
    print(f"    top-{TOPK_FRAC_B * 100:.0f}% overlap vs ref: mean={kb.mean():.4f}  std={kb.std():.4f}")
    print(f"    runtime: mean={np.mean(runtimes):.2f}s  ({np.mean(runtimes) / NUM_CANDIDATES * 1000:.1f} ms/candidate)")


def cross_group_agreement(score_list, label):
    corrs, scorrs = [], []
    for i in range(len(score_list)):
        for j in range(i + 1, len(score_list)):
            corrs.append(pearson_corr(score_list[i], score_list[j]))
            scorrs.append(spearman_corr(score_list[i], score_list[j]))
    print(f"  [{label}] cross-group agreement ({len(score_list)} groups, {len(corrs)} pairs): "
          f"mean Pearson={np.mean(corrs):.4f}  mean Spearman={np.mean(scorrs):.4f}")


# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_diagnostic_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage3_6_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage3_6_heldout", 500, seed=555, include_stray=False)

    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,) and torch.all(h_D > 0)
    print(f"Frozen h_D: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, max={h_D.max().item():.4g}")

    candidates = generate_candidates(denoiser, heldout_dataset, NUM_CANDIDATES, seed=42)
    print(f"Fixed candidate pool: {len(candidates)} candidates (real held-out x*, DiffusionSampler-imagined y*).")
    before_state = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    # ---- Reference: high-budget, independent per-candidate ----
    print("\n" + "=" * 88)
    print(f"REFERENCE: M_ref={M_REF}, independent per-candidate sampling")
    print("=" * 88)
    t0 = time.perf_counter()
    ref_scores = torch.tensor([
        candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=M_REF)
        for x_obs, x_act, y_star in candidates
    ])
    t_ref = time.perf_counter() - t0
    print(f"  computed in {t_ref:.1f}s ({t_ref / NUM_CANDIDATES * 1000:.1f} ms/candidate)")

    # ---- Independent sampling, M=3, several trials ----
    print("\n" + "=" * 88)
    print(f"INDEPENDENT SAMPLING, M={M}, {NUM_INDEP_TRIALS} trials")
    print("=" * 88)
    indep_scores_list, indep_pear, indep_spear, indep_topA, indep_topB, indep_rt = [], [], [], [], [], []
    for t in range(NUM_INDEP_TRIALS):
        t0 = time.perf_counter()
        scores = torch.tensor([
            candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=M)
            for x_obs, x_act, y_star in candidates
        ])
        dt = time.perf_counter() - t0
        indep_scores_list.append(scores)
        indep_pear.append(pearson_corr(scores, ref_scores))
        indep_spear.append(spearman_corr(scores, ref_scores))
        indep_topA.append(topk_overlap(scores, ref_scores, TOPK_FRAC_A))
        indep_topB.append(topk_overlap(scores, ref_scores, TOPK_FRAC_B))
        indep_rt.append(dt)
    summarize("independent M=3", indep_pear, indep_spear, indep_topA, indep_topB, indep_rt)
    cross_group_agreement(indep_scores_list, "independent M=3")

    # ---- CRN, M=3, several independently generated banks ----
    print("\n" + "=" * 88)
    print(f"COMMON RANDOM NUMBERS, M={M}, {NUM_CRN_BANKS} independently generated banks")
    print("=" * 88)
    y_shape = candidates[0][2].shape
    crn_weighted_list, crn_raw_list = [], []
    crn_pear, crn_spear, crn_topA, crn_topB, crn_rt = [], [], [], [], []
    for b in range(NUM_CRN_BANKS):
        bank = make_crn_bank(y_shape, device, seed=5000 + b, num_strata=M)
        t0 = time.perf_counter()
        weighted, raw = [], []
        for x_obs, x_act, y_star in candidates:
            w, r = score_with_bank(denoiser, params, h_D, bank, x_obs, x_act, y_star)
            weighted.append(w)
            raw.append(r)
        dt = time.perf_counter() - t0
        weighted_t, raw_t = torch.tensor(weighted), torch.tensor(raw)
        crn_weighted_list.append(weighted_t)
        crn_raw_list.append(raw_t)
        crn_pear.append(pearson_corr(weighted_t, ref_scores))
        crn_spear.append(spearman_corr(weighted_t, ref_scores))
        crn_topA.append(topk_overlap(weighted_t, ref_scores, TOPK_FRAC_A))
        crn_topB.append(topk_overlap(weighted_t, ref_scores, TOPK_FRAC_B))
        crn_rt.append(dt)
    summarize("CRN M=3 (each bank vs reference)", crn_pear, crn_spear, crn_topA, crn_topB, crn_rt)
    cross_group_agreement(crn_weighted_list, "CRN M=3 (bank-to-bank agreement)")

    # ---- Raw curvature vs LCG (h_D-weighted), same CRN probes ----
    print("\n" + "=" * 88)
    print("RAW sum(v^2) vs LCG sum(v^2/h_D), using IDENTICAL CRN probes per bank")
    print("=" * 88)
    raw_vs_weighted_pear = [pearson_corr(crn_raw_list[b], crn_weighted_list[b]) for b in range(NUM_CRN_BANKS)]
    raw_vs_weighted_spear = [spearman_corr(crn_raw_list[b], crn_weighted_list[b]) for b in range(NUM_CRN_BANKS)]
    raw_vs_ref_pear = [pearson_corr(crn_raw_list[b], ref_scores) for b in range(NUM_CRN_BANKS)]
    raw_vs_ref_spear = [spearman_corr(crn_raw_list[b], ref_scores) for b in range(NUM_CRN_BANKS)]
    print(f"  raw vs weighted (same bank, same probes): mean Pearson={np.mean(raw_vs_weighted_pear):.4f}  "
          f"mean Spearman={np.mean(raw_vs_weighted_spear):.4f}")
    print(f"  raw (unweighted) vs h_D-weighted reference correlation, for context:")
    print(f"    raw vs weighted-reference: mean Pearson={np.mean(raw_vs_ref_pear):.4f}  "
          f"mean Spearman={np.mean(raw_vs_ref_spear):.4f}")
    print(f"    (weighted CRN vs weighted-reference, from above): mean Pearson={np.mean(crn_pear):.4f}  "
          f"mean Spearman={np.mean(crn_spear):.4f}")

    for k, v in denoiser.state_dict().items():
        assert torch.equal(before_state[k], v), f"denoiser parameter {k} changed during Stage 3.6!"
    print("\n[integrity] denoiser parameters unchanged before vs after Stage 3.6: OK")

    print("\n" + "=" * 88)
    print("SUMMARY")
    print("=" * 88)
    print(f"  independent M=3: mean Spearman-vs-ref={np.mean(indep_spear):.3f} (std={np.std(indep_spear):.3f})")
    print(f"  CRN M=3:         mean Spearman-vs-ref={np.mean(crn_spear):.3f} (std={np.std(crn_spear):.3f})")
    print(f"  raw curvature (no h_D) vs reference:    mean Spearman={np.mean(raw_vs_ref_spear):.3f}")
    print(f"  h_D-weighted (LCG) vs raw, matched probes: mean Spearman={np.mean(raw_vs_weighted_spear):.3f}")
    print("\nStage 3.6 validation complete.")


if __name__ == "__main__":
    main()
