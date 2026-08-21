#! /usr/bin/env python
"""
LCG Stage 3.7: high-budget reference reliability + multi-bank CRN evaluation, the final
pre-integration diagnostic before wiring LCG into ActorCritic/WorldModelEnv/Trainer.

Does NOT change the LCG objective, WorldModelEnv, ActorCritic, or Trainer. Built entirely
from lcg.gauss_newton.compute_vjp (Stage 1, unmodified), lcg.candidate_score (Stage 3,
unmodified, used as-is for the independent M=64 reference draws), and
lcg.sigma_strata.sample_sigma_stratum (unmodified). Reuses the exact checkpoint/dataset/
candidate-generation recipe from Stage 3/3.5/3.6.

Part A: is a *single* M=64 estimate trustworthy as "ground truth"? Compute >=3 independent
M=64 draws, report their pairwise agreement, and average them into a consensus reference
(rather than treating any one draw as ground truth, per instruction).

Part B: K stratified CRN banks (K in {1,2,3,4}, each bank = the normal 3 stratified
probes, so K banks = 3K VJPs/candidate), all candidates sharing all K banks, final score
averaged over banks. Several independently generated bank sets per K, each compared to the
Part-A consensus reference, plus trial-to-trial agreement and runtime.

Part C: using the identical bank probes from Part B, compare raw sum(v^2) vs LCG
sum(v^2/h_D) against the (now more reliable) consensus reference, for every K.

Usage:
    python scripts/validate_lcg_stage3_7.py
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
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage3_7_heldout_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)

NUM_STEPS_CONDITIONING = 4
NUM_CANDIDATES = 40
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0

M_REF = 64
NUM_REF_DRAWS = 3
K_LIST = [1, 2, 3, 4]
NUM_TRIALS = 6
TOPK_FRAC_A = 0.10
TOPK_FRAC_B = 0.20


# --------------------------------------------------------------------------------------
# Setup (identical recipe to Stage 3 / 3.5 / 3.6)
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
# CRN bank machinery
# --------------------------------------------------------------------------------------


def make_bank(y_shape, device, seed, num_strata=3):
    torch.manual_seed(seed)
    bank = []
    for m in range(num_strata):
        sigma = sample_sigma_stratum(SIGMA_CFG, m, num_strata, 1, device)
        eps = torch.randn(y_shape, device=device)
        xi = torch.randn(y_shape, device=device)
        bank.append((sigma, eps, xi))
    return bank


def score_candidate_with_banks(denoiser, params, h_D, banks, x_obs, x_act, y_star):
    """banks: list of K banks (each a list of 3 (sigma,eps,xi) stratified triples). Returns
    (weighted, raw), each averaged over banks (of each bank's own 3-stratum average) --
    equivalent to a uniform average over all K*3 samples. Built only from compute_vjp.
    """
    weighted_per_bank, raw_per_bank = [], []
    for bank in banks:
        v_sq_per_stratum = []
        for sigma, eps, xi in bank:
            y_sigma = (y_star + sigma.view(-1, 1, 1, 1) * eps).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, x_obs, x_act, xi=xi)
            v_sq_per_stratum.append(v * v)
        v_sq = torch.stack(v_sq_per_stratum)  # (3, d_S)
        weighted_per_bank.append((v_sq / h_D.unsqueeze(0)).sum(dim=1).mean())
        raw_per_bank.append(v_sq.sum(dim=1).mean())
    weighted = torch.stack(weighted_per_bank).mean().item()
    raw = torch.stack(raw_per_bank).mean().item()
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


# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser = load_diagnostic_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage3_7_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage3_7_heldout", 500, seed=555, include_stray=False)

    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,) and torch.all(h_D > 0)
    print(f"Frozen h_D: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, max={h_D.max().item():.4g}")

    candidates = generate_candidates(denoiser, heldout_dataset, NUM_CANDIDATES, seed=42)
    print(f"Fixed candidate pool: {len(candidates)} candidates.")
    before_state = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    # ------------------------------------------------------------------------------
    # Part A: reference reliability
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"PART A: reference reliability -- {NUM_REF_DRAWS} independent M={M_REF} draws")
    print("=" * 88)
    ref_draws = []
    for d in range(NUM_REF_DRAWS):
        t0 = time.perf_counter()
        scores = torch.tensor([
            candidate_score(denoiser, params, h_D, SIGMA_CFG, x_obs, x_act, y_star, num_strata=M_REF)
            for x_obs, x_act, y_star in candidates
        ])
        dt = time.perf_counter() - t0
        ref_draws.append(scores)
        print(f"  draw {d}: {dt:.1f}s ({dt / NUM_CANDIDATES * 1000:.1f} ms/candidate)")

    pear_ref, spear_ref, top_ref = [], [], []
    for i in range(NUM_REF_DRAWS):
        for j in range(i + 1, NUM_REF_DRAWS):
            pear_ref.append(pearson_corr(ref_draws[i], ref_draws[j]))
            spear_ref.append(spearman_corr(ref_draws[i], ref_draws[j]))
            top_ref.append(topk_overlap(ref_draws[i], ref_draws[j], TOPK_FRAC_A))
    print(f"\n  pairwise agreement among the {NUM_REF_DRAWS} independent M={M_REF} draws:")
    print(f"    mean Pearson={np.mean(pear_ref):.4f}  mean Spearman={np.mean(spear_ref):.4f}  "
          f"mean top-{TOPK_FRAC_A * 100:.0f}% overlap={np.mean(top_ref):.4f}")

    consensus_ref = torch.stack(ref_draws).mean(dim=0)
    print(f"  consensus reference = mean of the {NUM_REF_DRAWS} draws (this replaces any single draw as 'ground truth').")

    # ------------------------------------------------------------------------------
    # Part B + C: K in {1,2,3,4} CRN banks, several independently generated bank sets
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART B/C: K-bank CRN sweep vs consensus reference (+ raw-vs-weighted, Part C)")
    print("=" * 88)
    y_shape = candidates[0][2].shape
    results = {}
    trial_scores_by_K = {}

    for K in K_LIST:
        weighted_trials, raw_trials, runtimes = [], [], []
        for t in range(NUM_TRIALS):
            banks = [make_bank(y_shape, device, seed=10_000 * K + 100 * t + b, num_strata=3) for b in range(K)]
            t0 = time.perf_counter()
            weighted, raw = [], []
            for x_obs, x_act, y_star in candidates:
                w, r = score_candidate_with_banks(denoiser, params, h_D, banks, x_obs, x_act, y_star)
                weighted.append(w)
                raw.append(r)
            dt = time.perf_counter() - t0
            weighted_trials.append(torch.tensor(weighted))
            raw_trials.append(torch.tensor(raw))
            runtimes.append(dt)

        pear = [pearson_corr(w, consensus_ref) for w in weighted_trials]
        spear = [spearman_corr(w, consensus_ref) for w in weighted_trials]
        topA = [topk_overlap(w, consensus_ref, TOPK_FRAC_A) for w in weighted_trials]
        topB = [topk_overlap(w, consensus_ref, TOPK_FRAC_B) for w in weighted_trials]
        raw_pear = [pearson_corr(r, consensus_ref) for r in raw_trials]
        raw_spear = [spearman_corr(r, consensus_ref) for r in raw_trials]

        trial_corrs, trial_scorrs = [], []
        for i in range(NUM_TRIALS):
            for j in range(i + 1, NUM_TRIALS):
                trial_corrs.append(pearson_corr(weighted_trials[i], weighted_trials[j]))
                trial_scorrs.append(spearman_corr(weighted_trials[i], weighted_trials[j]))

        ms_per_candidate = np.mean(runtimes) / NUM_CANDIDATES * 1000
        results[K] = dict(
            pearson=np.mean(pear), pearson_std=np.std(pear),
            spearman=np.mean(spear), spearman_std=np.std(spear),
            topA=np.mean(topA), topB=np.mean(topB),
            trial_pearson=np.mean(trial_corrs), trial_spearman=np.mean(trial_scorrs),
            raw_pearson=np.mean(raw_pear), raw_spearman=np.mean(raw_spear),
            ms_per_candidate=ms_per_candidate, vjps_per_candidate=3 * K,
        )
        trial_scores_by_K[K] = (weighted_trials, raw_trials)

        r = results[K]
        print(f"\n  K={K} ({3 * K} VJPs/candidate), {NUM_TRIALS} independently generated bank sets:")
        print(f"    LCG (h_D-weighted) vs consensus ref: Pearson={r['pearson']:.4f} (std={r['pearson_std']:.4f})  "
              f"Spearman={r['spearman']:.4f} (std={r['spearman_std']:.4f})")
        print(f"    top-{TOPK_FRAC_A * 100:.0f}% overlap={r['topA']:.4f}   top-{TOPK_FRAC_B * 100:.0f}% overlap={r['topB']:.4f}")
        print(f"    trial-to-trial agreement: Pearson={r['trial_pearson']:.4f}  Spearman={r['trial_spearman']:.4f}")
        print(f"    raw sum(v^2) vs consensus ref (same probes): Pearson={r['raw_pearson']:.4f}  "
              f"Spearman={r['raw_spearman']:.4f}")
        print(f"    runtime: {ms_per_candidate:.2f} ms/candidate")

    for k, v in denoiser.state_dict().items():
        assert torch.equal(before_state[k], v), f"denoiser parameter {k} changed during Stage 3.7!"
    print("\n[integrity] denoiser parameters unchanged before vs after Stage 3.7: OK")

    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("SUMMARY: cost/ranking tradeoff")
    print("=" * 88)
    print(f"  {'K':>3} {'VJPs/cand':>10} {'ms/cand':>9} {'Spearman':>9} {'top10%':>8} {'top20%':>8} "
          f"{'trial-Spearman':>15} {'raw-Spearman':>13}")
    prev_spear = None
    for K in K_LIST:
        r = results[K]
        delta = "" if prev_spear is None else f"  (Delta={r['spearman'] - prev_spear:+.3f})"
        print(f"  {K:>3} {r['vjps_per_candidate']:>10} {r['ms_per_candidate']:>9.2f} {r['spearman']:>9.4f}{delta} "
              f"{r['topA']:>8.4f} {r['topB']:>8.4f} {r['trial_spearman']:>15.4f} {r['raw_spearman']:>13.4f}")
        prev_spear = r["spearman"]

    print("\nStage 3.7 validation complete.")


if __name__ == "__main__":
    main()
