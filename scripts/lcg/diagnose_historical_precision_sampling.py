#! /usr/bin/env python
"""
Historical precision diagnostic: does h_D need 3-stratum sigma sampling, or does simple
IID Monte Carlo suffice, at equal VJP budgets? Diagnostic only -- does NOT modify
production historical_precision() or the default estimator. Calls the same production
primitives (apply_noise_from_samples, compute_vjp, sample_sigma_stratum, the corrected
sample_uniform_historical_transitions, the isolated LCG-local RNG generators) rather than
re-deriving any math.

Staged execution (per the task spec) via --until-stage, so compute is benchmarked before
committing to the full run:
  A: correctness (tiny budgets, sanity checks)
  B: timing projection for the full design
  C: high-budget reference agreement (Simple-MC M=48 vs stratified K=16=48 VJPs)
  D: low-budget comparison, K in {1,2,4} vs M in {3,6,12}, 8 seeds
  E: downstream candidate-score comparison (production Forward-JVP scorer, fixed CRN bank)

Usage:
    python scripts/lcg/diagnose_historical_precision_sampling.py --until-stage B
    python scripts/lcg/diagnose_historical_precision_sampling.py --until-stage E
"""
import argparse
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
from lcg.forward_jvp import (
    frozen_named_parameters,
    make_forward_jvp_simple_mc_bank,
    score_one_jvp_bank,
    selected_named_parameters,
)
from lcg.gauss_newton import compute_vjp
from lcg.precision import load_transition, sample_uniform_historical_transitions
from lcg.sigma_strata import sample_sigma_stratum
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
from models.diffusion.denoiser import apply_noise_from_samples
from models.diffusion.inner_model import InnerModelConfig

# --------------------------------------------------------------------------------------
# Fixed configuration (matches the checkpoint's own training recipe exactly)
# --------------------------------------------------------------------------------------
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DENOISER_PATH = SCRATCH_BASE / "lcg_diag_denoiser_converged.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
CANDIDATES_PATH = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "checkpoint" / "frozen_candidates_480.pt"
STRAY_EPISODE_PATH = _REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46" / "dataset" / "train" / "000" / "00" / "0" / "0.pt"

OUT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "historical_precision_sampling"
OUT_DIR.mkdir(parents=True, exist_ok=True)

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
NUM_STEPS_CONDITIONING = 4
TRAIN_TARGET_STEPS = 1500
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
DAMPING = 1e-4
BETA = 1.0

B = 160
NUM_SEEDS = 8
LOW_K = [1, 2, 4]
LOW_M = [3, 6, 12]
MAX_K, MAX_M = max(LOW_K), max(LOW_M)  # 4, 12 -- what we actually need to compute per low-budget seed
REF_M = 48
REF_K = 16
REF_SEED_BASE = 900000  # far from the 8 evaluation seeds (0..7), independent draws

CANDIDATE_SUBSET_SIZE = 160
CANDIDATE_M = 12  # production default
CANDIDATE_CHUNK_SIZE = 16  # production default
CANDIDATE_BANK_SEED = 424242


# --------------------------------------------------------------------------------------
# Dataset reconstruction (the original scratchpad dataset was wiped by OS Temp cleanup;
# same recipe as the deleted scripts/pre_implement_lcg_diagnostic/
# train_lcg_diagnostic_checkpoint.py, seed=0, so this reproduces the same replay the
# checkpoint was actually trained on)
# --------------------------------------------------------------------------------------


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
    if STRAY_EPISODE_PATH.is_file():
        dataset.add_episode(Episode.load(STRAY_EPISODE_PATH))
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


# --------------------------------------------------------------------------------------
# Per-transition VJP-squared accumulators (raw per-sample cache, offline-cumulative-
# reconstruction pattern used throughout this project: compute the MAX budget once,
# derive every smaller budget by averaging a prefix)
# --------------------------------------------------------------------------------------


def raw_stratified_vjp_sq(denoiser, params, obs, act, y, num_rounds, gen):
    """num_rounds independent complete 3-stratum sets -> returns a (num_rounds*3, d_S)
    tensor of raw v^2 rows (one per VJP), in round-major order so prefixes of length 3*K
    give exactly K rounds."""
    rows = []
    for _ in range(num_rounds):
        for s in range(3):
            sigma = sample_sigma_stratum(SIGMA_CFG, s, 3, 1, y.device, generator=gen)
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=gen)
            eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=y.device, generator=gen)
            y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, generator=gen)
            rows.append((v * v).detach())
    return torch.stack(rows)  # (num_rounds*3, d_S)


def raw_simple_mc_vjp_sq(denoiser, params, obs, act, y, num_samples, gen):
    """num_samples IID sigma draws from the FULL p_train(sigma) distribution (stratum_idx=0,
    num_strata=1 collapses to the unstratified distribution -- the same construction
    validated throughout the forward-JVP simple-MC diagnostics, reused here for the
    backward/historical-precision side)."""
    rows = []
    for _ in range(num_samples):
        sigma = sample_sigma_stratum(SIGMA_CFG, 0, 1, 1, y.device, generator=gen)
        eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=gen)
        eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=y.device, generator=gen)
        y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
        v, _ = compute_vjp(denoiser, params, y_sigma, sigma, obs, act, generator=gen)
        rows.append((v * v).detach())
    return torch.stack(rows)  # (num_samples, d_S)


def h_D_from_g_i_list(g_i_list, N, B_, device):
    d_S = g_i_list[0].numel()
    h = DAMPING * torch.ones(d_S, device=device)
    scale = BETA * (N / B_)
    for g_i in g_i_list:
        h = h + scale * g_i
    return h


# --------------------------------------------------------------------------------------
# Comparison metrics
# --------------------------------------------------------------------------------------


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
    rel_l2 = ((h_test - h_ref).norm() / h_ref.norm()).item()
    norm_ratio = (h_test.norm() / h_ref.norm()).item()
    log_pear = pearson(h_test.log(), h_ref.log())
    log_spear = spearman(h_test.log(), h_ref.log())
    inv_sqrt_test, inv_sqrt_ref = h_test.rsqrt(), h_ref.rsqrt()
    inv_pear = pearson(inv_sqrt_test, inv_sqrt_ref)
    inv_spear = spearman(inv_sqrt_test, inv_sqrt_ref)
    inv_rel_l2 = ((inv_sqrt_test - inv_sqrt_ref).norm() / inv_sqrt_ref.norm()).item()
    return dict(
        raw_pearson=pearson(h_test, h_ref), raw_spearman=spearman(h_test, h_ref),
        raw_rel_l2=rel_l2, raw_norm_ratio=norm_ratio,
        log_pearson=log_pear, log_spearman=log_spear,
        inv_sqrt_pearson=inv_pear, inv_sqrt_spearman=inv_spear, inv_sqrt_rel_l2=inv_rel_l2,
    )


# --------------------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--until-stage", choices=["A", "B", "C", "D", "E"], default="E")
    args = parser.parse_args()
    stage_order = ["A", "B", "C", "D", "E"]
    run_up_to = stage_order.index(args.until_stage)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim, ckpt_step = load_denoiser(device)
    denoiser_sha256 = sha256_of(DENOISER_PATH)
    print(f"checkpoint: {DENOISER_PATH} (total_step={ckpt_step}) sha256={denoiser_sha256}", flush=True)

    dataset = rebuild_or_load_train_dataset()
    N = dataset.num_steps
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)
    print(f"N={N}  d_S={d_S}  B={B}", flush=True)

    git_commit = subprocess.run(["git", "rev-parse", "HEAD"], cwd=_REPO_ROOT, capture_output=True, text=True).stdout.strip()

    manifest = dict(
        git_commit=git_commit, checkpoint_path=str(DENOISER_PATH), checkpoint_sha256=denoiser_sha256,
        checkpoint_total_step=ckpt_step, dataset_path=str(TRAIN_DATASET_DIR), dataset_N=N,
        theta_s_dim=d_S, sigma_cfg=dict(loc=SIGMA_CFG.loc, scale=SIGMA_CFG.scale, sigma_min=SIGMA_CFG.sigma_min, sigma_max=SIGMA_CFG.sigma_max),
        sigma_offset_noise=denoiser.cfg.sigma_offset_noise, beta=BETA, damping=DAMPING, B=B,
        num_seeds=NUM_SEEDS, low_K=LOW_K, low_M=LOW_M, ref_M=REF_M, ref_K=REF_K,
        candidate_subset_size=CANDIDATE_SUBSET_SIZE, candidate_M=CANDIDATE_M,
        candidate_chunk_size=CANDIDATE_CHUNK_SIZE, device=str(device),
    )

    # ------------------------------------------------------------------------------
    # Fixed historical transition subset (SAME for every estimator/comparison)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nSTAGE: fixed transition subset\n" + "=" * 88, flush=True)
    segment_ids = sample_uniform_historical_transitions(dataset, B, NUM_STEPS_CONDITIONING, seed=2024)
    transitions = [load_transition(dataset, sid, NUM_STEPS_CONDITIONING, device) for sid in segment_ids]
    transition_hash = hashlib.sha256(
        json.dumps(sorted([(s.episode_id, s.start, s.stop) for s in segment_ids])).encode()
    ).hexdigest()
    manifest["transition_subset_sha256"] = transition_hash
    print(f"fixed B={B} transitions, subset hash={transition_hash[:16]}...", flush=True)

    # ------------------------------------------------------------------------------
    # STAGE A: correctness
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nSTAGE A -- correctness (tiny budgets)\n" + "=" * 88, flush=True)
    gen_a = torch.Generator(device=device)
    gen_a.manual_seed(1)
    obs, act, y = transitions[0]
    strat_rows = raw_stratified_vjp_sq(denoiser, params, obs, act, y, num_rounds=1, gen=gen_a)
    simple_rows = raw_simple_mc_vjp_sq(denoiser, params, obs, act, y, num_samples=3, gen=gen_a)
    g_strat = strat_rows.mean(dim=0)
    g_simple = simple_rows.mean(dim=0)
    h_strat_1 = h_D_from_g_i_list([g_strat], N, B, device)
    h_simple_1 = h_D_from_g_i_list([g_simple], N, B, device)
    print(f"  stratified K=1: shape={tuple(h_strat_1.shape)} finite={torch.isfinite(h_strat_1).all().item()} "
          f"positive={(h_strat_1>0).all().item()} min={h_strat_1.min().item():.4g}")
    print(f"  simple M=3:     shape={tuple(h_simple_1.shape)} finite={torch.isfinite(h_simple_1).all().item()} "
          f"positive={(h_simple_1>0).all().item()} min={h_simple_1.min().item():.4g}")
    # determinism: same seed -> same result
    gen_a2 = torch.Generator(device=device)
    gen_a2.manual_seed(1)
    strat_rows_2 = raw_stratified_vjp_sq(denoiser, params, obs, act, y, num_rounds=1, gen=gen_a2)
    same_seed_ok = torch.equal(strat_rows, strat_rows_2)
    print(f"  same-seed determinism: {same_seed_ok}")
    print("STAGE A complete.", flush=True)
    if run_up_to == 0:
        return

    # ------------------------------------------------------------------------------
    # STAGE B: timing projection
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nSTAGE B -- timing projection\n" + "=" * 88, flush=True)
    torch.cuda.synchronize() if device.type == "cuda" else None
    t0 = time.perf_counter()
    NUM_TIMING_VJPS = 20
    gen_b = torch.Generator(device=device)
    gen_b.manual_seed(2)
    for _ in range(NUM_TIMING_VJPS):
        sigma = sample_sigma_stratum(SIGMA_CFG, 0, 1, 1, device, generator=gen_b)
        eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=gen_b)
        eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=y.device, generator=gen_b)
        y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
        compute_vjp(denoiser, params, y_sigma, sigma, obs, act, generator=gen_b)
    if device.type == "cuda":
        torch.cuda.synchronize()
    per_vjp_s = (time.perf_counter() - t0) / NUM_TIMING_VJPS
    print(f"  measured: {per_vjp_s * 1000:.1f} ms/VJP (mean of {NUM_TIMING_VJPS} calls)", flush=True)

    low_budget_vjps_per_transition = 3 * MAX_K + MAX_M  # nested reconstruction: compute max once
    low_budget_total_vjps = low_budget_vjps_per_transition * B * NUM_SEEDS
    ref_vjps_per_transition = 3 * REF_K + REF_M
    ref_total_vjps = ref_vjps_per_transition * B
    total_vjps = low_budget_total_vjps + ref_total_vjps
    projected_s = total_vjps * per_vjp_s

    print(f"  low-budget: {low_budget_vjps_per_transition} VJPs/transition/seed x B={B} x {NUM_SEEDS} seeds "
          f"= {low_budget_total_vjps} VJPs")
    print(f"  reference:  {ref_vjps_per_transition} VJPs/transition x B={B} = {ref_total_vjps} VJPs")
    print(f"  TOTAL: {total_vjps} VJPs, projected historical-precision-side runtime: "
          f"{projected_s:.1f}s ({projected_s/60:.1f} min)", flush=True)
    manifest["measured_ms_per_vjp"] = per_vjp_s * 1000
    manifest["projected_total_vjps"] = total_vjps
    manifest["projected_runtime_s"] = projected_s
    print("STAGE B complete.", flush=True)
    if run_up_to == 1:
        with open(OUT_DIR / "manifest_partial_AB.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        return

    # ------------------------------------------------------------------------------
    # STAGE C: high-budget reference agreement (independent draws from the low-budget
    # evaluation streams -- REF_SEED_BASE is far from the 8 evaluation seeds)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nSTAGE C -- high-budget reference agreement\n" + "=" * 88, flush=True)
    t_stage_c = time.perf_counter()
    g_simple_ref_list, g_strat_ref_list = [], []
    for i, (obs_i, act_i, y_i) in enumerate(transitions):
        gen_simple = torch.Generator(device=device)
        gen_simple.manual_seed(REF_SEED_BASE + i)
        rows_simple = raw_simple_mc_vjp_sq(denoiser, params, obs_i, act_i, y_i, REF_M, gen_simple)
        g_simple_ref_list.append(rows_simple.mean(dim=0))

        gen_strat = torch.Generator(device=device)
        gen_strat.manual_seed(REF_SEED_BASE + 500_000 + i)
        rows_strat = raw_stratified_vjp_sq(denoiser, params, obs_i, act_i, y_i, REF_K, gen_strat)
        g_strat_ref_list.append(rows_strat.mean(dim=0))
        if (i + 1) % 40 == 0:
            print(f"  reference progress: {i + 1}/{B} transitions, elapsed={time.perf_counter() - t_stage_c:.1f}s", flush=True)

    h_D_ref_simple = h_D_from_g_i_list(g_simple_ref_list, N, B, device)
    h_D_ref_strat = h_D_from_g_i_list(g_strat_ref_list, N, B, device)
    ref_agreement = h_D_comparison(h_D_ref_simple, h_D_ref_strat)
    print(f"\n  Reference A (simple MC, M={REF_M}) vs Reference B (stratified, K={REF_K}={3*REF_K} VJPs):")
    for k, v in ref_agreement.items():
        print(f"    {k}: {v:.4f}")
    references_agree = ref_agreement["raw_pearson"] > 0.9 and ref_agreement["inv_sqrt_pearson"] > 0.9
    print(f"\n  references agree strongly (raw & inv_sqrt Pearson > 0.9): {references_agree}")
    if not references_agree:
        print("  STOPPING per instructions: references do not agree strongly -- investigate before proceeding.")
        torch.save(dict(h_D_ref_simple=h_D_ref_simple.cpu(), h_D_ref_strat=h_D_ref_strat.cpu(), agreement=ref_agreement),
                   OUT_DIR / "reference_disagreement_DEBUG.pt")
        with open(OUT_DIR / "manifest_partial_ABC.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        sys.exit(1)

    h_D_reference = (h_D_ref_simple + h_D_ref_strat) / 2  # both agree strongly -> average for lowest-noise reference
    torch.save(dict(h_D_ref_simple=h_D_ref_simple.cpu(), h_D_ref_strat=h_D_ref_strat.cpu(),
                     h_D_reference=h_D_reference.cpu(), agreement=ref_agreement),
               OUT_DIR / "reference_h_D.pt")
    manifest["stage_c_reference_agreement"] = ref_agreement
    manifest["stage_c_runtime_s"] = time.perf_counter() - t_stage_c
    print(f"  Stage C runtime: {manifest['stage_c_runtime_s']:.1f}s", flush=True)
    print("STAGE C complete.", flush=True)
    if run_up_to == 2:
        with open(OUT_DIR / "manifest_partial_ABC.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        return

    # ------------------------------------------------------------------------------
    # STAGE D: low-budget comparison, K in {1,2,4} vs M in {3,6,12}, NUM_SEEDS seeds,
    # nested reconstruction (compute MAX_K/MAX_M once per transition per seed, derive
    # smaller budgets by averaging a prefix)
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + f"\nSTAGE D -- low-budget comparison, {NUM_SEEDS} seeds\n" + "=" * 88, flush=True)
    t_stage_d = time.perf_counter()
    h_D_by_condition_seed = {("strat", K): [] for K in LOW_K}
    h_D_by_condition_seed.update({("simple", M): [] for M in LOW_M})

    for seed_idx in range(NUM_SEEDS):
        g_strat_prefix = {K: [] for K in LOW_K}
        g_simple_prefix = {M: [] for M in LOW_M}
        for i, (obs_i, act_i, y_i) in enumerate(transitions):
            gen_strat = torch.Generator(device=device)
            gen_strat.manual_seed(seed_idx * 200_000 + i)
            rows_strat = raw_stratified_vjp_sq(denoiser, params, obs_i, act_i, y_i, MAX_K, gen_strat)  # (3*MAX_K, d_S)
            for K in LOW_K:
                g_strat_prefix[K].append(rows_strat[: 3 * K].mean(dim=0))

            gen_simple = torch.Generator(device=device)
            gen_simple.manual_seed(seed_idx * 200_000 + 100_000 + i)
            rows_simple = raw_simple_mc_vjp_sq(denoiser, params, obs_i, act_i, y_i, MAX_M, gen_simple)  # (MAX_M, d_S)
            for M in LOW_M:
                g_simple_prefix[M].append(rows_simple[:M].mean(dim=0))

        for K in LOW_K:
            h_D_by_condition_seed[("strat", K)].append(h_D_from_g_i_list(g_strat_prefix[K], N, B, device))
        for M in LOW_M:
            h_D_by_condition_seed[("simple", M)].append(h_D_from_g_i_list(g_simple_prefix[M], N, B, device))

        print(f"  seed {seed_idx} done, elapsed={time.perf_counter() - t_stage_d:.1f}s", flush=True)

    manifest["stage_d_runtime_s"] = time.perf_counter() - t_stage_d
    print(f"\n  Stage D runtime: {manifest['stage_d_runtime_s']:.1f}s "
          f"({manifest['stage_d_runtime_s'] / (NUM_SEEDS * B * (3 * MAX_K + MAX_M)) * 1000:.1f} ms/VJP effective)")

    print(f"\n  h_D agreement vs reference (mean +/- std over {NUM_SEEDS} seeds):")
    print(f"  {'condition':>12} {'raw_pearson':>12} {'raw_spearman':>13} {'raw_rel_l2':>11} "
          f"{'log_pearson':>12} {'inv_sqrt_pearson':>17} {'inv_sqrt_spearman':>18} {'inv_sqrt_rel_l2':>16}")
    h_D_metrics_summary = {}
    for cond in [("strat", K) for K in LOW_K] + [("simple", M) for M in LOW_M]:
        metrics_list = [h_D_comparison(h, h_D_reference) for h in h_D_by_condition_seed[cond]]
        summary = {k: (np.mean([m[k] for m in metrics_list]), np.std([m[k] for m in metrics_list])) for k in metrics_list[0]}
        h_D_metrics_summary[cond] = summary
        label = f"{cond[0]}_{cond[1]}"
        print(f"  {label:>12} {summary['raw_pearson'][0]:>7.4f}+-{summary['raw_pearson'][1]:<4.4f} "
              f"{summary['raw_spearman'][0]:>8.4f}+-{summary['raw_spearman'][1]:<4.4f} "
              f"{summary['raw_rel_l2'][0]:>6.4f}+-{summary['raw_rel_l2'][1]:<4.4f} "
              f"{summary['log_pearson'][0]:>7.4f}+-{summary['log_pearson'][1]:<4.4f} "
              f"{summary['inv_sqrt_pearson'][0]:>12.4f}+-{summary['inv_sqrt_pearson'][1]:<4.4f} "
              f"{summary['inv_sqrt_spearman'][0]:>13.4f}+-{summary['inv_sqrt_spearman'][1]:<4.4f} "
              f"{summary['inv_sqrt_rel_l2'][0]:>11.4f}+-{summary['inv_sqrt_rel_l2'][1]:<4.4f}")

    torch.save({f"{c[0]}_{c[1]}": [h.cpu() for h in hs] for c, hs in h_D_by_condition_seed.items()},
               OUT_DIR / "low_budget_h_D_by_seed.pt")
    print("STAGE D complete.", flush=True)
    if run_up_to == 3:
        with open(OUT_DIR / "manifest_partial_ABCD.json", "w") as f:
            json.dump(manifest, f, indent=2, default=str)
        return

    # ------------------------------------------------------------------------------
    # STAGE E: downstream candidate-score comparison. ONE fixed candidate CRN bank
    # shared across every h_D variant (only h_D changes) -- production Forward-JVP
    # scorer, unmodified.
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88 + "\nSTAGE E -- downstream candidate-score comparison\n" + "=" * 88, flush=True)
    t_stage_e = time.perf_counter()
    candidates_all = torch.load(CANDIDATES_PATH, map_location=device, weights_only=True)
    candidates = [(c[0].to(device), c[1].to(device), c[2].to(device)) for c in candidates_all[:CANDIDATE_SUBSET_SIZE]]
    candidates_hash = hashlib.sha256(str(list(range(CANDIDATE_SUBSET_SIZE))).encode()).hexdigest()
    manifest["candidate_subset_sha256"] = candidates_hash
    print(f"  using {len(candidates)}/{len(candidates_all)} candidates (first {CANDIDATE_SUBSET_SIZE}, deterministic)")

    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    bank = make_forward_jvp_simple_mc_bank(
        SIGMA_CFG, torch.Size([1, 3, 64, 64]), d_S, device, num_samples=CANDIDATE_M, seed=CANDIDATE_BANK_SEED
    )
    print(f"  fixed candidate CRN bank built once (M={CANDIDATE_M}, seed={CANDIDATE_BANK_SEED}), reused for every h_D")

    def score_with(h_D_variant):
        return score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D_variant,
                                   h_D_variant.rsqrt(), bank, candidates, CANDIDATE_CHUNK_SIZE)

    scores_reference = score_with(h_D_reference)
    print(f"  reference scores: finite={torch.isfinite(scores_reference).all().item()} "
          f"nonneg={(scores_reference>=0).all().item()} mean={scores_reference.mean().item():.3f}")

    def candidate_metrics(scores_test, scores_ref):
        return dict(
            spearman=spearman(scores_test, scores_ref), pearson=pearson(scores_test, scores_ref),
            top10=top_q_overlap(scores_test, scores_ref, 0.10), top20=top_q_overlap(scores_test, scores_ref, 0.20),
            rms_ratio=(scores_test.pow(2).mean().sqrt() / scores_ref.pow(2).mean().sqrt()).item(),
        )

    print(f"\n  candidate-score agreement vs reference-h_D scores (mean +/- std over {NUM_SEEDS} seeds):")
    print(f"  {'condition':>12} {'spearman':>16} {'pearson':>16} {'top10':>16} {'top20':>16} {'rms_ratio':>16}")
    candidate_metrics_summary = {}
    for cond in [("strat", K) for K in LOW_K] + [("simple", M) for M in LOW_M]:
        per_seed_metrics = []
        for h in h_D_by_condition_seed[cond]:
            scores_test = score_with(h)
            per_seed_metrics.append(candidate_metrics(scores_test, scores_reference))
        summary = {k: (np.mean([m[k] for m in per_seed_metrics]), np.std([m[k] for m in per_seed_metrics])) for k in per_seed_metrics[0]}
        candidate_metrics_summary[cond] = summary
        label = f"{cond[0]}_{cond[1]}"
        print(f"  {label:>12} " + " ".join(f"{summary[k][0]:>7.4f}+-{summary[k][1]:<6.4f}" for k in ["spearman", "pearson", "top10", "top20", "rms_ratio"]))

    manifest["stage_e_runtime_s"] = time.perf_counter() - t_stage_e
    print(f"\n  Stage E runtime: {manifest['stage_e_runtime_s']:.1f}s", flush=True)
    if device.type == "cuda":
        manifest["peak_cuda_memory_mb"] = torch.cuda.max_memory_allocated(device) / 1e6
        print(f"  peak CUDA memory: {manifest['peak_cuda_memory_mb']:.1f}MB")

    with open(OUT_DIR / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2, default=str)

    # human-readable summary of the metrics dicts (JSON can't hold torch/np scalars cleanly)
    results_summary = dict(
        reference_agreement=ref_agreement,
        h_D_metrics_by_condition={f"{c[0]}_{c[1]}": {k: list(v) for k, v in s.items()} for c, s in h_D_metrics_summary.items()},
        candidate_metrics_by_condition={f"{c[0]}_{c[1]}": {k: list(v) for k, v in s.items()} for c, s in candidate_metrics_summary.items()},
    )
    with open(OUT_DIR / "results_summary.json", "w") as f:
        json.dump(results_summary, f, indent=2, default=str)

    print("STAGE E complete.", flush=True)
    print("\nDiagnostic complete.", flush=True)


if __name__ == "__main__":
    main()
