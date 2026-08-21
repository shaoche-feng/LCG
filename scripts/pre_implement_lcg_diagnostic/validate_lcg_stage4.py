#! /usr/bin/env python
"""
LCG Stage 4 validation: imagined-transition exposure (WorldModelEnv.ImaginedCandidate) and
batched CRN candidate scoring infrastructure (lcg.batched_scoring / lcg.crn).

Does NOT train ActorCritic with LCG reward, does not replace WorldModelEnv's reward, does
not touch Trainer.run(). Reuses lcg.gauss_newton.compute_vjp (Stage 1), lcg.candidate_score
formula (Stage 3), and lcg.precision.historical_precision (Stage 2) unmodified. The only
non-additive-flag code change is the new opt-in `return_imagined_candidate` path in
WorldModelEnv.step(); its default (False) reproduces the exact prior behavior.

Part 4: consistency tests (batch==individual scoring, order invariance, determinism,
different-bank-set sensitivity, identical-probes-per-candidate, denoiser untouched, normal
WorldModelEnv reward/end behavior unaffected by the new flag).
Part 5: rollout-scale (B x H) profiling of sequential per-candidate CRN scoring, K=2 banks.

Usage:
    python scripts/validate_lcg_stage4.py
"""
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

from data import BatchSampler, Dataset, Episode, collate_segments_to_batch
from envs import ImaginedCandidate, WorldModelEnv, WorldModelEnvConfig
from envs.dm_control_env import make_dm_control_env
from lcg.batched_scoring import imagined_candidates_from_batch, score_candidate_with_banks, score_candidates_with_banks
from lcg.crn import make_crn_bank_set
from lcg.precision import historical_precision
from lcg.theta_s import selected_parameters
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DIAG_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage4_heldout_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)

NUM_STEPS_CONDITIONING = 4
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0
NUM_CRN_BANKS = 2  # Stage 3.7 initial integration default

# Consistency-test scale (fast) vs profiling scale (realistic, matches config/trainer.yaml
# world_model_env.horizon=15 and actor_critic.training.batch_size=32)
B_TEST, H_TEST = 8, 3
B_PROFILE, H_PROFILE = 32, 15


# --------------------------------------------------------------------------------------
# Setup
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
    return denoiser, ckpt["action_dim"]


def build_rew_end_model(device: torch.device, action_dim: int) -> RewEndModel:
    cfg = RewEndModelConfig(
        lstm_dim=512, img_channels=3, img_size=64, cond_channels=128,
        depths=[2, 2, 2, 2], channels=[32, 32, 32, 32], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim, continuous_reward=True,
    )
    model = RewEndModel(cfg).to(device)
    model.eval()
    return model


def make_wm_env_loader(dataset: Dataset, num_envs: int) -> DataLoader:
    bs = BatchSampler(dataset, 0, 1, num_envs, NUM_STEPS_CONDITIONING, sample_weights=None)
    return DataLoader(dataset=dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)


def build_world_model_env(denoiser, rew_end_model, dataset, num_envs, horizon, return_imagined_candidate):
    cfg = WorldModelEnvConfig(horizon=horizon, num_batches_to_preload=4, diffusion_sampler=SAMPLER_CFG)
    loader = make_wm_env_loader(dataset, num_envs)
    return WorldModelEnv(denoiser, rew_end_model, loader, cfg, return_imagined_candidate=return_imagined_candidate)


def run_rollout(env: WorldModelEnv, num_envs: int, horizon: int, action_dim: int, seed: int, device):
    torch.manual_seed(seed)
    np.random.seed(seed)
    obs0, _ = env.reset()
    outputs, candidates = [], []
    for _ in range(horizon):
        act = torch.rand(num_envs, action_dim, device=device) * 2 - 1
        next_obs, rew, end, trunc, info = env.step(act)
        outputs.append((next_obs.clone(), rew.clone(), end.clone(), trunc.clone()))
        if "imagined_candidate" in info:
            candidates.append(info["imagined_candidate"])
    return outputs, candidates


# --------------------------------------------------------------------------------------


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    denoiser, action_dim = load_diagnostic_denoiser(device)
    rew_end_model = build_rew_end_model(device, action_dim)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage4_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage4_heldout", 700, seed=555, include_stray=False)

    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,) and torch.all(h_D > 0)
    print(f"Frozen h_D: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, max={h_D.max().item():.4g}")

    before_state = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    # ------------------------------------------------------------------------------
    # PART 1 check: imagined candidate exposure + reward/end behavior unaffected
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART 1 CHECK: imagined-candidate exposure, additive-flag behavior preservation")
    print("=" * 88)

    env_with = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_TEST, H_TEST, True)
    outputs_with, candidates = run_rollout(env_with, B_TEST, H_TEST, action_dim, seed=123, device=device)

    env_without = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_TEST, H_TEST, False)
    outputs_without, candidates_none = run_rollout(env_without, B_TEST, H_TEST, action_dim, seed=123, device=device)

    same = all(
        torch.equal(a[0], b[0]) and torch.equal(a[1], b[1]) and torch.equal(a[2], b[2]) and torch.equal(a[3], b[3])
        for a, b in zip(outputs_with, outputs_without)
    )
    print(f"  next_obs/rew/end/trunc identical with vs without return_imagined_candidate: {same}")
    assert same
    assert len(candidates_none) == 0, "return_imagined_candidate=False must never populate info['imagined_candidate']"
    print(f"  collected {len(candidates)} ImaginedCandidate batches over {H_TEST} steps "
          f"({B_TEST} envs each -> {len(candidates) * B_TEST} candidates total).")

    c0 = candidates[0]
    print(f"  ImaginedCandidate shapes: x_obs={tuple(c0.x_obs.shape)}  x_act={tuple(c0.x_act.shape)}  "
          f"y_star={tuple(c0.y_star.shape)}")
    assert c0.x_obs.shape == (B_TEST, NUM_STEPS_CONDITIONING, 3, 64, 64)
    assert c0.y_star.shape == (B_TEST, 3, 64, 64)

    all_candidate_tuples = []
    for batch in candidates:
        all_candidate_tuples.extend(imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star))
    print(f"  flattened into {len(all_candidate_tuples)} per-candidate tuples for scoring.")

    # ------------------------------------------------------------------------------
    # PART 4: consistency tests
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"PART 4: consistency tests (K={NUM_CRN_BANKS} CRN banks, num_strata=3)")
    print("=" * 88)

    y_shape = all_candidate_tuples[0][2].shape
    banks_a = make_crn_bank_set(SIGMA_CFG, y_shape, device, num_crn_banks=NUM_CRN_BANKS, num_strata=3, seed=7000)

    batch_scores = score_candidates_with_banks(denoiser, params, h_D, banks_a, all_candidate_tuples)
    individual_scores = torch.tensor([
        score_candidate_with_banks(denoiser, params, h_D, banks_a, c) for c in all_candidate_tuples
    ])
    batch_eq_indiv = torch.equal(batch_scores, individual_scores)
    print(f"  [4a] batch scoring == individual scoring: {batch_eq_indiv}")
    assert batch_eq_indiv

    perm = torch.randperm(len(all_candidate_tuples)).tolist()
    shuffled = [all_candidate_tuples[i] for i in perm]
    shuffled_scores = score_candidates_with_banks(denoiser, params, h_D, banks_a, shuffled)
    reordered_back = torch.empty_like(shuffled_scores)
    for pos, orig_idx in enumerate(perm):
        reordered_back[orig_idx] = shuffled_scores[pos]
    order_invariant = torch.equal(reordered_back, batch_scores)
    print(f"  [4b] candidate order does not change individual scores: {order_invariant}")
    assert order_invariant

    repeat_scores = score_candidates_with_banks(denoiser, params, h_D, banks_a, all_candidate_tuples)
    deterministic = torch.equal(batch_scores, repeat_scores)
    print(f"  [4c] repeated scoring with same banks is deterministic (bit-identical): {deterministic}")
    assert deterministic

    banks_b = make_crn_bank_set(SIGMA_CFG, y_shape, device, num_crn_banks=NUM_CRN_BANKS, num_strata=3, seed=8000)
    scores_bank_b = score_candidates_with_banks(denoiser, params, h_D, banks_b, all_candidate_tuples)
    bank_changes_score = not torch.equal(batch_scores, scores_bank_b)
    rel_diff = ((batch_scores - scores_bank_b).abs() / batch_scores.abs().clamp_min(1e-12)).mean().item()
    print(f"  [4d] different CRN bank set changes the estimator: {bank_changes_score} "
          f"(mean relative difference={rel_diff:.4f})")
    assert bank_changes_score

    dup_candidates = [all_candidate_tuples[0], all_candidate_tuples[1], all_candidate_tuples[0]]
    dup_scores = score_candidates_with_banks(denoiser, params, h_D, banks_a, dup_candidates)
    identical_probes = torch.equal(dup_scores[0], dup_scores[2])
    print(f"  [4e] two duplicate candidates at different positions get identical scores "
          f"(same probes applied to every candidate): {identical_probes}")
    assert identical_probes

    for k, v in denoiser.state_dict().items():
        assert torch.equal(before_state[k], v), f"denoiser parameter {k} changed during Stage 4!"
    print("  [4f] denoiser parameters unchanged before vs after: True")

    print(f"  [4g] WorldModelEnv reward/end behavior unaffected by the new flag: {same} (checked above)")

    # ------------------------------------------------------------------------------
    # PART 5: rollout-scale profiling
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print(f"PART 5: rollout-scale profiling, B={B_PROFILE} x H={H_PROFILE} = {B_PROFILE * H_PROFILE} candidates")
    print("=" * 88)

    env_profile = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_PROFILE, H_PROFILE, True)
    t0 = time.perf_counter()
    _, profile_candidates = run_rollout(env_profile, B_PROFILE, H_PROFILE, action_dim, seed=999, device=device)
    t_rollout = time.perf_counter() - t0

    profile_tuples = []
    for batch in profile_candidates:
        profile_tuples.extend(imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star))
    print(f"  imagination rollout ({H_PROFILE} steps): {t_rollout:.2f}s -> {len(profile_tuples)} candidates "
          f"({t_rollout / len(profile_tuples) * 1000:.2f} ms/candidate for imagination alone)")

    banks_profile = make_crn_bank_set(SIGMA_CFG, y_shape, device, num_crn_banks=NUM_CRN_BANKS, num_strata=3, seed=9000)
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    t0 = time.perf_counter()
    profile_scores = score_candidates_with_banks(denoiser, params, h_D, banks_profile, profile_tuples)
    t_score = time.perf_counter() - t0
    peak_mem_mb = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None

    vjps_per_candidate = NUM_CRN_BANKS * 3
    total_vjps = vjps_per_candidate * len(profile_tuples)
    print(f"\n  LCG scoring ({vjps_per_candidate} VJPs/candidate, K={NUM_CRN_BANKS} banks x 3 strata):")
    print(f"    total scoring time: {t_score:.2f}s for {len(profile_tuples)} candidates")
    print(f"    ms/candidate: {t_score / len(profile_tuples) * 1000:.2f}")
    print(f"    total VJPs: {total_vjps}  ({t_score / total_vjps * 1000:.2f} ms/VJP)")
    if peak_mem_mb is not None:
        print(f"    peak GPU memory during scoring: {peak_mem_mb:.1f} MB")
    print(f"    scores: finite={torch.isfinite(profile_scores).all().item()}  "
          f"nonneg={(profile_scores >= 0).all().item()}  "
          f"min={profile_scores.min().item():.4g}  max={profile_scores.max().item():.4g}")
    print(f"\n  for comparison, the imagination rollout itself took {t_rollout:.2f}s; "
          f"LCG scoring of the resulting {len(profile_tuples)} candidates took {t_score:.2f}s "
          f"({t_score / t_rollout:.1f}x the rollout's own cost) with the current sequential-loop implementation.")

    for k, v in denoiser.state_dict().items():
        assert torch.equal(before_state[k], v), f"denoiser parameter {k} changed during Part 5 profiling!"
    print("\n[integrity] denoiser parameters unchanged after Part 5 profiling too: OK")

    print("\nStage 4 validation complete.")


if __name__ == "__main__":
    main()
