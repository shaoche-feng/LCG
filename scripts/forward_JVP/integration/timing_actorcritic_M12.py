#! /usr/bin/env python
"""
Stage 7 Part 10: integrated ActorCritic update timing, B=32/H=15 (480 imagined
transitions per rollout, matching the candidate-generation workload used throughout
this diagnostic effort), three conditions:

  A. baseline DIAMOND, LCG disabled
  B. unoptimized LCG (candidate_chunk_size=4, the pre-Stage-7 default), M=12
  C. optimized LCG (candidate_chunk_size=16, the Stage 7 production default), M=12

Uses the real production entry points (LCGLifecycle.refresh() + ActorCritic.
set_intrinsic_reward_fn(), exactly as Trainer.train_agent() wires them) for conditions
B/C, and ac.set_intrinsic_reward_fn(None) for condition A -- no hand-rolled scoring
path. One world-model round (one LCGLifecycle.refresh()) per condition, then several
timed ac() forward+backward updates.

Usage:
    python scripts/forward_JVP/integration/timing_actorcritic_M12.py
"""
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
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup
from envs.dm_control_env import make_dm_control_env
from lcg import LCGConfig, LCGLifecycle
from models.actor_critic import ActorCriticLossConfig
from utils import configure_opt

B_ENVS = 32
WM_HORIZON = 15
H_BACKUP_EVERY = 15  # one ac() call = one complete 15-step rollout = 480 imagined transitions
M = 12
NUM_TIMED_UPDATES = 5


def build_ac_and_env(denoiser, rew_end_model, dataset, action_dim, action_low, action_high, device):
    wm_env = setup.build_world_model_env(denoiser, rew_end_model, dataset, B_ENVS, WM_HORIZON)
    ac = setup.build_actor_critic(device, action_dim, action_low, action_high)
    loss_cfg = ActorCriticLossConfig(
        backup_every=H_BACKUP_EVERY, gamma=0.985, lambda_=0.95, weight_value_loss=1.0, weight_entropy_loss=0.001
    )
    ac.setup_training(wm_env, loss_cfg)
    opt = configure_opt(ac, lr=1e-4, weight_decay=0.0, eps=1e-8)
    return ac, opt


def time_updates(ac, opt, device, num_updates, label):
    losses = []
    times = []
    for i in range(num_updates):
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss, metrics = ac()
        opt.zero_grad()
        loss.backward()
        opt.step()
        if device.type == "cuda":
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        times.append(elapsed)
        losses.append(loss.item())
        print(f"  [{label}] update {i}: {elapsed:.3f}s  loss={loss.item():.4f}", flush=True)
    return times, losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  B_ENVS={B_ENVS} WM_HORIZON={WM_HORIZON} (480 imagined transitions/rollout)  M={M}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    rew_end_model = setup.build_rew_end_model(device, action_dim)
    train_dataset = setup.load_train_dataset()
    probe_env = make_dm_control_env(domain_name=setup.DOMAIN_NAME, task_name=setup.TASK_NAME, **setup.ENV_KWARGS)
    action_low, action_high = probe_env.action_low.tolist(), probe_env.action_high.tolist()

    results = {}

    # ------------------------------------------------------------------------------
    # A. baseline, LCG disabled
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88, flush=True)
    print("CONDITION A -- baseline DIAMOND, LCG disabled", flush=True)
    print("=" * 88, flush=True)
    ac_a, opt_a = build_ac_and_env(denoiser, rew_end_model, train_dataset, action_dim, action_low, action_high, device)
    # warm-up (compile/allocator warm-up, not timed)
    time_updates(ac_a, opt_a, device, 1, "A-warmup")
    times_a, losses_a = time_updates(ac_a, opt_a, device, NUM_TIMED_UPDATES, "A")
    results["baseline"] = dict(times=times_a, losses=losses_a)

    # ------------------------------------------------------------------------------
    # B. unoptimized LCG, candidate_chunk_size=4
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88, flush=True)
    print("CONDITION B -- unoptimized LCG (candidate_chunk_size=4), M=12", flush=True)
    print("=" * 88, flush=True)
    lcg_cfg_b = LCGConfig(
        enabled=True, h_d_batch_size=20, damping=1e-4, beta=1.0, chunk_size=4, rms_enabled=True,
        candidate_estimator="forward_jvp", candidate_sampling="simple_mc", candidate_num_mc=M,
        candidate_crn="full", candidate_chunk_size=4,
    )
    lifecycle_b = LCGLifecycle(lcg_cfg_b, setup.SIGMA_CFG, img_channels=3, img_size=64, device=device)
    ac_b, opt_b = build_ac_and_env(denoiser, rew_end_model, train_dataset, action_dim, action_low, action_high, device)
    t0 = time.time()
    lifecycle_b.refresh(denoiser, train_dataset)
    t_refresh_b = time.time() - t0
    ac_b.set_intrinsic_reward_fn(lifecycle_b.intrinsic_reward_fn)
    print(f"  refresh() time (h_D + bank build, NOT counted in per-update timing): {t_refresh_b:.2f}s", flush=True)
    time_updates(ac_b, opt_b, device, 1, "B-warmup")
    times_b, losses_b = time_updates(ac_b, opt_b, device, NUM_TIMED_UPDATES, "B")
    results["unoptimized_lcg_chunk4"] = dict(times=times_b, losses=losses_b, refresh_s=t_refresh_b)

    # ------------------------------------------------------------------------------
    # C. optimized LCG, candidate_chunk_size=16
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88, flush=True)
    print("CONDITION C -- optimized LCG (candidate_chunk_size=16), M=12", flush=True)
    print("=" * 88, flush=True)
    lcg_cfg_c = LCGConfig(
        enabled=True, h_d_batch_size=20, damping=1e-4, beta=1.0, chunk_size=4, rms_enabled=True,
        candidate_estimator="forward_jvp", candidate_sampling="simple_mc", candidate_num_mc=M,
        candidate_crn="full", candidate_chunk_size=16,
    )
    lifecycle_c = LCGLifecycle(lcg_cfg_c, setup.SIGMA_CFG, img_channels=3, img_size=64, device=device)
    ac_c, opt_c = build_ac_and_env(denoiser, rew_end_model, train_dataset, action_dim, action_low, action_high, device)
    t0 = time.time()
    lifecycle_c.refresh(denoiser, train_dataset)
    t_refresh_c = time.time() - t0
    ac_c.set_intrinsic_reward_fn(lifecycle_c.intrinsic_reward_fn)
    print(f"  refresh() time (h_D + bank build, NOT counted in per-update timing): {t_refresh_c:.2f}s", flush=True)
    time_updates(ac_c, opt_c, device, 1, "C-warmup")
    times_c, losses_c = time_updates(ac_c, opt_c, device, NUM_TIMED_UPDATES, "C")
    results["optimized_lcg_chunk16"] = dict(times=times_c, losses=losses_c, refresh_s=t_refresh_c)

    # ------------------------------------------------------------------------------
    # summary
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88, flush=True)
    print("SUMMARY", flush=True)
    print("=" * 88, flush=True)
    T_baseline = np.mean(results["baseline"]["times"])
    T_lcg_old = np.mean(results["unoptimized_lcg_chunk4"]["times"])
    T_lcg_new = np.mean(results["optimized_lcg_chunk16"]["times"])
    print(f"\n  T_baseline (mean of {NUM_TIMED_UPDATES} updates):                {T_baseline:.3f}s")
    print(f"  T_LCG_total (unoptimized, chunk=4, M=12):        {T_lcg_old:.3f}s   "
          f"overhead={T_lcg_old - T_baseline:.3f}s   ratio={T_lcg_old / T_baseline:.3f}x")
    print(f"  T_LCG_total (optimized, chunk=16, M=12):         {T_lcg_new:.3f}s   "
          f"overhead={T_lcg_new - T_baseline:.3f}s   ratio={T_lcg_new / T_baseline:.3f}x")
    print(f"\n  optimization speedup on the LCG update itself: {T_lcg_old / T_lcg_new:.3f}x")
    print(f"  optimization reduces LCG overhead from {T_lcg_old - T_baseline:.3f}s to {T_lcg_new - T_baseline:.3f}s "
          f"({100 * (1 - (T_lcg_new - T_baseline) / max(T_lcg_old - T_baseline, 1e-6)):.1f}% overhead reduction)")

    all_finite = (
        all(np.isfinite(v) for v in results["baseline"]["losses"])
        and all(np.isfinite(v) for v in results["unoptimized_lcg_chunk4"]["losses"])
        and all(np.isfinite(v) for v in results["optimized_lcg_chunk16"]["losses"])
    )
    print(f"\n  all losses finite across all 3 conditions: {all_finite}")

    print("\nIntegrated ActorCritic timing complete.", flush=True)


if __name__ == "__main__":
    main()
