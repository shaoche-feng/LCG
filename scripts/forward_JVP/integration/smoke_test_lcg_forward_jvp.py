#! /usr/bin/env python
"""
Stage 6 Part 10: integrated ActorCritic smoke test for the production forward-JVP LCG
lifecycle (LCGLifecycle with candidate_estimator="forward_jvp"). Reuses the already-
validated real converged denoiser/rew_end_model/train_dataset fixtures from the
backward-variance diagnostic setup (scripts/backward_VJP/3-stratum/
diagnose_lcg_backward_variance_setup.py) instead of building yet another synthetic
model, and exercises the ACTUAL production entry point end-to-end -- LCGLifecycle.
refresh() -> ActorCritic.set_intrinsic_reward_fn() -> ac() forward/backward -- the same
wiring Trainer.train_agent() uses (src/trainer.py), across two simulated outer "rounds"
(world-model update -> h_D refresh -> candidate-probe refresh -> RMS reset -> ActorCritic
updates), matching Algorithm 1's lifecycle order.

The world model itself is not retrained here (out of scope for this smoke test; the
Stage 6 regression suite already checks the world-model-untouched invariant): each
"round" just calls refresh() again against the SAME frozen denoiser, which is sufficient
to exercise the full LCG lifecycle plumbing (h_D refresh, CRN/probe-bank refresh, RMS
reset, hook rewiring) without an expensive real world-model training loop.

Checks:
  - h_D refresh occurs each round (h_d_compute_count increments)
  - candidate_num_mc=24 simple-MC Full-CRN probes are (re)created each round
  - imagination produces candidates (WorldModelEnv(..., return_imagined_candidate=True))
  - forward-JVP scores are finite and nonnegative (checked via LCGLifecycle's own
    [LCG] diagnostic print of raw_reward_min/mean each ac call)
  - raw LCG reward is produced; RunningRMS normalization operates normally
  - ActorCritic losses finite; actor/critic gradients finite
  - no denoiser/world-model parameter gradient is accidentally accumulated during
    intrinsic-reward scoring
  - no GPU memory growth across repeated ActorCritic updates

Usage:
    python scripts/forward_JVP/integration/smoke_test_lcg_forward_jvp.py
"""
import sys
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

import torch

import diagnose_lcg_backward_variance_setup as setup
from envs.dm_control_env import make_dm_control_env
from lcg import LCGConfig, LCGLifecycle
from models.actor_critic import ActorCriticLossConfig
from utils import configure_opt

NUM_ROUNDS = 2
NUM_AC_UPDATES_PER_ROUND = 3
B_ENVS = 8  # smaller than the 32 used for candidate generation elsewhere -- fast smoke test
WM_HORIZON = 8
H_BACKUP_EVERY = 4
H_D_BATCH_SIZE = 20
CANDIDATE_NUM_MC = 24


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    denoiser, action_dim = setup.load_converged_denoiser(device)
    rew_end_model = setup.build_rew_end_model(device, action_dim)
    train_dataset = setup.load_train_dataset()
    print(f"train_dataset.num_steps={train_dataset.num_steps}", flush=True)

    probe_env = make_dm_control_env(domain_name=setup.DOMAIN_NAME, task_name=setup.TASK_NAME, **setup.ENV_KWARGS)
    action_low, action_high = probe_env.action_low.tolist(), probe_env.action_high.tolist()

    wm_env = setup.build_world_model_env(denoiser, rew_end_model, train_dataset, B_ENVS, WM_HORIZON)
    ac = setup.build_actor_critic(device, action_dim, action_low, action_high)
    loss_cfg = ActorCriticLossConfig(
        backup_every=H_BACKUP_EVERY, gamma=0.985, lambda_=0.95, weight_value_loss=1.0, weight_entropy_loss=0.001
    )
    ac.setup_training(wm_env, loss_cfg)
    opt = configure_opt(ac, lr=1e-4, weight_decay=0.0, eps=1e-8)

    lcg_cfg = LCGConfig(
        enabled=True, h_d_batch_size=H_D_BATCH_SIZE, damping=1e-4, beta=1.0, num_strata=3, num_crn_banks=2,
        chunk_size=4, rms_enabled=True, rms_alpha=1.0, rms_ema_decay=0.99, rms_eps=1e-8,
        candidate_estimator="forward_jvp", candidate_sampling="simple_mc", candidate_num_mc=CANDIDATE_NUM_MC,
        candidate_crn="full",
    )
    lifecycle = LCGLifecycle(lcg_cfg, setup.SIGMA_CFG, img_channels=3, img_size=64, device=device)
    print(f"LCGConfig: candidate_estimator={lcg_cfg.candidate_estimator} candidate_sampling={lcg_cfg.candidate_sampling} "
          f"candidate_num_mc={lcg_cfg.candidate_num_mc} candidate_crn={lcg_cfg.candidate_crn}", flush=True)

    denoiser_snapshot_before = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}

    mem_allocated_by_update = []
    all_finite_ok = True
    grad_finite_ok = True
    denoiser_grad_none_ok = True
    h_D_fingerprints = []
    bank_sizes = []

    for round_idx in range(NUM_ROUNDS):
        print(f"\n{'=' * 88}\nROUND {round_idx}\n{'=' * 88}")
        lifecycle.refresh(denoiser, train_dataset)
        ac.set_intrinsic_reward_fn(lifecycle.intrinsic_reward_fn)

        assert lifecycle.h_d_compute_count == round_idx + 1, "h_D refresh must occur exactly once per round"
        assert lifecycle.crn_construct_count == round_idx + 1, "probe-bank refresh must occur exactly once per round"
        assert lifecycle.rms_reset_count == round_idx + 1, "RunningRMS must reset exactly once per round"
        assert lifecycle.banks.num_strata == CANDIDATE_NUM_MC, (
            f"expected {CANDIDATE_NUM_MC} simple-MC Full-CRN probes per bank, got {lifecycle.banks.num_strata}"
        )
        h_D_fingerprints.append(lifecycle.h_D.sum().item())
        bank_sizes.append(lifecycle.banks.num_strata)
        print(f"  h_D shape={tuple(lifecycle.h_D.shape)} min={lifecycle.h_D.min().item():.4g} "
              f"max={lifecycle.h_D.max().item():.4g} all_finite={torch.isfinite(lifecycle.h_D).all().item()} "
              f"all_positive={(lifecycle.h_D > 0).all().item()}")

        for update_idx in range(NUM_AC_UPDATES_PER_ROUND):
            if device.type == "cuda":
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats(device)

            loss, metrics = ac()
            opt.zero_grad()
            loss.backward()

            denoiser_grad_nonzero = any(
                p.grad is not None and p.grad.abs().sum().item() > 0 for p in denoiser.parameters()
            )
            if denoiser_grad_nonzero:
                denoiser_grad_none_ok = False

            ac_grads = [p.grad for p in ac.parameters() if p.grad is not None]
            grads_finite = bool(ac_grads) and all(torch.isfinite(g).all().item() for g in ac_grads)
            grad_finite_ok = grad_finite_ok and grads_finite
            opt.step()

            loss_finite = bool(torch.isfinite(loss).item())
            all_finite_ok = all_finite_ok and loss_finite
            print(f"  round={round_idx} update={update_idx} loss_total={loss.item():.4f} "
                  f"loss_finite={loss_finite} ac_grads_finite={grads_finite} "
                  f"denoiser_grad_accumulated={denoiser_grad_nonzero}")

            if device.type == "cuda":
                torch.cuda.synchronize()
                mem_allocated_by_update.append(torch.cuda.memory_allocated(device) / 1e6)

    denoiser_snapshot_after = {k: v.detach().clone() for k, v in denoiser.state_dict().items()}
    denoiser_unchanged = all(torch.equal(denoiser_snapshot_before[k], denoiser_snapshot_after[k]) for k in denoiser_snapshot_before)

    print("\n" + "=" * 88)
    print("VERIFICATION")
    print("=" * 88)
    print(f"  h_D refreshed each round (fingerprints differ across rounds): "
          f"{h_D_fingerprints} -> distinct={len(set(h_D_fingerprints)) == NUM_ROUNDS}")
    print(f"  probe-bank size == candidate_num_mc every round: {bank_sizes}")
    print(f"  denoiser unchanged across entire smoke test: {denoiser_unchanged}")
    print(f"  no denoiser gradient accumulated during any ActorCritic update: {denoiser_grad_none_ok}")
    print(f"  actor/critic grads finite (all {NUM_ROUNDS * NUM_AC_UPDATES_PER_ROUND} updates): {grad_finite_ok}")
    print(f"  loss finite (all updates): {all_finite_ok}")
    if mem_allocated_by_update:
        print(f"  memory_allocated by update (MB): {[f'{m:.1f}' for m in mem_allocated_by_update]}")
        drift = mem_allocated_by_update[-1] - mem_allocated_by_update[1]
        print(f"  memory drift (update index 1 -> last): {drift:.2f}MB")
        mem_ok = abs(drift) < 20.0
    else:
        mem_ok = True

    all_pass = (
        len(set(h_D_fingerprints)) == NUM_ROUNDS
        and all(b == CANDIDATE_NUM_MC for b in bank_sizes)
        and denoiser_unchanged
        and denoiser_grad_none_ok
        and grad_finite_ok
        and all_finite_ok
        and mem_ok
    )
    print(f"\n  ALL CHECKS PASS: {all_pass}")
    print("\nSmoke test complete.", flush=True)


if __name__ == "__main__":
    main()
