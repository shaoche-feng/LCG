#! /usr/bin/env python
"""
LCG Stage 5B: rollout-level LCG scoring (Part A) and intrinsic-reward scale normalization
(Part B). Correctness/diagnostic only, same tiny workload as Stage 5A.

Part A: lcg.intrinsic_reward.make_lcg_intrinsic_reward_fn now flattens all H x num_envs
candidates from one ActorCritic.forward() rollout into a single score_candidates_batched
call (chunk_size=4) instead of one call per rollout step -- a pure batching change; no
different VJP, CRN semantics, h_D, K, or num_strata. Verified here against a re-implemented
per-step reference (the exact Stage 5A logic) for numerical/ordering equivalence.

Part B: lcg.reward_normalization.RunningRMS implements the requested multiplicative
running-RMS scale (no batch z-score centering, no clipping), as an object completely
separate from the denoiser/ActorCritic modules. Three modes are compared on the same tiny
workload: baseline (world-model reward), raw LCG (Part A hook only), and RMS-normalized LCG
(Part A hook + RunningRMS, alpha=1).

Does not integrate into Trainer.run(), does not run long exploration, does not add
clipping. Reuses Stage 1-4.5's validated math throughout.

Usage:
    python scripts/validate_lcg_stage5b.py
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
from envs import WorldModelEnv, WorldModelEnvConfig
from envs.dm_control_env import make_dm_control_env
from lcg.batched_scoring import imagined_candidates_from_batch
from lcg.batched_vjp import score_candidates_batched
from lcg.crn import make_crn_bank_set
from lcg.intrinsic_reward import make_lcg_intrinsic_reward_fn
from lcg.precision import historical_precision
from lcg.reward_normalization import RunningRMS, RunningRMSConfig, wrap_with_running_rms
from lcg.theta_s import selected_parameters
import models.actor_critic as ac_module
from models.actor_critic import ActorCritic, ActorCriticConfig, ActorCriticLossConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig, SigmaDistributionConfig
from models.diffusion.inner_model import InnerModelConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from utils import configure_opt

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DIAG_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_stage5b_train_for_hD"
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage5b_heldout_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
NUM_STEPS_CONDITIONING = 4
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
SAMPLER_CFG = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0
NUM_CRN_BANKS = 2
CHUNK_SIZE = 4

# Same tiny diagnostic scale as Stage 5A
B_ENVS = 6
H_BACKUP_EVERY = 4
WM_HORIZON = 8
NUM_UPDATES = 3


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


def load_diagnostic_denoiser(device):
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
    return denoiser, ckpt["action_dim"]


def build_rew_end_model(device, action_dim):
    cfg = RewEndModelConfig(
        lstm_dim=512, img_channels=3, img_size=64, cond_channels=128,
        depths=[2, 2, 2, 2], channels=[32, 32, 32, 32], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim, continuous_reward=True,
    )
    model = RewEndModel(cfg).to(device)
    model.eval()
    return model


def build_world_model_env(denoiser, rew_end_model, dataset, num_envs, horizon):
    cfg = WorldModelEnvConfig(horizon=horizon, num_batches_to_preload=4, diffusion_sampler=SAMPLER_CFG)
    bs = BatchSampler(dataset, 0, 1, num_envs, NUM_STEPS_CONDITIONING, sample_weights=None)
    loader = DataLoader(dataset=dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    return WorldModelEnv(denoiser, rew_end_model, loader, cfg, return_imagined_candidate=True)


def build_actor_critic(device, action_dim, action_low, action_high):
    cfg = ActorCriticConfig(
        lstm_dim=512, img_channels=3, img_size=64, channels=[32, 32, 64, 64], down=[1, 1, 1, 1],
        continuous_action_dim=action_dim, action_low=action_low, action_high=action_high, continuous_reward=True,
    )
    return ActorCritic(cfg).to(device)


def snapshot(module):
    return {k: v.detach().clone() for k, v in module.state_dict().items()}


def unchanged(before, after):
    return all(torch.equal(before[k], after[k]) for k in before)


def pearson_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson_corr(ra, rb)


# --------------------------------------------------------------------------------------
# Part A reference: the exact Stage 5A per-step scoring logic, for comparison only
# --------------------------------------------------------------------------------------


def reference_per_step_scores(denoiser, params, h_D, banks, infos, env_rew, chunk_size):
    num_envs, num_steps = env_rew.shape
    per_step = []
    for t in range(num_steps):
        batch = infos[t]["imagined_candidate"]
        candidates = imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star)
        scores = score_candidates_batched(denoiser, params, h_D, banks, candidates, chunk_size)
        per_step.append(scores.to(device=env_rew.device, dtype=env_rew.dtype))
    return torch.stack(per_step, dim=1)


# --------------------------------------------------------------------------------------
# Part B: grad norms + monkeypatched compute_lambda_returns capture
# --------------------------------------------------------------------------------------


def compute_grad_norms(actor_critic):
    def norm_of(params):
        grads = [p.grad.detach().flatten() for p in params if p.grad is not None]
        return torch.cat(grads).norm().item() if grads else 0.0

    return (
        norm_of(actor_critic.actor_linear.parameters()),
        norm_of(actor_critic.critic_linear.parameters()),
        norm_of(actor_critic.parameters()),
    )


def run_mode(hook, denoiser, rew_end_model, heldout_dataset, action_dim, action_low, action_high, loss_cfg, device):
    original_clr = ac_module.compute_lambda_returns
    captured = {"rew": [], "lambda_returns": []}

    def capturing_clr(rew, end, trunc, val_bootstrap, gamma, lambda_, continuous_reward=False):
        captured["rew"].append(rew.detach().clone())
        result = original_clr(rew, end, trunc, val_bootstrap, gamma, lambda_, continuous_reward=continuous_reward)
        captured["lambda_returns"].append(result.detach().clone())
        return result

    wm_env = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_ENVS, WM_HORIZON)
    ac = build_actor_critic(device, action_dim, action_low, action_high)
    ac.setup_training(wm_env, loss_cfg)
    if hook is not None:
        ac.set_intrinsic_reward_fn(hook)
    opt = configure_opt(ac, lr=1e-4, weight_decay=0.0, eps=1e-8)

    ac_before = snapshot(ac)
    denoiser_before = snapshot(denoiser)

    ac_module.compute_lambda_returns = capturing_clr
    t0 = time.perf_counter()
    logs = []
    try:
        for _ in range(NUM_UPDATES):
            loss, metrics = ac()
            opt.zero_grad()
            loss.backward()
            actor_gn, critic_gn, total_gn = compute_grad_norms(ac)
            opt.step()
            m = {k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()}
            m.update(actor_grad_norm=actor_gn, critic_grad_norm=critic_gn, total_grad_norm=total_gn)
            logs.append(m)
    finally:
        ac_module.compute_lambda_returns = original_clr
    t_total = time.perf_counter() - t0

    ac_after = snapshot(ac)
    denoiser_after = snapshot(denoiser)

    return dict(
        logs=logs, t_total=t_total,
        rew_history=captured["rew"], lambda_returns_history=captured["lambda_returns"],
        params_changed=not unchanged(ac_before, ac_after),
        denoiser_unchanged=unchanged(denoiser_before, denoiser_after),
    )


def tensor_stats(tensors):
    flat = torch.cat([t.flatten().float() for t in tensors])
    return dict(
        mean=flat.mean().item(), std=flat.std().item(), rms=flat.square().mean().sqrt().item(),
        min=flat.min().item(), max=flat.max().item(),
    )


def print_mode_report(name, result):
    print(f"\n  --- {name} ---")
    for i, m in enumerate(result["logs"]):
        print(f"    update {i}: loss_actions={m['loss_actions']:.4g}  loss_values={m['loss_values']:.4g}  "
              f"loss_total={m['loss_total']:.4g}  actor_grad_norm={m['actor_grad_norm']:.4g}  "
              f"critic_grad_norm={m['critic_grad_norm']:.4g}  total_grad_norm={m['total_grad_norm']:.4g}")
    rs = tensor_stats(result["rew_history"])
    lrs = tensor_stats(result["lambda_returns_history"])
    print(f"    reward used by compute_lambda_returns: mean={rs['mean']:.4g} std={rs['std']:.4g} "
          f"rms={rs['rms']:.4g} range=[{rs['min']:.4g}, {rs['max']:.4g}]")
    print(f"    lambda_returns: mean={lrs['mean']:.4g} std={lrs['std']:.4g} "
          f"range=[{lrs['min']:.4g}, {lrs['max']:.4g}]")
    print(f"    params changed: {result['params_changed']}  denoiser unchanged: {result['denoiser_unchanged']}  "
          f"wall time: {result['t_total']:.2f}s")
    all_finite = all(torch.isfinite(r).all().item() for r in result["rew_history"])
    print(f"    reward finite (all updates): {all_finite}")


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    denoiser, action_dim = load_diagnostic_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)
    rew_end_model = build_rew_end_model(device, action_dim)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage5b_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage5b_heldout", 500, seed=555, include_stray=False)

    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,) and torch.all(h_D > 0)
    print(f"h_D computed and frozen: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, max={h_D.max().item():.4g}")

    y_shape = torch.Size([1, 3, 64, 64])
    banks = make_crn_bank_set(SIGMA_CFG, y_shape, device, num_crn_banks=NUM_CRN_BANKS, num_strata=3, seed=7000)
    print(f"CRN bank set frozen: K={NUM_CRN_BANKS} banks x 3 strata each.")

    probe_env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    action_low, action_high = probe_env.action_low.tolist(), probe_env.action_high.tolist()

    loss_cfg = ActorCriticLossConfig(
        backup_every=H_BACKUP_EVERY, gamma=0.985, lambda_=0.95, weight_value_loss=1.0, weight_entropy_loss=0.001
    )
    denoiser_before_all = snapshot(denoiser)

    # ------------------------------------------------------------------------------
    # PART A: rollout-level (flattened) scoring vs per-step reference
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART A: rollout-flattened scoring vs per-step reference")
    print("=" * 88)

    wm_env_probe = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_ENVS, WM_HORIZON)
    ac_probe = build_actor_critic(device, action_dim, action_low, action_high)
    ac_probe.setup_training(wm_env_probe, loss_cfg)
    _, act, env_rew, end, trunc, logits_act, val, val_bootstrap, infos = ac_probe.env_loop.send(H_BACKUP_EVERY)
    print(f"  captured one raw rollout: env_rew.shape={tuple(env_rew.shape)}, {len(infos)} info dicts")

    NUM_TIMING_REPS = 3
    old_scores = None
    t_old_list = []
    for _ in range(NUM_TIMING_REPS):
        t0 = time.perf_counter()
        old_scores = reference_per_step_scores(denoiser, params, h_D, banks, infos, env_rew, CHUNK_SIZE)
        t_old_list.append(time.perf_counter() - t0)

    flattened_hook = make_lcg_intrinsic_reward_fn(denoiser, params, h_D, banks, chunk_size=CHUNK_SIZE)
    new_scores = None
    t_new_list = []
    for _ in range(NUM_TIMING_REPS):
        t0 = time.perf_counter()
        new_scores = flattened_hook(infos, env_rew)
        t_new_list.append(time.perf_counter() - t0)

    abs_err = (new_scores - old_scores).abs()
    rel_err = abs_err / old_scores.abs().clamp_min(1e-12)
    pear = pearson_corr(new_scores.flatten(), old_scores.flatten())
    spear = spearman_corr(new_scores.flatten(), old_scores.flatten())
    order_match = torch.equal(torch.argsort(new_scores.flatten()), torch.argsort(old_scores.flatten()))
    elementwise_match = torch.allclose(new_scores, old_scores, rtol=1e-2, atol=1e-1)

    print(f"  old (per-step) shape={tuple(old_scores.shape)}   new (flattened) shape={tuple(new_scores.shape)}")
    print(f"  max_abs_err={abs_err.max().item():.4e}  max_rel_err={rel_err.max().item():.4e}  "
          f"Pearson={pear:.6f}  Spearman={spear:.6f}")
    print(f"  full ranking order identical (flattened, argsort): {order_match}")
    print(f"  elementwise close (rtol=1e-2, atol=1e-1): {elementwise_match}")
    print(f"  per-(env,step) shape/order check: new_scores[2,1]={new_scores[2, 1].item():.4f} "
          f"vs old_scores[2,1]={old_scores[2, 1].item():.4f}   "
          f"new_scores[0,3]={new_scores[0, 3].item():.4f} vs old_scores[0,3]={old_scores[0, 3].item():.4f}")
    print(f"  timing: per-step={np.mean(t_old_list) * 1000:.1f}ms (min {np.min(t_old_list) * 1000:.1f}ms)  "
          f"flattened={np.mean(t_new_list) * 1000:.1f}ms (min {np.min(t_new_list) * 1000:.1f}ms)  "
          f"speedup={np.mean(t_old_list) / np.mean(t_new_list):.2f}x")
    old_vjps = H_BACKUP_EVERY * -(-B_ENVS // CHUNK_SIZE) * NUM_CRN_BANKS * 3
    new_vjps = -(-(H_BACKUP_EVERY * B_ENVS) // CHUNK_SIZE) * NUM_CRN_BANKS * 3
    print(f"  VJP count: per-step={old_vjps} (has partial/wasted chunks per step)  "
          f"flattened={new_vjps} (fewer partial chunks)")

    # ------------------------------------------------------------------------------
    # PART B: three-mode comparison
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("PART B: baseline vs raw LCG vs RMS-normalized LCG")
    print("=" * 88)

    print("\n[mode 1/3] baseline (intrinsic_reward_fn=None)")
    result_baseline = run_mode(None, denoiser, rew_end_model, heldout_dataset, action_dim, action_low, action_high, loss_cfg, device)
    print_mode_report("baseline", result_baseline)

    print("\n[mode 2/3] raw LCG (Part A flattened hook, no normalization)")
    raw_hook = make_lcg_intrinsic_reward_fn(denoiser, params, h_D, banks, chunk_size=CHUNK_SIZE)
    result_raw = run_mode(raw_hook, denoiser, rew_end_model, heldout_dataset, action_dim, action_low, action_high, loss_cfg, device)
    print_mode_report("raw LCG", result_raw)

    print("\n[mode 3/3] RMS-normalized LCG (alpha=1, ema_decay=0.99, eps=1e-8)")
    raw_hook_for_norm = make_lcg_intrinsic_reward_fn(denoiser, params, h_D, banks, chunk_size=CHUNK_SIZE)
    rms = RunningRMS(RunningRMSConfig(enabled=True, alpha=1.0, ema_decay=0.99, eps=1e-8))
    raw_capture_log = []

    def instrumented_normalized_hook(infos_, env_rew_):
        raw = raw_hook_for_norm(infos_, env_rew_)
        raw_capture_log.append(raw.detach().clone())
        return rms(raw)

    result_norm = run_mode(
        instrumented_normalized_hook, denoiser, rew_end_model, heldout_dataset, action_dim, action_low, action_high, loss_cfg, device
    )
    print_mode_report("RMS-normalized LCG", result_norm)
    raw_stats = tensor_stats(raw_capture_log)
    norm_stats = tensor_stats(result_norm["rew_history"])
    print(f"    [pre-normalization] raw LCG:  mean={raw_stats['mean']:.4g} std={raw_stats['std']:.4g} "
          f"rms={raw_stats['rms']:.4g} range=[{raw_stats['min']:.4g}, {raw_stats['max']:.4g}]")
    print(f"    [post-normalization] r_int:   mean={norm_stats['mean']:.4g} std={norm_stats['std']:.4g} "
          f"rms={norm_stats['rms']:.4g} range=[{norm_stats['min']:.4g}, {norm_stats['max']:.4g}]")

    # ------------------------------------------------------------------------------
    # VERIFICATION
    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("VERIFICATION")
    print("=" * 88)

    order_preserved = all(
        torch.equal(torch.argsort(raw_capture_log[i].flatten()), torch.argsort(result_norm["rew_history"][i].flatten()))
        for i in range(len(raw_capture_log))
    )
    print(f"  normalization preserves LCG candidate ordering (all updates): {order_preserved}")

    finite_nonneg = all(
        torch.isfinite(r).all().item() and (r >= 0).all().item() for r in result_norm["rew_history"]
    )
    print(f"  normalized rewards finite and nonnegative (all updates): {finite_nonneg}")

    print(f"  denoiser unchanged -- baseline: {result_baseline['denoiser_unchanged']}  "
          f"raw LCG: {result_raw['denoiser_unchanged']}  RMS-normalized: {result_norm['denoiser_unchanged']}")
    denoiser_after_all = snapshot(denoiser)
    print(f"  denoiser unchanged across entire Stage 5B run: {unchanged(denoiser_before_all, denoiser_after_all)}")

    baseline_sane = (
        all(np.isfinite(m["loss_total"]) for m in result_baseline["logs"])
        and result_baseline["params_changed"]
        and result_baseline["denoiser_unchanged"]
    )
    print(f"  baseline mode behaves as in Stage 5A (finite/updated/denoiser-untouched): {baseline_sane}")

    disabled_rms = RunningRMS(RunningRMSConfig(enabled=False))
    disabled_hook = wrap_with_running_rms(flattened_hook, disabled_rms)
    disabled_out = disabled_hook(infos, env_rew)
    raw_out = flattened_hook(infos, env_rew)
    disabling_matches_raw = torch.equal(disabled_out, raw_out)
    print(f"  disabling normalization (enabled=False) reproduces raw-LCG hook output exactly: {disabling_matches_raw}")

    print(f"  rollout-flattened scoring matches per-step reference (Part A): "
          f"order_match={order_match}  elementwise_close={elementwise_match}  max_rel_err={rel_err.max().item():.4e}")

    print("\n" + "=" * 88)
    print("SUMMARY: is normalized LCG's lambda-return/value-loss magnitude reasonable?")
    print("=" * 88)
    base_lrs = tensor_stats(result_baseline["lambda_returns_history"])
    raw_lrs = tensor_stats(result_raw["lambda_returns_history"])
    norm_lrs = tensor_stats(result_norm["lambda_returns_history"])
    print(f"  baseline    lambda_returns range: [{base_lrs['min']:.4g}, {base_lrs['max']:.4g}]  "
          f"loss_values(last update)={result_baseline['logs'][-1]['loss_values']:.4g}")
    print(f"  raw LCG     lambda_returns range: [{raw_lrs['min']:.4g}, {raw_lrs['max']:.4g}]  "
          f"loss_values(last update)={result_raw['logs'][-1]['loss_values']:.4g}")
    print(f"  norm. LCG   lambda_returns range: [{norm_lrs['min']:.4g}, {norm_lrs['max']:.4g}]  "
          f"loss_values(last update)={result_norm['logs'][-1]['loss_values']:.4g}")

    print("\nStage 5B diagnostic complete.")


if __name__ == "__main__":
    main()
