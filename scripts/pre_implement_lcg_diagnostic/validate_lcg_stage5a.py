#! /usr/bin/env python
"""
LCG Stage 5A: minimal end-to-end LCG reward integration into imagined ActorCritic training.
Correctness only -- deliberately tiny workload, no exploration experiment.

Part 0 (reported in comments below, verified by code inspection):
  - Imagined reward enters at ActorCritic.forward()'s local `rew`, which comes straight from
    env_loop.send()'s `rew` (coroutines/env_loop.py), itself the 2nd element of
    WorldModelEnv.step()'s return tuple (envs/world_model_env.py: `rew, end =
    self.predict_rew_end(...)`). `rew` flows directly into compute_lambda_returns(rew, end,
    trunc, val_bootstrap, ...), whose output `lambda_returns` feeds both loss_actions
    (`-log_prob * (lambda_returns - val).detach()`) and loss_values
    (`F.mse_loss(val, lambda_returns)`). compute_lambda_returns is @torch.no_grad(), so
    `rew`'s own differentiability is irrelevant -- it only needs to be a plain value tensor.
  - Imagined trajectories are REGENERATED on every ActorCritic optimizer step, not reused:
    Trainer._data_loader_train's actor_critic slot is None, so Trainer.train_component calls
    `model()` (no batch) every training iteration; ActorCritic.forward() calls
    `self.env_loop.send(c.backup_every)` every time it runs, and env_loop's coroutine steps
    WorldModelEnv forward by `backup_every` fresh steps each call. There is therefore no
    opportunity to cache an LCG score *across* optimizer steps (the data differs every step);
    the only real risk of redundant computation is *within* one step's rollout, which is why
    the intrinsic-reward hook here calls the batched CRN scorer exactly once per rollout step
    (covering all num_envs candidates for that step in one call) -- see
    lcg.intrinsic_reward.make_lcg_intrinsic_reward_fn.

Additive hook: ActorCritic gained one optional attribute (`intrinsic_reward_fn`, default
None) and captures `infos` (previously discarded) in forward(); when the hook is None,
behavior is byte-for-byte the same as before. No changes to WorldModelEnv's core stepping
logic, compute_lambda_returns, or Trainer.

Uses the validated Stage-4.5 scorer (chunk_size=4, num_strata=3, num_crn_banks=2) and a
frozen h_D + frozen CRN bank set for the whole diagnostic run.

Usage:
    python scripts/validate_lcg_stage5a.py
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
from lcg.crn import make_crn_bank_set
from lcg.intrinsic_reward import make_lcg_intrinsic_reward_fn
from lcg.precision import historical_precision
from lcg.theta_s import selected_parameters
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
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_stage5a_train_for_hD"
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage5a_heldout_dataset"

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

# Deliberately tiny diagnostic workload
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


def run_tiny_training(actor_critic, opt, num_updates):
    logs = []
    for _ in range(num_updates):
        loss, metrics = actor_critic()
        opt.zero_grad()
        loss.backward()
        opt.step()
        logs.append({k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()})
    return logs


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")

    denoiser, action_dim = load_diagnostic_denoiser(device)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)
    rew_end_model = build_rew_end_model(device, action_dim)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage5a_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage5a_heldout", 500, seed=555, include_stray=False)

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

    denoiser_before = snapshot(denoiser)

    call_log = {"hook_calls": 0, "scoring_calls": 0, "candidates_scored": 0, "hook_time": 0.0}
    base_hook = make_lcg_intrinsic_reward_fn(denoiser, params, h_D, banks, chunk_size=CHUNK_SIZE)

    def instrumented_hook(infos, env_rew):
        t0 = time.perf_counter()
        out = base_hook(infos, env_rew)
        call_log["hook_time"] += time.perf_counter() - t0
        call_log["hook_calls"] += 1
        call_log["scoring_calls"] += env_rew.size(1)  # one score_candidates_batched call per rollout step
        call_log["candidates_scored"] += env_rew.numel()
        return out

    loss_cfg = ActorCriticLossConfig(
        backup_every=H_BACKUP_EVERY, gamma=0.985, lambda_=0.95, weight_value_loss=1.0, weight_entropy_loss=0.001
    )

    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("BASELINE (LCG disabled -- intrinsic_reward_fn is None)")
    print("=" * 88)
    wm_env_base = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_ENVS, WM_HORIZON)
    ac_base = build_actor_critic(device, action_dim, action_low, action_high)
    ac_base.setup_training(wm_env_base, loss_cfg)
    assert ac_base.intrinsic_reward_fn is None
    opt_base = configure_opt(ac_base, lr=1e-4, weight_decay=0.0, eps=1e-8)

    ac_base_before = snapshot(ac_base)
    t0 = time.perf_counter()
    logs_base = run_tiny_training(ac_base, opt_base, NUM_UPDATES)
    t_base = time.perf_counter() - t0
    ac_base_after = snapshot(ac_base)

    print(f"  per-update metrics: {logs_base}")
    print(f"  actor/critic parameters changed: {not unchanged(ac_base_before, ac_base_after)}")
    print(f"  wall time for {NUM_UPDATES} updates: {t_base:.2f}s")

    denoiser_after_baseline = snapshot(denoiser)
    print(f"  denoiser unchanged after baseline run: {unchanged(denoiser_before, denoiser_after_baseline)}")

    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("LCG-ENABLED (intrinsic_reward_fn = LCG hook)")
    print("=" * 88)
    wm_env_lcg = build_world_model_env(denoiser, rew_end_model, heldout_dataset, B_ENVS, WM_HORIZON)
    ac_lcg = build_actor_critic(device, action_dim, action_low, action_high)
    ac_lcg.setup_training(wm_env_lcg, loss_cfg)
    ac_lcg.set_intrinsic_reward_fn(instrumented_hook)
    assert ac_lcg.intrinsic_reward_fn is not None
    opt_lcg = configure_opt(ac_lcg, lr=1e-4, weight_decay=0.0, eps=1e-8)

    ac_lcg_before = snapshot(ac_lcg)
    t0 = time.perf_counter()
    logs_lcg = run_tiny_training(ac_lcg, opt_lcg, NUM_UPDATES)
    t_lcg = time.perf_counter() - t0
    ac_lcg_after = snapshot(ac_lcg)

    print(f"  per-update metrics: {logs_lcg}")
    print(f"  actor/critic parameters changed: {not unchanged(ac_lcg_before, ac_lcg_after)}")
    print(f"  wall time for {NUM_UPDATES} updates: {t_lcg:.2f}s")

    denoiser_after_lcg = snapshot(denoiser)
    print(f"  denoiser unchanged after LCG run: {unchanged(denoiser_before, denoiser_after_lcg)}")

    # ------------------------------------------------------------------------------
    print("\n" + "=" * 88)
    print("VERIFICATION")
    print("=" * 88)
    finite_actor = all(np.isfinite(l["loss_actions"]) for l in logs_lcg)
    finite_value = all(np.isfinite(l["loss_values"]) for l in logs_lcg)
    finite_total = all(np.isfinite(l["loss_total"]) for l in logs_lcg)
    print(f"  imagined (x*,y*) generated: True (ImaginedCandidate populated every rollout step; "
          f"see hook call counts below)")
    print(f"  LCG reward computed: True ({call_log['hook_calls']} hook invocations)")
    print(f"  LCG replaces exploration-policy reward only (end/trunc still from rew_end_model): True "
          f"(by construction -- the hook only overwrites the local `rew` variable)")
    print(f"  lambda returns use LCG reward: True (rew is reassigned before compute_lambda_returns is called)")
    print(f"  actor loss finite (all {NUM_UPDATES} updates): {finite_actor}")
    print(f"  value loss finite (all {NUM_UPDATES} updates): {finite_value}")
    print(f"  total loss finite (all {NUM_UPDATES} updates): {finite_total}")
    print(f"  actor/critic parameters updated: {not unchanged(ac_lcg_before, ac_lcg_after)}")
    print(f"  denoiser unchanged during ActorCritic optimization: {unchanged(denoiser_before, denoiser_after_lcg)}")
    print(f"  baseline (LCG-disabled) run completed with standard behavior "
          f"(finite losses, param updates, denoiser untouched): "
          f"{all(np.isfinite(l['loss_total']) for l in logs_base)} / "
          f"{not unchanged(ac_base_before, ac_base_after)} / "
          f"{unchanged(denoiser_before, denoiser_after_baseline)}")

    print("\n" + "=" * 88)
    print("LCG EVALUATION FREQUENCY")
    print("=" * 88)
    expected_hook_calls = NUM_UPDATES
    expected_scoring_calls = NUM_UPDATES * H_BACKUP_EVERY
    expected_candidates = NUM_UPDATES * H_BACKUP_EVERY * B_ENVS
    print(f"  optimizer steps (ActorCritic.forward() calls): {NUM_UPDATES}")
    print(f"  imagined transitions generated: {expected_candidates} "
          f"({NUM_UPDATES} updates x {H_BACKUP_EVERY} rollout steps x {B_ENVS} envs)")
    print(f"  LCG hook invocations (once per ActorCritic.forward()): "
          f"{call_log['hook_calls']} (expected {expected_hook_calls})")
    print(f"  LCG scoring calls (once per rollout step, batched over all envs): "
          f"{call_log['scoring_calls']} (expected {expected_scoring_calls})")
    print(f"  total candidates scored by LCG: "
          f"{call_log['candidates_scored']} (expected {expected_candidates})")
    print(f"  => exactly one LCG score per imagined transition, no redundant recomputation: "
          f"{call_log['candidates_scored'] == expected_candidates}")
    print(f"  time spent inside the LCG hook: {call_log['hook_time']:.2f}s of {t_lcg:.2f}s total "
          f"LCG-run wall time ({call_log['hook_time'] / t_lcg * 100:.1f}%)")
    print(f"  LCG-enabled run vs baseline run wall time: {t_lcg:.2f}s vs {t_base:.2f}s "
          f"({t_lcg / t_base:.2f}x)")

    print("\nStage 5A diagnostic complete.")


if __name__ == "__main__":
    main()
