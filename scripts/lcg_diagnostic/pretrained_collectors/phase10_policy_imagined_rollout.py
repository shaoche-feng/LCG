"""Phase 10: multi-step self-generated (imagined) rollout, open-loop-action design.

Drives the REAL dm_control environment with an actual trained TD-MPC2 walk/run
specialist policy (the same collector used to originally collect the source pools,
see collect_source_pools.py) to get a realistic, in-distribution action sequence,
then feeds that EXACT action sequence into Seed A's diffusion world model via the
production DiffusionSampler.sample() (the same class WorldModelEnv wraps for
production imagination). After an n_cond-frame real burn-in, every subsequent
frame is 100% self-generated -- recursively conditioned on the model's own prior
predictions, exactly mirroring WorldModelEnv.step()'s buffer-roll semantics
(act_buffer[:, -1] overwritten with the fresh action, next_obs sampled, both
buffers rolled and next_obs appended) -- just hand-rolled here instead of
instantiating WorldModelEnv itself (which needs a DataLoader + rew_end_model
burn-in machinery not needed for pure visualization).

Design note (confirmed with user before building): this is OPEN-LOOP -- TD-MPC2
chooses each action from the REAL environment's true state, not from the world
model's own imagined frames, because TD-MPC2 only ever consumes native
proprioceptive state observations, never images. True closed-loop control (the
policy reacting to the model's own imagined frames) would require a new
image-to-state estimator that does not exist anywhere in this repo.

Saves a side-by-side (imagined | real) mp4 per behavior, both driven by the exact
same action sequence, so the imagined rollout's plausibility can be checked
directly against what actually happened in the real environment.

No production-code changes -- reuses DiffusionSampler, DMControlEnv, and
TDMPC2Collector exactly as they already exist.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/phase10_policy_imagined_rollout.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import sys
from pathlib import Path

import cv2
import numpy as np
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))

from tdmpc2_adapter import load_pretrained_collector  # noqa: E402
from dm_control_env import DMControlEnv  # noqa: E402
from models.diffusion.diffusion_sampler import DiffusionSampler  # noqa: E402
from phase5_lcg_scoring import load_agent  # noqa: E402
from phase6_full_ensemble import checkpoint_path  # noqa: E402

DOMAINS = ["walker", "quadruped", "hopper"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]
# Same alias as collect_source_pools.py's DOMAIN_BEHAVIOR_TASK: hopper has no walk/run
# tasks, so "walk" behavior -> dm_control task "stand", "run" behavior -> "hop".
DOMAIN_BEHAVIOR_TASK = {
    "walker": {"walk": "walk", "run": "run"},
    "quadruped": {"walk": "walk", "run": "run"},
    "hopper": {"walk": "stand", "run": "hop"},
}
SEED_LABEL = "A"
BEHAVIORS = ["walk", "run"]
ROLLOUT_LEN = 150
ACTION_REPEAT = 2
IMG_SIZE = 64
CAMERA_ID = 0
COLLECTOR_SEED = 1  # same TD-MPC2 checkpoint-seed convention as collect_source_pools.py
ENV_SEEDS = {"walk": 90_000, "run": 90_001}  # new documented namespace, distinct from
                                              # collect_source_pools' 10k/20k/30k ranges
FPS = 15
MEAN_SAMPLES = 5  # average this many independent diffusion samples per step, instead of
                   # feeding a single noisy draw back in autoregressively -- same idea as
                   # phase6_full_ensemble.py's cum_means, applied within one model instead
                   # of across an ensemble, to see the model's central-tendency rollout
                   # rather than one potentially-unlucky single-sample trajectory.
TORCH_SEED_BASE = 13579  # fixes the diffusion sampler's noise draws for reproducibility
                          # (the single-sample version had no seed at all -- a real gap:
                          # a rerun silently produced a different, non-reproducible rollout)

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "phase10_policy_imagined_rollout"


def to_uint8_bgr(frame_pm1: torch.Tensor) -> np.ndarray:
    """frame_pm1: (c,h,w) tensor in [-1,1] -> HWC uint8 BGR for cv2."""
    arr = frame_pm1.permute(1, 2, 0).cpu().numpy()
    arr = np.clip((arr + 1) / 2 * 255, 0, 255).astype(np.uint8)
    return cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)


def real_frame_to_tensor(obs_hwc_uint8: np.ndarray, device) -> torch.Tensor:
    t = torch.from_numpy(obs_hwc_uint8.copy()).float().div(255).mul(2).sub(1).permute(2, 0, 1)
    return t.to(device)


def run_rollout(domain: str, behavior: str, condition: str, sampler: DiffusionSampler, n_cond: int, device,
                 dmc_task: str = None) -> None:
    """dmc_task is the ACTUAL dm_control/TD-MPC2-checkpoint task name for `behavior` in this
    domain -- defaults to `behavior` itself (identity), unchanged for walker/quadruped (whose
    "walk"/"run" behavior labels ARE real dm_control tasks). hopper has no walk/run tasks, so
    its caller passes dmc_task="stand" for behavior="walk" and dmc_task="hop" for behavior="run"
    (see collect_source_pools.py's DOMAIN_BEHAVIOR_TASK for the same alias used everywhere
    else in this pipeline). `behavior` itself keeps naming output files/ENV_SEEDS lookups."""
    dmc_task = dmc_task if dmc_task is not None else behavior
    out_path = OUT_ROOT / f"{domain}_{cond_dir(domain, condition)}_{SEED_LABEL}_{slot_dir(domain, behavior)}_policy_mean{MEAN_SAMPLES}_imagined_vs_real.mp4"
    if out_path.exists():
        print(f"  SKIPPING {out_path.name} (already exists)")
        return

    env_seed = ENV_SEEDS[behavior]
    torch.manual_seed(TORCH_SEED_BASE + env_seed)  # fixes every diffusion-sampler noise draw below
    dm_env = DMControlEnv(domain_name=domain, task_name=dmc_task, size=IMG_SIZE, camera_id=CAMERA_ID,
                           action_repeat=ACTION_REPEAT, time_limit=None, seed=env_seed)
    collector = load_pretrained_collector(source="tdmpc2", domain=domain, task=dmc_task, seed=COLLECTOR_SEED)

    obs, _ = dm_env.reset(seed=env_seed)
    try:
        collector.reset_episode(seed=env_seed)
    except TypeError:
        collector.reset_episode()

    # --- real burn-in: n_cond real (frame, action) pairs seed the conditioning window ---
    obs_list, act_list = [], []
    for _ in range(n_cond):
        state_obs = dm_env._dm_env.task.get_observation(dm_env._dm_env.physics)
        action = collector.act(state_obs, deterministic=True)
        obs_list.append(real_frame_to_tensor(obs, device))
        act_list.append(torch.from_numpy(action).float().to(device))
        obs, _, _, _, _ = dm_env.step(action)

    obs_buffer = torch.stack(obs_list).unsqueeze(0)  # (1, n_cond, c, h, w)
    act_buffer = torch.stack(act_list).unsqueeze(0)  # (1, n_cond, act_dim)

    imagined_frames, real_frames = [], []
    with torch.no_grad():
        for step in range(ROLLOUT_LEN):
            state_obs = dm_env._dm_env.task.get_observation(dm_env._dm_env.physics)
            action = collector.act(state_obs, deterministic=True)
            action_t = torch.from_numpy(action).float().to(device)

            act_buffer = act_buffer.clone()
            act_buffer[:, -1] = action_t
            draws = [sampler.sample(obs_buffer, act_buffer)[0] for _ in range(MEAN_SAMPLES)]
            next_obs_imagined = torch.stack(draws).mean(dim=0)
            imagined_frames.append(next_obs_imagined[0].clone())

            real_next_obs, _, terminated, truncated, _ = dm_env.step(action)
            real_frames.append(real_frame_to_tensor(real_next_obs, device))

            obs_buffer = obs_buffer.roll(-1, dims=1)
            obs_buffer[:, -1] = next_obs_imagined
            act_buffer = act_buffer.roll(-1, dims=1)

            if terminated or truncated:
                print(f"  [{behavior}] real env terminated/truncated at step {step + 1} -- "
                      f"stopping (real action sequence can't continue past this point)")
                break

    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (IMG_SIZE * 2, IMG_SIZE))
    for im_f, re_f in zip(imagined_frames, real_frames):
        frame = np.concatenate([to_uint8_bgr(im_f), to_uint8_bgr(re_f)], axis=1)
        writer.write(frame)
    writer.release()
    print(f"  Saved: {out_path} ({len(imagined_frames)} frames, left=imagined right=real)")


def main() -> None:
    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    diffusion_cfg = instantiate(cfg.world_model_env.diffusion_sampler)

    for domain in DOMAINS:
        probe_task = DOMAIN_BEHAVIOR_TASK[domain]["walk"]  # any valid task works for action-space probing
        for condition in CONDITIONS:
            ckpt = checkpoint_path(domain, condition, SEED_LABEL)
            agent, sigma_cfg, _ = load_agent(domain, ckpt, probe_task=probe_task)
            denoiser = agent.denoiser
            device = denoiser.device
            n_cond = denoiser.cfg.inner_model.num_steps_conditioning
            sampler = DiffusionSampler(denoiser, diffusion_cfg)

            print(f"\n{'#' * 70}\n[{domain}/{condition}/{SEED_LABEL}] n_cond={n_cond}, rollout_len={ROLLOUT_LEN}\n{'#' * 70}")
            for behavior in BEHAVIORS:
                dmc_task = DOMAIN_BEHAVIOR_TASK[domain][behavior]
                print(f"\n=== driven by TD-MPC2 {behavior} policy (dm_control task={dmc_task!r}, "
                      f"env_seed={ENV_SEEDS[behavior]}) ===")
                run_rollout(domain, behavior, condition, sampler, n_cond, device, dmc_task=dmc_task)

            del agent, denoiser, sampler
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
