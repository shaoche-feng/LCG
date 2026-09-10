#! /usr/bin/env python
"""
Build a new, larger (N~5000) matched replay for the historical-precision B-scaling
diagnostic's N-scaling follow-up (see docs/lcg_diagnostic/historical_precision_B_scaling/).

Uses EXACTLY the same collection recipe as the trustworthy N=1540 diagnostic snapshot
(recovered from git history: scripts/pre_implement_lcg_diagnostic/train_lcg_diagnostic_checkpoint.py,
deleted in the LCG script-cleanup pass but preserved in git log) -- dm_control cheetah/run,
action_repeat=2, time_limit=1.0, size=64, camera_id=0, random uniform-action policy,
np.random.default_rng(seed), and the SAME one stray real episode
(outputs/2026-08-17/15-35-46/dataset/train/000/00/0/0.pt) prepended as episode 0 before the
freshly-collected episodes -- only the target step count changed (1500 -> ~5000). Because
seed=0 drives the same deterministic rng stream, this new dataset's first 31 episodes are
identical to the N=1540 dataset's episodes; additional episodes are appended past that point
until the larger target is reached (expected and reported, not a defect).

Saved under a durable repository-side directory (docs/lcg_diagnostic/.../N5000/), NOT under
OS Temp or the Claude scratchpad, per the durability requirement (Temp cleanup has destroyed
diagnostic artifacts more than once this project).

Usage:
    python scripts/historical_precision_diagnostic/build_replay_N5000.py
"""
import shutil
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

import numpy as np
import torch

from data import Dataset, Episode
from envs.dm_control_env import make_dm_control_env

STRAY_EPISODE_PATH = _REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46" / "dataset" / "train" / "000" / "00" / "0" / "0.pt"

OUT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "historical_precision_B_scaling" / "N5000"
TRAIN_DATASET_DIR = OUT_DIR / "train_dataset"
VAL_DATASET_DIR = OUT_DIR / "val_dataset"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
TRAIN_TARGET_STEPS = 5000
VAL_TARGET_STEPS = 300  # unchanged from the N=1540 recipe: a fixed validation probe, not part of N
TRAIN_SEED = 0
VAL_SEED = 999


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
        assert STRAY_EPISODE_PATH.is_file(), f"stray episode missing: {STRAY_EPISODE_PATH}"
        dataset.add_episode(Episode.load(STRAY_EPISODE_PATH))
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    return dataset


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"building train replay (target N={TRAIN_TARGET_STEPS}, seed={TRAIN_SEED}, stray episode included)...", flush=True)
    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_diag_train_N5000", TRAIN_TARGET_STEPS, TRAIN_SEED, include_stray=True)
    print(f"train replay: {train_dataset.num_episodes} episodes, N={train_dataset.num_steps} steps -> {TRAIN_DATASET_DIR}", flush=True)

    print(f"\nbuilding val replay (target N={VAL_TARGET_STEPS}, seed={VAL_SEED}, no stray episode)...", flush=True)
    val_dataset = build_dataset(VAL_DATASET_DIR, "lcg_diag_val_N5000", VAL_TARGET_STEPS, VAL_SEED, include_stray=False)
    print(f"val replay: {val_dataset.num_episodes} episodes, N={val_dataset.num_steps} steps -> {VAL_DATASET_DIR}", flush=True)

    action_dim = int(train_dataset.load_episode(0).act.shape[-1])
    print(f"\naction_dim={action_dim}", flush=True)
    print("Done.", flush=True)


if __name__ == "__main__":
    main()
