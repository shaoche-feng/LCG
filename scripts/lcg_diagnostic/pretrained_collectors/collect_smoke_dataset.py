"""Tiny smoke test: collect ~100 transitions per pretrained TD-MPC2 policy
through our actual DMControlEnv wrapper, save them using DIAMOND's existing
Episode/Dataset format (unmodified src/data/*.py), then reload and confirm
they're directly consumable by the normal DIAMOND data-loading pipeline
(DatasetTraverser -> Batch), exactly as the real Trainer would.

Does NOT invent a new dataset format, and does NOT modify any DIAMOND/LCG
production code -- this script only *uses* src/data and src/envs as they are.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/collect_smoke_dataset.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))

from tdmpc2_adapter import load_pretrained_collector  # noqa: E402
from dm_control_env import DMControlEnv  # noqa: E402 -- unmodified production wrapper
from data import Dataset, Episode, DatasetTraverser  # noqa: E402 -- unmodified production data pipeline

TASKS = [("walker", "walk"), ("walker", "run"), ("quadruped", "walk"), ("quadruped", "run")]
NUM_TRANSITIONS = 100
SMOKE_SEED_BASE = 20_000  # disjoint from both adapter smoke tests (seed=0) and validation (10000+)
OUT_DIR = _THIS_DIR / "smoke_dataset"


def _to_episode_tensors(frames, actions, rewards, ends, truncs, final_obs):
    # Replicates envs/env.py's TorchEnv._to_tensor conventions exactly:
    # HWC uint8 [0,255] -> CHW float32 [-1,1] for obs; uint8 for end/trunc; float32 for act/rew.
    obs = torch.from_numpy(np.stack(frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    act = torch.from_numpy(np.stack(actions)).float()
    rew = torch.tensor(rewards, dtype=torch.float32)
    end = torch.tensor(ends, dtype=torch.uint8)
    trunc = torch.tensor(truncs, dtype=torch.uint8)
    info = {}
    if final_obs is not None:
        fo = torch.from_numpy(final_obs.copy()).float().div(255).mul(2).sub(1).permute(2, 0, 1).contiguous()
        info["final_observation"] = fo
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info=info)


def collect_one(domain: str, task: str) -> Path:
    print(f"\n=== collecting {domain}/{task} ({NUM_TRANSITIONS} transitions) ===")
    collector = load_pretrained_collector(source="tdmpc2", domain=domain, task=task)
    dm_env = DMControlEnv(
        domain_name=domain, task_name=task, size=64, camera_id=0,
        action_repeat=2, time_limit=None, seed=SMOKE_SEED_BASE,
    )
    obs, _ = dm_env.reset(seed=SMOKE_SEED_BASE)
    collector.reset_episode()

    frames, actions, rewards, ends, truncs = [], [], [], [], []
    final_obs = None
    for t in range(NUM_TRANSITIONS):
        state_obs = dm_env._dm_env.task.get_observation(dm_env._dm_env.physics)
        action = collector.act(state_obs, deterministic=True)

        frames.append(obs)
        actions.append(action)

        next_obs, reward, terminated, truncated, info = dm_env.step(action)
        is_last = t == NUM_TRANSITIONS - 1
        end_flag = 1 if terminated else 0
        trunc_flag = 1 if (truncated or is_last) else 0

        rewards.append(reward)
        ends.append(end_flag)
        truncs.append(trunc_flag)
        obs = next_obs

        if end_flag or trunc_flag:
            final_obs = next_obs
            break

    episode = _to_episode_tensors(frames, actions, rewards, ends, truncs, final_obs)

    print(f"  obs shape={tuple(episode.obs.shape)} dtype={episode.obs.dtype} "
          f"range=[{episode.obs.min():.3f}, {episode.obs.max():.3f}]")
    print(f"  act shape={tuple(episode.act.shape)} dtype={episode.act.dtype}")
    print(f"  rew shape={tuple(episode.rew.shape)} dtype={episode.rew.dtype} sum={episode.rew.sum().item():.2f}")
    print(f"  end sum={episode.end.sum().item()}  trunc sum={episode.trunc.sum().item()}")
    print(f"  length={len(episode)}")

    ds_dir = OUT_DIR / f"{domain}_{task}"
    if ds_dir.exists():
        import shutil
        shutil.rmtree(ds_dir)
    dataset = Dataset(ds_dir, name=f"{domain}_{task}_smoke", cache_in_ram=True)
    episode_id = dataset.add_episode(episode)
    dataset.save_to_default_path()
    print(f"  saved episode_id={episode_id} to {ds_dir}")
    return ds_dir


def verify_round_trip(domain: str, task: str, ds_dir: Path) -> bool:
    print(f"  --- round-trip check: {domain}/{task} ---")
    fresh = Dataset(ds_dir, name=f"{domain}_{task}_smoke_reloaded", cache_in_ram=True)
    fresh.load_from_default_path()
    assert fresh.num_episodes == 1, f"expected 1 episode, got {fresh.num_episodes}"
    assert fresh.num_steps == NUM_TRANSITIONS, f"expected {NUM_TRANSITIONS} steps, got {fresh.num_steps}"

    reloaded = fresh.load_episode(0)
    assert reloaded.obs.dtype == torch.float32
    assert reloaded.obs.min() >= -1.0 - 1e-4 and reloaded.obs.max() <= 1.0 + 1e-4, "obs not in [-1,1] after reload"
    assert reloaded.obs.shape[1:] == (3, 64, 64), f"unexpected obs shape {reloaded.obs.shape}"
    assert reloaded.act.dtype == torch.float32
    assert torch.isfinite(reloaded.obs).all(), "non-finite values in reloaded obs"
    assert torch.isfinite(reloaded.act).all(), "non-finite values in reloaded act"

    # Strongest check: run it through the actual DIAMOND batch-loading pipeline
    # (DatasetTraverser -> make_segment -> Batch), exactly as Trainer's test
    # data loaders do (see trainer.py: DatasetTraverser(self.test_dataset, ...)).
    traverser = DatasetTraverser(fresh, batch_num_samples=2, chunk_size=8)
    num_batches = 0
    for batch in traverser:
        num_batches += 1
        assert batch.obs.shape[-3:] == (3, 64, 64)
        assert torch.isfinite(batch.obs.float()).all()
    print(f"  DatasetTraverser produced {num_batches} batches successfully -- round trip OK")
    return True


def main() -> None:
    OUT_DIR.mkdir(exist_ok=True)
    results = {}
    for domain, task in TASKS:
        ds_dir = collect_one(domain, task)
        ok = verify_round_trip(domain, task, ds_dir)
        results[f"{domain}-{task}"] = ok

    print("\n=== SMOKE COLLECTION SUMMARY ===")
    for k, ok in results.items():
        print(f"  {k}: {'PASSED' if ok else 'FAILED'}")
    assert all(results.values()), "one or more smoke collections failed round-trip verification"
    print("\nALL 4 SMOKE DATASET COLLECTIONS PASSED ROUND-TRIP VERIFICATION")


if __name__ == "__main__":
    main()
