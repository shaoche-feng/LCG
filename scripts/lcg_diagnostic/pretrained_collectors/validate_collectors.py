"""Validate the 4 TD-MPC2 pretrained collectors against our actual DIAMOND
DM-Control environment wrapper (src/envs/dm_control_env.py, unmodified).

Runs 10 deterministic evaluation episodes per (domain, task) on seeds that
were never used anywhere else in this diagnostic (10000+), reports return/
length/action-range/finite-check/forward-velocity statistics, and saves a
short PNG frame sequence per policy for visual sanity checking.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/validate_collectors.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent  # scripts/lcg_diagnostic/pretrained_collectors -> LCG/
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))  # import the module directly, bypassing
# src/envs/__init__.py's package-level imports (e.g. ale_py/AsyncVectorEnv), which are
# irrelevant here and would needlessly couple this isolated script to the Atari path.

from tdmpc2_adapter import load_pretrained_collector  # noqa: E402
from dm_control_env import DMControlEnv  # noqa: E402 -- unmodified production wrapper

TASKS = [("walker", "walk"), ("walker", "run"), ("quadruped", "walk"), ("quadruped", "run")]
NUM_EVAL_EPISODES = 10
UNSEEN_SEED_BASE = 10_000  # disjoint from seed=0 used in adapter smoke tests
FRAME_DUMP_COUNT = 40

OUT_DIR = _THIS_DIR / "validation_results"
OUT_DIR.mkdir(exist_ok=True)


def _forward_velocity(domain: str, dm_env: DMControlEnv) -> float:
    physics = dm_env._dm_env.physics  # read-only access, no modification to DMControlEnv
    if domain == "walker":
        return float(physics.horizontal_velocity())
    elif domain == "quadruped":
        return float(physics.torso_velocity()[0])
    raise ValueError(domain)


def validate_one(domain: str, task: str) -> dict:
    print(f"\n=== {domain}/{task} ===")
    collector = load_pretrained_collector(source="tdmpc2", domain=domain, task=task)

    returns, lengths, fwd_vels = [], [], []
    action_min, action_max = None, None
    all_finite = True
    frames = []

    for ep in range(NUM_EVAL_EPISODES):
        seed = UNSEEN_SEED_BASE + ep
        dm_env = DMControlEnv(
            domain_name=domain, task_name=task, size=64, camera_id=0,
            action_repeat=2, time_limit=None, seed=seed,
        )
        dm_env.reset(seed=seed)
        collector.reset_episode()

        ep_return, ep_len, terminated, truncated = 0.0, 0, False, False
        while not (terminated or truncated):
            state_obs = dm_env._dm_env.task.get_observation(dm_env._dm_env.physics)
            action = collector.act(state_obs, deterministic=True)

            finite = bool(np.all(np.isfinite(action)))
            all_finite = all_finite and finite
            action_min = action.copy() if action_min is None else np.minimum(action_min, action)
            action_max = action.copy() if action_max is None else np.maximum(action_max, action)

            obs_rgb, reward, terminated, truncated, info = dm_env.step(action)
            ep_return += float(reward)
            ep_len += 1
            fwd_vels.append(_forward_velocity(domain, dm_env))

            if ep == 0 and len(frames) < FRAME_DUMP_COUNT:
                frames.append(obs_rgb.copy())

        returns.append(ep_return)
        lengths.append(ep_len)
        print(f"  episode {ep} (seed={seed}): return={ep_return:.2f} length={ep_len}")

    frame_dir = OUT_DIR / f"{domain}_{task}_frames"
    frame_dir.mkdir(exist_ok=True)
    for i, frame in enumerate(frames):
        Image.fromarray(frame).save(frame_dir / f"frame_{i:03d}.png")

    result = {
        "domain": domain,
        "task": task,
        "return_mean": float(np.mean(returns)),
        "return_std": float(np.std(returns)),
        "return_median": float(np.median(returns)),
        "return_min": float(np.min(returns)),
        "return_max": float(np.max(returns)),
        "length_mean": float(np.mean(lengths)),
        "action_min": action_min.tolist(),
        "action_max": action_max.tolist(),
        "all_actions_finite": all_finite,
        "forward_velocity_mean": float(np.mean(fwd_vels)),
        "forward_velocity_std": float(np.std(fwd_vels)),
        "returns": returns,
        "lengths": lengths,
        "frame_dir": str(frame_dir),
    }
    print(f"  MEAN return={result['return_mean']:.2f} +/- {result['return_std']:.2f}, "
          f"fwd_vel={result['forward_velocity_mean']:.4f} +/- {result['forward_velocity_std']:.4f}")
    return result


def main() -> None:
    all_results = {}
    for domain, task in TASKS:
        result = validate_one(domain, task)
        all_results[f"{domain}-{task}"] = result

    with open(OUT_DIR / "summary.json", "w") as f:
        json.dump(all_results, f, indent=2)

    print("\n\n=== SUMMARY ===")
    for key, r in all_results.items():
        print(f"{key:20s} return={r['return_mean']:7.2f}+/-{r['return_std']:6.2f}  "
              f"fwd_vel={r['forward_velocity_mean']:7.4f}+/-{r['forward_velocity_std']:6.4f}  "
              f"finite={r['all_actions_finite']}")

    for domain in ("walker", "quadruped"):
        v_walk = all_results[f"{domain}-walk"]["forward_velocity_mean"]
        v_run = all_results[f"{domain}-run"]["forward_velocity_mean"]
        print(f"{domain}: v_run ({v_run:.4f}) > v_walk ({v_walk:.4f}) -> {v_run > v_walk}")


if __name__ == "__main__":
    main()
