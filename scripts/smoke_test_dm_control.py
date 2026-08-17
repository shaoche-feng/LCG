#! /usr/bin/env python
"""
Smoke test for the general DM Control Suite environment adapter (Stage 1-3).

Verifies, for a given domain/task pair, that the adapter:
  - loads via suite.load with configurable domain_name/task_name,
  - resets successfully,
  - reports the RGB observation shape/dtype,
  - reports the inferred action dimension and bounds (not hard-coded),
  - accepts valid random continuous actions within bounds,
  - steps for >= 100 environment steps,
  - returns finite rewards,
  - keeps returning correctly shaped observations, including across resets.

Usage:
    python scripts/smoke_test_dm_control.py
"""
import sys
from pathlib import Path

# NOTE: imported directly from the module file (not `from envs import ...`) because
# `src/envs/__init__.py` eagerly imports the full model/data/wandb stack via
# `world_model_env`, and the pinned wandb==0.17.0 is currently incompatible with
# numpy==2.2.5 in this environment (`np.float_` removed in NumPy 2.0). That is a
# pre-existing issue affecting the Atari path too (`from envs import make_atari_env`
# fails identically) and is out of scope for this stage. The adapter module itself
# (dm_control_env.py) has no dependency on that stack.
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src" / "envs"))

import numpy as np

from dm_control_env import make_dm_control_env


def run_smoke_test(
    domain_name: str,
    task_name: str,
    num_steps: int = 100,
    size: int = 64,
    camera_id: int = 0,
    action_repeat: int = 2,
) -> None:
    print(f"\n{'=' * 60}")
    print(f"Smoke test: domain_name={domain_name!r} task_name={task_name!r}")
    print(f"{'=' * 60}")

    env = make_dm_control_env(
        domain_name=domain_name,
        task_name=task_name,
        size=size,
        camera_id=camera_id,
        action_repeat=action_repeat,
        seed=0,
    )

    obs, info = env.reset(seed=0)
    print(f"Reset OK. Observation shape: {obs.shape}, dtype: {obs.dtype}")
    assert obs.shape == (size, size, 3), f"Unexpected obs shape {obs.shape}"
    assert obs.dtype == np.uint8, f"Unexpected obs dtype {obs.dtype}"

    action_dim = env.action_dim
    low, high = env.action_low, env.action_high
    print(f"Action dim: {action_dim}")
    print(f"Action bounds: low={low}, high={high}")
    assert action_dim > 0, "Action dimension must be positive"
    assert low.shape == (action_dim,) and high.shape == (action_dim,)

    rng = np.random.default_rng(0)
    rewards = []
    num_resets = 0

    for t in range(num_steps):
        action = rng.uniform(low=low, high=high).astype(np.float32)
        assert env.action_space.contains(action), f"Sampled action {action} not in action_space"

        obs, reward, terminated, truncated, info = env.step(action)

        assert obs.shape == (size, size, 3), f"Step {t}: unexpected obs shape {obs.shape}"
        assert obs.dtype == np.uint8, f"Step {t}: unexpected obs dtype {obs.dtype}"
        assert np.isfinite(reward), f"Step {t}: non-finite reward {reward}"
        rewards.append(reward)

        if terminated or truncated:
            num_resets += 1
            obs, info = env.reset()
            assert obs.shape == (size, size, 3), f"Post-reset obs shape {obs.shape}"
            assert obs.dtype == np.uint8, f"Post-reset obs dtype {obs.dtype}"

    print(
        f"Stepped {num_steps} steps OK ({num_resets} episode reset(s) along the way). "
        f"Reward stats: min={min(rewards):.4f} max={max(rewards):.4f} mean={np.mean(rewards):.4f}"
    )
    print("All checks passed.")


if __name__ == "__main__":
    run_smoke_test("cheetah", "run")
    run_smoke_test("hopper", "hop")  # action_dim=4, vs cheetah's 6 -> proves no hard-coded action dimension
