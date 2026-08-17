from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import gymnasium
from gymnasium.spaces import Box
import numpy as np

from dm_control import suite


class DMControlEnv(gymnasium.Env):
    """General DM Control Suite adapter producing RGB observations.

    Wraps `dm_control.suite.load(domain_name, task_name)` and exposes the
    standard gymnasium.Env `reset`/`step` interface expected downstream
    (mirrors the role of `AtariPreprocessing` for the Atari path). Action
    dimension and bounds are inferred from `env.action_spec()`, never
    hard-coded.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        domain_name: str,
        task_name: str,
        size: int = 64,
        camera_id: int = 0,
        action_repeat: int = 2,
        time_limit: Optional[float] = None,
        seed: Optional[int] = None,
    ) -> None:
        super().__init__()
        assert size > 0
        assert action_repeat > 0

        self.domain_name = domain_name
        self.task_name = task_name
        self.size = size
        self.camera_id = camera_id
        self.action_repeat = action_repeat
        self._time_limit = time_limit

        self._dm_env = self._make_dm_env(seed)

        action_spec = self._dm_env.action_spec()
        self.action_dim: int = int(np.prod(action_spec.shape))
        self.action_low: np.ndarray = np.asarray(action_spec.minimum, dtype=np.float32)
        self.action_high: np.ndarray = np.asarray(action_spec.maximum, dtype=np.float32)

        self.action_space = Box(low=self.action_low, high=self.action_high, shape=action_spec.shape, dtype=np.float32)
        self.observation_space = Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)

        self._action_dtype = action_spec.dtype

    def _make_dm_env(self, seed: Optional[int]):
        task_kwargs: Dict[str, Any] = {}
        if self._time_limit is not None:
            task_kwargs["time_limit"] = self._time_limit
        if seed is not None:
            task_kwargs["random"] = seed
        return suite.load(self.domain_name, self.task_name, task_kwargs=task_kwargs or None)

    def _render(self) -> np.ndarray:
        return self._dm_env.physics.render(height=self.size, width=self.size, camera_id=self.camera_id)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # dm_control has no in-place RNG reseed hook, so a new seed rebuilds the task.
        if seed is not None:
            self._dm_env = self._make_dm_env(seed)
        self._dm_env.reset()
        obs = self._render()
        return obs, {}

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action = np.asarray(action, dtype=np.float32).reshape(self.action_space.shape)
        tol = 1e-6
        assert np.all(action >= self.action_low - tol) and np.all(action <= self.action_high + tol), (
            f"action {action} outside bounds [{self.action_low}, {self.action_high}]"
        )
        dm_action = action.astype(self._action_dtype)

        total_reward = 0.0
        terminated = False
        truncated = False
        discount = None

        for _ in range(self.action_repeat):
            timestep = self._dm_env.step(dm_action)
            total_reward += float(timestep.reward)
            if timestep.last():
                discount = timestep.discount
                # DM Control convention: discount == 0.0 on true termination,
                # discount != 0.0 (typically 1.0) when the episode hit its time_limit.
                terminated = bool(discount == 0.0)
                truncated = not terminated
                break

        obs = self._render()
        info: Dict[str, Any] = {"discount": discount}
        return obs, total_reward, terminated, truncated, info


def make_dm_control_env(
    domain_name: str,
    task_name: str,
    size: int = 64,
    camera_id: int = 0,
    action_repeat: int = 2,
    time_limit: Optional[float] = None,
    seed: Optional[int] = None,
) -> DMControlEnv:
    return DMControlEnv(
        domain_name=domain_name,
        task_name=task_name,
        size=size,
        camera_id=camera_id,
        action_repeat=action_repeat,
        time_limit=time_limit,
        seed=seed,
    )
