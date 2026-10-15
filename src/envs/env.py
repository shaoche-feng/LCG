from __future__ import annotations
from typing import Any, Dict, Optional, Sequence, Tuple

import gymnasium
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
from torch import Tensor

from .dm_control_env import DMControlEnv, validate_task_names


def make_dm_control_env(
    num_envs: int,
    device: torch.device,
    domain_name: str,
    task_names: Sequence[str],
    size: int,
    camera_id: int,
    action_repeat: int,
    time_limit: Optional[float],
) -> TorchEnv:
    # Fail here, before spawning workers, on an invalid or unsupported task selection.
    task_names = validate_task_names(domain_name, task_names)

    def env_fn():
        return DMControlEnv(
            domain_name=domain_name,
            task_names=task_names,
            size=size,
            camera_id=camera_id,
            action_repeat=action_repeat,
            time_limit=time_limit,
        )

    # Spawn avoids inheriting the Trainer's initialized CUDA context (fork deadlocks).
    env = AsyncVectorEnv([env_fn for _ in range(num_envs)], context="spawn")
    env = TorchEnv(env, device, task_names)

    return env


class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device, task_names: Optional[Sequence[str]] = None) -> None:
        super().__init__(env)
        self.device = device
        # When set, each step returns info["task_rewards"] of shape (num_envs, num_tasks).
        self.task_names = None if task_names is None else tuple(task_names)
        self.num_envs = env.observation_space.shape[0]

        single_action_space = env.unwrapped.single_action_space
        assert isinstance(single_action_space, gymnasium.spaces.Box)
        self.action_dim = int(single_action_space.shape[0])
        self.action_low = torch.as_tensor(single_action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(single_action_space.high, dtype=torch.float32, device=device)

        b, h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        return self._to_tensor(obs), info

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        dead = np.logical_or(end, trunc)
        if dead.any():
            info["final_observation"] = self._to_tensor(np.stack(info["final_observation"][dead]))
        if self.task_names is not None:
            info["task_rewards"] = self._task_rewards(info, dead)
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        if self.task_names is not None and not torch.equal(info["task_rewards"][:, 0], rew):
            raise RuntimeError("First task reward label does not match the environment reward")
        return obs, rew, end, trunc, info

    def _task_rewards(self, info: Dict[str, Any], dead: np.ndarray) -> Tensor:
        # A finished sub-environment autoresets inside the vector env: the labels of its
        # last transition are then in final_info, and the top-level info is the reset's.
        rows = []
        for i in range(self.num_envs):
            if dead[i]:
                rows.append(info["final_info"][i]["task_rewards"])
            else:
                if not info["_task_rewards"][i]:
                    raise RuntimeError(f"Sub-environment {i} returned no task reward labels")
                rows.append(info["task_rewards"][i])
        info.pop("_task_rewards", None)
        task_rewards = np.stack(rows)
        if task_rewards.shape != (self.num_envs, len(self.task_names)):
            raise RuntimeError(f"Task reward labels of shape {task_rewards.shape} for tasks {self.task_names}")
        return torch.tensor(task_rewards, dtype=torch.float32, device=self.device)

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
