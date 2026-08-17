from __future__ import annotations
from typing import Any, Dict, Optional, Tuple

import ale_py
import gymnasium
from gymnasium.vector import AsyncVectorEnv
import numpy as np
import torch
from torch import Tensor

from .atari_preprocessing import AtariPreprocessing
from .dm_control_env import DMControlEnv


def make_atari_env(
    id: str,
    num_envs: int,
    device: torch.device,
    done_on_life_loss: bool,
    size: int,
    max_episode_steps: Optional[int],
) -> TorchEnv:
    def env_fn():
        env = gymnasium.make(
            id,
            full_action_space=False,
            frameskip=1,
            render_mode="rgb_array",
            max_episode_steps=max_episode_steps,
        )
        env = AtariPreprocessing(
            env=env,
            noop_max=30,
            frame_skip=4,
            screen_size=size,
        )
        return env

    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])

    # The AsyncVectorEnv resets the env on termination, which means that it will
    # reset the environment if we use the default AtariPreprocessing of gymnasium with
    # terminate_on_life_loss=True (which means that we will only see the first life).
    # Hence a separate wrapper for life_loss, coming after the AsyncVectorEnv.

    if done_on_life_loss:
        env = DoneOnLifeLoss(env)

    env = TorchEnv(env, device)

    return env


def make_dm_control_env(
    num_envs: int,
    device: torch.device,
    domain_name: str,
    task_name: str,
    size: int,
    camera_id: int,
    action_repeat: int,
    time_limit: Optional[float],
) -> TorchEnv:
    def env_fn():
        return DMControlEnv(
            domain_name=domain_name,
            task_name=task_name,
            size=size,
            camera_id=camera_id,
            action_repeat=action_repeat,
            time_limit=time_limit,
        )

    env = AsyncVectorEnv([env_fn for _ in range(num_envs)])
    env = TorchEnv(env, device)

    return env


class DoneOnLifeLoss(gymnasium.Wrapper):
    def __init__(self, env: AsyncVectorEnv) -> None:
        super().__init__(env)

    def step(self, actions: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions)
        life_loss = info["life_loss"]
        if life_loss.any():
            end[life_loss] = True
            info["final_observation"] = obs
        return obs, rew, end, trunc, info


class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device) -> None:
        super().__init__(env)
        self.device = device
        self.num_envs = env.observation_space.shape[0]

        single_action_space = env.unwrapped.single_action_space
        self.is_discrete = isinstance(single_action_space, gymnasium.spaces.Discrete)
        if self.is_discrete:
            self.num_actions = single_action_space.n
            self.action_dim = None
            self.action_low = None
            self.action_high = None
        else:
            assert isinstance(single_action_space, gymnasium.spaces.Box)
            self.num_actions = None
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
        obs, rew, end, trunc = (self._to_tensor(x) for x in (obs, rew, end, trunc))
        return obs, rew, end, trunc, info

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
