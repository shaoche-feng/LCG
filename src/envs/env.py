from __future__ import annotations
from typing import Any, Dict, List, Optional, Sequence, Tuple

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
    record_physical_state: bool = False,
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
            record_physical_state=record_physical_state,
        )

    # Spawn avoids inheriting the Trainer's initialized CUDA context (fork deadlocks).
    env = AsyncVectorEnv([env_fn for _ in range(num_envs)], context="spawn")
    env = TorchEnv(env, device, task_names, record_physical_state)

    return env


PhysicalState = Dict[str, Tensor]  # {"qpos": (nq,), "qvel": (nv,)} float64 CPU copies


class TorchEnv(gymnasium.Wrapper):
    def __init__(self, env: gymnasium.Env, device: torch.device, task_names: Optional[Sequence[str]] = None,
                 record_physical_state: bool = False) -> None:
        super().__init__(env)
        self.device = device
        # When set, each step returns info["task_rewards"] of shape (num_envs, num_tasks).
        self.task_names = None if task_names is None else tuple(task_names)
        self.num_envs = env.observation_space.shape[0]
        # When set (the sub-environments must record too), each step's info also holds
        # "physical_state": per env, the state of the observation the action was taken from (None
        # if unknown), and "final_physical_state": per finished env, in final_observation's order,
        # the state of its final observation. Analysis only: states stay on the CPU.
        self.record_physical_state = bool(record_physical_state)
        self._physical_states: List[Optional[PhysicalState]] = [None] * self.num_envs

        single_action_space = env.unwrapped.single_action_space
        assert isinstance(single_action_space, gymnasium.spaces.Box)
        self.action_dim = int(single_action_space.shape[0])
        self.action_low = torch.as_tensor(single_action_space.low, dtype=torch.float32, device=device)
        self.action_high = torch.as_tensor(single_action_space.high, dtype=torch.float32, device=device)

        b, h, w, c = env.observation_space.shape
        self.observation_space = gymnasium.spaces.Box(low=-1, high=1, shape=(b, c, h, w))

    def reset(self, *args, **kwargs) -> Tuple[Tensor, Dict[str, Any]]:
        obs, info = self.env.reset(*args, **kwargs)
        if self.record_physical_state:
            self._physical_states = [self._physical_state(info, i) for i in range(self.num_envs)]
            self._pop_physical_state(info)
        return self._to_tensor(obs), info

    def get_env_states(self) -> List[Dict[str, Any]]:
        """Each sub-environment's DMControlEnv.collection_state."""
        return list(self.env.get_attr("collection_state"))

    def set_env_states(self, states: List[Dict[str, Any]]) -> None:
        if len(states) != self.num_envs:
            raise ValueError(f"Got {len(states)} saved env states for {self.num_envs} envs")
        self.env.set_attr("collection_state", list(states))
        if self.record_physical_state:
            # The restored state is the current observation's (read only: no step, reset or draw).
            self._physical_states = [self._copy_physical_state(s) for s in self.env.get_attr("physical_state")]

    def physical_states(self) -> List[Optional[PhysicalState]]:
        """The state of each env's current observation (None if unknown or not recorded)."""
        return list(self._physical_states)

    def step(self, actions: Tensor) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        obs, rew, end, trunc, info = self.env.step(actions.cpu().numpy())
        dead = np.logical_or(end, trunc)
        if self.record_physical_state:
            # A finished env's own step info is final_info (its top-level info is the autoreset's,
            # whose state belongs to the new episode's first observation).
            info["final_physical_state"] = [self._physical_state(info["final_info"][i], None)
                                            for i in range(self.num_envs) if dead[i]]
            info["physical_state"] = self._physical_states
            self._physical_states = [self._physical_state(info, i) for i in range(self.num_envs)]
            self._pop_physical_state(info)
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

    @staticmethod
    def _physical_state(info: Dict[str, Any], i: Optional[int]) -> PhysicalState:
        """Env i's state from a vector info (i=None: a single env's info). A recording env always
        reports one, so a missing state is an error rather than a silent gap."""
        if i is None:
            qpos, qvel = info.get("qpos"), info.get("qvel")
        else:
            if not (info.get("_qpos", np.zeros(i + 1, bool))[i] and info.get("_qvel", np.zeros(i + 1, bool))[i]):
                raise RuntimeError(f"Sub-environment {i} returned no physical state")
            qpos, qvel = info["qpos"][i], info["qvel"][i]
        if qpos is None or qvel is None:
            raise RuntimeError("A recording environment returned no physical state")
        return TorchEnv._copy_physical_state(dict(qpos=qpos, qvel=qvel))

    @staticmethod
    def _copy_physical_state(state: Dict[str, np.ndarray]) -> PhysicalState:
        return {k: torch.tensor(np.asarray(state[k]), dtype=torch.float64) for k in ("qpos", "qvel")}  # copies

    @staticmethod
    def _pop_physical_state(info: Dict[str, Any]) -> None:
        for key in ("qpos", "qvel", "_qpos", "_qvel"):
            info.pop(key, None)

    def _to_tensor(self, x: Tensor) -> Tensor:
        if x.ndim == 4:
            return torch.tensor(x, device=self.device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
        elif x.dtype is np.dtype("bool"):
            return torch.tensor(x, dtype=torch.uint8, device=self.device)
        else:
            return torch.tensor(x, dtype=torch.float32, device=self.device)
