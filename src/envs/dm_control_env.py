from __future__ import annotations
from collections.abc import Sequence
import hashlib
import pickle
from typing import Any, Dict, Optional, Tuple

import gymnasium
from gymnasium.spaces import Box
import mujoco
import numpy as np

from dm_control import suite


# Task groups that can share one collected trajectory, each task's reward recorded on it.
# Evidence and exclusions: docs/DM_CONTROL_TASK_COMPATIBILITY.md (dm_control 1.0.44).
REWARD_LABEL_TASK_GROUPS = {
    "acrobot": [("swingup", "swingup_sparse")],
    "cartpole": [("balance", "balance_sparse"), ("swingup", "swingup_sparse")],
    "hopper": [("stand", "hop")],
    "humanoid": [("stand", "walk", "run", "run_pure_state")],
    "humanoid_CMU": [("stand", "walk", "run")],
    "walker": [("stand", "walk", "run")],
}

# Domains whose tasks end only at the time limit: no task overrides get_termination (only
# lqr does). Exploration imagines no true endings, so it accepts only these domains.
TIME_LIMIT_ONLY_DOMAINS = (
    "acrobot", "ball_in_cup", "cartpole", "cheetah", "dog", "finger", "fish", "hopper", "humanoid",
    "humanoid_CMU", "manipulator", "pendulum", "point_mass", "quadruped", "reacher", "stacker",
    "swimmer", "walker",
)


def validate_task_names(
    domain_name: str,
    task_names: Sequence[str],
) -> Tuple[str, ...]:
    """The first task drives collection; every task is reward-labeled on its physics."""
    if isinstance(task_names, str) or not isinstance(task_names, Sequence):
        raise ValueError(f"task_names must be a list of task names, got {task_names!r}")

    if not task_names or not all(isinstance(name, str) for name in task_names):
        raise ValueError(f"task_names must contain one or more strings, got {list(task_names)}")

    if len(set(task_names)) != len(task_names):
        raise ValueError(f"task_names contains duplicates: {list(task_names)}")

    available = {task for domain, task in suite.ALL_TASKS if domain == domain_name}
    unknown = set(task_names) - available
    if unknown:
        raise ValueError(f"Unknown tasks for {domain_name}: {sorted(unknown)}")

    if len(task_names) > 1:
        groups = REWARD_LABEL_TASK_GROUPS.get(domain_name, [])
        if not any(set(task_names) <= set(group) for group in groups):
            raise ValueError(
                f"Unsupported task combination for {domain_name}: {list(task_names)}; "
                f"verified groups: {groups or 'none'}"
            )
    return tuple(task_names)


class DMControlEnv(gymnasium.Env):
    """DM Control Suite adapter producing RGB observations.

    info["task_rewards"] holds each task's reward for the step, ordered as task_names.
    """

    metadata: Dict[str, Any] = {"render_modes": []}

    def __init__(
        self,
        domain_name: str,
        task_names: Sequence,
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
        self.task_names = validate_task_names(domain_name, task_names)
        self.size = size
        self.camera_id = camera_id
        self.action_repeat = action_repeat
        self._time_limit = time_limit

        self._dm_env = self._make_dm_env(self.task_names[0], seed)
        self._label_tasks = self._load_label_tasks()

        action_spec = self._dm_env.action_spec()
        self.action_dim: int = int(np.prod(action_spec.shape))
        self.action_low: np.ndarray = np.asarray(action_spec.minimum, dtype=np.float32)
        self.action_high: np.ndarray = np.asarray(action_spec.maximum, dtype=np.float32)

        self.action_space = Box(low=self.action_low, high=self.action_high, shape=action_spec.shape, dtype=np.float32)
        self.observation_space = Box(low=0, high=255, shape=(size, size, 3), dtype=np.uint8)

        self._action_dtype = action_spec.dtype

    def _make_dm_env(self, task_name: str, seed: Optional[int]):
        task_kwargs: Dict[str, Any] = {}
        if self._time_limit is not None:
            task_kwargs["time_limit"] = self._time_limit
        if seed is not None:
            task_kwargs["random"] = seed
        return suite.load(self.domain_name, task_name, task_kwargs=task_kwargs or None)

    def _load_label_tasks(self) -> tuple:
        """Load reward-label tasks compatible with the collection environment."""
        reference = self._dm_env
        tasks = []
        for name in self.task_names[1:]:
            # Only the task object is kept; a fixed seed avoids drawing OS entropy.
            env = self._make_dm_env(name, seed=0)
            same_model = pickle.dumps(env.physics.model.ptr) == pickle.dumps(reference.physics.model.ptr)
            checks = {
                "task class": type(env.task) is type(reference.task),
                "physics class": type(env.physics) is type(reference.physics),
                "MuJoCo model": same_model,
                "control timestep": env.control_timestep() == reference.control_timestep(),
                "time limit": env._step_limit == reference._step_limit,
                "action spec": env.action_spec() == reference.action_spec(),
            }
            for field, matches in checks.items():
                if not matches:
                    raise ValueError(
                        f"Task {self.domain_name}/{name} has a different {field} "
                        f"from collection task {self.task_names[0]}"
                    )
            tasks.append(env.task)
        return tuple(tasks)

    @property
    def collection_state(self) -> Dict[str, Any]:
        """Everything needed to continue this env exactly: the full MuJoCo data (including
        solver warm start, time and derived quantities), the task's attributes and RNG, and
        the episode step count. qpos/qvel alone would not reproduce the next steps."""
        env = self._dm_env
        return dict(
            model=self._model_digest(),
            data=pickle.dumps(env.physics.data.ptr),
            task=pickle.dumps(env.task.__dict__),
            step_count=env._step_count,
            reset_next_step=env._reset_next_step,
        )

    @collection_state.setter
    def collection_state(self, state: Dict[str, Any]) -> None:
        # A different model (other task, or one a reset edited) cannot be restored exactly.
        if state["model"] != self._model_digest():
            raise ValueError(f"Saved state of {self.domain_name}/{self.task_names[0]} belongs to a different MuJoCo model")
        env = self._dm_env
        mujoco.mj_copyData(env.physics.data.ptr, env.physics.model.ptr, pickle.loads(state["data"]))
        env.task.__dict__.update(pickle.loads(state["task"]))
        env._step_count = state["step_count"]
        env._reset_next_step = state["reset_next_step"]

    def _model_digest(self) -> str:
        return hashlib.sha256(pickle.dumps(self._dm_env.physics.model.ptr)).hexdigest()

    def _render(self) -> np.ndarray:
        return self._dm_env.physics.render(height=self.size, width=self.size, camera_id=self.camera_id)

    def reset(
        self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        # dm_control has no in-place RNG reseed hook, so a new seed rebuilds the task.
        if seed is not None:
            self._dm_env = self._make_dm_env(self.task_names[0], seed)
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
        label_rewards = [0.0] * len(self._label_tasks)
        terminated = False
        truncated = False
        discount = None

        for _ in range(self.action_repeat):
            timestep = self._dm_env.step(dm_action)
            total_reward += float(timestep.reward)
            # Same physics state at which dm_control computed timestep.reward.
            for i, task in enumerate(self._label_tasks):
                label_rewards[i] += float(task.get_reward(self._dm_env.physics))
            if timestep.last():
                discount = timestep.discount
                # DM Control convention: discount == 0.0 on true termination,
                # discount != 0.0 (typically 1.0) when the episode hit its time_limit.
                terminated = bool(discount == 0.0)
                truncated = not terminated
                break

        obs = self._render()
        task_rewards = np.array([total_reward, *label_rewards], dtype=np.float64)
        info: Dict[str, Any] = {"discount": discount, "task_rewards": task_rewards}
        return obs, total_reward, terminated, truncated, info

