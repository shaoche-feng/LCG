from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from . import coroutine
from data import Episode, Dataset
from envs import TorchEnv
from .env_loop import make_env_loop
from utils import Logs


COLLECTION_MODES = ("uniform", "policy")


class ActionSource:
    """Stands in for the policy in make_env_loop. In uniform mode it samples actions
    uniformly within the environment's bounds and never evaluates the policy. Both modes
    share one env loop, so a mode switch continues the episodes in progress."""

    def __init__(self, env: TorchEnv, model: nn.Module) -> None:
        self.env = env
        self.model = model
        self.frame_stack = model.frame_stack
        self.uniform = False

    def predict_act_value(self, obs: torch.Tensor):
        if not self.uniform:
            return self.model.predict_act_value(obs)
        # Placeholders: the collector never reads policy outputs or values.
        return obs.new_zeros(obs.size(0), 0), obs.new_zeros(obs.size(0))

    def sample_action(self, logits_act: torch.Tensor) -> torch.Tensor:
        if not self.uniform:
            return self.model.sample_action(logits_act)
        unit = torch.rand(logits_act.size(0), self.env.action_dim, device=logits_act.device)
        return self.env.action_low + unit * (self.env.action_high - self.env.action_low)


@coroutine
def make_collector(
    env: TorchEnv,
    model: nn.Module,
    dataset: Dataset,
    epsilon: float = 0.0,
    reset_every_collect: bool = False,
    verbose: bool = True,
) -> Generator[Logs, int, None]:
    num_envs = env.num_envs
    task_names = getattr(env, "task_names", None)  # None: env records no task rewards
    action_source = ActionSource(env, model)

    env_loop, buffer, episode_ids, dead = (None,) * 4
    num_steps, num_episodes, to_log, pbar = (None,) * 4

    def setup_new_collect():
        nonlocal num_steps, num_episodes, buffer, to_log, pbar
        num_steps = 0
        num_episodes = 0
        buffer = defaultdict(list)
        to_log = []
        pbar = tqdm(
            total=num_to_collect.total,
            unit=num_to_collect.unit,
            desc=f"Collect {dataset.name}",
            disable=not verbose,
        )

    def reset():
        nonlocal env_loop, episode_ids, dead
        env_loop = make_env_loop(env, action_source, epsilon)
        episode_ids = defaultdict(lambda: None)
        dead = [None] * num_envs

    num_to_collect = yield
    setup_new_collect()
    reset()
    action_source.uniform = num_to_collect.mode == "uniform"

    while True:
        with torch.no_grad():
            all_obs, act, rew, end, trunc, *_, [infos] = env_loop.send(1)

        num_steps += num_envs
        pbar.update(num_envs if num_to_collect.steps is not None else 0)

        for i in range(num_envs):
            transition = [all_obs[i], act[i], rew[i], end[i], trunc[i]]
            if task_names is not None:
                transition.append(infos["task_rewards"][i : i + 1])
            buffer[i].append(transition)
            dead[i] = (end[i] + trunc[i]).clip(max=1).item()

        num_episodes += sum(dead)

        can_stop = num_to_collect.can_stop(num_steps, num_episodes)

        count_dead = 0
        for i in range(num_envs):
            # Store incomplete episodes only when reset_every_collect is set to False (train)
            add_to_dataset = dead[i] or (can_stop and not reset_every_collect)
            if add_to_dataset:
                info = {"final_observation": infos["final_observation"][count_dead]} if dead[i] else {}
                columns = [torch.cat(x, dim=0) for x in zip(*buffer[i])]
                task_rew = columns[5] if task_names is not None else None
                ep = Episode(*columns[:5], info, task_rew=task_rew, task_names=task_names).to("cpu")
                if episode_ids[i] is not None:
                    ep = dataset.load_episode(episode_ids[i]) + ep
                episode_ids[i] = dataset.add_episode(ep, episode_id=episode_ids[i])

            if dead[i]:
                to_log.append(
                    {
                        f"{dataset.name}/episode_id": episode_ids[i],
                        **ep.compute_metrics(),
                    }
                )
                buffer[i] = []
                episode_ids[i] = None
                pbar.update(1 if num_to_collect.episodes is not None else 0)

            count_dead += dead[i]

        if can_stop:
            pbar.close()
            metrics = {
                "num_steps": dataset.num_steps,
                "counts/end_0": dataset.counts_end[0],
                "counts/end_1": dataset.counts_end[1],
            }
            to_log.append({f"{dataset.name}/{k}": v for k, v in metrics.items()})
            num_to_collect = yield to_log
            setup_new_collect()
            if reset_every_collect:
                reset()
            action_source.uniform = num_to_collect.mode == "uniform"


@dataclass
class NumToCollect:
    steps: Optional[int] = None
    episodes: Optional[int] = None
    mode: str = "policy"  # one of COLLECTION_MODES

    def __post_init__(self) -> None:
        assert (self.steps is None) != (self.episodes is None)
        if self.mode not in COLLECTION_MODES:
            raise ValueError(f"Collection mode must be one of {COLLECTION_MODES}, got {self.mode!r}")

    def can_stop(self, num_steps: int, num_episodes: int) -> bool:
        return num_steps >= self.steps if self.steps is not None else num_episodes >= self.episodes

    @property
    def unit(self) -> str:
        return "steps" if self.steps is not None else "eps"

    @property
    def total(self) -> int:
        return self.steps if self.steps is not None else self.episodes
