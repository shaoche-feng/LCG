from collections import defaultdict
from dataclasses import dataclass
from typing import Generator, Optional

import torch
import torch.nn as nn
from tqdm import tqdm

from . import coroutine
from data import Episode, Dataset
from envs import TorchEnv
from .env_loop import EnvResetSeedState, make_env_loop
from utils import Logs


@coroutine
def make_collector(
    env: TorchEnv,
    model: nn.Module,
    dataset: Dataset,
    epsilon: float = 0.0,
    reset_every_collect: bool = False,
    verbose: bool = True,
    reset_seed_state: Optional[EnvResetSeedState] = None,
    flush_before_reset: bool = False,
) -> Generator[Logs, int, None]:
    """reset_seed_state=None, flush_before_reset=False (every existing call site) preserve the
    exact prior behavior. Passing both together (see Trainer.__init__'s train collector) turns
    on a checkpoint-safe collection mode: at the end of EVERY `.send()` call (i.e. every epoch's
    collection batch, where Trainer.save_checkpoint() runs), whatever's been buffered for each
    env since its last flush is ALWAYS persisted (flush_before_reset overrides
    reset_every_collect's usual "discard the in-progress episode" behavior for that env) and
    treated as closed (episode_ids cleared, buffer cleared, logged) even if the real env hasn't
    actually terminated there, and then the whole collector (env_loop, hence env.reset(), hence
    the model's frame-stack/LSTM state) is rebuilt from scratch via a DETERMINISTIC seed drawn
    from reset_seed_state -- so a checkpoint/resume cycle and an uninterrupted run reach that
    boundary in an IDENTICAL, freshly-reset state, and no collector-local Python state (buffer,
    episode_ids, dead) or hidden env_loop/env state needs to be separately checkpointed at all.
    The cost: episodes are now cut off at collection-batch boundaries rather than running to
    their natural length whenever that's shorter than steps_per_epoch -- see
    Trainer.__init__'s docstring/comment for this tradeoff. See tests/coroutines/
    test_collector_resume_fidelity.py for the uninterrupted-vs-resumed equivalence this buys."""
    num_envs = env.num_envs

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
        env_loop = make_env_loop(env, model, epsilon, reset_seed_state=reset_seed_state)
        episode_ids = defaultdict(lambda: None)
        dead = [None] * num_envs

    num_to_collect = yield
    setup_new_collect()
    reset()

    while True:
        with torch.no_grad():
            all_obs, act, rew, end, trunc, *_, [infos] = env_loop.send(1)

        num_steps += num_envs
        pbar.update(num_envs if num_to_collect.steps is not None else 0)

        for i, (o, a, r, e, t) in enumerate(zip(all_obs, act, rew, end, trunc)):
            buffer[i].append((o, a, r, e, t))
            dead[i] = (e + t).clip(max=1).item()

        num_episodes += sum(dead)

        can_stop = num_to_collect.can_stop(num_steps, num_episodes)

        count_dead = 0
        for i in range(num_envs):
            # Store incomplete episodes when reset_every_collect is False (the original train
            # behavior: episodes grow seamlessly across collection calls) OR when
            # flush_before_reset is True (the new checkpoint-safe train mode: always persist
            # what's been collected so far, then close out below regardless of reset_every_collect).
            add_to_dataset = dead[i] or (can_stop and (flush_before_reset or not reset_every_collect))
            if add_to_dataset:
                info = {"final_observation": infos["final_observation"][count_dead]} if dead[i] else {}
                ep = Episode(*(torch.cat(x, dim=0) for x in zip(*buffer[i])), info).to("cpu")
                if episode_ids[i] is not None:
                    ep = dataset.load_episode(episode_ids[i]) + ep
                episode_ids[i] = dataset.add_episode(ep, episode_id=episode_ids[i])

            # Natural termination always closes the episode; flush_before_reset additionally
            # forces a close at the collection-batch boundary even if the real env didn't
            # actually terminate there (that env's NEXT collected step, after reset() below,
            # starts a genuinely new episode rather than silently continuing the old one under
            # a stale episode_id whose underlying env/frame-stack state no longer matches it).
            closing = dead[i] or (can_stop and flush_before_reset)
            if closing:
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
                "counts/rew_-1": dataset.counts_rew[0],
                "counts/rew__0": dataset.counts_rew[1],
                "counts/rew_+1": dataset.counts_rew[2],
                "counts/end_0": dataset.counts_end[0],
                "counts/end_1": dataset.counts_end[1],
            }
            to_log.append({f"{dataset.name}/{k}": v for k, v in metrics.items()})
            num_to_collect = yield to_log
            setup_new_collect()
            if reset_every_collect:
                reset()


@dataclass
class NumToCollect:
    steps: Optional[int] = None
    episodes: Optional[int] = None

    def __post_init__(self) -> None:
        assert (self.steps is None) != (self.episodes is None)

    def can_stop(self, num_steps: int, num_episodes: int) -> bool:
        return num_steps >= self.steps if self.steps is not None else num_episodes >= self.episodes

    @property
    def unit(self) -> str:
        return "steps" if self.steps is not None else "eps"

    @property
    def total(self) -> int:
        return self.steps if self.steps is not None else self.episodes
