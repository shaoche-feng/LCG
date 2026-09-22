from typing import Any, Dict, Generator, List, Optional, Union

import numpy as np
import torch

from .dataset import Dataset
from .segment import SegmentId

# Fixed, explicit component identifiers for deriving independent per-component RNG streams
# (see utils.derive_component_seed). NEVER use Python's randomized hash() for this -- it's
# salted per-process by default (PYTHONHASHSEED), so the SAME component name would derive a
# DIFFERENT seed on every run, defeating the entire point of a reproducible, component-local
# stream.
COMPONENT_SEED_ID = {
    "denoiser": 0,
    "rew_end_model": 1,
    "actor_critic": 2,
    # DrQActorCritic's own exploration-noise streams (models.drq_actor_critic) -- deliberately
    # separate from each other so consuming noise in one loop (e.g. real-env collection) can
    # never perturb another loop's (e.g. imagined-training's) future draws. See
    # DrQActorCritic's module docstring, RNG-isolation section.
    "drq_imagination_noise": 3,
    "drq_real_collection_noise": 4,
    "drq_eval_noise": 5,
}


class BatchSampler(torch.utils.data.Sampler):
    def __init__(
        self,
        dataset: Dataset,
        rank: int,
        world_size: int,
        batch_size: int,
        seq_length: int,
        sample_weights: Optional[List[float]] = None,
        can_sample_beyond_end: bool = False,
        rng: Optional[Union[np.random.Generator, np.random.SeedSequence, int]] = None,
    ) -> None:
        super().__init__(dataset)
        assert isinstance(dataset, Dataset)
        self.dataset = dataset
        self.rank = rank
        self.world_size = world_size
        self.sample_weights = sample_weights
        self.batch_size = batch_size
        self.seq_length = seq_length
        self.can_sample_beyond_end = can_sample_beyond_end
        # rng=None (the historical default) falls back to np.random.default_rng(None), which
        # seeds from OS entropy -- i.e. still an independent stream per BatchSampler instance,
        # just no longer the single process-global np.random state every caller used to share.
        # This is a real behavior change (see derive_component_seed's docstring for why it's
        # required), but for the *unseeded* case it is strictly a decorrelation improvement:
        # no existing call site relied on drawing from a SHARED stream with any other sampler.
        self._rng: np.random.Generator = (
            rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        )

    def __len__(self):
        raise NotImplementedError

    def __iter__(self) -> Generator[List[SegmentId], None, None]:
        while True:
            yield self.sample()

    def state_dict(self) -> Dict[str, Any]:
        return {"bit_generator_state": self._rng.bit_generator.state}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        self._rng.bit_generator.state = state_dict["bit_generator_state"]

    def sample(self) -> List[SegmentId]:
        num_episodes = self.dataset.num_episodes

        if (self.sample_weights is None) or num_episodes < len(self.sample_weights):
            weights = self.dataset.lengths / self.dataset.num_steps
        else:
            weights = self.sample_weights
            num_weights = len(self.sample_weights)
            assert all([0 <= x <= 1 for x in weights]) and sum(weights) == 1
            sizes = [
                num_episodes // num_weights + (num_episodes % num_weights) * (i == num_weights - 1)
                for i in range(num_weights)
            ]
            weights = [w / s for (w, s) in zip(weights, sizes) for _ in range(s)]

        episodes_partition = np.arange(self.rank, num_episodes, self.world_size)
        weights = np.array(weights[self.rank::self.world_size])
        episode_ids = self._rng.choice(episodes_partition, size=self.batch_size, replace=True, p=weights / weights.sum())
        timesteps = self._rng.integers(low=0, high=self.dataset.lengths[episode_ids])

        # padding allowed, both before start and after end
        if self.can_sample_beyond_end:
            starts = timesteps - self._rng.integers(0, self.seq_length, len(timesteps))
            stops = starts + self.seq_length

        # padding allowed only before start
        else:
            stops = np.minimum(
                self.dataset.lengths[episode_ids], timesteps + 1 + self._rng.integers(0, self.seq_length, len(timesteps))
            )
            starts = stops - self.seq_length

        return [SegmentId(*x) for x in zip(episode_ids, starts, stops)]
