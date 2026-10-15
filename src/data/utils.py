import math
from typing import Generator, List

import torch
import torch.nn.functional as F

from .batch import Batch
from .episode import Episode
from .segment import Segment, SegmentId


def collate_segments_to_batch(segments: List[Segment]) -> Batch:
    attrs = ("obs", "act", "rew", "end", "trunc", "mask_padding")
    stack = (torch.stack([getattr(s, x) for s in segments]) for x in attrs)
    num_labeled = sum(s.task_rew is not None for s in segments)
    if num_labeled == 0:
        task_rew = None
    elif num_labeled == len(segments):
        task_rew = torch.stack([s.task_rew for s in segments])
    else:
        raise ValueError("Cannot batch segments with and without task reward labels")
    return Batch(*stack, [s.info for s in segments], [s.id for s in segments], task_rew)


def collate_segments_without_rewards(segments: List[Segment]) -> Batch:
    """Batch for reward-free learners: task labels dropped, rew a zero placeholder, and no
    episode metadata (no exploration learner reads batch.info)."""
    batch = collate_segments_to_batch(segments)
    batch.rew = torch.zeros_like(batch.rew)
    batch.task_rew = None
    batch.info = [{} for _ in segments]  # new dicts: the episodes' own info is untouched
    return batch


def make_segment(episode: Episode, segment_id: SegmentId, should_pad: bool = True) -> Segment:
    assert segment_id.start < len(episode) and segment_id.stop > 0 and segment_id.start < segment_id.stop
    pad_len_right = max(0, segment_id.stop - len(episode))
    pad_len_left = max(0, -segment_id.start)
    assert pad_len_right == pad_len_left == 0 or should_pad

    def pad(x):
        right = F.pad(x, [0 for _ in range(2 * x.ndim - 1)] + [pad_len_right]) if pad_len_right > 0 else x
        return F.pad(right, [0 for _ in range(2 * x.ndim - 2)] + [pad_len_left, 0]) if pad_len_left > 0 else right

    start = max(0, segment_id.start)
    stop = min(len(episode), segment_id.stop)
    mask_padding = torch.cat((torch.zeros(pad_len_left), torch.ones(stop - start), torch.zeros(pad_len_right))).bool()

    return Segment(
        pad(episode.obs[start:stop]),
        pad(episode.act[start:stop]),
        pad(episode.rew[start:stop]),
        pad(episode.end[start:stop]),
        pad(episode.trunc[start:stop]),
        mask_padding,
        info=episode.info,
        id=SegmentId(segment_id.episode_id, start, stop),
        task_rew=None if episode.task_rew is None else pad(episode.task_rew[start:stop]),
    )


class DatasetTraverser:
    def __init__(self, dataset, batch_num_samples: int, chunk_size: int, collate_fn=collate_segments_to_batch) -> None:
        self.dataset = dataset
        self.batch_num_samples = batch_num_samples
        self.chunk_size = chunk_size
        self.collate_fn = collate_fn

    def __len__(self):
        return math.ceil(
            sum(
                [
                    math.ceil(self.dataset.lengths[episode_id] / self.chunk_size)
                    - int(self.dataset.lengths[episode_id] % self.chunk_size == 1)
                    for episode_id in range(self.dataset.num_episodes)
                ]
            )
            / self.batch_num_samples
        )

    def __iter__(self) -> Generator[Batch, None, None]:
        chunks = []
        for episode_id in range(self.dataset.num_episodes):
            episode = self.dataset.load_episode(episode_id)
            chunks.extend(
                make_segment(
                    episode,
                    SegmentId(episode_id, start=i * self.chunk_size, stop=(i + 1) * self.chunk_size),
                    should_pad=True,
                )
                for i in range(math.ceil(len(episode) / self.chunk_size))
            )
            if chunks[-1].effective_size < 2:
                chunks.pop()

            while len(chunks) >= self.batch_num_samples:
                yield self.collate_fn(chunks[: self.batch_num_samples])
                chunks = chunks[self.batch_num_samples :]

        if len(chunks) > 0:
            yield self.collate_fn(chunks)
