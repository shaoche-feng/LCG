from collections import Counter
import hashlib
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment
from utils import StateDictMixin


class Dataset(StateDictMixin, torch.utils.data.Dataset):
    def __init__(
        self,
        directory: Path,
        name: Optional[str] = None,
        cache_in_ram: bool = False,
        use_manager: bool = False,
        save_on_disk: bool = True,
    ) -> None:
        super().__init__()

        # State
        self.is_static = False
        self.num_episodes = None
        self.num_steps = None
        self.start_idx = None
        self.lengths = None
        self.counter_rew = None
        self.counter_end = None

        self._directory = Path(directory).expanduser()
        self._name = name if name is not None else self._directory.stem
        self._cache_in_ram = cache_in_ram
        self._save_on_disk = save_on_disk
        self._default_path = self._directory / "info.pt"
        self._cache = mp.Manager().dict() if use_manager else {}
        self._reset()

    def __len__(self) -> int:
        return self.num_steps

    def __getitem__(self, segment_id: SegmentId) -> Segment:
        episode = self.load_episode(segment_id.episode_id)
        return make_segment(episode, segment_id, should_pad=True)

    def __str__(self) -> str:
        return f"{self.name}: {self.num_episodes} episodes, {self.num_steps} steps."

    @property
    def name(self) -> str:
        return self._name

    @property
    def counts_rew(self) -> List[int]:
        return [self.counter_rew[r] for r in [-1, 0, 1]]

    @property
    def counts_end(self) -> List[int]:
        return [self.counter_end[e] for e in [0, 1]]

    def _reset(self) -> None:
        self.num_episodes = 0
        self.num_steps = 0
        self.start_idx = np.array([], dtype=np.int64)
        self.lengths = np.array([], dtype=np.int64)
        self.counter_rew = Counter()
        self.counter_end = Counter()
        self._cache.clear()

    def clear(self) -> None:
        self.assert_not_static()
        if self._directory.is_dir():
            shutil.rmtree(self._directory)
        self._reset()

    def load_episode(self, episode_id: int) -> Episode:
        if self._cache_in_ram and episode_id in self._cache:
            episode = self._cache[episode_id]
        else:
            episode = Episode.load(self._get_episode_path(episode_id))
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def add_episode(self, episode: Episode, *, episode_id: Optional[int] = None) -> int:
        self.assert_not_static()
        episode = episode.to("cpu")

        if episode_id is None:
            episode_id = self.num_episodes
            self.start_idx = np.concatenate((self.start_idx, np.array([self.num_steps])))
            self.lengths = np.concatenate((self.lengths, np.array([len(episode)])))
            self.num_steps += len(episode)
            self.num_episodes += 1

        else:
            assert episode_id < self.num_episodes
            old_episode = self.load_episode(episode_id)
            incr_num_steps = len(episode) - len(old_episode)
            self.lengths[episode_id] = len(episode)
            self.start_idx[episode_id + 1 :] += incr_num_steps
            self.num_steps += incr_num_steps
            self.counter_rew.subtract(old_episode.rew.sign().tolist())
            self.counter_end.subtract(old_episode.end.tolist())

        self.counter_rew.update(episode.rew.sign().tolist())
        self.counter_end.update(episode.end.tolist())

        if self._save_on_disk:
            episode.save(self._get_episode_path(episode_id))

        if self._cache_in_ram:
            self._cache[episode_id] = episode

        return episode_id

    def _get_episode_path(self, episode_id: int) -> Path:
        n = 3  # number of hierarchies
        powers = np.arange(n)
        subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
        subfolders = [int(x) for x in subfolders[::-1]]
        subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
        return self._directory / subfolders / f"{episode_id}.pt"

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self._cache.clear()

    def assert_not_static(self) -> None:
        assert not self.is_static, "Trying to modify a static dataset."

    def save_to_default_path(self) -> None:
        self._default_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), self._default_path)

    def compute_manifest(self) -> Dict[str, Any]:
        """Episode-level integrity manifest: per-episode sha256 of the saved file, cross-
        checked against this dataset's own sampler-visible bookkeeping length. Exists to
        catch a real bug found during the detach_value_trunk experiment: an in-progress
        episode's on-disk file can grow (add_episode() with an explicit episode_id saves the
        file unconditionally) without info.pt's `lengths` entry for that episode ever being
        re-persisted before a process stop, silently leaving BatchSampler unable to draw the
        tail of that episode even though the bytes are sitting right there on disk. Purely
        diagnostic/provenance -- computing this never mutates the dataset."""
        manifest: Dict[str, Any] = {
            "num_episodes": int(self.num_episodes),
            "num_steps": int(self.num_steps),
            "episodes": {},
            "length_mismatches": [],
        }
        for episode_id in range(self.num_episodes):
            path = self._get_episode_path(episode_id)
            if not path.is_file():
                continue
            sha256 = hashlib.sha256(path.read_bytes()).hexdigest()
            bookkeeping_length = int(self.lengths[episode_id])
            actual_length = len(Episode.load(path))
            manifest["episodes"][episode_id] = {
                "sha256": sha256,
                "bookkeeping_length": bookkeeping_length,
                "actual_length": actual_length,
            }
            if actual_length != bookkeeping_length:
                manifest["length_mismatches"].append(episode_id)
        return manifest

    def load_from_default_path(self) -> None:
        if self._default_path.is_file():
            # Trusted, locally generated DIAMOND dataset state (numpy arrays for start_idx/lengths,
            # Counter objects for counter_rew/counter_end), not just tensor weights -- weights_only=False
            # is required for PyTorch >=2.6, whose default changed to weights_only=True.
            self.load_state_dict(torch.load(self._default_path, weights_only=False))
