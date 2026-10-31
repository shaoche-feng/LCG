from collections import Counter
import multiprocessing as mp
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Set

import numpy as np
import torch

from .episode import Episode
from .segment import Segment, SegmentId
from .utils import make_segment
from utils import file_sha256, StateDictMixin


class Dataset(StateDictMixin, torch.utils.data.Dataset):
    """Replay of episodes. Each saved episode file is an immutable version named by episode
    id and length (`<id>_<length>.pt`): extending an unfinished episode writes a new version
    and keeps the old one, so a saved index (this state dict) always reads exactly the
    transitions it recorded, whatever was collected later."""

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
        self.counter_end = None
        self.task_names = None  # task labels shared by every episode, None if unlabeled
        self.versioned = True  # False for datasets saved before versions (`<id>.pt` files)
        self.file_sha256 = None  # checksum of each episode's current version file, taken when written
        self.generation = 0  # clear(keep_files=True) count: new episode files go to a new folder

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
    def counts_end(self) -> List[int]:
        return [self.counter_end[e] for e in [0, 1]]

    def _reset(self) -> None:
        self.num_episodes = 0
        self.num_steps = 0
        self.start_idx = np.array([], dtype=np.int64)
        self.lengths = np.array([], dtype=np.int64)
        self.counter_end = Counter()
        self.task_names = None
        self.versioned = True
        self.file_sha256 = []
        self._cache.clear()

    def clear(self, keep_files: bool = False) -> None:
        """Empty the index. keep_files: leave the episode files (a saved index may still need
        them; prune_versions removes them later) and write new ones to a new generation folder."""
        self.assert_not_static()
        generation = self.generation + 1 if keep_files else 0
        if not keep_files and self._directory.is_dir():
            shutil.rmtree(self._directory)
        self._reset()
        self.generation = generation

    def load_episode(self, episode_id: int) -> Episode:
        if self._cache_in_ram and episode_id in self._cache:
            episode = self._cache[episode_id]
        else:
            # Replay tensors live on CPU. Older episodes can carry CUDA final-
            # observation metadata even when their observations/actions are CPU.
            episode = Episode.load(self._get_episode_path(episode_id), map_location=torch.device("cpu"))
            if self._cache_in_ram:
                self._cache[episode_id] = episode
        return episode

    def add_episode(self, episode: Episode, *, episode_id: Optional[int] = None) -> int:
        self.assert_not_static()
        episode = episode.to("cpu")

        if self.num_episodes == 0:
            self.task_names = episode.task_names
        elif episode.task_names != self.task_names:
            raise ValueError(f"Episode task labels {episode.task_names} do not match {self.name}'s {self.task_names}")

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
            self.counter_end.subtract(old_episode.end.tolist())

        self.counter_end.update(episode.end.tolist())

        sha256 = None
        if self._save_on_disk:
            path = self._get_episode_path(episode_id)
            episode.save(path)
            sha256 = file_sha256(path)  # once per immutable version
        if episode_id == len(self.file_sha256):
            self.file_sha256.append(sha256)
        else:
            self.file_sha256[episode_id] = sha256

        if self._cache_in_ram:
            self._cache[episode_id] = episode

        return episode_id

    def _get_episode_path(self, episode_id: int) -> Path:
        """Path of the episode version this index refers to."""
        n = 3  # number of hierarchies
        powers = np.arange(n)
        subfolders = np.floor((episode_id % 10 ** (1 + powers)) / 10**powers) * 10**powers
        subfolders = [int(x) for x in subfolders[::-1]]
        subfolders = "/".join([f"{x:0{n - i}d}" for i, x in enumerate(subfolders)])
        if not self.versioned:
            return self._directory / subfolders / f"{episode_id}.pt"
        if self.generation:
            subfolders = f"generation_{self.generation}/{subfolders}"
        return self._directory / subfolders / f"{episode_id}_{self.lengths[episode_id]}.pt"

    @property
    def directory(self) -> Path:
        return self._directory

    def episode_files(self) -> List[Path]:
        """The episode versions this index refers to, in episode order."""
        return [self._get_episode_path(i) for i in range(self.num_episodes)]

    def verify_episode_files(self) -> None:
        """Validate this checkpoint's immutable replay once on resume, bypassing the cache."""
        if len(self.file_sha256) != self.num_episodes or any(s is None for s in self.file_sha256):
            raise ValueError(f"{self.name}: checkpoint has no complete replay checksums; cannot resume faithfully")
        for episode_id, (path, expected) in enumerate(zip(self.episode_files(), self.file_sha256)):
            try:
                actual = file_sha256(path)
            except FileNotFoundError as error:
                raise FileNotFoundError(f"{self.name}: checkpoint episode {path} is missing") from error
            except OSError as error:
                raise ValueError(f"{self.name}: missing or unreadable checkpoint episode {path}: {error}") from error
            if actual != expected:
                raise ValueError(f"{self.name}: checkpoint episode {path} does not match its checksum")
            try:
                episode = Episode.load(path, map_location=torch.device("cpu"))
                if len(episode) != self.lengths[episode_id] or episode.task_names != self.task_names:
                    raise ValueError("episode length or task labels differ from the checkpoint index")
            except Exception as error:
                raise ValueError(f"{self.name}: unreadable checkpoint episode {path}: {error}") from error

    def prune_versions(self, keep: Set[Path]) -> int:
        """Delete episode versions (and interrupted writes) not in keep; returns the count."""
        self.assert_not_static()
        if not self.versioned:
            raise ValueError(f"{self.name} has unversioned episode files; nothing can be pruned safely")
        keep = {Path(p).resolve() for p in keep}
        removed = 0
        for pattern in ("*.pt", "*.tmp"):
            for path in self._directory.rglob(pattern):
                if path != self._default_path and path.resolve() not in keep:
                    path.unlink()
                    removed += 1
        return removed

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        state_dict = {k: v for k, v in state_dict.items() if k != "counter_rew"}
        state_dict.setdefault("task_names", None)  # datasets saved before task labels existed
        state_dict.setdefault("versioned", False)  # datasets saved before episode versions
        state_dict.setdefault("file_sha256", [None] * state_dict["num_episodes"])
        state_dict.setdefault("generation", 0)  # datasets saved before generations
        super().load_state_dict(state_dict)
        self._cache.clear()

    def assert_not_static(self) -> None:
        assert not self.is_static, "Trying to modify a static dataset."

    def save_to_default_path(self) -> None:
        self._default_path.parent.mkdir(exist_ok=True, parents=True)
        torch.save(self.state_dict(), self._default_path)

    def load_from_default_path(self) -> None:
        if self._default_path.is_file():
            # Trusted, locally generated DIAMOND dataset state (numpy arrays for start_idx/lengths,
            # Counter object for counter_end), not just tensor weights -- weights_only=False
            # is required for PyTorch >=2.6, whose default changed to weights_only=True.
            self.load_state_dict(torch.load(self._default_path, weights_only=False))
