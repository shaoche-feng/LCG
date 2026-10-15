from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch


@dataclass
class Episode:
    obs: torch.FloatTensor
    act: torch.Tensor  # float32 (T, action_dim), in environment action bounds
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    info: Dict[str, Any]
    # Per-transition rewards of each recorded task, (T, len(task_names)), aligned with rew;
    # both None for episodes recorded without task labels.
    task_rew: Optional[torch.FloatTensor] = None
    task_names: Optional[Tuple[str, ...]] = None

    def __post_init__(self) -> None:
        if (self.task_rew is None) != (self.task_names is None):
            raise ValueError("Episode task_rew and task_names must be given together")
        if self.task_names is not None:
            self.task_names = tuple(self.task_names)
            if self.task_rew.shape != (len(self), len(self.task_names)):
                raise ValueError(f"Episode task_rew shape {tuple(self.task_rew.shape)} does not match "
                                 f"{len(self)} steps and tasks {self.task_names}")

    def __len__(self) -> int:
        return self.obs.size(0)

    def __add__(self, other: Episode) -> Episode:
        assert self.dead.sum() == 0
        if self.task_names != other.task_names:
            raise ValueError(f"Cannot extend an episode labeled {self.task_names} with one labeled {other.task_names}")
        d = {k: torch.cat((self.__dict__[k], other.__dict__[k]), dim=0) for k in ("obs", "act", "rew", "end", "trunc")}
        task_rew = None
        if self.task_rew is not None:
            task_rew = torch.cat((self.task_rew, other.task_rew), dim=0)
        return Episode(**d, info=merge_info(self.info, other.info), task_rew=task_rew, task_names=self.task_names)

    def to(self, device) -> Episode:
        return Episode(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in self.__dict__.items()})

    @property
    def dead(self) -> torch.ByteTensor:
        return (self.end + self.trunc).clip(max=1)

    def compute_metrics(self) -> Dict[str, Any]:
        return {"length": len(self), "return": self.rew.sum().item()}

    @classmethod
    def load(cls, path: Path, map_location: Optional[torch.device] = None) -> Episode:
        return cls(
            **{
                k: v.div(255).mul(2).sub(1) if k == "obs" else v
                for k, v in torch.load(Path(path), map_location=map_location).items()
            }
        )

    def save(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        d = {k: v.add(1).div(2).mul(255).byte() if k == "obs" else v for k, v in self.__dict__.items()}
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)


def merge_info(info_a, info_b):
    keys_a = set(info_a)
    keys_b = set(info_b)
    intersection = keys_a & keys_b
    info = {
        **{k: info_a[k] for k in keys_a if k not in intersection},
        **{k: info_b[k] for k in keys_b if k not in intersection},
        **{k: torch.cat((info_a[k], info_b[k]), dim=0) for k in intersection},
    }
    return info
