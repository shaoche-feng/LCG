from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import torch

from .segment import SegmentId


@dataclass
class Batch:
    obs: torch.ByteTensor
    act: torch.Tensor  # float32 (B, T, action_dim), in environment action bounds
    rew: torch.FloatTensor
    end: torch.LongTensor
    trunc: torch.LongTensor
    mask_padding: torch.BoolTensor
    info: List[Dict[str, Any]]
    segment_ids: List[SegmentId]
    task_rew: Optional[torch.FloatTensor] = None  # (B, T, num_tasks), None for unlabeled data

    def pin_memory(self) -> Batch:
        return Batch(**{k: v.pin_memory() if torch.is_tensor(v) else v for k, v in self.__dict__.items()})

    def to(self, device: torch.device) -> Batch:
        return Batch(**{k: v.to(device) if torch.is_tensor(v) else v for k, v in self.__dict__.items()})
