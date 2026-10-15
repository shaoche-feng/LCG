from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, Optional

import torch


@dataclass
class SegmentId:
    episode_id: int
    start: int
    stop: int


@dataclass
class Segment:
    obs: torch.FloatTensor
    act: torch.Tensor  # float32 (T, action_dim), in environment action bounds
    rew: torch.FloatTensor
    end: torch.ByteTensor
    trunc: torch.ByteTensor
    mask_padding: torch.BoolTensor
    info: Dict[str, Any]
    id: SegmentId
    task_rew: Optional[torch.FloatTensor] = None  # (T, num_tasks), padded like rew

    @property
    def effective_size(self):
        return self.mask_padding.sum().item()
