from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

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
    # Analysis only (never read by training): the simulator state of each frame, see
    # PHYSICAL_STATE_KEYS. None for episodes collected without recording (all older episodes).
    physical_state: Optional[Dict[str, torch.Tensor]] = None

    def __post_init__(self) -> None:
        if (self.task_rew is None) != (self.task_names is None):
            raise ValueError("Episode task_rew and task_names must be given together")
        if self.task_names is not None:
            self.task_names = tuple(self.task_names)
            if self.task_rew.shape != (len(self), len(self.task_names)):
                raise ValueError(f"Episode task_rew shape {tuple(self.task_rew.shape)} does not match "
                                 f"{len(self)} steps and tasks {self.task_names}")
        if self.physical_state is not None:
            check_physical_state(self.physical_state, len(self))

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
        return Episode(**d, info=merge_info(self.info, other.info), task_rew=task_rew, task_names=self.task_names,
                       physical_state=join_physical_states(self.physical_state, len(self), other.physical_state, len(other)))

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
        # Round, not truncate: frames are uint8 levels, and truncation lowered some by one on
        # every save, drifting the prefix of an extended episode.
        d = {k: v.add(1).div(2).mul(255).round().byte() if k == "obs" else v for k, v in self.__dict__.items()
             if not (k == "physical_state" and v is None)}  # unrecorded: the file is as before recording existed
        torch.save(d, path.with_suffix(".tmp"))
        path.with_suffix(".tmp").replace(path)


# Episode.physical_state, a dict of CPU tensors copied from MuJoCo's mjData (float64, as simulated):
#   qpos (T, nq), qvel (T, nv): the state at which obs[t] was rendered, i.e. before act[t];
#   recorded (T,) bool: False where that frame's state is unknown; its qpos/qvel rows are NaN;
#   final_qpos (nq,), final_qvel (nv,): the state of info["final_observation"] (after the last
#       action, before the vector env's autoreset); present only if the episode ended with it recorded.
# The state of the frame after an unfinished episode's last step is a milestone's successor state.
PHYSICAL_STATE_KEYS = ("qpos", "qvel", "recorded")
FINAL_PHYSICAL_STATE_KEYS = ("final_qpos", "final_qvel")


def check_physical_state(state: Dict[str, torch.Tensor], length: int) -> None:
    keys = set(state)
    final = keys & set(FINAL_PHYSICAL_STATE_KEYS)
    if not set(PHYSICAL_STATE_KEYS) <= keys or keys - set(PHYSICAL_STATE_KEYS) - final or final not in (
            set(), set(FINAL_PHYSICAL_STATE_KEYS)):
        raise ValueError(f"Episode physical_state has keys {sorted(keys)}; expected {list(PHYSICAL_STATE_KEYS)} "
                         f"and optionally both of {list(FINAL_PHYSICAL_STATE_KEYS)}")
    qpos, qvel, recorded = (state[k] for k in PHYSICAL_STATE_KEYS)
    if qpos.ndim != 2 or qvel.ndim != 2 or qpos.size(0) != length or qvel.size(0) != length:
        raise ValueError(f"Episode physical_state qpos {tuple(qpos.shape)} / qvel {tuple(qvel.shape)} "
                         f"do not have one row per step ({length})")
    if recorded.dtype is not torch.bool or recorded.shape != (length,):
        raise ValueError(f"Episode physical_state recorded must be a bool tensor of shape ({length},)")
    if qpos.dtype is not torch.float64 or qvel.dtype is not torch.float64:
        raise ValueError("Episode physical_state qpos/qvel must be float64 (MuJoCo's precision)")
    if not (qpos[~recorded].isnan().all() and qvel[~recorded].isnan().all()):
        raise ValueError("Episode physical_state rows not recorded must be NaN")
    if final and (state["final_qpos"].shape != qpos.shape[1:] or state["final_qvel"].shape != qvel.shape[1:]
                  or state["final_qpos"].dtype is not torch.float64 or state["final_qvel"].dtype is not torch.float64):
        raise ValueError("Episode physical_state final_qpos/final_qvel do not match qpos/qvel")


def stack_physical_states(states: List[Optional[Dict[str, torch.Tensor]]],
                          final: Optional[Dict[str, torch.Tensor]] = None) -> Optional[Dict[str, torch.Tensor]]:
    """Episode.physical_state of consecutive frames, from each frame's {qpos, qvel} (None: unknown)
    and the final state, if any. None if no state is known at all: there is nothing to size it by."""
    known = [s for s in states if s is not None] + ([final] if final is not None else [])
    if not known:
        return None
    nq, nv = known[0]["qpos"].numel(), known[0]["qvel"].numel()
    if any(s["qpos"].shape != (nq,) or s["qvel"].shape != (nv,) for s in known):
        raise ValueError("Physical states of one episode have different dimensions")
    nan = torch.full((nq + nv,), float("nan"), dtype=torch.float64)
    rows = [nan if s is None else torch.cat((s["qpos"], s["qvel"])).to(torch.float64) for s in states]
    stacked = torch.stack(rows) if rows else nan.new_empty(0, nq + nv)
    state = dict(qpos=stacked[:, :nq].contiguous(), qvel=stacked[:, nq:].contiguous(),
                 recorded=torch.tensor([s is not None for s in states], dtype=torch.bool))
    if final is not None:
        state.update(final_qpos=final["qpos"].to(torch.float64).clone(), final_qvel=final["qvel"].to(torch.float64).clone())
    return state


def join_physical_states(first: Optional[Dict[str, torch.Tensor]], first_len: int,
                         second: Optional[Dict[str, torch.Tensor]], second_len: int) -> Optional[Dict[str, torch.Tensor]]:
    """physical_state of an episode extended by a continuation. A part without states becomes
    explicit unrecorded (NaN) rows; mismatched dimensions are refused rather than misaligned."""
    if first is None and second is None:
        return None
    if first is None:
        first = missing_physical_state(first_len, second["qpos"].size(1), second["qvel"].size(1))
    if second is None:
        second = missing_physical_state(second_len, first["qpos"].size(1), first["qvel"].size(1))
    if first["qpos"].size(1) != second["qpos"].size(1) or first["qvel"].size(1) != second["qvel"].size(1):
        raise ValueError("Cannot extend an episode's physical states with ones of different dimensions")
    if any(k in first for k in FINAL_PHYSICAL_STATE_KEYS):
        raise ValueError("Cannot extend an episode whose physical states already end")
    state = {k: torch.cat((first[k], second[k]), dim=0) for k in PHYSICAL_STATE_KEYS}
    state.update({k: second[k] for k in FINAL_PHYSICAL_STATE_KEYS if k in second})
    return state


def missing_physical_state(length: int, nq: int, nv: int) -> Dict[str, torch.Tensor]:
    return dict(qpos=torch.full((length, nq), float("nan"), dtype=torch.float64),
                qvel=torch.full((length, nv), float("nan"), dtype=torch.float64),
                recorded=torch.zeros(length, dtype=torch.bool))


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
