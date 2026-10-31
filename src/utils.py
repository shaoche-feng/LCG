
from argparse import Namespace
from collections import OrderedDict
from contextlib import contextmanager, nullcontext
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import hashlib
from importlib.metadata import version as package_version
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple, Union

from omegaconf import DictConfig, OmegaConf, open_dict
import numpy as np
import torch
import torch.distributed as dist
from torch import Tensor
from torch.optim.lr_scheduler import LambdaLR
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
import wandb


Logs = List[Dict[str, float]]
LossAndLogs = Tuple[Tensor, Dict[str, Any]]


class StateDictMixin:
    def _init_fields(self) -> None:
        def has_sd(x: str) -> bool:
            return callable(getattr(x, "state_dict", None)) and callable(getattr(x, "load_state_dict", None))

        self._all_fields = {k for k in vars(self) if not k.startswith("_")}
        self._fields_sd = {k for k in self._all_fields if has_sd(getattr(self, k))}

    def _get_field(self, k: str) -> Any:
        return getattr(self, k).state_dict() if k in self._fields_sd else getattr(self, k)

    def _set_field(self, k: str, v: Any) -> None:
        getattr(self, k).load_state_dict(v) if k in self._fields_sd else setattr(self, k, v)

    def state_dict(self) -> Dict[str, Any]:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        return {k: self._get_field(k) for k in self._all_fields}

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        if not hasattr(self, "_all_fields"):
            self._init_fields()
        assert set(list(state_dict.keys())) == self._all_fields
        for k, v in state_dict.items():
            self._set_field(k, v)


@dataclass
class CommonTools(StateDictMixin):
    denoiser: Any
    rew_end_model: Any
    actor_critic: Any

    def get(self, name: str) -> Any:
        return getattr(self, name)

    def set(self, name: str, value: Any):
        return setattr(self, name, value)


def broadcast_if_needed(*args):
    objects = list(args)
    if dist.is_initialized():
        dist.broadcast_object_list(objects, src=0) 
        # the list `objects` now contains the version of rank 0
    return objects


def build_ddp_wrapper(**modules_dict: Dict[str, nn.Module]) -> Namespace:
    return Namespace(**{name: DDP(module) for name, module in modules_dict.items()})


def compute_classification_metrics(confusion_matrix: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    num_classes = confusion_matrix.size(0)
    precision = torch.zeros(num_classes)
    recall = torch.zeros(num_classes)
    f1_score = torch.zeros(num_classes)

    for i in range(num_classes):
        true_positive = confusion_matrix[i, i].item()
        false_positive = confusion_matrix[:, i].sum().item() - true_positive
        false_negative = confusion_matrix[i, :].sum().item() - true_positive

        precision[i] = true_positive / (true_positive + false_positive) if (true_positive + false_positive) != 0 else 0
        recall[i] = true_positive / (true_positive + false_negative) if (true_positive + false_negative) != 0 else 0
        f1_score[i] = (
            2 * (precision[i] * recall[i]) / (precision[i] + recall[i]) if (precision[i] + recall[i]) != 0 else 0
        )

    return precision, recall, f1_score


def configure_opt(model: nn.Module, lr: float, weight_decay: float, eps: float, *blacklist_module_names: str) -> AdamW:
    """Credits to https://github.com/karpathy/minGPT"""
    # separate out all parameters to those that will and won't experience regularizing weight decay
    decay = set()
    no_decay = set()
    whitelist_weight_modules = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.LSTMCell, nn.LSTM)
    blacklist_weight_modules = (nn.LayerNorm, nn.Embedding, nn.GroupNorm)
    for mn, m in model.named_modules():
        for pn, p in m.named_parameters():
            fpn = "%s.%s" % (mn, pn) if mn else pn  # full param name
            if any([fpn.startswith(module_name) for module_name in blacklist_module_names]):
                no_decay.add(fpn)
            elif "bias" in pn:
                # all biases will not be decayed
                no_decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, whitelist_weight_modules):
                # weights of whitelist modules will be weight decayed
                decay.add(fpn)
            elif (pn.endswith("weight") or pn.startswith("weight_")) and isinstance(m, blacklist_weight_modules):
                # weights of blacklist modules will NOT be weight decayed
                no_decay.add(fpn)

    # validate that we considered every parameter
    param_dict = {pn: p for pn, p in model.named_parameters()}
    inter_params = decay & no_decay
    union_params = decay | no_decay
    assert len(inter_params) == 0, f"parameters {str(inter_params)} made it into both decay/no_decay sets!"
    assert (
        len(param_dict.keys() - union_params) == 0
    ), f"parameters {str(param_dict.keys() - union_params)} were not separated into either decay/no_decay set!"

    # create the pytorch optimizer object
    optim_groups = [
        {"params": [param_dict[pn] for pn in sorted(list(decay))], "weight_decay": weight_decay},
        {"params": [param_dict[pn] for pn in sorted(list(no_decay))], "weight_decay": 0.0},
    ]
    optimizer = AdamW(optim_groups, lr=lr, eps=eps)
    return optimizer


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def extract_state_dict(state_dict: OrderedDict, module_name: str) -> OrderedDict:
    return OrderedDict({k.split(".", 1)[1]: v for k, v in state_dict.items() if k.startswith(module_name)})


def get_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> LambdaLR:
    def lr_lambda(current_step: int):
        return 1 if current_step >= num_warmup_steps else current_step / max(1, num_warmup_steps)

    return LambdaLR(opt, lr_lambda, last_epoch=-1)


def init_lstm(model: nn.Module) -> None:
    for name, p in model.named_parameters():
        if "weight_ih" in name:
            nn.init.xavier_uniform_(p.data)
        elif "weight_hh" in name:
            nn.init.orthogonal_(p.data)
        elif "bias_ih" in name:
            p.data.fill_(0)
            # Set forget-gate bias to 1
            n = p.size(0)
            p.data[(n // 4) : (n // 2)].fill_(1)
        elif "bias_hh" in name:
            p.data.fill_(0)


def get_path_agent_ckpt(path_ckpt_dir: Union[str, Path], epoch: int, num_zeros: int = 5) -> Path:
    d = Path(path_ckpt_dir) / "agent_versions"
    if epoch >= 0:
        return d / f"agent_epoch_{epoch:0{num_zeros}d}.pt"
    else:
        all_ = sorted(list(d.iterdir()))
        assert len(all_) >= -epoch
        return all_[epoch]


def keep_agent_copies_every(
    agent_sd: Dict[str, Any],
    epoch: int,
    path_ckpt_dir: Path,
    every: int,
    num_to_keep: Optional[int],
) -> None:
    assert every > 0
    assert num_to_keep is None or num_to_keep > 0
    get_path = partial(get_path_agent_ckpt, path_ckpt_dir)
    get_path(0).parent.mkdir(parents=False, exist_ok=True)

    # Save agent
    save_atomically(agent_sd, get_path(epoch))

    # Clean oldest
    if (num_to_keep is not None) and (epoch % every == 0):
        get_path(max(0, epoch - num_to_keep * every)).unlink(missing_ok=True)

    # Clean previous
    if (epoch - 1) % every != 0:
        get_path(max(0, epoch - 1)).unlink(missing_ok=True)


def process_confusion_matrices_if_any_and_compute_classification_metrics(logs: Logs) -> None:
    cm = [x.pop("confusion_matrix") for x in logs if "confusion_matrix" in x]
    if len(cm) > 0:
        confusion_matrices = {k: sum([d[k] for d in cm]) for k in cm[0]}  # accumulate confusion matrices
        metrics = {}
        for key, confusion_matrix in confusion_matrices.items():
            precision, recall, f1_score = compute_classification_metrics(confusion_matrix)
            metrics.update(
                {
                    **{f"classification_metrics/{key}_precision_class_{i}": v for i, v in enumerate(precision)},
                    **{f"classification_metrics/{key}_recall_class_{i}": v for i, v in enumerate(recall)},
                    **{f"classification_metrics/{key}_f1_score_class_{i}": v for i, v in enumerate(f1_score)},
                }
            )

        logs.append(metrics)  # Append the obtained metrics to logs (in place)


def save_info_for_import_script(epoch: int, run_name: str, path_ckpt_dir: Path) -> None:
    with (path_ckpt_dir / "info_for_import_script.json").open("w") as f:
        json.dump({"epoch": epoch, "name": run_name}, f)


def save_atomically(obj: Any, path: Path):
    """Write aside, then replace: an interrupted save leaves the previous file intact."""
    tmp = path.with_suffix(".tmp")
    torch.save(obj, tmp)
    tmp.replace(path)


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    random.seed(seed)


def skip_if_run_is_over(func: Callable) -> Callable:
    def inner(*args, **kwargs):
        path_run_is_over = Path(".run_is_over")
        if not path_run_is_over.is_file():
            func(*args, **kwargs)
            path_run_is_over.touch()
        else:
            print(f"Run is marked as finished. To unmark, remove '{str(path_run_is_over)}'.")

    return inner


def try_until_no_except(func: Callable) -> None:
    while True:
        try:
            func()
        except KeyboardInterrupt:
            break
        except Exception:
            continue
        else:
            break


def wandb_log(logs: Logs, epoch: int):
    for d in logs:
        wandb.log({"epoch": epoch, **d})


# The local W&B row mirror (wandb_history.jsonl): one JSON record per line, {"step", "epoch",
# "segment", "values"}, appended and fsynced before the checkpoint that commits it. A checkpoint
# stores the file's byte length at its commit: records before that offset are committed, records
# after it come from an interrupted segment. Read line by line (the file can be large).

def parse_history_record(line: bytes, path: Path, number: int, offset: int) -> Dict:
    try:
        record = json.loads(line)
    except ValueError as error:
        raise ValueError(f"{path}: record {number} (byte offset {offset}) is not valid JSON ({error}); "
                         "not resuming over a corrupt W&B mirror") from None
    if not (isinstance(record, dict) and isinstance(record.get("step"), int) and isinstance(record.get("values"), dict)):
        raise ValueError(f"{path}: record {number} (byte offset {offset}) is not a W&B mirror record")
    return record


def history_records(path: Path, end: Optional[int] = None):
    """Yield every complete record in file order, up to byte `end` (a record boundary; None: the
    whole file). Corruption raises ValueError with its location."""
    offset = 0
    with open(path, "rb") as f:
        for number, line in enumerate(f, start=1):
            if end is not None and offset >= end:
                break
            if not line.endswith(b"\n"):
                raise ValueError(f"{path}: record {number} (byte offset {offset}) is incomplete")
            yield parse_history_record(line, path, number, offset)
            offset += len(line)
    if end is not None and offset != end:
        raise ValueError(f"{path}: the checkpoint committed {end} bytes but the complete records end at byte {offset}")


def repair_history(path: Path, committed_bytes: Optional[int]) -> Optional[str]:
    """Make the mirror safe to append to and backfill from, as recover_metrics_log does for the
    downstream log: an interrupted last record is kept byte-for-byte in <name>.interrupted-<n> and
    truncated away; a whole last record that only lost its newline gets it. Raises ValueError on
    corruption before the last line, or if records the checkpoint committed are missing, cut or do
    not end on a record boundary. Returns what was done (None: nothing). Idempotent."""
    if not path.is_file():
        if committed_bytes:
            raise ValueError(f"{path} is missing but the checkpoint committed {committed_bytes} bytes of W&B rows")
        return None
    boundary, offset, tail, number = committed_bytes in (None, 0), 0, None, 0
    with open(path, "rb") as f:
        for number, line in enumerate(f, start=1):
            if not line.endswith(b"\n"):
                tail = line  # only the last line can lack its newline
                break
            parse_history_record(line, path, number, offset)
            offset += len(line)
            boundary = boundary or offset == committed_bytes
    if committed_bytes is not None and (offset < committed_bytes or not boundary):
        raise ValueError(f"{path}: the checkpoint committed {committed_bytes} bytes of W&B rows, but the complete "
                         f"records end at byte {offset}{'' if offset < committed_bytes else ' with no record boundary there'}; "
                         "committed rows are missing or damaged, not resuming over them")
    if tail is None:
        return None
    try:
        parse_history_record(tail, path, number, offset)
    except ValueError:
        pass
    else:
        with open(path, "ab") as f:
            f.write(b"\n")
            f.flush()
            os.fsync(f.fileno())
        return f"completed the last record of {path.name} (its newline was missing)"
    n = 1
    while True:
        recovery = path.with_name(f"{path.name}.interrupted-{n}")
        try:
            with open(recovery, "xb") as f:  # exclusive: never overwrites an earlier recovery
                f.write(tail)
                f.flush()
                os.fsync(f.fileno())
            break
        except FileExistsError:
            n += 1
    os.truncate(path, offset)
    return f"kept the interrupted last record of {path.name} ({len(tail)} bytes) in {recovery.name} and removed it"


# ---------------------------------------------------------------- terminal output

PROGRESS_MODES = ("bar", "none")
_progress = dict(mode="bar")
_terminal = dict(blank=True)  # the last stdout line was blank (or nothing was written yet)


def configure_progress(mode: str = "bar") -> None:
    """bar: tqdm progress bars; none: no progress output."""
    if mode not in PROGRESS_MODES:
        raise ValueError(f"logging.progress must be one of {PROGRESS_MODES}, got {mode!r}")
    _progress["mode"] = mode


def say(message: str = "") -> None:
    """Immediately flushed stdout line(s). Written through tqdm, so an active bar is cleared
    and redrawn below the message instead of being overwritten."""
    import sys
    from tqdm import tqdm
    tqdm.write(message, file=sys.stdout)
    sys.stdout.flush()
    _terminal["blank"] = message.endswith("\n") or message == ""


def phase() -> None:
    """A blank line before an epoch heading or a block outside the epochs (never two in a row)."""
    if not _terminal["blank"]:
        say()


PROGRESS_UNITS = dict(steps="step", updates="update", trans="trans", ep="ep")  # unit -> singular


def rate_text(fmt: Dict[str, Any], unit: str) -> str:
    """tqdm's rate with a space before the unit, inverted below one per second like tqdm's own
    (e.g. "295.55 steps/s", "3.85 s/update", "? updates/s" before the first measurement)."""
    rate = fmt.get("rate")
    if rate is None and fmt.get("elapsed"):
        rate = (fmt["n"] - fmt.get("initial", 0)) / fmt["elapsed"]
    if not rate:
        return f"? {unit}/s"
    if rate < 1:
        return f"{1 / rate:.2f} s/{PROGRESS_UNITS.get(unit, unit)}"
    return f"{rate:.2f} {unit}/s"


class Progress:
    """A tqdm bar: phase, percentage, completed/total, elapsed and remaining time, and rate
    (rate_text). No postfix (losses and diagnostics go to W&B). Written to stdout, so it stays
    ordered with say(); each bar is closed (left complete on its own line) when its phase ends.
    Display only: phase timings in W&B are measured separately."""

    def __init__(self, iterable=None, total: Optional[int] = None, desc: str = "", unit: str = "steps",
                 disable: bool = False, initial: int = 0) -> None:
        import sys
        from tqdm import tqdm

        class Bar(tqdm):
            @property
            def format_dict(self):
                fmt = super().format_dict
                fmt["rate_text"] = rate_text(fmt, unit)
                return fmt

        if total is None and hasattr(iterable, "__len__"):
            total = len(iterable)
        disable = disable or _progress["mode"] == "none"
        bar_format = ("{desc}: {percentage:3.0f}%|{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_text}]"
                      if total else "{desc}: {n_fmt} " + unit + " [{elapsed}, {rate_text}]")
        self._bar = Bar(iterable, total=total, desc=desc, unit=unit, bar_format=bar_format, file=sys.stdout,
                        leave=True, dynamic_ncols=True, disable=disable, initial=initial)  # initial: resumed count
        if not disable:
            _terminal["blank"] = False

    def __iter__(self):
        return iter(self._bar)

    def update(self, n: int = 1) -> None:
        self._bar.update(n)

    def close(self) -> None:
        self._bar.close()


def normalize_saved_config(cfg: DictConfig) -> DictConfig:
    """Drop only obsolete schema fields; reject unsupported learning paths.

    Works on a complete Trainer config or its ``agent`` subtree. The input and
    historical YAML files are left intact. Environment-derived metadata is still
    supplied by AgentConfig, exactly as in the original PMPO implementation.
    """
    cfg = deepcopy(cfg)
    agent = cfg.get("agent", cfg)
    if agent.actor_critic.get("_target_") != "models.pmpo_beta.PMPOBetaConfig":
        raise ValueError("Only saved PMPO-Beta configurations are supported")
    # A retained PMPO field may refer to an obsolete loss field. Materialize
    # those values on the copy before deleting their interpolation targets.
    OmegaConf.resolve(cfg)

    with open_dict(cfg):
        if "training" in cfg:
            if cfg.training.get("model_free", False) is not False:
                raise ValueError("Saved training.model_free must be false for PMPO")
            cfg.training.pop("model_free", None)
        if "env" in cfg:
            # Validate both before deleting train.type: test.type may refer to it.
            for split in ("train", "test"):
                if cfg.env[split].get("type", "dm_control") != "dm_control":
                    raise ValueError("Only DM Control environments are supported")
            for split in ("test", "train"):
                if "type" in cfg.env[split]:
                    del cfg.env[split]["type"]
        if "collection" in cfg:
            first = cfg.collection.train.first_epoch
            for name in ("max", "threshold_rew"):
                if name in first:
                    value = first[name]
                    if not (name == "max" and value is None) and (type(value) is not int or value < 0):
                        raise ValueError(f"Invalid obsolete collection.train.first_epoch.{name}")
            first.pop("max", None)
            first.pop("threshold_rew", None)
        if "actor_critic" in cfg and "agent" in cfg:
            # These settings never participated in PMPO's losses/optimizers.
            if "actor_critic_loss" in cfg.actor_critic:
                loss = cfg.actor_critic.actor_critic_loss
                fields = {"_target_", "backup_every", "gamma", "lambda_",
                          "weight_value_loss", "weight_entropy_loss"}
                if not isinstance(loss, DictConfig) or set(loss) != fields:
                    raise ValueError("Unrecognized obsolete actor_critic_loss fields")
                if loss._target_ != "models.actor_critic.ActorCriticLossConfig":
                    raise ValueError("Unrecognized obsolete actor_critic_loss target")
                if type(loss.backup_every) is not int or loss.backup_every < 1:
                    raise ValueError("Invalid obsolete actor_critic_loss.backup_every")
                for name in fields - {"_target_", "backup_every"}:
                    value = loss[name]
                    if type(value) not in (int, float) or not np.isfinite(value) or value < 0:
                        raise ValueError(f"Invalid obsolete actor_critic_loss.{name}")
                    if name in ("gamma", "lambda_") and value > 1:
                        raise ValueError(f"Invalid obsolete actor_critic_loss.{name}")
                del cfg.actor_critic["actor_critic_loss"]
            if "optimizer" in cfg.actor_critic:
                optimizer = cfg.actor_critic.optimizer
                if not isinstance(optimizer, DictConfig) or set(optimizer) != {"lr", "weight_decay", "eps"}:
                    raise ValueError("Unrecognized obsolete actor_critic.optimizer fields")
                for name, value in optimizer.items():
                    if type(value) not in (int, float) or not np.isfinite(value):
                        raise ValueError(f"Invalid obsolete actor_critic.optimizer.{name}")
                    if value < 0 or (name != "weight_decay" and value == 0):
                        raise ValueError(f"Invalid obsolete actor_critic.optimizer.{name}")
                del cfg.actor_critic["optimizer"]

        for component in (agent, agent.denoiser.inner_model, agent.rew_end_model, agent.actor_critic):
            if component.get("num_actions") is not None:
                raise ValueError("Discrete actions are unsupported")
            component.pop("num_actions", None)
        if "continuous_reward" in agent.denoiser.inner_model:
            raise ValueError("continuous_reward was never an InnerModelConfig field")
        for component in (agent, agent.rew_end_model, agent.actor_critic):
            if component.get("continuous_reward", True) is not True:
                raise ValueError("Reward-sign classification is unsupported")
            component.pop("continuous_reward", None)
        # The inherited policy LSTM width was unused by feedforward PMPO.
        if "lstm_dim" in agent.actor_critic:
            value = agent.actor_critic.lstm_dim
            if type(value) is not int or value < 1:
                raise ValueError("Invalid obsolete policy lstm_dim")
        agent.actor_critic.pop("lstm_dim", None)
    return cfg


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# Packages whose versions must match for a faithful resume.
CHECKED_PACKAGES = ("torch", "numpy", "mujoco", "dm_control", "gymnasium")


def source_identity(root_dir: Path) -> Dict[str, Any]:
    def git(*args):
        try:
            return subprocess.run(["git", "-C", str(root_dir), *args], capture_output=True, text=True,
                                  check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return None

    commit = git("rev-parse", "HEAD")
    return dict(git_commit=commit, git_dirty=None if commit is None else bool(git("status", "--porcelain")),
                packages={name: package_version(name) for name in CHECKED_PACKAGES})


def check_same_packages(source: Dict[str, Any]) -> None:
    current = {name: package_version(name) for name in CHECKED_PACKAGES}
    changed = {k: (v, current[k]) for k, v in source["packages"].items() if current.get(k) != v}
    if changed:
        raise ValueError(f"Cannot resume faithfully: package versions changed (saved, now): {changed}")


def rng_state() -> Dict[str, Any]:
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def set_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None:
        if not torch.cuda.is_available() or torch.cuda.device_count() != len(state["cuda"]):
            raise ValueError("Checkpoint CUDA RNG state needs the same number of visible GPUs")
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def setup_visible_cuda_devices(devices: Union[str, int, List[int]]) -> None:
    if isinstance(devices, str):
        if devices == "cpu":
            devices = []
        else:
            assert devices == "all"
            return
    elif isinstance(devices, int):
        devices = [devices]
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, devices))


class RegionTimer:
    """Accumulated time of marked regions (monitoring only). Only the outermost region counts,
    so work inside nested regions is counted once. On CUDA each region records two events on
    the current stream (no host synchronization); seconds() waits for the last one. On CPU,
    wall-clock time. reset() starts a new accumulation."""

    def __init__(self, device: torch.device) -> None:
        self.cuda = device.type == "cuda"
        self.reset()

    def reset(self) -> None:
        self._events, self._seconds, self._depth = [], 0.0, 0

    def _mark(self) -> Any:
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    @contextmanager
    def region(self):
        self._depth += 1
        start = self._mark() if self._depth == 1 else None
        try:
            yield
        finally:
            self._depth -= 1
            if start is not None:
                end = self._mark()
                if self.cuda:
                    self._events.append((start, end))
                else:
                    self._seconds += end - start

    def seconds(self) -> float:
        if self._events:
            self._events[-1][1].synchronize()
            self._seconds += sum(start.elapsed_time(end) for start, end in self._events) / 1000
            self._events = []
        return self._seconds


def timed(timer: Optional[RegionTimer]):
    """timer.region(), or nothing without a timer."""
    return nullcontext() if timer is None else timer.region()


# ---------------------------------------------------------------------- compute and rendering devices
# (formerly src/device_report.py) Compute and rendering devices for the startup banners.
# Physical GPU numbers are nvidia-smi's indexes, matched by UUID; a process-local device such as
# cuda:0 is never taken as a physical index (CUDA_VISIBLE_DEVICES renumbers devices). Rendering
# runs in the env worker processes; for EGL a subprocess with this process's environment asks the
# EGL device that dm_control selects (MUJOCO_EGL_DEVICE_ID) for its CUDA device and that device
# for its UUID, without creating a GL or CUDA context. Whatever cannot be verified is reported as
# configured. Nothing here runs at import time: the probes run only when called.

# Run in a subprocess (same environment as the env workers): EGL device -> CUDA ordinal
# (EGL_NV_device_cuda, process-local numbering) -> UUID (CUDA driver API, no context).
EGL_PROBE = r"""
import ctypes, json, os
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
from dm_control._render.pyopengl import egl_ext as EGL
out = {}
devices = EGL.eglQueryDevicesEXT()
out["num_devices"] = len(devices)
index = int(os.environ["MUJOCO_EGL_DEVICE_ID"])
attrib = ctypes.CFUNCTYPE(EGL.EGLBoolean, EGL.EGLDeviceEXT, EGL.EGLint, ctypes.POINTER(ctypes.c_ssize_t))(
    EGL.eglGetProcAddress("eglQueryDeviceAttribEXT"))
ordinal = ctypes.c_ssize_t(-1)
if 0 <= index < len(devices) and attrib(devices[index], 0x323A, ctypes.byref(ordinal)) and ordinal.value >= 0:  # EGL_CUDA_DEVICE_NV
    cuda = ctypes.CDLL("libcuda.so.1")
    dev, raw = ctypes.c_int(), (ctypes.c_ubyte * 16)()
    if cuda.cuInit(0) == 0 and cuda.cuDeviceGet(ctypes.byref(dev), int(ordinal.value)) == 0 and cuda.cuDeviceGetUuid(raw, dev) == 0:
        h = bytes(raw).hex()
        out["uuid"] = f"GPU-{h[:8]}-{h[8:12]}-{h[12:16]}-{h[16:20]}-{h[20:]}"
print(json.dumps(out))
"""


def nvidia_smi_gpus() -> Optional[List[Dict[str, str]]]:
    """Physical GPUs (nvidia-smi index, UUID, name), or None if nvidia-smi is unavailable."""
    try:
        out = subprocess.run(["nvidia-smi", "--query-gpu=index,uuid,name", "--format=csv,noheader"],
                             capture_output=True, text=True, timeout=20, check=True).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    gpus = []
    for line in out.strip().splitlines():
        index, uuid, name = [part.strip() for part in line.split(",", 2)]
        gpus.append(dict(index=index, uuid=uuid, name=name))
    return gpus


def cuda_uuid(device: torch.device) -> Optional[str]:
    uuid = getattr(torch.cuda.get_device_properties(device), "uuid", None)
    if uuid is None:
        return None
    uuid = str(uuid)
    return uuid if uuid.startswith("GPU-") else f"GPU-{uuid}"


def physical_index(uuid: Optional[str], gpus: Optional[List[Dict[str, str]]]) -> Optional[str]:
    matches = [g["index"] for g in gpus or [] if uuid is not None and g["uuid"].lower() == uuid.lower()]
    return matches[0] if len(matches) == 1 else None


def physical_gpu_index(device: torch.device) -> Optional[int]:
    """nvidia-smi index of this process's CUDA device (matched by UUID); None on CPU or if unverified."""
    if device.type != "cuda":
        return None
    index = physical_index(cuda_uuid(device), nvidia_smi_gpus())
    return None if index is None else int(index)


def compute_device_line(device: torch.device, name: Optional[str], uuid: Optional[str],
                        gpus: Optional[List[Dict[str, str]]]) -> str:
    if device.type != "cuda":
        return "Compute device: CPU"
    index = physical_index(uuid, gpus)
    where = f"physical GPU {index}" if index is not None else \
        f"UUID {uuid or 'unknown'} (physical index not verified)"
    return f"Compute device: {name} | {where} | process {device}"


def probe_egl(env: Mapping[str, str]) -> Optional[Dict]:
    """Run EGL_PROBE with this environment; None if it fails."""
    try:
        out = subprocess.run([sys.executable, "-c", EGL_PROBE], capture_output=True, text=True, timeout=60,
                             env=dict(env), check=True).stdout
        return json.loads(out.strip().splitlines()[-1])
    except (OSError, subprocess.SubprocessError, ValueError, IndexError):
        return None


def rendering_device_line(env: Mapping[str, str], probe: Optional[Dict],
                          gpus: Optional[List[Dict[str, str]]]) -> str:
    backend = env.get("MUJOCO_GL")
    if backend is None:
        return "Rendering device: MuJoCo default backend (MUJOCO_GL not set; device not verified)"
    backend = backend.lower()
    if backend == "osmesa":
        return "Rendering device: OSMesa | CPU (software rendering)"
    if backend != "egl":
        return f"Rendering device: {backend.upper()} | default display device (configured; physical GPU not verified)"
    device_id = env.get("MUJOCO_EGL_DEVICE_ID")
    if device_id is None:
        return "Rendering device: EGL | first usable EGL device (configured; physical GPU not verified)"
    index = physical_index((probe or {}).get("uuid"), gpus)
    if index is not None:
        return f"Rendering device: EGL | physical GPU {index}"
    if probe and probe.get("uuid"):
        return f"Rendering device: EGL | EGL device {device_id} = UUID {probe['uuid']} (physical index not verified)"
    return f"Rendering device: EGL | EGL device {device_id} (configured; physical GPU not verified)"


def device_lines(device: torch.device, env: Optional[Mapping[str, str]] = None) -> List[str]:
    """The two banner lines for this process (rank 0) and its env workers."""
    env = os.environ if env is None else env
    gpus = nvidia_smi_gpus() if device.type == "cuda" or env.get("MUJOCO_GL", "").lower() == "egl" else None
    name = torch.cuda.get_device_name(device) if device.type == "cuda" else None
    uuid = cuda_uuid(device) if device.type == "cuda" else None
    egl = env.get("MUJOCO_GL", "").lower() == "egl" and env.get("MUJOCO_EGL_DEVICE_ID") is not None
    probe = probe_egl(env) if egl else None
    return [compute_device_line(device, name, uuid, gpus), rendering_device_line(env, probe, gpus)]
