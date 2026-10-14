
from argparse import Namespace
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from functools import partial
import json
from pathlib import Path
import random
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    save_with_backup(agent_sd, get_path(epoch))

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


def save_with_backup(obj: Any, path: Path):
    bk = path.with_suffix(".bk")
    if path.is_file():
        path.rename(bk)
    torch.save(obj, path)
    bk.unlink(missing_ok=True)


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
