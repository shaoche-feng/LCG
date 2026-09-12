import fnmatch
from dataclasses import dataclass
from typing import Dict, List, Tuple

import torch
import torch.nn as nn

from models.diffusion.denoiser import Denoiser


@dataclass
class ThetaSConfig:
    include: Tuple[str, ...]
    exclude: Tuple[str, ...] = ()


def selected_named_parameters(denoiser: Denoiser, cfg: ThetaSConfig) -> Dict[str, nn.Parameter]:
    named_params = list(denoiser.inner_model.named_parameters())
    all_names = [name for name, _ in named_params]

    for pattern in cfg.include:
        if not any(fnmatch.fnmatchcase(name, pattern) for name in all_names):
            raise ValueError(f"Theta-S include pattern {pattern!r} matched no parameters.")

    selected: Dict[str, nn.Parameter] = {}
    for name, param in named_params:
        included = any(fnmatch.fnmatchcase(name, p) for p in cfg.include)
        excluded = any(fnmatch.fnmatchcase(name, p) for p in cfg.exclude)
        if included and not excluded:
            selected[name] = param

    if not selected:
        raise ValueError(
            "Theta-S selection is empty after applying include/exclude patterns "
            f"(include={cfg.include!r}, exclude={cfg.exclude!r})."
        )
    return selected


def frozen_named_parameters(denoiser: Denoiser, theta_s_named: Dict[str, nn.Parameter]) -> Dict[str, torch.Tensor]:
    """The complement of theta_s_named: every inner_model parameter NOT selected into
    theta_S, keyed the same way (denoiser.inner_model.named_parameters()'s own dotted
    names). Needed by forward_jvp.jvp_through_F's functional_call, which requires the
    FULL parameter dict (frozen + theta_S merged back together) to run a forward pass,
    even though only theta_S is differentiated."""
    inner = denoiser.inner_model
    selected_ids = {id(p) for p in theta_s_named.values()}
    return {name: p for name, p in inner.named_parameters() if id(p) not in selected_ids}


# For testing
def selected_parameters(denoiser: Denoiser, cfg: ThetaSConfig) -> List[nn.Parameter]:
    return list(selected_named_parameters(denoiser, cfg).values())


def selected_parameter_names(denoiser: Denoiser, cfg: ThetaSConfig) -> List[str]:
    return list(selected_named_parameters(denoiser, cfg).keys())


def selected_dim(denoiser: Denoiser, cfg: ThetaSConfig) -> int:
    return sum(p.numel() for p in selected_named_parameters(denoiser, cfg).values())
