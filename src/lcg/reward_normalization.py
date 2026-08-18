from dataclasses import dataclass
from typing import Callable, Dict, List, Optional

import torch
from torch import Tensor


@dataclass
class RunningRMSConfig:
    enabled: bool = True
    alpha: float = 1.0
    ema_decay: float = 0.99  # rho
    eps: float = 1e-8


class RunningRMS:
    """Multiplicative running-RMS scale for LCG reward normalization (no batch z-score
    centering, no clipping):

        s_t^2 = rho * s_{t-1}^2 + (1 - rho) * mean(r_LCG^2)
        r_int = alpha / (sqrt(s_t^2) + eps) * r_LCG

    State (`s2`) is a detached, no-grad scalar owned entirely by this object -- it never
    touches the denoiser or any nn.Module, and is not itself trainable. Since scale is
    always a positive scalar multiplier, normalization is monotonic per call and preserves
    the relative ordering of candidates within a batch exactly.

    Initialized from the first observed batch's own mean square (not zero), so the first
    normalized reward isn't artificially huge from an under-warmed running estimate.
    """

    def __init__(self, cfg: RunningRMSConfig):
        self.cfg = cfg
        self.s2: Optional[Tensor] = None

    @torch.no_grad()
    def __call__(self, r: Tensor) -> Tensor:
        if not self.cfg.enabled:
            return r
        batch_mean_sq = r.detach().square().mean()
        if self.s2 is None:
            self.s2 = batch_mean_sq.clone()
        else:
            self.s2 = self.cfg.ema_decay * self.s2 + (1 - self.cfg.ema_decay) * batch_mean_sq
        scale = self.cfg.alpha / (self.s2.sqrt() + self.cfg.eps)
        return r * scale


def wrap_with_running_rms(
    hook_fn: Callable[[List[Dict], Tensor], Tensor], rms: RunningRMS
) -> Callable[[List[Dict], Tensor], Tensor]:
    """Composes an existing intrinsic_reward_fn (e.g. from make_lcg_intrinsic_reward_fn)
    with RunningRMS normalization as a separate, optional layer. Raw-LCG (Stage 5A/5B Part
    A) behavior is reproduced exactly by simply not wrapping, or by wrapping with
    RunningRMSConfig(enabled=False).
    """

    def normalized_hook(infos, env_rew):
        raw = hook_fn(infos, env_rew)
        return rms(raw)

    return normalized_hook
