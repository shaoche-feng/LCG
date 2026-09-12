from dataclasses import dataclass, field
from typing import Dict, Optional

import torch
from torch import Tensor

from data import Dataset
from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig

from .forward_jvp import make_jvp_bank
from .intrinsic_reward import make_lcg_intrinsic_reward_fn
from .precision import assert_setup_valid, historical_precision
from .reward_normalization import RunningRMS, RunningRMSConfig
from .theta_s import ThetaSConfig, frozen_named_parameters, selected_named_parameters


@dataclass
class LCGConfig:
    enabled: bool = False
    precision_reference_size: int = 320
    precision_num_mc: int = 3  # historical precision: IID sigma draws per historical transition (backward VJP)
    damping: float = 1e-4
    beta: float = 1.0
    candidate_num_mc: int = 12  # candidate scoring: IID sigma draws per candidate (forward JVP, Full CRN)
    candidate_chunk_size: int = 16
    rms_enabled: bool = True
    rms_alpha: float = 1.0
    rms_ema_decay: float = 0.99
    rms_eps: float = 1e-8
    theta_s: ThetaSConfig = field(kw_only=True)  # which parameters belong to theta_S -- see config/intrinsic_reward/lcg.yaml

    def __post_init__(self) -> None:
        assert self.precision_reference_size > 0, (
            f"precision_reference_size must be > 0, got {self.precision_reference_size}"
        )
        assert self.precision_num_mc > 0, f"precision_num_mc must be > 0, got {self.precision_num_mc}"
        assert self.candidate_num_mc > 0, (
            f"candidate_num_mc must be > 0, got {self.candidate_num_mc} -- a zero-sample "
            "candidate bank would silently produce an empty JVP bank and zero intrinsic reward."
        )
        assert self.candidate_chunk_size > 0, f"candidate_chunk_size must be > 0, got {self.candidate_chunk_size}"
        assert self.damping > 0, f"damping must be > 0, got {self.damping}"
        if self.rms_enabled:
            assert self.rms_alpha > 0, f"rms_alpha must be > 0, got {self.rms_alpha}"
            assert 0.0 <= self.rms_ema_decay < 1.0, f"rms_ema_decay must be in [0, 1), got {self.rms_ema_decay}"
            assert self.rms_eps > 0, f"rms_eps must be > 0, got {self.rms_eps}"


class LCGLifecycle:
    """theta_S is selected once per lifecycle/model instance (cached on the first
    refresh()); h_D, h_D_inv_sqrt, and the candidate JVP/CRN bank are rebuilt every outer
    round. `round_identifier` (passed into refresh() by the caller, e.g. Trainer.epoch)
    seeds both the historical-precision draw and the candidate bank draw (in separate
    namespaces, offset by 1_000_000) -- it must be a value the CALLER already persists
    and restores across checkpoint/resume, since LCGLifecycle itself is not checkpointed:
    using a purely internal, non-persisted counter here would silently restart the seed
    sequence from 0 after every resume, reusing seeds (and therefore historical
    subsets/candidate banks) that a pre-resume round already used."""

    def __init__(self, cfg: LCGConfig, sigma_cfg: SigmaDistributionConfig, img_channels: int, img_size: int, device: torch.device):
        self.cfg = cfg
        self.sigma_cfg = sigma_cfg
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device

        self.round_id = -1

        self._theta_s_named: Optional[Dict[str, torch.nn.Parameter]] = None
        self._theta_s_dim: Optional[int] = None
        self._frozen_named: Optional[Dict[str, torch.Tensor]] = None
        self._denoiser_ref: Optional[Denoiser] = None

        self.h_D: Optional[Tensor] = None
        self.h_D_inv_sqrt: Optional[Tensor] = None
        self.bank = None
        self.rms: Optional[RunningRMS] = None
        self.intrinsic_reward_fn = None

    def refresh(self, denoiser: Denoiser, dataset: Dataset, round_identifier: int) -> None:
        assert self.cfg.enabled
        self.round_id = round_identifier
        seed = round_identifier

        # theta_S selection and frozen parameter caching is done once per lifecycle instance.
        # The denoiser instance must be the same across refreshes.
        if self._theta_s_named is None:
            theta_s_named = selected_named_parameters(denoiser, self.cfg.theta_s)
            self._theta_s_named = theta_s_named
            self._theta_s_dim = sum(p.numel() for p in theta_s_named.values())
            self._frozen_named = frozen_named_parameters(denoiser, theta_s_named)
            self._denoiser_ref = denoiser
            print(
                f"[LCG] theta_S: parameter tensors={len(theta_s_named)} "
                f"scalar parameters={self._theta_s_dim} "
                f"include={list(self.cfg.theta_s.include)} exclude={list(self.cfg.theta_s.exclude)}",
                flush=True,
            )
        elif denoiser is not self._denoiser_ref:
            raise RuntimeError(
                "LCGLifecycle.refresh() was called with a different denoiser instance "
                "than the one theta_S was cached against on this lifecycle's first "
                "refresh(). theta_S selection (membership/order/parameter object "
                "identity) is resolved once per LCGLifecycle instance and reused on every "
                "later round. If the trainer genuinely needs to replace the denoiser "
                "mid-run, construct a new LCGLifecycle for it instead."
            )

        theta_s_named = self._theta_s_named
        frozen_named = self._frozen_named
        params = list(theta_s_named.values())

        self.h_D = historical_precision(
            denoiser, params, dataset, self.sigma_cfg,
            B=self.cfg.precision_reference_size, N=dataset.num_steps, num_mc=self.cfg.precision_num_mc,
            beta=self.cfg.beta, damping=self.cfg.damping, seed=seed,
        )
        assert_setup_valid(theta_s_named, self.h_D, d_S=self._theta_s_dim)
        self.h_D_inv_sqrt = self.h_D.rsqrt()  # fixed for the whole round -- computed once, not per AC scoring call

        y_shape = torch.Size([1, self.img_channels, self.img_size, self.img_size])
        self.bank = make_jvp_bank(
            self.sigma_cfg, y_shape, self._theta_s_dim, self.device,
            num_samples=self.cfg.candidate_num_mc, seed=seed + 1_000_000,
        )
        base_hook = make_lcg_intrinsic_reward_fn(
            denoiser, theta_s_named, frozen_named, self.h_D_inv_sqrt, self.bank, chunk_size=self.cfg.candidate_chunk_size,
        )

        self.rms = RunningRMS(RunningRMSConfig(
            enabled=self.cfg.rms_enabled, alpha=self.cfg.rms_alpha,
            ema_decay=self.cfg.rms_ema_decay, eps=self.cfg.rms_eps,
        ))

        print(
            f"[LCG] round={self.round_id} refresh: N={dataset.num_steps} "
            f"h_D_fingerprint={self.h_D.sum().item():.6f} banks_fingerprint={sum(s.item() for s in self.bank.sigmas):.6f}",
            flush=True,
        )

        def hook(infos, env_rew):
            raw = base_hook(infos, env_rew)
            return self.rms(raw)

        self.intrinsic_reward_fn = hook
