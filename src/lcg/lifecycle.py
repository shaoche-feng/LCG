from dataclasses import dataclass, field
import time
from typing import Dict, Optional, Tuple

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
    precision_reference_size: int = 40
    precision_num_mc: int = 3  # historical precision: IID sigma draws per historical transition (backward VJP)
    damping: float = 1e-4
    beta: float = 1.0
    candidate_num_mc: int = 24  # candidate scoring: IID sigma draws per candidate (forward JVP, Full CRN)
    candidate_chunk_size: int = 16
    rms_enabled: bool = True
    rms_alpha: float = 1.0
    rms_ema_decay: float = 0.99
    rms_eps: float = 1e-8
    theta_s: ThetaSConfig = field(kw_only=True)  # which parameters belong to theta_S -- see config/intrinsic_reward/lcg.yaml


class LCGLifecycle:
    def __init__(self, cfg: LCGConfig, sigma_cfg: SigmaDistributionConfig, img_channels: int, img_size: int, device: torch.device):
        self.cfg = cfg
        self.sigma_cfg = sigma_cfg
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device

        self.round_id = -1

        self._theta_s_named: Optional[Dict[str, torch.nn.Parameter]] = None
        self._theta_s_names: Optional[Tuple[str, ...]] = None
        self._theta_s_dim: Optional[int] = None
        self._frozen_named: Optional[Dict[str, torch.Tensor]] = None
        self._denoiser_ref: Optional[Denoiser] = None

        self.h_D: Optional[Tensor] = None
        self.bank = None
        self.rms: Optional[RunningRMS] = None
        self.intrinsic_reward_fn = None

    def refresh(self, denoiser: Denoiser, dataset: Dataset) -> None:
        assert self.cfg.enabled

        self.round_id += 1
        seed = self.round_id

        # theta_S selection and frozen parameter caching is done once per lifecycle instance
        # The denoiser instance must be the same across refreshes.
        if self._theta_s_named is None:
            theta_s_named = selected_named_parameters(denoiser, self.cfg.theta_s)
            self._theta_s_named = theta_s_named
            self._theta_s_names = tuple(theta_s_named.keys())
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
            raise RuntimeError("LCGLifecycle.refresh() was called with a different denoiser instance ")

        theta_s_named = self._theta_s_named
        frozen_named = self._frozen_named
        params = list(theta_s_named.values())

        # Compute historical precision h_D (backward VJP)
        t0 = time.time()
        self.h_D = historical_precision(
            denoiser, params, dataset, self.sigma_cfg,
            B=self.cfg.precision_reference_size, N=dataset.num_steps, num_mc=self.cfg.precision_num_mc,
            beta=self.cfg.beta, damping=self.cfg.damping, seed=seed,
        )
        t_h_d = time.time() - t0
        assert_setup_valid(theta_s_named, self.h_D, d_S=self._theta_s_dim)

        y_shape = torch.Size([1, self.img_channels, self.img_size, self.img_size])
        t0 = time.time()
        self.bank = make_jvp_bank(
            self.sigma_cfg, y_shape, self._theta_s_dim, self.device,
            num_samples=self.cfg.candidate_num_mc, seed=seed + 1_000_000,
        )
        base_hook = make_lcg_intrinsic_reward_fn(
            denoiser, theta_s_named, frozen_named, self.h_D, self.bank, chunk_size=self.cfg.candidate_chunk_size,
        )
        t_crn = time.time() - t0

        self.rms = RunningRMS(RunningRMSConfig(
            enabled=self.cfg.rms_enabled, alpha=self.cfg.rms_alpha,
            ema_decay=self.cfg.rms_ema_decay, eps=self.cfg.rms_eps,
        ))

        h_d_fingerprint = self.h_D.sum().item()
        banks_fingerprint = sum(s.item() for s in self.bank.sigmas)
        print(
            f"[LCG] round={self.round_id} refresh: N={dataset.num_steps} "
            f"h_D_fingerprint={h_d_fingerprint:.6f} (t={t_h_d:.2f}s)  "
            f"banks_fingerprint={banks_fingerprint:.6f} (t={t_crn:.2f}s)",
            flush=True,
        )

        call_index = {"n": 0}

        def hook(infos, env_rew):
            t0_ = time.time()
            raw = base_hook(infos, env_rew)
            t_score = time.time() - t0_
            normalized = self.rms(raw)
            call_index["n"] += 1
            print(
                f"[LCG] round={self.round_id} ac_call={call_index['n']} "
                f"h_D_fingerprint={self.h_D.sum().item():.6f} banks_fingerprint="
                f"{sum(s.item() for s in self.bank.sigmas):.6f} "
                f"rms_s2={self.rms.s2.item():.6f} raw_reward_mean={raw.mean().item():.4f} "
                f"raw_reward_min={raw.min().item():.4f} norm_reward_mean={normalized.mean().item():.4f} "
                f"norm_reward_min={normalized.min().item():.4f} scoring_time={t_score:.3f}s",
                flush=True,
            )
            return normalized

        self.intrinsic_reward_fn = hook
