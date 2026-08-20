from dataclasses import dataclass
import time
from typing import Optional

import torch
from torch import Tensor

from data import Dataset
from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig

from .crn import make_crn_bank_set
from .intrinsic_reward import make_lcg_intrinsic_reward_fn
from .precision import historical_precision
from .reward_normalization import RunningRMS, RunningRMSConfig
from .theta_s import selected_parameters


@dataclass
class LCGConfig:
    enabled: bool = False
    h_d_batch_size: int = 40
    damping: float = 1e-4
    beta: float = 1.0
    num_strata: int = 3
    num_crn_banks: int = 2
    chunk_size: int = 4
    rms_enabled: bool = True
    rms_alpha: float = 1.0
    rms_ema_decay: float = 0.99
    rms_eps: float = 1e-8


class LCGLifecycle:
    """Owns the outer-round LCG state and refreshes it once per completed world-model
    update round, per Algorithm 1's lifecycle:

        world-model update -> h_D refresh -> CRN-bank refresh -> RunningRMS reset
        -> LCG ActorCritic inner training (fixed theta/h_D/banks; RMS evolves normally
           across the round's ActorCritic optimizer steps)
        -> next world-model round -> refresh everything

    h_D is estimated via lcg.precision.historical_precision (Stage 2, unmodified), which
    already samples candidate transitions uniformly (sample_weights=None) rather than with
    DIAMOND's training-time recency weighting, and already applies the N/B correction using
    N=dataset.num_steps by default -- both passed explicitly here for clarity. theta_S,
    the CRN estimator, and the RunningRMS formula are untouched (Stages 1-3, 5B).

    Instrumented for lifecycle verification: `round_id` identifies the current outer round
    (0-indexed, set at the start of refresh() and held constant until the next refresh());
    h_d_compute_count / crn_construct_count / rms_reset_count each increment exactly once
    per refresh() call. `refresh()` and the returned intrinsic_reward_fn print `[LCG]`-
    prefixed diagnostic lines (round id, content fingerprints of h_D/banks, RMS state,
    timing) so an external harness can verify exactly-once-per-round refresh and
    within-round persistence from process stdout alone.
    """

    def __init__(self, cfg: LCGConfig, sigma_cfg: SigmaDistributionConfig, img_channels: int, img_size: int, device: torch.device):
        self.cfg = cfg
        self.sigma_cfg = sigma_cfg
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device

        self.round_id = -1
        self.h_d_compute_count = 0
        self.crn_construct_count = 0
        self.rms_reset_count = 0

        self.h_D: Optional[Tensor] = None
        self.banks = None
        self.rms: Optional[RunningRMS] = None
        self.intrinsic_reward_fn = None

    def refresh(self, denoiser: Denoiser, dataset: Dataset) -> None:
        assert self.cfg.enabled
        self.round_id += 1
        params = selected_parameters(denoiser)
        seed = self.round_id

        t0 = time.time()
        self.h_D = historical_precision(
            denoiser, params, dataset, self.sigma_cfg,
            B=self.cfg.h_d_batch_size, N=dataset.num_steps, num_strata=self.cfg.num_strata,
            beta=self.cfg.beta, damping=self.cfg.damping, seed=seed,
        )
        t_h_d = time.time() - t0
        self.h_d_compute_count += 1

        y_shape = torch.Size([1, self.img_channels, self.img_size, self.img_size])
        t0 = time.time()
        self.banks = make_crn_bank_set(
            self.sigma_cfg, y_shape, self.device,
            num_crn_banks=self.cfg.num_crn_banks, num_strata=self.cfg.num_strata, seed=seed + 1_000_000,
        )
        t_crn = time.time() - t0
        self.crn_construct_count += 1

        self.rms = RunningRMS(RunningRMSConfig(
            enabled=self.cfg.rms_enabled, alpha=self.cfg.rms_alpha,
            ema_decay=self.cfg.rms_ema_decay, eps=self.cfg.rms_eps,
        ))
        self.rms_reset_count += 1

        h_d_fingerprint = self.h_D.sum().item()
        banks_fingerprint = sum(s.item() for bank in self.banks for s in bank.sigmas)
        print(
            f"[LCG] round={self.round_id} refresh: N={dataset.num_steps} "
            f"h_d_compute_count={self.h_d_compute_count} h_D_fingerprint={h_d_fingerprint:.6f} (t={t_h_d:.2f}s)  "
            f"crn_construct_count={self.crn_construct_count} banks_fingerprint={banks_fingerprint:.6f} (t={t_crn:.2f}s)  "
            f"rms_reset_count={self.rms_reset_count}",
            flush=True,
        )

        base_hook = make_lcg_intrinsic_reward_fn(denoiser, params, self.h_D, self.banks, chunk_size=self.cfg.chunk_size)
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
                f"{sum(s.item() for bank in self.banks for s in bank.sigmas):.6f} "
                f"rms_s2={self.rms.s2.item():.6f} raw_reward_mean={raw.mean().item():.4f} "
                f"raw_reward_min={raw.min().item():.4f} norm_reward_mean={normalized.mean().item():.4f} "
                f"norm_reward_min={normalized.min().item():.4f} scoring_time={t_score:.3f}s",
                flush=True,
            )
            return normalized

        self.intrinsic_reward_fn = hook
