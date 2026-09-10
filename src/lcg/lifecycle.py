from dataclasses import dataclass
import time
from typing import Optional

import torch
from torch import Tensor

from data import Dataset
from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig

from .forward_jvp import assert_setup_valid, frozen_named_parameters, make_jvp_bank, selected_named_parameters
from .intrinsic_reward import make_lcg_intrinsic_reward_fn
from .precision import historical_precision
from .reward_normalization import RunningRMS, RunningRMSConfig


@dataclass
class LCGConfig:
    enabled: bool = False
    h_d_batch_size: int = 40
    precision_num_mc: int = 3  # historical precision: IID sigma draws per historical transition (backward VJP)
    damping: float = 1e-4
    beta: float = 1.0
    candidate_num_mc: int = 24  # candidate scoring: IID sigma draws per candidate (forward JVP, Full CRN)
    candidate_chunk_size: int = 16
    rms_enabled: bool = True
    rms_alpha: float = 1.0
    rms_ema_decay: float = 0.99
    rms_eps: float = 1e-8


class LCGLifecycle:
    """Owns the outer-round LCG state and refreshes it once per completed world-model
    update round:

        world-model update -> h_D refresh (backward VJP, simple MC) -> candidate JVP-bank
        refresh (forward JVP, simple MC, Full CRN) -> RunningRMS reset
        -> LCG ActorCritic inner training (fixed theta_S/h_D/bank; RMS evolves normally
           across the round's ActorCritic optimizer steps)
        -> next world-model round -> refresh everything

    h_D is estimated via lcg.precision.historical_precision, which samples historical
    transitions uniformly without replacement (lcg.precision.
    sample_uniform_historical_transitions) and applies the N/B correction using
    N=dataset.num_steps by default -- both passed explicitly here for clarity. theta_S
    and the RunningRMS formula are fixed by the method, not configurable here.

    Candidate scoring is always forward-mode JVP + simple Monte Carlo + Full CRN -- there
    is no alternative estimator to select.

    Instrumented for lifecycle verification: `round_id` identifies the current outer round
    (0-indexed, set at the start of refresh() and held constant until the next refresh());
    h_d_compute_count / crn_construct_count / rms_reset_count each increment exactly once
    per refresh() call. `refresh()` and the returned intrinsic_reward_fn print `[LCG]`-
    prefixed diagnostic lines (round id, content fingerprints of h_D/bank, RMS state,
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
        self.bank = None
        self.rms: Optional[RunningRMS] = None
        self.intrinsic_reward_fn = None

    def refresh(self, denoiser: Denoiser, dataset: Dataset) -> None:
        assert self.cfg.enabled
        self.round_id += 1
        theta_s_named = selected_named_parameters(denoiser)
        params = list(theta_s_named.values())
        seed = self.round_id

        t0 = time.time()
        self.h_D = historical_precision(
            denoiser, params, dataset, self.sigma_cfg,
            B=self.cfg.h_d_batch_size, N=dataset.num_steps, num_mc=self.cfg.precision_num_mc,
            beta=self.cfg.beta, damping=self.cfg.damping, seed=seed,
        )
        t_h_d = time.time() - t0
        self.h_d_compute_count += 1

        y_shape = torch.Size([1, self.img_channels, self.img_size, self.img_size])
        t0 = time.time()
        frozen_named = frozen_named_parameters(denoiser, theta_s_named)
        assert_setup_valid(theta_s_named, self.h_D, d_S=self.h_D.numel())
        self.bank = make_jvp_bank(
            self.sigma_cfg, y_shape, self.h_D.numel(), self.device,
            num_samples=self.cfg.candidate_num_mc, seed=seed + 1_000_000,
        )
        base_hook = make_lcg_intrinsic_reward_fn(
            denoiser, theta_s_named, frozen_named, self.h_D, self.bank, chunk_size=self.cfg.candidate_chunk_size,
        )
        t_crn = time.time() - t0
        self.crn_construct_count += 1

        self.rms = RunningRMS(RunningRMSConfig(
            enabled=self.cfg.rms_enabled, alpha=self.cfg.rms_alpha,
            ema_decay=self.cfg.rms_ema_decay, eps=self.cfg.rms_eps,
        ))
        self.rms_reset_count += 1

        h_d_fingerprint = self.h_D.sum().item()
        banks_fingerprint = sum(s.item() for s in self.bank.sigmas)
        print(
            f"[LCG] round={self.round_id} refresh: N={dataset.num_steps} "
            f"h_d_compute_count={self.h_d_compute_count} h_D_fingerprint={h_d_fingerprint:.6f} (t={t_h_d:.2f}s)  "
            f"crn_construct_count={self.crn_construct_count} banks_fingerprint={banks_fingerprint:.6f} (t={t_crn:.2f}s)  "
            f"rms_reset_count={self.rms_reset_count}",
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
