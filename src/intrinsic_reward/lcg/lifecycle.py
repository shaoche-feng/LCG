from dataclasses import dataclass, field
import math
import time
from typing import Dict, Optional

import torch
from torch import Tensor

from data import Dataset
from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig

from .forward_jvp import JVPBank, make_jvp_bank, score_one_jvp_bank
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
    # Policy reward from the raw LCG score: "rms" (RunningRMS), "fixed" (raw / fixed_divisor,
    # nothing else) or "none" (raw). None: the meaning before this field existed, i.e. "rms"
    # if rms_enabled else "none". An explicit mode must agree with rms_enabled.
    reward_scaling: Optional[str] = None
    fixed_divisor: Optional[float] = None
    theta_s: ThetaSConfig = field(kw_only=True)  # which parameters belong to theta_S -- see config/intrinsic_reward/lcg.yaml

    def __post_init__(self) -> None:
        legacy = "rms" if self.rms_enabled else "none"
        if self.reward_scaling is None:
            self.reward_scaling = legacy
        if self.reward_scaling not in REWARD_SCALINGS:
            raise ValueError(f"reward_scaling must be one of {REWARD_SCALINGS}, got {self.reward_scaling!r}")
        if (self.reward_scaling == "rms") != bool(self.rms_enabled):
            raise ValueError(f"reward_scaling={self.reward_scaling} conflicts with rms_enabled={self.rms_enabled}; "
                             f"set rms_enabled={self.reward_scaling == 'rms'}")
        if self.reward_scaling == "fixed":
            d = self.fixed_divisor
            if isinstance(d, bool) or not isinstance(d, (int, float)) or not math.isfinite(d) or d <= 0:
                raise ValueError(f"fixed_divisor must be a finite positive number, got {d!r}")
        elif self.fixed_divisor is not None:
            raise ValueError(f"fixed_divisor is used only with reward_scaling=fixed, got {self.fixed_divisor!r}")
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

    def reward_scaling_settings(self) -> Dict:
        """What turns raw LCG scores into policy reward; a resume must not change it."""
        settings = dict(mode=self.reward_scaling)
        if self.reward_scaling == "fixed":
            settings["fixed_divisor"] = float(self.fixed_divisor)
        if self.reward_scaling == "rms":
            settings.update(alpha=self.rms_alpha, ema_decay=self.rms_ema_decay, eps=self.rms_eps)
        return settings


REWARD_SCALINGS = ("rms", "fixed", "none")


class LCGLifecycle:
    """theta_S is selected once per lifecycle/model instance (cached on the first
    refresh()); h_D, h_D_inv_sqrt, and the candidate JVP/CRN bank are rebuilt every outer
    round. `round_identifier` (passed into refresh() by the caller, e.g. Trainer.epoch)
    seeds both the historical-precision draw and the candidate bank draw (in separate
    namespaces, offset by 1_000_000) -- it must be a value the CALLER already persists
    and restores across checkpoint/resume: using a purely internal, non-persisted counter here
    would silently restart the seed sequence from 0 after every resume, reusing seeds (and
    therefore historical subsets/candidate banks) that a pre-resume round already used.
    state_dict()/load_state_dict() checkpoint only the current round's frozen scorer (h_D, the
    candidate bank, RunningRMS), so a resume can evaluate with it without refreshing."""

    def __init__(self, cfg: LCGConfig, sigma_cfg: SigmaDistributionConfig, img_channels: int, img_size: int, device: torch.device):
        self.cfg = cfg
        self.sigma_cfg = sigma_cfg
        self.img_channels = img_channels
        self.img_size = img_size
        self.device = device
        self.verbose = True  # refresh details on stdout (the exploration trainer logs them to W&B)

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
        # Monitoring only: the last scored batch (raw and policy reward) and timings.
        self.last_raw: Optional[Tensor] = None
        self.last_scaled: Optional[Tensor] = None
        self.precision_seconds = 0.0
        self.scoring_seconds = 0.0

    def refresh(self, denoiser: Denoiser, dataset: Dataset, round_identifier: int) -> None:
        assert self.cfg.enabled
        self.round_id = round_identifier
        seed = round_identifier
        self._select_theta_s(denoiser)

        theta_s_named = self._theta_s_named
        params = list(theta_s_named.values())

        start = _synchronized_time(self.device)
        self.h_D = lcg_historical_precision(denoiser, params, dataset, self.sigma_cfg, self.cfg, round_identifier)
        self.precision_seconds = _synchronized_time(self.device) - start
        self.scoring_seconds = 0.0
        assert_setup_valid(theta_s_named, self.h_D, d_S=self._theta_s_dim, verbose=self.verbose)
        self.h_D_inv_sqrt = self.h_D.rsqrt()  # fixed for the whole round -- computed once, not per AC scoring call

        y_shape = torch.Size([1, self.img_channels, self.img_size, self.img_size])
        self.bank = make_jvp_bank(
            self.sigma_cfg, y_shape, self._theta_s_dim, self.device,
            num_samples=self.cfg.candidate_num_mc, seed=seed + 1_000_000,
        )
        self.rms = self._new_rms()

        if self.verbose:
            print(
                f"[LCG] round={self.round_id} refresh: N={dataset.num_steps} "
                f"h_D_fingerprint={self.h_D.sum().item():.6f} banks_fingerprint={sum(s.item() for s in self.bank.sigmas):.6f}",
                flush=True,
            )
        self.intrinsic_reward_fn = self._make_hook(denoiser)

    def _select_theta_s(self, denoiser: Denoiser) -> None:
        # theta_S selection and frozen parameter caching is done once per lifecycle instance.
        # The denoiser instance must be the same across refreshes.
        if self._theta_s_named is None:
            theta_s_named = selected_named_parameters(denoiser, self.cfg.theta_s)
            self._theta_s_named = theta_s_named
            self._theta_s_dim = sum(p.numel() for p in theta_s_named.values())
            self._frozen_named = frozen_named_parameters(denoiser, theta_s_named)
            self._denoiser_ref = denoiser
            if self.verbose:
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

    def _new_rms(self) -> Optional[RunningRMS]:
        if self.cfg.reward_scaling != "rms":
            return None
        return RunningRMS(RunningRMSConfig(enabled=True, alpha=self.cfg.rms_alpha,
                                           ema_decay=self.cfg.rms_ema_decay, eps=self.cfg.rms_eps))

    def _make_hook(self, denoiser: Denoiser):
        """The PMPO hook of the current round (precision, bank and RunningRMS)."""
        base_hook = make_lcg_intrinsic_reward_fn(
            denoiser, self._theta_s_named, self._frozen_named, self.h_D_inv_sqrt, self.bank,
            chunk_size=self.cfg.candidate_chunk_size,
        )

        def hook(infos, env_rew):
            start = _synchronized_time(self.device)
            raw = base_hook(infos, env_rew)
            self.scoring_seconds += _synchronized_time(self.device) - start
            reward = self.scale(raw)
            self.last_raw, self.last_scaled = raw.detach(), reward.detach()
            return reward

        return hook

    def settings(self) -> Dict:
        """What a saved round's scores depend on besides the denoiser; a restore must match."""
        c = self.cfg
        return dict(reward_scaling=c.reward_scaling_settings(), candidate_num_mc=c.candidate_num_mc,
                    candidate_chunk_size=c.candidate_chunk_size, precision_reference_size=c.precision_reference_size,
                    precision_num_mc=c.precision_num_mc, damping=c.damping, beta=c.beta,
                    theta_s=dict(include=list(c.theta_s.include), exclude=list(c.theta_s.exclude)),
                    image=[self.img_channels, self.img_size])

    def state_dict(self) -> Optional[Dict]:
        """The current round's frozen scorer (None before the first refresh), on CPU: everything
        evaluation needs to score exactly as this round's PMPO hook did."""
        if not self.ready:
            return None
        bank = {name: [t.cpu() for t in getattr(self.bank, name)] for name in ("sigmas", "epsilons", "epsilons_offset", "etas")}
        return dict(settings=self.settings(), round_id=self.round_id, theta_s_names=list(self._theta_s_named),
                    theta_s_dim=self._theta_s_dim, h_D=self.h_D.cpu(), bank=bank,
                    rms_s2=None if self.rms is None or self.rms.s2 is None else self.rms.s2.cpu())

    def load_state_dict(self, state: Dict, denoiser: Denoiser) -> None:
        """Restore a saved round on this denoiser (its weights already loaded): nothing is
        recomputed or redrawn; the PMPO hook is rebuilt around the restored state."""
        if state["settings"] != self.settings():
            raise ValueError(f"Checkpoint's LCG scorer was built with {state['settings']}, the config gives "
                             f"{self.settings()}; a resume must not change them")
        self._select_theta_s(denoiser)
        if list(self._theta_s_named) != state["theta_s_names"] or self._theta_s_dim != state["theta_s_dim"]:
            raise ValueError("Checkpoint's LCG theta_S selection differs from this denoiser's")
        self.round_id = state["round_id"]
        self.h_D = state["h_D"].to(self.device)
        self.h_D_inv_sqrt = self.h_D.rsqrt()
        self.bank = JVPBank(*(tuple(t.to(self.device) for t in state["bank"][name])
                              for name in ("sigmas", "epsilons", "epsilons_offset", "etas")))
        self.rms = self._new_rms()
        if self.rms is not None and state["rms_s2"] is not None:
            self.rms.s2 = state["rms_s2"].to(self.device)
        self.intrinsic_reward_fn = self._make_hook(denoiser)

    @property
    def ready(self) -> bool:
        """A round's precision and candidate bank exist (refreshed, or restored from a checkpoint)."""
        return self.bank is not None

    def score_candidates(self, candidates) -> Tensor:
        """Raw LCG scores of (x_obs_flat, x_act, y) candidates with this round's frozen precision
        and bank, exactly as the PMPO hook scores imagined ones; no timing, last-batch or
        normalization state changes (evaluation)."""
        if not self.ready:
            raise RuntimeError("LCG has no refreshed or restored round")
        return score_one_jvp_bank(self._denoiser_ref, self._theta_s_named, self._frozen_named, self.h_D_inv_sqrt,
                                  self.bank, candidates, self.cfg.candidate_chunk_size)

    def scale_frozen(self, raw: Tensor) -> Optional[Tensor]:
        """Policy reward of raw scores without updating RunningRMS (None before its first update)."""
        if self.cfg.reward_scaling == "rms":
            return None if self.rms is None else self.rms.apply_frozen(raw)
        return self.scale(raw)

    def scale(self, raw: Tensor) -> Tensor:
        """Policy reward from raw LCG scores. fixed: raw / fixed_divisor only -- no running
        statistics, centering or clipping; the same divisor for every batch and epoch."""
        if self.cfg.reward_scaling == "fixed":
            return raw / self.cfg.fixed_divisor
        if self.cfg.reward_scaling == "rms":
            return self.rms(raw)
        return raw


def lcg_historical_precision(denoiser: Denoiser, params, dataset: Dataset, sigma_cfg: SigmaDistributionConfig, cfg,
                             round_identifier: int) -> Tensor:
    """LCG's historical precision h_D of one refresh round, exactly as LCG uses it (shared with
    the Laplace ensemble). cfg supplies precision_reference_size (B), precision_num_mc, beta and
    damping; N is the replay size; the round identifier seeds the reference draw and probes.
    Diagonal: damping + beta * (N / B) * sum over B reference transitions of the probe-averaged
    squared VJP sqrt(2) xi^T F_theta (see precision.historical_precision)."""
    return historical_precision(
        denoiser, params, dataset, sigma_cfg,
        B=cfg.precision_reference_size, N=dataset.num_steps, num_mc=cfg.precision_num_mc,
        beta=cfg.beta, damping=cfg.damping, seed=round_identifier,
    )


def _synchronized_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()
