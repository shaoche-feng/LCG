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
    # nothing else), "none" (raw), or one of two per-round divisors that keep the reward's scale
    # steady as h_D grows with the replay (frozen for the round, like h_D):
    #   "data_scaled": raw / fixed_divisor * N / data_reference_steps, N the replay size at the
    #     refresh (h_D grows about linearly in N; at N = data_reference_steps this is "fixed");
    #   "reference": raw / (mean raw score of reference_num_prompts imagined uniform-action
    #     rollouts of this round's denoiser, scored by this round's scorer; see calibrate()).
    # None: the meaning before this field existed, i.e. "rms" if rms_enabled else "none". An
    # explicit mode must agree with rms_enabled.
    reward_scaling: Optional[str] = None
    fixed_divisor: Optional[float] = None
    data_reference_steps: Optional[int] = None
    reference_num_prompts: Optional[int] = None
    theta_s: ThetaSConfig = field(kw_only=True)  # which parameters belong to theta_S -- see config/intrinsic_reward/lcg.yaml

    def __post_init__(self) -> None:
        if not isinstance(self.rms_enabled, bool):
            raise ValueError(f"rms_enabled must be a boolean, got {self.rms_enabled!r}")
        legacy = "rms" if self.rms_enabled else "none"
        if self.reward_scaling is None:
            self.reward_scaling = legacy
        if self.reward_scaling not in REWARD_SCALINGS:
            raise ValueError(f"reward_scaling must be one of {REWARD_SCALINGS}, got {self.reward_scaling!r}")
        if (self.reward_scaling == "rms") != bool(self.rms_enabled):
            raise ValueError(f"reward_scaling={self.reward_scaling} conflicts with rms_enabled={self.rms_enabled}; "
                             f"set rms_enabled={self.reward_scaling == 'rms'}")
        if self.reward_scaling in ("fixed", "data_scaled"):
            d = self.fixed_divisor
            if not _positive_finite_number(d):
                raise ValueError(f"fixed_divisor must be a finite positive number, got {d!r}")
        elif self.fixed_divisor is not None:
            raise ValueError(f"fixed_divisor is used only with reward_scaling=fixed or data_scaled, "
                             f"got {self.fixed_divisor!r}")
        for name, mode in (("data_reference_steps", "data_scaled"), ("reference_num_prompts", "reference")):
            value = getattr(self, name)
            if self.reward_scaling == mode:
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} must be a positive integer with reward_scaling={mode}, got {value!r}")
            elif value is not None:
                raise ValueError(f"{name} is used only with reward_scaling={mode}, got {value!r}")
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
            assert _positive_finite_number(self.rms_alpha), f"rms_alpha must be finite and > 0, got {self.rms_alpha!r}"
            assert (not isinstance(self.rms_ema_decay, bool) and isinstance(self.rms_ema_decay, (int, float))
                    and math.isfinite(self.rms_ema_decay) and 0 <= self.rms_ema_decay < 1), (
                f"rms_ema_decay must be finite and in [0, 1), got {self.rms_ema_decay!r}")
            assert _positive_finite_number(self.rms_eps), f"rms_eps must be finite and > 0, got {self.rms_eps!r}"

    def reward_scaling_settings(self) -> Dict:
        """What turns raw LCG scores into policy reward; a resume must not change it."""
        settings = dict(mode=self.reward_scaling)
        if self.reward_scaling in ("fixed", "data_scaled"):
            settings["fixed_divisor"] = float(self.fixed_divisor)
        if self.reward_scaling == "data_scaled":
            settings["data_reference_steps"] = int(self.data_reference_steps)
        if self.reward_scaling == "reference":
            settings["reference_num_prompts"] = int(self.reference_num_prompts)
        if self.reward_scaling == "rms":
            settings.update(alpha=self.rms_alpha, ema_decay=self.rms_ema_decay, eps=self.rms_eps)
        return settings


REWARD_SCALINGS = ("rms", "fixed", "none", "data_scaled", "reference")


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
        # data_scaled / reference: this round's divisor (frozen for the round, checkpointed); the
        # reference mode's mean raw reference score is kept for monitoring.
        self.reward_divisor: Optional[float] = None
        self.reference_raw_mean: Optional[float] = None
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
        self.reward_divisor = self.reference_raw_mean = None
        if self.cfg.reward_scaling == "data_scaled":
            self.reward_divisor = self.cfg.fixed_divisor * (self.cfg.data_reference_steps / dataset.num_steps)
            if not _positive_finite_number(self.reward_divisor):
                raise ValueError(f"data_scaled produced an invalid reward divisor: {self.reward_divisor!r}")
        # reference: the caller imagines this round's reference rollouts and passes their scores
        # to calibrate() before PMPO; the hook refuses to score until then.

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

    @property
    def needs_reference(self) -> bool:
        """reference mode after a refresh: this round's divisor awaits calibrate()."""
        return self.cfg.reward_scaling == "reference" and self.ready and self.reward_divisor is None

    def calibrate(self, reference_raw: Tensor) -> None:
        """reference mode: this round's divisor is the mean raw score of its reference rollouts
        (imagined uniform-action transitions of this round's denoiser, scored by this round's
        precision and bank). Called once per refresh, before PMPO; frozen for the round."""
        if self.cfg.reward_scaling != "reference":
            raise RuntimeError(f"calibrate() is for reward_scaling=reference, not {self.cfg.reward_scaling}")
        if not self.ready:
            raise RuntimeError("Reference calibration requires a refreshed scorer")
        if self.reward_divisor is not None:
            raise RuntimeError("This reference round is already calibrated; its divisor is frozen until refresh()")
        if reference_raw.numel() == 0 or not torch.isfinite(reference_raw).all() or (reference_raw < 0).any():
            raise ValueError("Reference scores must be non-empty, finite and non-negative with a finite positive mean")
        mean = float(reference_raw.detach().double().mean().item())
        if not _positive_finite_number(mean):
            raise ValueError(f"Reference scores must be non-empty with a finite positive mean, got {mean!r}")
        self.reference_raw_mean = self.reward_divisor = mean

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
                    rms_s2=None if self.rms is None or self.rms.s2 is None else self.rms.s2.cpu(),
                    reward_divisor=self.reward_divisor, reference_raw_mean=self.reference_raw_mean)

    def load_state_dict(self, state: Dict, denoiser: Denoiser) -> None:
        """Restore a saved round on this denoiser (its weights already loaded): nothing is
        recomputed or redrawn; the PMPO hook is rebuilt around the restored state."""
        if state["settings"] != self.settings():
            raise ValueError(f"Checkpoint's LCG scorer was built with {state['settings']}, the config gives "
                             f"{self.settings()}; a resume must not change them")
        divisor, reference_mean = state.get("reward_divisor"), state.get("reference_raw_mean")
        if self.cfg.reward_scaling in ("data_scaled", "reference"):
            if divisor is None:
                raise ValueError(f"Checkpoint's LCG scorer has no reward divisor for reward_scaling={self.cfg.reward_scaling}")
            if not _positive_finite_number(divisor):
                raise ValueError(f"Checkpoint's LCG scorer has an invalid reward divisor: {divisor!r}")
        if self.cfg.reward_scaling == "reference" and (
                not _positive_finite_number(reference_mean) or reference_mean != divisor):
            raise ValueError("Checkpoint's reference raw mean must be finite, positive and equal to its reward divisor")
        rms_s2 = state.get("rms_s2")
        if self.cfg.reward_scaling == "rms" and rms_s2 is not None and (
                not torch.is_tensor(rms_s2) or rms_s2.numel() != 1 or not torch.isfinite(rms_s2).all()
                or (rms_s2 < 0).any()):
            raise ValueError("Checkpoint's RMS mean square must be a finite non-negative scalar")
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
        # Checkpoints from before the per-round divisors have none (their modes do not use one).
        self.reward_divisor, self.reference_raw_mean = divisor, reference_mean
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
        statistics, centering or clipping; the same divisor for every batch and epoch.
        data_scaled / reference: raw / this round's divisor (the same for every batch of the round)."""
        if self.cfg.reward_scaling == "fixed":
            return raw / self.cfg.fixed_divisor
        if self.cfg.reward_scaling in ("data_scaled", "reference"):
            if self.reward_divisor is None:
                raise RuntimeError(f"reward_scaling={self.cfg.reward_scaling}: this round has no divisor yet "
                                   "(reference mode needs calibrate() after refresh)")
            return raw / self.reward_divisor
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


def _positive_finite_number(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def _synchronized_time(device: torch.device) -> float:
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()
