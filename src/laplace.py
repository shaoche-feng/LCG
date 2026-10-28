"""Laplace-inspired sampled diffusion ensemble disagreement (design: docs/LAPLACE_ENSEMBLE_DESIGN.md).

One trained denoiser. After each denoiser phase, LCG's historical precision h (diagonal; see
lcg.lcg_historical_precision, used unchanged) over the LCG-selected parameters theta_S defines
    q(theta_S) = Normal(theta_hat_S, T * diag(h)^-1),   theta_not_S = theta_hat_not_S,
with T = posterior_temperature. K samples theta_S[k] = theta_hat_S + sqrt(T) * eps_k / sqrt(h)
(eps_k ~ N(0, I) from a private generator) are drawn once per refresh and frozen for the epoch's
PMPO phase. This is a curvature-based approximation, not a calibrated posterior.

Imagination: every member runs the complete diffusion chain from the same initial noise with its
sampled parameters fixed for all denoising steps (functional parameter substitution: the base
module is never modified). Reward and next observation are those of the full ensemble
(population pixel variance; one uniformly drawn member per trajectory and transition). For a
selection inside the last decoder level (e.g. R1), the first denoising evaluation, whose input is
identical for all members, runs the shared U-Net prefix once and only the selected suffix per
member; later evaluations are full per-member passes.
"""
from dataclasses import asdict, dataclass, field
import math
import time
from typing import Any, Dict, List, Optional

import numpy as np
import torch
from torch import Tensor
from torch.func import functional_call
import torch.nn.functional as F

from data import Dataset
from ensemble import EnsembleScorer
from lcg import lcg_historical_precision, selected_named_parameters, ThetaSConfig
from models.diffusion import Denoiser, DiffusionSampler
from models.diffusion.denoiser import SigmaDistributionConfig

REWARD_SCALINGS = ("fixed", "none")
MEMBER_SELECTIONS = ("uniform_per_transition",)


@dataclass
class LaplaceConfig:
    """All laplace_disagreement settings (config/intrinsic_reward/laplace_disagreement.yaml)."""
    method: str = "laplace_disagreement"
    enabled: bool = True
    num_members: int = 5  # sampled parameter sets (the central model is not a member)
    posterior_temperature: float = 1.0  # covariance multiplier T (std scales with sqrt(T))
    # LCG's selection and historical-precision settings, same names and meaning.
    theta_s: ThetaSConfig = field(default_factory=lambda: ThetaSConfig(include=("unet.u_blocks.3.resblocks.0.*",)))
    precision_reference_size: int = 320
    precision_num_mc: int = 3
    damping: float = 1e-4
    beta: float = 1.0
    num_noise_samples: int = 1
    member_selection: str = "uniform_per_transition"
    shared_prefix: bool = True  # exact first-evaluation prefix reuse when the selection allows it
    reward_scaling: str = "fixed"  # fixed: raw / fixed_divisor; none: raw
    fixed_divisor: Optional[float] = 1.0

    def __post_init__(self) -> None:
        if self.method != "laplace_disagreement":
            raise ValueError(f"LaplaceConfig.method must be laplace_disagreement, got {self.method!r}")
        if isinstance(self.theta_s, dict):
            self.theta_s = ThetaSConfig(**{k: v for k, v in self.theta_s.items() if k != "_target_"})
        self.theta_s.include, self.theta_s.exclude = tuple(self.theta_s.include), tuple(self.theta_s.exclude)
        if not is_int(self.num_members, 2):
            raise ValueError(f"intrinsic_reward.num_members must be an integer >= 2, got {self.num_members!r}")
        for name in ("precision_reference_size", "precision_num_mc", "num_noise_samples"):
            if not is_int(getattr(self, name), 1):
                raise ValueError(f"intrinsic_reward.{name} must be a positive integer, got {getattr(self, name)!r}")
        for name in ("posterior_temperature", "damping", "beta"):
            if not is_positive(getattr(self, name)):
                raise ValueError(f"intrinsic_reward.{name} must be a finite positive number, got {getattr(self, name)!r}")
        if self.member_selection not in MEMBER_SELECTIONS:
            raise ValueError(f"intrinsic_reward.member_selection must be one of {MEMBER_SELECTIONS}")
        if not isinstance(self.shared_prefix, bool):
            raise ValueError(f"intrinsic_reward.shared_prefix must be true or false, got {self.shared_prefix!r}")
        if self.reward_scaling not in REWARD_SCALINGS:
            raise ValueError(f"intrinsic_reward.reward_scaling must be one of {REWARD_SCALINGS}, got {self.reward_scaling!r}")
        if self.reward_scaling == "fixed" and not is_positive(self.fixed_divisor):
            raise ValueError(f"intrinsic_reward.fixed_divisor must be a finite positive number, got {self.fixed_divisor!r}")

    def reward_scaling_settings(self) -> Dict:
        """What turns member disagreement into policy reward; a resume must not change it."""
        settings = dict(method=self.method, num_noise_samples=self.num_noise_samples,
                        member_selection=self.member_selection, mode=self.reward_scaling,
                        variance="population (K), mean over pixels")
        if self.reward_scaling == "fixed":
            settings["fixed_divisor"] = float(self.fixed_divisor)
        return settings

    def settings(self) -> Dict[str, Any]:
        d = asdict(self)
        d.pop("enabled")
        d["theta_s"] = dict(include=list(self.theta_s.include), exclude=list(self.theta_s.exclude),
                            expected_num_tensors=self.theta_s.expected_num_tensors, expected_dim=self.theta_s.expected_dim)
        return d


def is_int(value, minimum: int) -> bool:
    return not isinstance(value, bool) and isinstance(value, int) and value >= minimum


def is_positive(value) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(value) and value > 0


def perturbation_scale(precision: Tensor, temperature: float) -> Tensor:
    """Standard deviation of each selected coordinate: sqrt(T) / sqrt(h) (diagonal precision)."""
    if precision.ndim != 1:
        raise ValueError("Only the diagonal precision LCG produces is supported")
    if not torch.isfinite(precision).all() or not (precision > 0).all():
        raise ValueError("Historical precision must be finite and strictly positive; no fallback covariance is used")
    return math.sqrt(temperature) * precision.rsqrt()


class LaplaceScorer(EnsembleScorer):
    """The full ensemble's PMPO hook (population pixel variance from WorldModelEnv, scaled)."""
    metric_prefix = "laplace"


class LaplaceEnsemble:
    """Selected parameters, precision, frozen sample bank, member predictions; checkpointed by
    ExplorationTrainer (state_dict)."""

    def __init__(self, cfg: LaplaceConfig, denoiser: Denoiser, sigma_cfg: SigmaDistributionConfig, seed: int,
                 device: torch.device) -> None:
        self.cfg, self.denoiser, self.sigma_cfg, self.seed, self.device = cfg, denoiser, sigma_cfg, int(seed), device
        named = selected_named_parameters(denoiser, cfg.theta_s)  # LCG's selector and guards
        self.names = list(named)
        self.shapes = [tuple(p.shape) for p in named.values()]
        self.dim = sum(p.numel() for p in named.values())
        last = len(denoiser.inner_model.unet.u_blocks) - 1
        suffix = (f"unet.u_blocks.{last}.", "norm_out.", "conv_out.")
        self.prefix_supported = all(n.startswith(suffix) for n in self.names)
        self.use_prefix = cfg.shared_prefix and self.prefix_supported
        if cfg.shared_prefix and not self.prefix_supported:
            print(f"[LAPLACE] shared prefix unsupported for selection {self.names[:3]}...: full-forward members", flush=True)
        self.precision: Optional[Tensor] = None  # (dim,) diagonal, on CPU in checkpoints
        self.perturbations: Optional[Tensor] = None  # (K, dim) theta_S[k] - theta_hat_S
        self.refreshes, self.refresh_epoch = 0, None
        # Monitoring only.
        self.prediction_seconds, self.prediction_calls, self.prefix_evaluations = 0.0, 0, 0

    def settings(self) -> Dict[str, Any]:
        return dict(config=self.cfg.settings(), names=list(self.names), shapes=[list(s) for s in self.shapes],
                    dim=self.dim, precision=dict(source="lcg.lcg_historical_precision", representation="diagonal",
                                                 seed="round identifier = epoch"),
                    sampling="theta_S[k] = theta_hat_S + sqrt(T) * eps_k / sqrt(h); eps_k ~ N(0, I) from "
                             "torch.Generator(cpu).manual_seed(SeedSequence([seed, 21, epoch]))")

    def bank_seed(self, epoch: int) -> int:
        return int(np.random.SeedSequence([self.seed, 21, int(epoch)]).generate_state(1)[0])

    def selected_flat(self) -> Tensor:
        params = dict(self.denoiser.inner_model.named_parameters())
        return torch.cat([params[n].detach().reshape(-1) for n in self.names])

    # ---------------------------------------------------------------- refresh

    def refresh(self, dataset: Dataset, epoch: int) -> Dict[str, Any]:
        t0 = synchronized_time(self.device)
        named = selected_named_parameters(self.denoiser, self.cfg.theta_s)
        if list(named) != self.names:
            raise RuntimeError("The selected parameters changed since construction")
        precision = lcg_historical_precision(self.denoiser, list(named.values()), dataset, self.sigma_cfg, self.cfg,
                                             round_identifier=epoch).detach()
        t1 = synchronized_time(self.device)
        scale = perturbation_scale(precision, self.cfg.posterior_temperature)
        generator = torch.Generator(device="cpu").manual_seed(self.bank_seed(epoch))
        eps = torch.randn(self.cfg.num_members, self.dim, generator=generator)
        self.precision = precision
        self.perturbations = eps.to(self.device) * scale
        self.refreshes += 1
        self.refresh_epoch = int(epoch)
        self.prediction_seconds, self.prediction_calls, self.prefix_evaluations = 0.0, 0, 0
        t2 = synchronized_time(self.device)
        base_norm = self.selected_flat().norm()
        norms = self.perturbations.norm(dim=1)
        q = torch.quantile(precision.float().cpu(), torch.tensor([0.0, 0.01, 0.5, 0.99, 1.0]))
        s = torch.quantile(scale.float().cpu(), torch.tensor([0.0, 0.5, 1.0]))
        logs = {"laplace/selected_dim": self.dim, "laplace/posterior_temperature": self.cfg.posterior_temperature,
                "laplace/refreshes": self.refreshes, "laplace/precision_mean": float(precision.mean()),
                **{f"laplace/precision_{k}": float(v) for k, v in zip(("min", "p01", "median", "p99", "max"), q)},
                **{f"laplace/perturbation_std_{k}": float(v) for k, v in zip(("min", "median", "max"), s)},
                "laplace/selected_weight_norm": float(base_norm),
                "timing/laplace_precision_seconds": t1 - t0, "timing/laplace_sampling_seconds": t2 - t1}
        for k in range(self.cfg.num_members):
            logs[f"laplace/member_{k}_perturbation_norm"] = float(norms[k])
            logs[f"laplace/member_{k}_relative_perturbation"] = float(norms[k] / base_norm.clamp_min(1e-12))
        return logs

    # ---------------------------------------------------------------- members

    def member_params(self) -> List[Dict[str, Tensor]]:
        """Each member's selected parameter values (base + its perturbation), reshaped; the base
        weights are read, never written."""
        if self.perturbations is None:
            raise RuntimeError("The Laplace sample bank has not been drawn")
        base = self.selected_flat()
        members = []
        for delta in self.perturbations:
            flat, values, offset = base + delta, {}, 0
            for name, shape in zip(self.names, self.shapes):
                n = int(np.prod(shape))
                values[name] = flat[offset:offset + n].view(shape)
                offset += n
            members.append(values)
        return members

    @torch.no_grad()
    def predict_members(self, sampler: DiffusionSampler, obs: Tensor, act: Tensor, noise: Tensor,
                        reuse_prefix: Optional[bool] = None) -> Tensor:
        """(K, b, c, h, w) completed next-frame predictions, all members from the same initial
        noise, each with its sampled parameters for every denoising step. reuse_prefix=False is
        the straightforward reference (full forward per member at every evaluation)."""
        if sampler.cfg.s_churn != 0:
            raise ValueError("Laplace members share the initial noise only; it needs s_churn=0")
        start = synchronized_time(self.device)
        params = self.member_params()
        reuse = self.use_prefix if reuse_prefix is None else reuse_prefix
        if reuse and not self.prefix_supported:
            raise ValueError("Shared-prefix reuse is not supported for this parameter selection")
        first = [None] * len(params)
        if reuse:
            b, t, c, h, w = obs.shape
            first = shared_prefix_first_evaluation(self.denoiser, sampler.sigmas[0], noise,
                                                   obs.reshape(b, t * c, h, w), act, params)
            self.prefix_evaluations += 1
        predictions = torch.stack([sampler.sample(obs, act, noise, params=p, first_denoised=f)[0]
                                   for p, f in zip(params, first)])
        self.prediction_seconds += synchronized_time(self.device) - start
        self.prediction_calls += 1
        return predictions

    # ---------------------------------------------------------------- state

    def state_dict(self) -> Dict[str, Any]:
        return dict(settings=self.settings(), seed=self.seed,
                    precision=None if self.precision is None else self.precision.cpu(),
                    perturbations=None if self.perturbations is None else self.perturbations.cpu(),
                    refreshes=self.refreshes, refresh_epoch=self.refresh_epoch,
                    bank_seed=None if self.refresh_epoch is None else self.bank_seed(self.refresh_epoch))

    def check_settings(self, state: Optional[Dict[str, Any]]) -> None:
        saved = None if state is None else state["settings"]
        if saved != self.settings() or (state is not None and state["seed"] != self.seed):
            raise ValueError(f"Checkpoint was trained with Laplace settings {saved}, the config gives "
                             f"{self.settings()}; a resume must not change them")

    def load_state_dict(self, state: Dict[str, Any]) -> None:
        self.check_settings(state)
        self.precision = None if state["precision"] is None else state["precision"].to(self.device)
        self.perturbations = None if state["perturbations"] is None else state["perturbations"].to(self.device)
        self.refreshes, self.refresh_epoch = state["refreshes"], state["refresh_epoch"]


@torch.no_grad()
def shared_prefix_first_evaluation(denoiser: Denoiser, sigma: Tensor, x: Tensor, obs: Tensor, act: Tensor,
                                   member_params: List[Dict[str, Tensor]]) -> List[Tensor]:
    """D_k(x, sigma) for every member at an input shared by all members (the first denoising
    evaluation): conditioning, conv_in, the down path, the middle blocks, every decoder level but
    the last, and its upsampling run once; the last decoder level, norm_out and conv_out run per
    member with its parameters. Valid only when every sampled parameter lies in that suffix.
    Mirrors InnerModel.forward / UNet.forward."""
    inner, unet = denoiser.inner_model, denoiser.inner_model.unet
    cs = denoiser.compute_conditioners(sigma)
    noisy, rescaled_obs = x * cs.c_in, obs / denoiser.cfg.sigma_data
    cond = inner.cond_proj(inner.noise_emb(cs.c_noise) + inner.act_emb(act))
    h = inner.conv_in(torch.cat((rescaled_obs, noisy), dim=1))
    *_, height, width = h.size()
    n = unet._num_down
    h = F.pad(h, (0, math.ceil(width / 2 ** n) * 2 ** n - width, 0, math.ceil(height / 2 ** n) * 2 ** n - height))
    d_outputs = []
    for block, down in zip(unet.d_blocks, unet.downsamples):
        x_down = down(h)
        h, block_outputs = block(x_down, cond)
        d_outputs.append((x_down, *block_outputs))
    h, _ = unet.mid_blocks(h, cond)
    skips = list(reversed(d_outputs))
    last = len(unet.u_blocks) - 1
    for i in range(last):
        h, _ = unet.u_blocks[i](unet.upsamples[i](h), cond, skips[i][::-1])
    x_up = unet.upsamples[last](h)

    def sub(params, prefix):
        return {k[len(prefix):]: v for k, v in params.items() if k.startswith(prefix)}

    outputs = []
    for params in member_params:
        y, _ = functional_call(unet.u_blocks[last], sub(params, f"unet.u_blocks.{last}."), (x_up, cond, skips[last][::-1]))
        y = y[..., :height, :width]
        y = functional_call(inner.norm_out, sub(params, "norm_out."), (y,))
        y = functional_call(inner.conv_out, sub(params, "conv_out."), (F.silu(y),))
        outputs.append(denoiser.wrap_model_output(x, y, cs))
    return outputs


def synchronized_time(device: torch.device) -> float:
    if torch.device(device).type == "cuda":
        torch.cuda.synchronize(device)
    return time.perf_counter()
