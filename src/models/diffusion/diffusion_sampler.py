from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import torch
from torch import Tensor

from .denoiser import Denoiser


@dataclass
class DiffusionSamplerConfig:
    num_steps_denoising: int
    sigma_min: float = 2e-3
    sigma_max: float = 5
    rho: int = 7
    order: int = 1
    s_churn: float = 0
    s_tmin: float = 0
    s_tmax: float = float("inf")
    s_noise: float = 1


class DiffusionSampler:
    def __init__(self, denoiser: Denoiser, cfg: DiffusionSamplerConfig) -> None:
        self.denoiser = denoiser
        self.cfg = cfg
        self.sigmas = build_sigmas(cfg.num_steps_denoising, cfg.sigma_min, cfg.sigma_max, cfg.rho, denoiser.device)

    @torch.no_grad()
    def sample(self, prev_obs: Tensor, prev_act: Tensor, noise: Optional[Tensor] = None,
               params: Optional[Dict[str, Tensor]] = None,
               first_denoised: Optional[Tensor] = None) -> Tuple[Tensor, List[Tensor]]:
        """noise: optional initial noise (b, c, h, w), e.g. shared across ensemble members;
        with s_churn=0 it is the only randomness of the sample. params: optional InnerModel
        parameter values used for every denoiser evaluation of this chain (functional
        substitution; the denoiser is not modified). first_denoised: the already computed
        output of the first evaluation, D(noise, sigma_0) with these params (s_churn=0 only)."""
        if first_denoised is not None and self.cfg.s_churn != 0:
            raise ValueError("first_denoised requires s_churn=0 (the first input must be the given noise)")
        device = prev_obs.device
        b, t, c, h, w = prev_obs.size()
        prev_obs = prev_obs.reshape(b, t * c, h, w)
        s_in = torch.ones(b, device=device)
        gamma_ = min(self.cfg.s_churn / (len(self.sigmas) - 1), 2**0.5 - 1)
        x = torch.randn(b, c, h, w, device=device) if noise is None else noise.clone()
        trajectory = [x]
        for i, (sigma, next_sigma) in enumerate(zip(self.sigmas[:-1], self.sigmas[1:])):
            gamma = gamma_ if self.cfg.s_tmin <= sigma <= self.cfg.s_tmax else 0
            sigma_hat = sigma * (gamma + 1)
            if gamma > 0:
                eps = torch.randn_like(x) * self.cfg.s_noise
                x = x + eps * (sigma_hat**2 - sigma**2) ** 0.5
            if i == 0 and first_denoised is not None:
                denoised = first_denoised
            elif params is None:
                denoised = self.denoiser.denoise(x, sigma, prev_obs, prev_act)
            else:
                denoised = self.denoiser.denoise(x, sigma, prev_obs, prev_act, params=params)
            d = (x - denoised) / sigma_hat
            dt = next_sigma - sigma_hat
            if self.cfg.order == 1 or next_sigma == 0:
                # Euler method
                x = x + d * dt
            else:
                # Heun's method
                x_2 = x + d * dt
                if params is None:
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, prev_obs, prev_act)
                else:
                    denoised_2 = self.denoiser.denoise(x_2, next_sigma * s_in, prev_obs, prev_act, params=params)
                d_2 = (x_2 - denoised_2) / next_sigma
                d_prime = (d + d_2) / 2
                x = x + d_prime * dt
            trajectory.append(x)
        return x, trajectory


def build_sigmas(num_steps: int, sigma_min: float, sigma_max: float, rho: int, device: torch.device) -> Tensor:
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    l = torch.linspace(0, 1, num_steps, device=device)
    sigmas = (max_inv_rho + l * (min_inv_rho - max_inv_rho)) ** rho
    return torch.cat((sigmas, sigmas.new_zeros(1)))

