from dataclasses import dataclass
from typing import Optional

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F

from data import Batch
from .inner_model import InnerModel, InnerModelConfig
from utils import LossAndLogs


def add_dims(input: Tensor, n: int) -> Tensor:
    return input.reshape(input.shape + (1,) * (n - input.ndim))


def apply_noise_from_samples(x: Tensor, sigma: Tensor, eps: Tensor, eps_offset: Tensor, sigma_offset_noise: float) -> Tensor:
    """The single authoritative DIAMOND corruption formula:

        y_sigma = x + sigma * eps + sigma_offset_noise * eps_offset

    eps/eps_offset are supplied by the caller rather than drawn internally, so callers
    outside training (e.g. LCG's historical-precision/forward-JVP estimators) can control
    them explicitly -- under Full CRN, LCG shares one (sigma, eps, eps_offset) triple
    across every candidate in a scoring batch. `Denoiser.apply_noise` draws its own fresh
    eps/eps_offset and calls this; the two paths must never diverge into separate
    hand-written formulas.

    sigma broadcasts via add_dims against x's shape (e.g. (B,) -> (B,1,1,1)). eps must
    already match x's shape. eps_offset must be broadcastable against x -- DIAMOND
    training uses (B,C,1,1) (one independent draw per training example, spatially
    constant within each channel); LCG's Full CRN uses (1,C,1,1) (one draw shared across
    the whole candidate batch via ordinary broadcasting)."""
    return x + add_dims(sigma, x.ndim) * eps + sigma_offset_noise * eps_offset


@dataclass
class Conditioners:
    c_in: Tensor
    c_out: Tensor
    c_skip: Tensor
    c_noise: Tensor


@dataclass
class SigmaDistributionConfig:
    loc: float
    scale: float
    sigma_min: float
    sigma_max: float


@dataclass
class DenoiserConfig:
    inner_model: InnerModelConfig
    sigma_data: float
    sigma_offset_noise: float


class Denoiser(nn.Module):
    def __init__(self, cfg: DenoiserConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.inner_model = InnerModel(cfg.inner_model)
        self.sample_sigma_training = None

    @property
    def device(self) -> torch.device:
        return self.inner_model.noise_emb.weight.device

    def setup_training(self, cfg: SigmaDistributionConfig) -> None:
        assert self.sample_sigma_training is None

        def sample_sigma(n: int, device: torch.device):
            s = torch.randn(n, device=device) * cfg.scale + cfg.loc
            return s.exp().clip(cfg.sigma_min, cfg.sigma_max)

        self.sample_sigma_training = sample_sigma
    
    def apply_noise(self, x: Tensor, sigma: Tensor, sigma_offset_noise: float) -> Tensor:
        b, c, _, _ = x.shape
        # Draw order preserved exactly (offset noise first, then pixel noise) so this
        # remains bit-identical to the pre-refactor inline formula under a fixed seed.
        eps_offset = torch.randn(b, c, 1, 1, device=self.device)
        eps = torch.randn_like(x)
        return apply_noise_from_samples(x, sigma, eps, eps_offset, sigma_offset_noise)

    def compute_conditioners(self, sigma: Tensor) -> Conditioners:
        sigma = (sigma**2 + self.cfg.sigma_offset_noise**2).sqrt()
        c_in = 1 / (sigma**2 + self.cfg.sigma_data**2).sqrt()
        c_skip = self.cfg.sigma_data**2 / (sigma**2 + self.cfg.sigma_data**2)
        c_out = sigma * c_skip.sqrt()
        c_noise = sigma.log() / 4
        return Conditioners(*(add_dims(c, n) for c, n in zip((c_in, c_out, c_skip, c_noise), (4, 4, 4, 1, 1))))

    def compute_model_output(self, noisy_next_obs: Tensor, obs: Tensor, act: Tensor, cs: Conditioners) -> Tensor:
        rescaled_obs = obs / self.cfg.sigma_data
        rescaled_noise = noisy_next_obs * cs.c_in
        return self.inner_model(rescaled_noise, cs.c_noise, rescaled_obs, act)
    
    @torch.no_grad()
    def wrap_model_output(self, noisy_next_obs: Tensor, model_output: Tensor, cs: Conditioners) -> Tensor:
        d = cs.c_skip * noisy_next_obs + cs.c_out * model_output
        # Quantize to {0, ..., 255}, then back to [-1, 1]
        d = d.clamp(-1, 1).add(1).div(2).mul(255).byte().div(255).mul(2).sub(1)
        return d
    
    @torch.no_grad()
    def denoise(self, noisy_next_obs: Tensor, sigma: Tensor, obs: Tensor, act: Tensor) -> Tensor:
        cs = self.compute_conditioners(sigma)
        model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)
        denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
        return denoised

    def forward(self, batch: Batch) -> LossAndLogs:
        n = self.cfg.inner_model.num_steps_conditioning
        seq_length = batch.obs.size(1) - n

        all_obs = batch.obs.clone()
        loss = 0

        for i in range(seq_length):
            obs = all_obs[:, i : n + i]
            next_obs = all_obs[:, n + i]
            act = batch.act[:, i : n + i]
            mask = batch.mask_padding[:, n + i]

            b, t, c, h, w = obs.shape
            obs = obs.reshape(b, t * c, h, w)

            sigma = self.sample_sigma_training(b, self.device)
            noisy_next_obs = self.apply_noise(next_obs, sigma, self.cfg.sigma_offset_noise)

            cs = self.compute_conditioners(sigma)
            model_output = self.compute_model_output(noisy_next_obs, obs, act, cs)

            target = (next_obs - cs.c_skip * noisy_next_obs) / cs.c_out
            loss += F.mse_loss(model_output[mask], target[mask])

            denoised = self.wrap_model_output(noisy_next_obs, model_output, cs)
            all_obs[:, n + i] = denoised

        loss /= seq_length
        return loss, {"loss_denoising": loss.detach()}
