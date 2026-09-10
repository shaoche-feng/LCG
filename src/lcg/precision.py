from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from data import BatchSampler, Dataset, SegmentId
from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig, apply_noise_from_samples

from .gauss_newton import compute_vjp
from .sigma_strata import sample_sigma_stratum


def sample_valid_transitions(
    dataset: Dataset,
    batch_size: int,
    num_steps_conditioning: int,
    seed: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> List[SegmentId]:
    """B candidate LCG transitions (x_i, y_i), drawn through DIAMOND's existing replay
    sampling machinery (BatchSampler + Dataset) rather than a new pipeline.

    One valid transition is a segment of `num_steps_conditioning + 1` frames whose *last*
    frame (the target y_i) is a real, unpadded environment step; the preceding
    `num_steps_conditioning` frames (x_i's conditioning window) may be left-padded with
    zeros near an episode's start -- exactly what Denoiser.forward already tolerates via
    mask_padding (it masks the loss on the target position, never on the conditioning
    window). With `can_sample_beyond_end=False`, BatchSampler only ever produces segments
    whose stop position is a real, in-episode step, so every draw satisfies this by
    construction; `load_transition` still checks it defensively.

    Uses natural (unweighted, i.e. `sample_weights=None`) episode-length-proportional
    sampling -- not the training-time recency-biased `sample_weights` curriculum -- since
    h_D is meant to summarize the whole historical dataset D, not a training curriculum.
    """
    if seed is not None:
        np.random.seed(seed)
    sampler = BatchSampler(
        dataset,
        rank,
        world_size,
        batch_size,
        num_steps_conditioning + 1,
        sample_weights=None,
        can_sample_beyond_end=False,
    )
    return sampler.sample()


def load_transition(
    dataset: Dataset, segment_id: SegmentId, num_steps_conditioning: int, device: torch.device
) -> Tuple[Tensor, Tensor, Tensor]:
    """Materializes one (x_i, y_i) pair for segment_id: x_i = (obs window, act window) of
    length num_steps_conditioning ending just before the target, y_i = the target frame.
    Shapes match what Denoiser.compute_model_output/compute_conditioners expect for a
    batch of size 1: obs (1, num_steps_conditioning*img_channels, H, W), act (1,
    num_steps_conditioning, ...), y (1, img_channels, H, W).
    """
    segment = dataset[segment_id]
    n = num_steps_conditioning
    assert segment.mask_padding[n], f"target frame is padding for segment {segment_id}"
    obs = segment.obs[:n].reshape(1, -1, *segment.obs.shape[-2:]).to(device)
    act = segment.act[:n].unsqueeze(0).to(device)
    y = segment.obs[n].unsqueeze(0).to(device)
    return obs, act, y


def historical_precision(
    denoiser: Denoiser,
    theta_s_params: List[nn.Parameter],
    dataset: Dataset,
    sigma_cfg: SigmaDistributionConfig,
    B: int,
    N: Optional[int] = None,
    num_strata: int = 3,
    beta: float = 1.0,
    damping: float = 1e-4,
    seed: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tensor:
    """Offline diagonal Laplace/Gauss-Newton precision estimate, Algorithm 2:

        h_D_hat = damping * 1_{d_S} + beta * (N / B) * sum_{i in B} g_i
        g_i = (1 / num_strata) * sum_{m=1}^{num_strata} v_i^(m) (.) v_i^(m)

    B is a random subset of size |B| drawn (with replacement, matching DIAMOND's own
    minibatch sampling convention) from the full dataset D of size N (defaults to
    `dataset.num_steps`: see `sample_valid_transitions`/`load_transition` for what counts
    as one valid transition in DIAMOND's replay representation, and why `dataset.num_steps`
    is exactly that count under this pipeline's left-padding convention). The N/B factor
    makes h_D_hat an unbiased estimate of the full-dataset sum for any B, so its scale
    should not depend systematically on the precision-estimation batch size.

    Uses the exact DIAMOND training corruption law, y_sigma = y + sigma*eps +
    sigma_offset_noise*eps_offset (models.diffusion.denoiser.apply_noise_from_samples,
    the same helper Denoiser.apply_noise calls) -- previously this used y + sigma*eps
    only, omitting the offset-noise term DIAMOND training actually applies; fixed so
    historical precision evaluates D_theta/F_theta at the same corrupted-input
    distribution the denoiser was trained/queried on. compute_vjp/differentiable_denoise
    are unmodified: compute_conditioners already derives the correct effective sigma
    (sqrt(sigma^2+sigma_offset_noise^2)) from the bare sigma passed alongside y_sigma, so
    only the corruption construction here needed fixing, not the preconditioning math.
    Caller is responsible for `denoiser.eval()`/frozen weights; this function never calls
    `.backward()`, never touches `.grad`, and never modifies denoiser parameters (only
    `torch.autograd.grad(..., retain_graph=False, create_graph=False)` inside `compute_vjp`
    is used, and each transition's graph is built and discarded independently).
    """
    assert B > 0 and num_strata > 0
    device = denoiser.device
    N = dataset.num_steps if N is None else N
    num_steps_conditioning = denoiser.cfg.inner_model.num_steps_conditioning

    d_S = sum(p.numel() for p in theta_s_params)
    h = damping * torch.ones(d_S, device=device)

    segment_ids = sample_valid_transitions(dataset, B, num_steps_conditioning, seed, rank, world_size)
    scale = beta * (N / B)

    for segment_id in segment_ids:
        obs, act, y = load_transition(dataset, segment_id, num_steps_conditioning, device)
        g_i = torch.zeros(d_S, device=device)
        for m in range(num_strata):
            sigma = sample_sigma_stratum(sigma_cfg, m, num_strata, 1, device)
            eps = torch.randn_like(y)
            eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=device)
            y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            v, _ = compute_vjp(denoiser, theta_s_params, y_sigma, sigma, obs, act)
            g_i = g_i + (v * v) / num_strata
        h = h + scale * g_i

    return h
