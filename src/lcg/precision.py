import math
from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from data import Dataset, SegmentId
from models.diffusion.denoiser import (
    Denoiser,
    SigmaDistributionConfig,
    apply_noise_from_samples,
    sample_sigma_training_distribution,
)

_SQRT_2 = math.sqrt(2.0)


def sample_uniform_historical_transitions(
    dataset: Dataset,
    batch_size: int,
    num_steps_conditioning: int,
    seed: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    replace: bool = False,
) -> List[SegmentId]:
    if world_size > 1:
        eligible_episodes = np.arange(rank, dataset.num_episodes, world_size)
    else:
        eligible_episodes = np.arange(dataset.num_episodes)
    eligible_lengths = dataset.lengths[eligible_episodes]
    n_available = int(eligible_lengths.sum())

    if not replace:
        assert batch_size <= n_available, (
            f"requested batch_size={batch_size} historical transitions without replacement, "
            f"but only {n_available} distinct transitions are available "
            f"(rank={rank}, world_size={world_size}); pass replace=True to explicitly allow "
            f"duplicate transitions, or reduce batch_size."
        )

    rng = np.random.default_rng(seed)
    global_indices = rng.choice(n_available, size=batch_size, replace=replace)

    # cumulative frame offsets WITHIN the eligible-episode subset, for global index -> (episode, t)
    eligible_start_idx = np.concatenate(([0], np.cumsum(eligible_lengths)[:-1]))
    seq_length = num_steps_conditioning + 1

    segment_ids = []
    for g in global_indices:
        local_ep_idx = int(np.searchsorted(eligible_start_idx, g, side="right") - 1)
        episode_id = int(eligible_episodes[local_ep_idx])
        t = int(g - eligible_start_idx[local_ep_idx])
        stop = t + 1
        start = stop - seq_length
        segment_ids.append(SegmentId(episode_id, start, stop))
    return segment_ids


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


def _backward_vjp_probe(
    denoiser: Denoiser,
    params: List[nn.Parameter],
    y_sigma: Tensor,
    sigma: Tensor,
    obs: Tensor,
    act: Tensor,
    xi: Optional[Tensor] = None,
    generator: Optional[torch.Generator] = None,
) -> Tensor:
    assert y_sigma.size(0) == 1, "_backward_vjp_probe handles one transition (batch size 1) at a time"
    cs = denoiser.compute_conditioners(sigma)
    model_output = denoiser.compute_model_output(y_sigma, obs, act, cs)  # raw F_theta
    if xi is None:
        xi = torch.randn(model_output.shape, dtype=model_output.dtype, device=model_output.device, generator=generator)
    scalar = _SQRT_2 * (xi * model_output).sum()
    grads = torch.autograd.grad(scalar, params, retain_graph=False, create_graph=False)
    return torch.cat([g.reshape(-1) for g in grads])


def historical_precision(
    denoiser: Denoiser,
    theta_s_params: List[nn.Parameter],
    dataset: Dataset,
    sigma_cfg: SigmaDistributionConfig,
    B: int,
    N: Optional[int] = None,
    num_mc: int = 3,
    beta: float = 1.0,
    damping: float = 1e-4,
    seed: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
) -> Tensor:
    assert B > 0 and num_mc > 0
    device = denoiser.device
    N = dataset.num_steps if N is None else N
    num_steps_conditioning = denoiser.cfg.inner_model.num_steps_conditioning

    d_S = sum(p.numel() for p in theta_s_params)
    h = damping * torch.ones(d_S, device=device)

    # Local, device-matched torch.Generator -- RNG isolation from DIAMOND: seeded (not
    # torch.manual_seed'd globally) when `seed` is given, else auto-seeded from entropy;
    # either way, every torch.randn* draw below uses THIS generator explicitly, so this
    # call never mutates the global torch RNG stream DIAMOND's own code observes.
    torch_gen = torch.Generator(device=device)
    if seed is not None:
        torch_gen.manual_seed(seed)

    segment_ids = sample_uniform_historical_transitions(dataset, B, num_steps_conditioning, seed, rank, world_size)
    scale = beta * (N / B)

    for segment_id in segment_ids:
        obs, act, y = load_transition(dataset, segment_id, num_steps_conditioning, device)
        g_i = torch.zeros(d_S, device=device)
        for _ in range(num_mc):
            sigma = sample_sigma_training_distribution(sigma_cfg, 1, device, generator=torch_gen)
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=torch_gen)
            eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=device, generator=torch_gen)
            y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            v = _backward_vjp_probe(denoiser, theta_s_params, y_sigma, sigma, obs, act, generator=torch_gen)
            g_i = g_i + (v * v) / num_mc
        h = h + scale * g_i

    return h
