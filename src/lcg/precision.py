from typing import List, Optional, Tuple

import numpy as np
import torch
from torch import Tensor
import torch.nn as nn

from data import Dataset, SegmentId
from models.diffusion.denoiser import Denoiser, SigmaDistributionConfig, apply_noise_from_samples

from .gauss_newton import compute_vjp
from .sigma_strata import sample_sigma_stratum


def sample_uniform_historical_transitions(
    dataset: Dataset,
    batch_size: int,
    num_steps_conditioning: int,
    seed: Optional[int] = None,
    rank: int = 0,
    world_size: int = 1,
    replace: bool = False,
) -> List[SegmentId]:
    """B historical LCG transitions (x_i, y_i), sampled uniformly over every valid
    transition in the (rank's partition of the) dataset -- P((episode, t)) = 1/N for
    every valid (episode_id, t), N = dataset.num_steps (summed over the rank's eligible
    episodes).

    LCG-specific: does NOT use DIAMOND's generic BatchSampler. BatchSampler.sample()'s
    can_sample_beyond_end=False branch draws a uniform timestep t and THEN shifts it by an
    independent random offset in [0, seq_length) before clipping to the episode boundary
    (stop = min(L_e, t+1+offset)) -- appropriate for training-time window diversity (which
    of the seq_length frames in a window is the "last" one should vary), but wrong for
    LCG's "one uniformly-sampled historical transition = one curvature sample" semantics:
    the transition that ends up as the actual VJP target is stop-1, not t, so the
    resulting distribution over targets is NOT uniform -- it under-samples the first
    seq_length-1 transitions of every episode and produces a probability spike at each
    episode's final transition (all overflowing (t, offset) combinations collapse onto
    it via the min-clip). See docs/lcg_diagnostic/historical_sampling_fix/ for the
    quantified bias and an old-vs-new comparison diagnostic.

    Every frame s_t (t in [0, L_e)) of every episode e is a valid target here (the
    num_steps_conditioning frames before it may be left-padded near an episode's start,
    exactly as DIAMOND's own Denoiser.forward tolerates via mask_padding) -- so N =
    dataset.num_steps = sum_e L_e is exactly the number of valid (episode, t) pairs, and
    a uniform draw of a global frame index in [0, N) followed by mapping it back to
    (episode_id, t) via dataset.start_idx gives P((episode, t)) = 1/N directly, with no
    further randomization: the SegmentId's stop is built as t+1 (not a randomized
    offset), so the sampled t IS the transition whose curvature gets measured.

    batch_size transitions are drawn WITHOUT replacement by default (replace=False),
    matching the N/B unbiased-subset-sum estimator's "B distinct historical transitions"
    interpretation; raises if batch_size exceeds the number of available transitions
    unless replace=True is passed explicitly. Uses a local np.random.Generator (not
    global np.random state) -- this reduces, not adds to, the codebase's existing global-
    RNG-contamination surface (a separate, not-yet-fixed issue tracked elsewhere).
    """
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

    B is a subset of size |B| drawn WITHOUT replacement, uniformly over every valid
    historical transition (P((episode,t))=1/N for every valid (episode,t) pair -- see
    sample_uniform_historical_transitions, which replaced the previous BatchSampler-based
    sampler after discovering it did not actually sample the transition it claimed to:
    BatchSampler's endpoint-shifting logic made the ACTUAL VJP target a randomized
    function of the originally-drawn timestep, non-uniform over historical transitions),
    from the full dataset D of size N (defaults to `dataset.num_steps`: see
    `sample_uniform_historical_transitions`/`load_transition` for what counts as one
    valid transition in DIAMOND's replay representation, and why `dataset.num_steps` is
    exactly that count under this pipeline's left-padding convention). The N/B factor
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
        for m in range(num_strata):
            sigma = sample_sigma_stratum(sigma_cfg, m, num_strata, 1, device, generator=torch_gen)
            eps = torch.randn(y.shape, dtype=y.dtype, device=y.device, generator=torch_gen)
            eps_offset = torch.randn(y.shape[0], y.shape[1], 1, 1, device=device, generator=torch_gen)
            y_sigma = apply_noise_from_samples(y, sigma, eps, eps_offset, denoiser.cfg.sigma_offset_noise).detach()
            v, _ = compute_vjp(denoiser, theta_s_params, y_sigma, sigma, obs, act, generator=torch_gen)
            g_i = g_i + (v * v) / num_strata
        h = h + scale * g_i

    return h
