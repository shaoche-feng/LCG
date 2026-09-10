"""
Permanent regression tests for Full-CRN sharing in the production forward-JVP scorer.
A complete CRN entry is now (sigma, eps, eps_offset, eta) -- verify the same complete
entry is reused across every logical candidate and every computational chunk,
independent of how candidates happen to be chunked.
"""
import torch
from lcg.forward_jvp import (
    frozen_named_parameters,
    make_jvp_bank,
    score_one_jvp_bank,
    selected_named_parameters,
)
from models.diffusion import SigmaDistributionConfig

TINY_IMG_CHANNELS = 3
TINY_IMG_SIZE = 8
TINY_SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)


def _make_candidates(denoiser, n):
    torch.manual_seed(123)
    candidates = []
    for _ in range(n):
        obs = torch.randn(1, denoiser.cfg.inner_model.num_steps_conditioning * TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE)
        act = torch.randn(1, denoiser.cfg.inner_model.num_steps_conditioning, 2)
        y = torch.randn(1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE)
        candidates.append((obs, act, y))
    return candidates


def test_eps_offset_shared_not_per_candidate(tiny_denoiser):
    theta_s_named = selected_named_parameters(tiny_denoiser)
    d_S = sum(p.numel() for p in theta_s_named.values())
    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE]), d_S,
        device=torch.device("cpu"), num_samples=4, seed=0,
    )
    for eps_offset in bank.epsilons_offset:
        assert eps_offset.shape[0] == 1, "eps_offset must broadcast across candidates, not one per candidate"
        assert eps_offset.shape[1] == TINY_IMG_CHANNELS


def test_duplicate_candidate_scores_identically_across_chunks(tiny_denoiser):
    """The defining Full-CRN property: a candidate's score must not depend on which
    computational chunk it lands in."""
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    h_D = torch.rand(d_S) + 0.5
    h_D_inv_sqrt = h_D.rsqrt()

    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE]), d_S,
        device=torch.device("cpu"), num_samples=3, seed=1,
    )
    candidates = _make_candidates(denoiser, 6)
    # place an exact duplicate of candidate 0 at index 4, forced into a different chunk
    # under chunk_size=2 (chunks are [0,1], [2,3], [4,5])
    candidates[4] = tuple(t.clone() for t in candidates[0])

    scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=2)
    assert torch.equal(scores[0], scores[4])


def test_chunk_size_does_not_change_scores(tiny_denoiser):
    """Chunk size is purely a batching/throughput choice -- it must not change which
    logical CRN samples are used or the resulting scores."""
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    h_D = torch.rand(d_S) + 0.5
    h_D_inv_sqrt = h_D.rsqrt()

    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE]), d_S,
        device=torch.device("cpu"), num_samples=3, seed=2,
    )
    candidates = _make_candidates(denoiser, 6)

    scores_c1 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=1)
    scores_c3 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=3)
    scores_c6 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=6)

    assert torch.allclose(scores_c1, scores_c3, atol=1e-5)
    assert torch.allclose(scores_c1, scores_c6, atol=1e-5)
