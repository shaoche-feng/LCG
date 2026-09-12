"""
Permanent regression tests for LCG's RNG isolation from DIAMOND's global random-number
streams. Every LCG operation that draws randomness now uses a local, explicit generator
(np.random.default_rng for NumPy, torch.Generator for Torch) instead of mutating global
state (np.random.seed/torch.manual_seed) -- these tests protect that invariant directly,
plus same-seed reproducibility and different-seed distinctness, which must hold exactly
as before despite no longer touching global state.
"""
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


sys.path.insert(0, str(_find_repo_root(Path(__file__).parent) / "src"))

from data import Dataset, Episode  # noqa: E402
from lcg.forward_jvp import (  # noqa: E402
    make_jvp_bank,
    score_one_jvp_bank,
)
from lcg.precision import historical_precision, sample_uniform_historical_transitions  # noqa: E402
from lcg.theta_s import ThetaSConfig, frozen_named_parameters, selected_named_parameters, selected_parameters  # noqa: E402
from models.diffusion import SigmaDistributionConfig  # noqa: E402

IMG_CHANNELS, IMG_SIZE, ACTION_DIM = 3, 8, 2
NUM_STEPS_CONDITIONING = 1
TINY_SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)


def _default_theta_s_config(denoiser) -> ThetaSConfig:
    """Current-production-equivalent ThetaSConfig for whichever denoiser is passed --
    duplicated per test file, matching this suite's existing convention (see conftest.py's
    default_theta_s_config, the same helper)."""
    last_idx = len(denoiser.inner_model.unet.u_blocks) - 1
    return ThetaSConfig(include=(f"unet.u_blocks.{last_idx}.*", "norm_out.*", "conv_out.*"), exclude=())


def _make_episode(length):
    return Episode(
        obs=torch.randn(length, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
        act=torch.randn(length, ACTION_DIM),
        rew=torch.zeros(length),
        end=torch.zeros(length, dtype=torch.uint8),
        trunc=torch.zeros(length, dtype=torch.uint8),
        info={},
    )


@pytest.fixture
def small_dataset(tmp_path):
    dataset = Dataset(tmp_path / "ds", "rng_test_ds", cache_in_ram=True)
    for L in [2, 3, 5]:
        dataset.add_episode(_make_episode(L))
    return dataset


# --------------------------------------------------------------------------------------
# A/B/C: global RNG preservation (NumPy, Torch CPU, Torch CUDA)
# --------------------------------------------------------------------------------------


def test_numpy_global_rng_preserved_by_transition_sampling(small_dataset):
    np.random.seed(123)
    expected = np.random.random(50)

    np.random.seed(123)
    sample_uniform_historical_transitions(small_dataset, batch_size=5, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=999)
    actual = np.random.random(50)

    np.testing.assert_array_equal(actual, expected)


def test_torch_cpu_global_rng_preserved_by_candidate_bank(tiny_denoiser):
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    d_S = sum(p.numel() for p in theta_s_named.values())
    y_shape = torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE])

    torch.manual_seed(456)
    expected = torch.rand(50)

    torch.manual_seed(456)
    make_jvp_bank(TINY_SIGMA_CFG, y_shape, d_S, device=torch.device("cpu"), num_samples=4, seed=999)
    actual = torch.rand(50)

    assert torch.equal(actual, expected)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_torch_cuda_global_rng_preserved_by_candidate_bank(tiny_denoiser):
    device = torch.device("cuda")
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    d_S = sum(p.numel() for p in theta_s_named.values())
    y_shape = torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE])

    torch.cuda.manual_seed(456)
    expected = torch.rand(50, device=device)

    torch.cuda.manual_seed(456)
    make_jvp_bank(TINY_SIGMA_CFG, y_shape, d_S, device=device, num_samples=4, seed=999)
    actual = torch.rand(50, device=device)

    assert torch.equal(actual, expected)


# --------------------------------------------------------------------------------------
# D/E: same-seed reproducibility, different-seed distinctness
# --------------------------------------------------------------------------------------


def test_same_seed_bank_is_identical(tiny_denoiser):
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    d_S = sum(p.numel() for p in theta_s_named.values())
    y_shape = torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE])

    bank1 = make_jvp_bank(TINY_SIGMA_CFG, y_shape, d_S, device=torch.device("cpu"), num_samples=3, seed=7)
    bank2 = make_jvp_bank(TINY_SIGMA_CFG, y_shape, d_S, device=torch.device("cpu"), num_samples=3, seed=7)

    for a, b in zip(bank1.sigmas, bank2.sigmas):
        assert torch.equal(a, b)
    for a, b in zip(bank1.epsilons, bank2.epsilons):
        assert torch.equal(a, b)
    for a, b in zip(bank1.epsilons_offset, bank2.epsilons_offset):
        assert torch.equal(a, b)
    for a, b in zip(bank1.etas, bank2.etas):
        assert torch.equal(a, b)


def test_different_seed_bank_is_different(tiny_denoiser):
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    d_S = sum(p.numel() for p in theta_s_named.values())
    y_shape = torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE])

    bank1 = make_jvp_bank(TINY_SIGMA_CFG, y_shape, d_S, device=torch.device("cpu"), num_samples=3, seed=7)
    bank2 = make_jvp_bank(TINY_SIGMA_CFG, y_shape, d_S, device=torch.device("cpu"), num_samples=3, seed=8)

    assert not all(torch.equal(a, b) for a, b in zip(bank1.etas, bank2.etas))


# --------------------------------------------------------------------------------------
# F/G: historical transition sampling and historical_precision reproducibility
# --------------------------------------------------------------------------------------


def test_historical_transition_sampling_reproducible(small_dataset):
    ids1 = sample_uniform_historical_transitions(small_dataset, batch_size=6, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=42)
    ids2 = sample_uniform_historical_transitions(small_dataset, batch_size=6, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=42)
    assert [(s.episode_id, s.start, s.stop) for s in ids1] == [(s.episode_id, s.start, s.stop) for s in ids2]


def test_historical_precision_reproducible(tiny_denoiser, small_dataset):
    denoiser = tiny_denoiser
    dataset = small_dataset
    params = selected_parameters(denoiser, _default_theta_s_config(denoiser))
    N = dataset.num_steps

    h_D_1 = historical_precision(denoiser, params, dataset, TINY_SIGMA_CFG, B=N, N=N, num_mc=3, seed=99)
    h_D_2 = historical_precision(denoiser, params, dataset, TINY_SIGMA_CFG, B=N, N=N, num_mc=3, seed=99)
    assert torch.equal(h_D_1, h_D_2)


# --------------------------------------------------------------------------------------
# H: candidate score reproducibility
# --------------------------------------------------------------------------------------


def test_candidate_score_reproducible(tiny_denoiser):
    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    h_D = torch.rand(d_S) + 0.5
    h_D_inv_sqrt = h_D.rsqrt()

    n = denoiser.cfg.inner_model.num_steps_conditioning
    obs = torch.randn(1, n * IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    act = torch.randn(1, n, ACTION_DIM)
    y = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    candidates = [(obs, act, y)]

    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE]), d_S, device=torch.device("cpu"),
        num_samples=3, seed=13,
    )
    scores1 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=1)
    scores2 = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=1)
    assert torch.equal(scores1, scores2)


# --------------------------------------------------------------------------------------
# Section 13: explicit DIAMOND-sequence-independence regressions (the direct proof that
# enabling/running LCG does not shift unrelated DIAMOND randomness downstream)
# --------------------------------------------------------------------------------------


def test_diamond_torch_sequence_independent_of_historical_precision(tiny_denoiser, small_dataset):
    S = 24680
    torch.manual_seed(S)
    baseline_sequence = torch.rand(30)

    denoiser = tiny_denoiser
    dataset = small_dataset
    params = selected_parameters(denoiser, _default_theta_s_config(denoiser))
    N = dataset.num_steps

    torch.manual_seed(S)
    h_D = historical_precision(denoiser, params, dataset, TINY_SIGMA_CFG, B=N, N=N, num_mc=3, seed=17)
    sequence_after_lcg = torch.rand(30)

    assert torch.equal(sequence_after_lcg, baseline_sequence)
    assert torch.isfinite(h_D).all() and (h_D > 0).all()  # sanity: the run actually did something valid


def test_diamond_numpy_sequence_independent_of_historical_precision(tiny_denoiser, small_dataset):
    S = 13579
    np.random.seed(S)
    baseline_sequence = np.random.random(30)

    denoiser = tiny_denoiser
    dataset = small_dataset
    params = selected_parameters(denoiser, _default_theta_s_config(denoiser))
    N = dataset.num_steps

    np.random.seed(S)
    historical_precision(denoiser, params, dataset, TINY_SIGMA_CFG, B=N, N=N, num_mc=3, seed=17)
    sequence_after_lcg = np.random.random(30)

    np.testing.assert_array_equal(sequence_after_lcg, baseline_sequence)


def test_diamond_torch_sequence_independent_of_candidate_bank_and_scoring(tiny_denoiser):
    S = 97531
    torch.manual_seed(S)
    baseline_sequence = torch.rand(30)

    denoiser = tiny_denoiser
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    h_D = torch.rand(d_S) + 0.5  # NOTE: this itself consumes global RNG (test setup, not LCG)
    h_D_inv_sqrt = h_D.rsqrt()
    n = denoiser.cfg.inner_model.num_steps_conditioning
    obs = torch.randn(1, n * IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    act = torch.randn(1, n, ACTION_DIM)
    y = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    candidates = [(obs, act, y)]

    torch.manual_seed(S)
    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE]), d_S, device=torch.device("cpu"),
        num_samples=3, seed=21,
    )
    scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, chunk_size=1)
    sequence_after_lcg = torch.rand(30)

    assert torch.equal(sequence_after_lcg, baseline_sequence)
    assert torch.isfinite(scores).all() and (scores >= 0).all()


# --------------------------------------------------------------------------------------
# Section 14: small integration smoke -- both RNG preservation and output validity
# --------------------------------------------------------------------------------------


def test_integration_smoke_rng_preserved_and_outputs_valid(tiny_denoiser, small_dataset):
    denoiser = tiny_denoiser
    dataset = small_dataset
    params = selected_parameters(denoiser, _default_theta_s_config(denoiser))
    N = dataset.num_steps

    # test-fixture data (candidate tensors) built BEFORE capturing RNG state -- their
    # construction legitimately consumes global RNG and must not be mistaken for LCG leakage
    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    n = denoiser.cfg.inner_model.num_steps_conditioning
    obs = torch.randn(1, n * IMG_CHANNELS, IMG_SIZE, IMG_SIZE)
    act = torch.randn(1, n, ACTION_DIM)
    y = torch.randn(1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE)

    np_state_before = np.random.get_state()
    torch_state_before = torch.get_rng_state()

    h_D = historical_precision(denoiser, params, dataset, TINY_SIGMA_CFG, B=N, N=N, num_mc=3, seed=5)
    assert h_D.shape == (sum(p.numel() for p in params),)
    assert torch.isfinite(h_D).all() and (h_D > 0).all()

    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE]), d_S, device=torch.device("cpu"),
        num_samples=2, seed=6,
    )
    scores = score_one_jvp_bank(denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt=h_D.rsqrt(),
                                 bank=bank, candidates=[(obs, act, y)], chunk_size=1)
    assert scores.shape == (1,)
    assert torch.isfinite(scores).all() and (scores >= 0).all()

    np_state_after = np.random.get_state()
    torch_state_after = torch.get_rng_state()
    assert np_state_before[1].tolist() == np_state_after[1].tolist()  # numpy MT19937 state array unchanged
    assert torch.equal(torch_state_before, torch_state_after)
