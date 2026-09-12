"""
Permanent regression tests for lcg.precision.sample_uniform_historical_transitions, the
LCG-specific replacement for DIAMOND's generic BatchSampler in historical_precision().

BatchSampler's can_sample_beyond_end=False branch draws a uniform timestep t and then
shifts it by an independent random offset before clipping to the episode boundary, so the
transition that actually becomes the VJP target (the segment's last frame) is NOT t --
it is a randomized, boundary-biased function of t. These tests protect the fix: every
valid (episode, t) pair must be reachable, the sampled t must be exactly the transition
used (no silent shift), sampling without replacement must give exactly B distinct
transitions, and the resulting distribution must be uniform (both by construction and,
as a secondary sanity check, empirically).
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
from data.utils import make_segment  # noqa: E402
from lcg.precision import historical_precision, load_transition, sample_uniform_historical_transitions  # noqa: E402
from lcg.theta_s import ThetaSConfig, selected_parameters  # noqa: E402
from models.diffusion import SigmaDistributionConfig  # noqa: E402


def _default_theta_s_config(denoiser) -> ThetaSConfig:
    """Current-production-equivalent ThetaSConfig for whichever denoiser is passed --
    duplicated per test file, matching this suite's existing convention (see conftest.py's
    default_theta_s_config, the same helper)."""
    last_idx = len(denoiser.inner_model.unet.u_blocks) - 1
    return ThetaSConfig(include=(f"unet.u_blocks.{last_idx}.*", "norm_out.*", "conv_out.*"), exclude=())


IMG_CHANNELS = 3
IMG_SIZE = 8
ACTION_DIM = 2
NUM_STEPS_CONDITIONING = 1  # small: only need to exercise the padding boundary, not a big window


def _make_episode(length):
    return Episode(
        obs=torch.randn(length, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
        act=torch.randn(length, ACTION_DIM),
        rew=torch.zeros(length),
        end=torch.zeros(length, dtype=torch.uint8),
        trunc=torch.zeros(length, dtype=torch.uint8),
        info={},
    )


def _build_dataset(tmp_path, lengths):
    dataset = Dataset(tmp_path / "ds", "test_ds", cache_in_ram=True, save_on_disk=True)
    for L in lengths:
        dataset.add_episode(_make_episode(L))
    return dataset


def _all_valid_transitions(lengths):
    """The complete, independently-computed set of valid (episode_id, t) pairs."""
    return {(e, t) for e, L in enumerate(lengths) for t in range(L)}


@pytest.fixture
def small_dataset(tmp_path):
    lengths = [2, 3, 5]  # N = 10, matches the task's example
    return _build_dataset(tmp_path, lengths), lengths


def test_every_transition_is_representable(small_dataset):
    dataset, lengths = small_dataset
    N = sum(lengths)
    segment_ids = sample_uniform_historical_transitions(
        dataset, reference_size=N, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=0, replace=False
    )
    sampled_pairs = {(sid.episode_id, sid.stop - 1) for sid in segment_ids}
    assert sampled_pairs == _all_valid_transitions(lengths)


def test_exact_endpoint_preservation(small_dataset):
    """The single most important regression: if transition t is selected, the resulting
    segment's target frame must be exactly episode.obs[t], not obs[t+k] for any k!=0."""
    dataset, lengths = small_dataset
    segment_ids = sample_uniform_historical_transitions(
        dataset, reference_size=sum(lengths), num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=1, replace=False
    )
    for sid in segment_ids:
        t = sid.stop - 1
        episode = dataset.load_episode(sid.episode_id)
        expected_y = episode.obs[t]
        obs, act, y = load_transition(dataset, sid, NUM_STEPS_CONDITIONING, device=torch.device("cpu"))
        assert torch.equal(y.squeeze(0), expected_y)
        # and the action immediately preceding the target, when not padding, is exactly a_{t-1}
        if t - 1 >= 0:
            assert torch.equal(act.squeeze(0)[-1], episode.act[t - 1])


def test_no_duplicate_transitions_without_replacement(small_dataset):
    dataset, lengths = small_dataset
    N = sum(lengths)
    B = N - 2
    segment_ids = sample_uniform_historical_transitions(
        dataset, reference_size=B, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=2, replace=False
    )
    pairs = [(sid.episode_id, sid.stop - 1) for sid in segment_ids]
    assert len(pairs) == len(set(pairs))


def test_correct_sample_count(small_dataset):
    dataset, lengths = small_dataset
    for B in [1, 4, sum(lengths)]:
        segment_ids = sample_uniform_historical_transitions(
            dataset, reference_size=B, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=3, replace=False
        )
        assert len(segment_ids) == B


def test_reference_size_exceeding_available_raises_without_replace(small_dataset):
    dataset, lengths = small_dataset
    N = sum(lengths)
    with pytest.raises(AssertionError):
        sample_uniform_historical_transitions(
            dataset, reference_size=N + 1, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=4, replace=False
        )
    # explicit replace=True must not raise
    segment_ids = sample_uniform_historical_transitions(
        dataset, reference_size=N + 5, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=4, replace=True
    )
    assert len(segment_ids) == N + 5


def test_episode_start_left_padding(small_dataset):
    """t=0 (the very first frame of an episode) must produce a fully-left-padded
    conditioning window, matching DIAMOND's existing mask_padding convention."""
    dataset, lengths = small_dataset
    n = NUM_STEPS_CONDITIONING
    from data.segment import SegmentId
    sid = SegmentId(episode_id=0, start=0 - n, stop=1)
    segment = make_segment(dataset.load_episode(0), sid, should_pad=True)
    assert segment.mask_padding[:n].sum().item() == 0  # conditioning window is all padding
    assert segment.mask_padding[n].item() is True  # target frame is real
    obs, act, y = load_transition(dataset, sid, n, device=torch.device("cpu"))
    assert torch.equal(y.squeeze(0), dataset.load_episode(0).obs[0])


def test_final_transition_no_extra_probability_from_clipping(small_dataset):
    """The old BatchSampler-based sampler produced a probability spike at each episode's
    LAST transition from boundary clipping. Verify the new sampler's empirical frequency
    at each episode's final transition is NOT inflated relative to other transitions.

    Uses independent reference_size=1 draws (NOT reference_size=N without replacement, which
    would trivially force a full permutation -- i.e. every transition exactly once --
    regardless of whether the underlying sampling is biased or not)."""
    dataset, lengths = small_dataset
    counts = {k: 0 for k in _all_valid_transitions(lengths)}
    num_draws = 20000
    for seed in range(num_draws):
        segment_ids = sample_uniform_historical_transitions(
            dataset, reference_size=1, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=1000 + seed, replace=False
        )
        sid = segment_ids[0]
        counts[(sid.episode_id, sid.stop - 1)] += 1

    final_transitions = [(e, L - 1) for e, L in enumerate(lengths)]
    other_transitions = [k for k in _all_valid_transitions(lengths) if k not in final_transitions]
    mean_final = np.mean([counts[k] for k in final_transitions])
    mean_other = np.mean([counts[k] for k in other_transitions])
    # with a correct uniform sampler these should be close; allow generous MC tolerance
    assert mean_final < 1.5 * mean_other


def test_empirical_frequencies_approximately_uniform(small_dataset):
    """Diagnostic sanity check (not the primary correctness proof, see the deterministic
    tests above): empirical transition frequencies over many draws should be roughly
    uniform across all N valid transitions."""
    dataset, lengths = small_dataset
    N = sum(lengths)
    rng_seeds = range(2000)
    counts = {k: 0 for k in _all_valid_transitions(lengths)}
    for seed in rng_seeds:
        segment_ids = sample_uniform_historical_transitions(
            dataset, reference_size=1, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=seed, replace=False
        )
        sid = segment_ids[0]
        counts[(sid.episode_id, sid.stop - 1)] += 1

    expected = len(rng_seeds) / N
    observed = np.array(list(counts.values()))
    assert observed.min() > 0  # every transition was reached at least once
    max_rel_dev = np.max(np.abs(observed - expected)) / expected
    assert max_rel_dev < 1.0  # generous tolerance -- this is a sanity check, not a precise test


def test_historical_precision_end_to_end_with_fixed_sampler(tiny_denoiser, small_dataset):
    """historical_precision() itself, run end-to-end on a tiny real model + tiny dataset,
    with BOTH fixes active (corrected corruption + corrected uniform transition sampler).
    Not a full-scale h_D recomputation -- B is tiny and the model is tiny."""
    dataset, lengths = small_dataset
    denoiser = tiny_denoiser
    N = sum(lengths)
    B = N  # exercise the "B == N, without replacement" boundary explicitly
    params = selected_parameters(denoiser, _default_theta_s_config(denoiser))
    d_S = sum(p.numel() for p in params)
    sigma_cfg = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

    h_D = historical_precision(
        denoiser, params, dataset, sigma_cfg, B=B, N=N, num_mc=3, beta=1.0, damping=1e-4, seed=42,
    )

    assert h_D.shape == (d_S,)
    assert torch.isfinite(h_D).all()
    assert (h_D > 0).all()  # strictly positive after damping
    assert (h_D >= 1e-4).all()  # damping is a floor: h = damping*ones + beta*(N/B)*sum(...), sum(...) >= 0
