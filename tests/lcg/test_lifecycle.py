"""
Permanent regression tests for LCGLifecycle's theta_S caching: theta_S selection is
resolved ONCE per lifecycle/model instance (on the first refresh()) and reused on every
later round, while h_D / the candidate JVP-CRN bank / RunningRMS still refresh every
round. See lcg.lifecycle.LCGLifecycle's docstring for the intended cadence.
"""
import sys
from pathlib import Path
from unittest import mock

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
from lcg import lifecycle as lcg_lifecycle  # noqa: E402
from lcg.lifecycle import LCGConfig, LCGLifecycle  # noqa: E402
from lcg.theta_s import ThetaSConfig, frozen_named_parameters, selected_named_parameters  # noqa: E402
from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig  # noqa: E402
from models.diffusion.inner_model import InnerModelConfig  # noqa: E402

IMG_CHANNELS, IMG_SIZE, ACTION_DIM = 3, 8, 2
NUM_STEPS_CONDITIONING = 1
SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
THETA_S_CFG = ThetaSConfig(include=("conv_out.*",))  # tiny, fast: caching mechanics don't depend on which subset


def _build_denoiser() -> Denoiser:
    torch.manual_seed(0)
    inner_cfg = InnerModelConfig(
        img_channels=IMG_CHANNELS, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=16,
        depths=[1], channels=[8], attn_depths=[False], continuous_action_dim=ACTION_DIM,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg)
    denoiser.eval()
    return denoiser


def _build_dataset(tmp_path) -> Dataset:
    dataset = Dataset(tmp_path / "ds", "lifecycle_test_ds", cache_in_ram=True)
    for L in [4, 6]:
        ep = Episode(
            obs=torch.randn(L, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), act=torch.randn(L, ACTION_DIM), rew=torch.zeros(L),
            end=torch.zeros(L, dtype=torch.uint8), trunc=torch.zeros(L, dtype=torch.uint8), info={},
        )
        dataset.add_episode(ep)
    return dataset


def _build_lifecycle() -> LCGLifecycle:
    cfg = LCGConfig(
        enabled=True, precision_reference_size=10, precision_num_mc=2, candidate_num_mc=2, candidate_chunk_size=2,
        theta_s=THETA_S_CFG,
    )
    return LCGLifecycle(cfg, SIGMA_CFG, IMG_CHANNELS, IMG_SIZE, torch.device("cpu"))


# --------------------------------------------------------------------------------------
# Item 8: selector's returned references stay live across an external in-place mutation
# (a direct property of lcg.theta_s.selected_named_parameters, exercised without going
# through LCGLifecycle, to isolate the invariant the caching relies on)
# --------------------------------------------------------------------------------------


def test_selected_named_parameters_returns_live_references_across_mutation():
    denoiser = _build_denoiser()
    named = selected_named_parameters(denoiser, THETA_S_CFG)
    name, cached_param = next(iter(named.items()))
    old_value = cached_param.detach().clone()

    with torch.no_grad():
        cached_param.add_(5.0)  # simulate an optimizer step, mutating the live parameter

    current_param = dict(denoiser.inner_model.named_parameters())[name]
    assert cached_param is current_param  # same object -- selection did not snapshot
    assert not torch.equal(cached_param, old_value)  # and it sees the new value


# --------------------------------------------------------------------------------------
# Item 9: theta_S is selected exactly once across multiple refresh() rounds; h_D and the
# candidate bank are still rebuilt every round.
# --------------------------------------------------------------------------------------


def test_lifecycle_selects_theta_s_exactly_once_across_rounds(tmp_path):
    denoiser = _build_denoiser()
    dataset = _build_dataset(tmp_path)
    lifecycle = _build_lifecycle()

    with mock.patch.object(lcg_lifecycle, "selected_named_parameters", wraps=selected_named_parameters) as spy_select, \
         mock.patch.object(lcg_lifecycle, "frozen_named_parameters", wraps=frozen_named_parameters) as spy_frozen:
        lifecycle.refresh(denoiser, dataset, round_identifier=0)
        h_D_round0 = lifecycle.h_D.clone()
        bank_round0 = lifecycle.bank

        lifecycle.refresh(denoiser, dataset, round_identifier=1)
        h_D_round1 = lifecycle.h_D.clone()
        bank_round1 = lifecycle.bank

        lifecycle.refresh(denoiser, dataset, round_identifier=2)
        h_D_round2 = lifecycle.h_D.clone()
        bank_round2 = lifecycle.bank

    assert spy_select.call_count == 1  # theta_S resolved exactly once across 3 rounds
    assert spy_frozen.call_count == 1  # its complement (frozen_named) likewise cached once

    # h_D and the bank are still rebuilt every round (different seeds -> different draws)
    assert not torch.equal(h_D_round0, h_D_round1)
    assert not torch.equal(h_D_round1, h_D_round2)
    assert bank_round0 is not bank_round1
    assert bank_round1 is not bank_round2


def test_lifecycle_round_id_tracks_caller_supplied_identifier(tmp_path):
    """round_id is now a direct copy of the caller-supplied round_identifier (e.g.
    Trainer.epoch), not an internally auto-incremented counter -- this is what makes the
    seed sequence survive checkpoint/resume (see test_lifecycle_resume_reproduces_seeds)."""
    denoiser = _build_denoiser()
    dataset = _build_dataset(tmp_path)
    lifecycle = _build_lifecycle()

    assert lifecycle.round_id == -1
    lifecycle.refresh(denoiser, dataset, round_identifier=0)
    assert lifecycle.round_id == 0
    lifecycle.refresh(denoiser, dataset, round_identifier=1)
    assert lifecycle.round_id == 1
    # a caller may pass any identifier sequence -- e.g. resuming straight to round 7
    lifecycle.refresh(denoiser, dataset, round_identifier=7)
    assert lifecycle.round_id == 7


def test_lifecycle_resume_reproduces_seeds(tmp_path):
    """Simulates checkpoint/resume: LCGLifecycle itself is never checkpointed (it is a
    underscore-prefixed Trainer attribute, see StateDictMixin), so a resumed run
    reconstructs a brand-new LCGLifecycle and calls refresh() directly at the round
    identifier it resumed from. Since historical_precision/make_jvp_bank are pure
    functions of (denoiser weights, dataset, seed) with no persistent state carried by
    LCGLifecycle across rounds, jumping straight to round_identifier=2 on a fresh
    lifecycle must reproduce bit-identical h_D/bank to having reached round 2 by calling
    refresh() at 0, 1, 2 in sequence on a long-lived lifecycle."""
    denoiser = _build_denoiser()
    dataset = _build_dataset(tmp_path)

    uninterrupted = _build_lifecycle()
    for r in (0, 1, 2):
        uninterrupted.refresh(denoiser, dataset, round_identifier=r)
    h_D_uninterrupted = uninterrupted.h_D.clone()
    bank_uninterrupted = uninterrupted.bank

    resumed = _build_lifecycle()  # fresh instance, as after a process restart
    resumed.refresh(denoiser, dataset, round_identifier=2)
    h_D_resumed = resumed.h_D.clone()
    bank_resumed = resumed.bank

    assert torch.equal(h_D_uninterrupted, h_D_resumed)
    assert bank_uninterrupted.sigmas == bank_resumed.sigmas
    assert all(torch.equal(a, b) for a, b in zip(bank_uninterrupted.epsilons, bank_resumed.epsilons))
    assert all(torch.equal(a, b) for a, b in zip(bank_uninterrupted.epsilons_offset, bank_resumed.epsilons_offset))
    assert all(torch.equal(a, b) for a, b in zip(bank_uninterrupted.etas, bank_resumed.etas))


# --------------------------------------------------------------------------------------
# Item 10: cached theta_S sees updated parameter values (live references, not snapshots)
# --------------------------------------------------------------------------------------


def test_cached_theta_s_reflects_parameter_mutations_across_rounds(tmp_path):
    denoiser = _build_denoiser()
    dataset = _build_dataset(tmp_path)
    lifecycle = _build_lifecycle()

    lifecycle.refresh(denoiser, dataset, round_identifier=0)  # round 0: cache created
    name = next(iter(lifecycle._theta_s_named.keys()))
    live_param = dict(denoiser.inner_model.named_parameters())[name]
    old_value = live_param.detach().clone()

    with torch.no_grad():
        live_param.add_(3.0)  # simulate an optimizer step directly on the model

    cached_param = lifecycle._theta_s_named[name]
    assert cached_param is live_param  # cache did not snapshot
    assert not torch.equal(cached_param, old_value)  # cache observes the new value

    lifecycle.refresh(denoiser, dataset, round_identifier=1)  # round 1: must reuse cache, not reselect
    assert torch.isfinite(lifecycle.h_D).all()
    assert (lifecycle.h_D > 0).all()


# --------------------------------------------------------------------------------------
# Item 11: denoiser replacement is rejected loudly, not silently reselected
# --------------------------------------------------------------------------------------


def test_lifecycle_raises_on_denoiser_replacement(tmp_path):
    denoiser_a = _build_denoiser()
    denoiser_b = _build_denoiser()
    dataset = _build_dataset(tmp_path)
    lifecycle = _build_lifecycle()

    lifecycle.refresh(denoiser_a, dataset, round_identifier=0)
    with pytest.raises(RuntimeError, match="different denoiser instance"):
        lifecycle.refresh(denoiser_b, dataset, round_identifier=1)


# --------------------------------------------------------------------------------------
# Item 7: dimension invariant
# --------------------------------------------------------------------------------------


def test_lifecycle_caches_dimension_matching_h_D(tmp_path):
    denoiser = _build_denoiser()
    dataset = _build_dataset(tmp_path)
    lifecycle = _build_lifecycle()

    lifecycle.refresh(denoiser, dataset, round_identifier=0)
    assert lifecycle._theta_s_dim == sum(p.numel() for p in lifecycle._theta_s_named.values())
    assert lifecycle.h_D.numel() == lifecycle._theta_s_dim


# --------------------------------------------------------------------------------------
# Item 8: LCGConfig.__post_init__ validation -- reject nonsensical configs loudly instead
# of letting them silently degrade (in particular candidate_num_mc=0, which would
# otherwise silently produce an empty JVP bank and a zero intrinsic reward everywhere).
# --------------------------------------------------------------------------------------


def _valid_kwargs(**overrides):
    kwargs = dict(
        enabled=True, precision_reference_size=10, precision_num_mc=2, candidate_num_mc=2,
        candidate_chunk_size=2, theta_s=THETA_S_CFG,
    )
    kwargs.update(overrides)
    return kwargs


@pytest.mark.parametrize(
    "override",
    [
        {"precision_reference_size": 0},
        {"precision_reference_size": -1},
        {"precision_num_mc": 0},
        {"precision_num_mc": -1},
        {"candidate_chunk_size": 0},
        {"candidate_chunk_size": -1},
        {"damping": 0.0},
        {"damping": -1e-4},
    ],
)
def test_lcg_config_rejects_non_positive_fields(override):
    with pytest.raises(AssertionError):
        LCGConfig(**_valid_kwargs(**override))


def test_lcg_config_rejects_non_positive_candidate_num_mc():
    """candidate_num_mc=0 must not silently produce a zero-sample JVP bank and thus zero
    intrinsic reward everywhere -- it must fail loudly at config time instead."""
    with pytest.raises(AssertionError, match="candidate_num_mc"):
        LCGConfig(**_valid_kwargs(candidate_num_mc=0))
    with pytest.raises(AssertionError):
        LCGConfig(**_valid_kwargs(candidate_num_mc=-3))


@pytest.mark.parametrize(
    "override",
    [
        {"rms_alpha": 0.0},
        {"rms_alpha": -1.0},
        {"rms_ema_decay": -0.1},
        {"rms_ema_decay": 1.0},
        {"rms_ema_decay": 1.5},
        {"rms_eps": 0.0},
        {"rms_eps": -1e-8},
    ],
)
def test_lcg_config_rejects_invalid_rms_params_when_enabled(override):
    with pytest.raises(AssertionError):
        LCGConfig(**_valid_kwargs(rms_enabled=True, **override))


def test_lcg_config_allows_invalid_rms_params_when_rms_disabled():
    # rms_* fields are irrelevant when rms_enabled=False -- must not be validated then
    LCGConfig(**_valid_kwargs(rms_enabled=False, rms_alpha=-1.0, rms_ema_decay=5.0, rms_eps=-1.0))


def test_lcg_config_accepts_valid_defaults():
    LCGConfig(**_valid_kwargs())
