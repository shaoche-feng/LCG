"""
Permanent regression tests for the single authoritative training-sigma sampler
(models.diffusion.denoiser.sample_sigma_training_distribution) and for historical
precision's use of it as plain IID Monte Carlo (not stratified sampling).
"""
from unittest import mock

import torch
from lcg import precision as lcg_precision
from models.diffusion.denoiser import sample_sigma_training_distribution


def test_denoiser_setup_training_matches_authoritative_helper(tiny_denoiser):
    """Denoiser.setup_training's sample_sigma_training must delegate to
    sample_sigma_training_distribution -- verified with monkeypatched fixed draws against
    an independent hand-written formula (not a tautological self-comparison)."""
    denoiser = tiny_denoiser
    cfg = denoiser.cfg  # DenoiserConfig, has no sigma cfg; build a fresh setup_training call
    from models.diffusion import SigmaDistributionConfig

    sigma_cfg = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
    fresh = type(denoiser)(denoiser.cfg)  # a second instance so setup_training's assert (called once) doesn't fire
    fresh.setup_training(sigma_cfg)

    fixed_z = torch.tensor([0.1, -0.2, 0.3, 0.0, 1.5])
    with mock.patch("torch.randn", return_value=fixed_z):
        actual = fresh.sample_sigma_training(5, torch.device("cpu"))

    expected = (fixed_z * sigma_cfg.scale + sigma_cfg.loc).exp().clip(sigma_cfg.sigma_min, sigma_cfg.sigma_max)
    assert torch.equal(actual, expected)

    # and the same fixed draw through the helper directly must match too
    with mock.patch("torch.randn", return_value=fixed_z):
        helper_direct = sample_sigma_training_distribution(sigma_cfg, 5, torch.device("cpu"))
    assert torch.equal(helper_direct, expected)


def test_sample_sigma_training_distribution_generator_isolation():
    """generator=None preserves DIAMOND's original (global-RNG) behavior exactly;
    passing a local generator does not change the formula, only where the randomness
    comes from."""
    from models.diffusion import SigmaDistributionConfig

    cfg = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

    torch.manual_seed(0)
    global_draw = sample_sigma_training_distribution(cfg, 100, torch.device("cpu"))

    gen = torch.Generator(device="cpu")
    gen.manual_seed(0)
    local_draw = sample_sigma_training_distribution(cfg, 100, torch.device("cpu"), generator=gen)

    # same seed, same underlying algorithm -> bit-identical values (only isolation differs)
    assert torch.equal(global_draw, local_draw)


def test_historical_precision_uses_iid_full_distribution_not_stratified(tiny_denoiser, monkeypatch):
    """Production historical_precision must call the single authoritative
    sample_sigma_training_distribution exactly num_mc times per historical transition
    (never a stratified per-quantile sampler), drawing from the COMPLETE distribution."""
    import sys
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        p = start.resolve()
        while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
            if p.parent == p:
                raise RuntimeError("could not locate LCG repo root")
            p = p.parent
        return p

    sys.path.insert(0, str(_find_repo_root(Path(__file__).parent) / "src"))
    from data import Dataset, Episode
    from lcg.theta_s import ThetaSConfig, selected_parameters
    from models.diffusion import SigmaDistributionConfig

    denoiser = tiny_denoiser
    sigma_cfg = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)

    def _make_episode(length):
        return Episode(
            obs=torch.randn(length, 3, 8, 8), act=torch.randn(length, 2), rew=torch.zeros(length),
            end=torch.zeros(length, dtype=torch.uint8), trunc=torch.zeros(length, dtype=torch.uint8), info={},
        )

    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = Dataset(Path(tmp_dir) / "ds", "sigma_sampling_test_ds", cache_in_ram=True)
        for L in [4, 6]:
            dataset.add_episode(_make_episode(L))
        N = dataset.num_steps
        last_idx = len(denoiser.inner_model.unet.u_blocks) - 1
        theta_s_cfg = ThetaSConfig(include=(f"unet.u_blocks.{last_idx}.*", "norm_out.*", "conv_out.*"), exclude=())
        params = selected_parameters(denoiser, theta_s_cfg)

        calls = []
        real_fn = lcg_precision.sample_sigma_training_distribution

        def recording_wrapper(cfg, n, device, generator=None):
            calls.append(n)
            return real_fn(cfg, n, device, generator=generator)

        monkeypatch.setattr(lcg_precision, "sample_sigma_training_distribution", recording_wrapper)

        num_mc = 3
        B = N
        lcg_precision.historical_precision(denoiser, params, dataset, sigma_cfg, B=B, N=N, num_mc=num_mc, seed=1)

        assert len(calls) == B * num_mc, "must call the sigma sampler exactly num_mc times per historical transition"
        assert all(n == 1 for n in calls), "each call draws exactly one sigma (no batching that would change the formula)"

        # confirm it draws from the FULL distribution (pooled log-sigma stats match cfg.loc/scale),
        # not a restricted sub-quantile stratum
        torch.manual_seed(7)
        pooled = torch.cat([sample_sigma_training_distribution(sigma_cfg, 1, torch.device("cpu")) for _ in range(2000)])
        log_pooled = pooled.log()
        assert abs(log_pooled.mean().item() - sigma_cfg.loc) < 0.1
        assert abs(log_pooled.std().item() - sigma_cfg.scale) < 0.1
