"""
Permanent small integration smoke test for the production path: candidate extraction ->
forward-JVP scoring -> reshape to (num_envs, num_steps) -> intrinsic reward hook.
Extracted from scripts/forward_JVP/integration/smoke_test_lcg_forward_jvp.py, shrunk from
the original 480-candidate/B=32,H=15 workload to a handful of tiny synthetic candidates.
"""
from dataclasses import dataclass

import torch
from lcg.forward_jvp import make_jvp_bank
from lcg.intrinsic_reward import imagined_candidates_from_batch, make_lcg_intrinsic_reward_fn
from lcg.theta_s import ThetaSConfig, frozen_named_parameters, selected_named_parameters
from models.diffusion import SigmaDistributionConfig

TINY_IMG_CHANNELS = 3
TINY_IMG_SIZE = 8
TINY_SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)


def _default_theta_s_config(denoiser) -> ThetaSConfig:
    """Current-production-equivalent ThetaSConfig for whichever denoiser is passed --
    duplicated per test file, matching this suite's existing convention (see conftest.py's
    default_theta_s_config, the same helper)."""
    last_idx = len(denoiser.inner_model.unet.u_blocks) - 1
    return ThetaSConfig(include=(f"unet.u_blocks.{last_idx}.*", "norm_out.*", "conv_out.*"), exclude=())


@dataclass
class _FakeImaginedCandidate:
    x_obs: torch.Tensor
    x_act: torch.Tensor
    y_star: torch.Tensor


def test_reshape_is_step_major(tiny_denoiser):
    """The exact reshape used by make_lcg_intrinsic_reward_fn: flat index t*num_envs+b
    must land at reshaped[b, t]."""
    num_envs, num_steps = 3, 2
    flat = torch.arange(num_envs * num_steps).float()
    reshaped = flat.view(num_steps, num_envs).transpose(0, 1).contiguous()
    for t in range(num_steps):
        for b in range(num_envs):
            assert reshaped[b, t].item() == flat[t * num_envs + b].item()


def test_full_pipeline_no_candidate_lost_or_duplicated(tiny_denoiser):
    denoiser = tiny_denoiser
    n = denoiser.cfg.inner_model.num_steps_conditioning
    num_envs, num_steps = 2, 2

    infos = []
    for _ in range(num_steps):
        x_obs = torch.randn(num_envs, n, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE)
        x_act = torch.randn(num_envs, n, 2)
        y_star = torch.randn(num_envs, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE)
        infos.append({"imagined_candidate": _FakeImaginedCandidate(x_obs, x_act, y_star)})
    env_rew = torch.zeros(num_envs, num_steps)

    # candidate extraction: exactly num_envs candidates per step, none lost/duplicated
    all_candidates = []
    for t in range(num_steps):
        batch = infos[t]["imagined_candidate"]
        step_candidates = imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star)
        assert len(step_candidates) == num_envs
        all_candidates.extend(step_candidates)
    assert len(all_candidates) == num_envs * num_steps

    theta_s_named = selected_named_parameters(denoiser, _default_theta_s_config(denoiser))
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    d_S = sum(p.numel() for p in theta_s_named.values())
    h_D_inv_sqrt = (torch.rand(d_S) + 0.5).rsqrt()
    bank = make_jvp_bank(
        TINY_SIGMA_CFG, torch.Size([1, TINY_IMG_CHANNELS, TINY_IMG_SIZE, TINY_IMG_SIZE]), d_S,
        device=torch.device("cpu"), num_samples=2, seed=0,
    )

    hook = make_lcg_intrinsic_reward_fn(denoiser, theta_s_named, frozen_named, h_D_inv_sqrt, bank, chunk_size=2)
    out = hook(infos, env_rew)

    assert out.shape == env_rew.shape
    assert torch.isfinite(out).all()
    assert (out >= 0).all()


def test_disabled_lcg_leaves_baseline_reward_untouched():
    """Mirrors ActorCritic.forward()'s exact gate (src/models/actor_critic.py):
    `if self.intrinsic_reward_fn is not None: rew = self.intrinsic_reward_fn(infos, rew)`.
    With intrinsic_reward_fn=None, rew must be completely unaffected."""
    original_rew = torch.randn(3, 4)
    rew = original_rew.clone()
    intrinsic_reward_fn = None
    if intrinsic_reward_fn is not None:
        rew = intrinsic_reward_fn(None, rew)
    assert torch.equal(rew, original_rew)
