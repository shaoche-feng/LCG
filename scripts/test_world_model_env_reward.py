#! /usr/bin/env python
"""
Stage 5C verification: WorldModelEnv support for the continuous-reward output
introduced in Stage 5B.

Builds real (small) Denoiser + RewEndModel + WorldModelEnv instances directly
(no Agent/AgentConfig, no Trainer, no ActorCritic/env_loop, no real replay
dataset -- a tiny in-memory fake data loader stands in for it) and verifies:
  - Atari (discrete, continuous_reward=False): reward is still sampled via
    Categorical over 3 classes and mapped to {-1, 0, 1}.
  - Continuous (continuous_reward=True): the RewEndModel's scalar prediction
    passes through unchanged -- no Categorical, no sign(), no clipping --
    verified with an exact deterministic passthrough check for specific
    values (0.17, 0.53) via a monkeypatched RewEndModel.predict_rew_end.
  - Returned reward shape is (num_envs,) in both modes.
  - End prediction is unaffected in both modes.
  - WorldModelEnv.step() runs end-to-end with continuous actions of
    action_dim=4 and action_dim=6.

Runs on CPU, and on CUDA too if available.

Usage:
    python scripts/test_world_model_env_reward.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from data import Batch, SegmentId
from envs import WorldModelEnv, WorldModelEnvConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig
from models.diffusion.inner_model import InnerModelConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig


IMG_CHANNELS = 3
IMG_SIZE = 8
NUM_STEPS_CONDITIONING = 4
COND_CHANNELS = 16
LSTM_DIM = 32
NUM_ENVS = 2


class FakeBatchSampler:
    def __init__(self, batch_size: int) -> None:
        self.batch_size = batch_size


class FakeDataLoader:
    """Minimal stand-in for the real replay DataLoader: WorldModelEnv only needs
    `.batch_sampler.batch_size` and an infinite iterable of Batch objects shaped
    (batch_size, num_steps_conditioning, ...)."""

    def __init__(self, batch_size: int, make_batch_fn) -> None:
        self.batch_sampler = FakeBatchSampler(batch_size)
        self._make_batch_fn = make_batch_fn

    def __iter__(self):
        while True:
            yield self._make_batch_fn()


def make_fake_batch(action_dim, num_actions, device) -> Batch:
    T = NUM_STEPS_CONDITIONING
    obs = torch.rand(NUM_ENVS, T, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device) * 2 - 1
    if action_dim is None:
        act = torch.randint(0, num_actions, (NUM_ENVS, T), dtype=torch.int64, device=device)
    else:
        act = torch.randn(NUM_ENVS, T, action_dim, device=device)
    rew = torch.zeros(NUM_ENVS, T, device=device)
    end = torch.zeros(NUM_ENVS, T, dtype=torch.int64, device=device)
    trunc = torch.zeros(NUM_ENVS, T, dtype=torch.int64, device=device)
    mask_padding = torch.ones(NUM_ENVS, T, dtype=torch.bool, device=device)
    info = [{} for _ in range(NUM_ENVS)]
    segment_ids = [SegmentId(0, 0, T) for _ in range(NUM_ENVS)]
    return Batch(obs, act, rew, end, trunc, mask_padding, info, segment_ids)


def make_denoiser(num_actions=None, continuous_action_dim=None) -> Denoiser:
    cfg = DenoiserConfig(
        inner_model=InnerModelConfig(
            img_channels=IMG_CHANNELS, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=COND_CHANNELS,
            depths=[1, 1], channels=[8, 8], attn_depths=[0, 0],
            num_actions=num_actions, continuous_action_dim=continuous_action_dim,
        ),
        sigma_data=0.5, sigma_offset_noise=0.3,
    )
    return Denoiser(cfg)


def make_rew_end_model(num_actions=None, continuous_action_dim=None, continuous_reward=False) -> RewEndModel:
    cfg = RewEndModelConfig(
        lstm_dim=LSTM_DIM, img_channels=IMG_CHANNELS, img_size=IMG_SIZE, cond_channels=COND_CHANNELS,
        depths=[1, 1], channels=[8, 8], attn_depths=[0, 0],
        num_actions=num_actions, continuous_action_dim=continuous_action_dim, continuous_reward=continuous_reward,
    )
    return RewEndModel(cfg)


def make_world_model_env(denoiser, rew_end_model, action_dim, num_actions, device, horizon=1000) -> WorldModelEnv:
    data_loader = FakeDataLoader(NUM_ENVS, lambda: make_fake_batch(action_dim, num_actions, device))
    cfg = WorldModelEnvConfig(
        horizon=horizon,
        num_batches_to_preload=1,
        diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=3),
    )
    return WorldModelEnv(denoiser, rew_end_model, data_loader, cfg)


def sample_action(action_dim, num_actions, device):
    if action_dim is None:
        return torch.randint(0, num_actions, (NUM_ENVS,), dtype=torch.int64, device=device)
    return torch.randn(NUM_ENVS, action_dim, device=device)


def check_atari_reward_unchanged(device) -> None:
    print(f"\n--- Atari (discrete, continuous_reward=False) [{device}] ---")
    denoiser = make_denoiser(num_actions=18).to(device)
    rew_end_model = make_rew_end_model(num_actions=18, continuous_reward=False).to(device)
    wm_env = make_world_model_env(denoiser, rew_end_model, action_dim=None, num_actions=18, device=device)

    wm_env.reset(seed=0)
    all_rew = []
    for _ in range(10):
        act = sample_action(None, 18, device)
        obs, rew, end, trunc, info = wm_env.step(act)
        assert obs.shape == (NUM_ENVS, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), obs.shape
        assert rew.shape == (NUM_ENVS,), f"expected reward shape (num_envs,), got {rew.shape}"
        assert end.shape == (NUM_ENVS,) and trunc.shape == (NUM_ENVS,)
        all_rew.append(rew)
    all_rew = torch.cat(all_rew)
    allowed = torch.tensor([-1.0, 0.0, 1.0], device=device)
    assert torch.isin(all_rew, allowed).all(), f"Atari reward left {{-1,0,1}}: {all_rew.unique()}"
    print(f"  reward shape=(num_envs,)={tuple(rew.shape)}, all sampled values in {{-1,0,1}}: {all_rew.unique().tolist()}  OK")


def check_exact_continuous_passthrough(device) -> None:
    print(f"\n--- Continuous reward: exact passthrough of specific float values [{device}] ---")
    denoiser = make_denoiser(continuous_action_dim=4).to(device)
    rew_end_model = make_rew_end_model(continuous_action_dim=4, continuous_reward=True).to(device)
    wm_env = make_world_model_env(denoiser, rew_end_model, action_dim=4, num_actions=None, device=device)
    wm_env.reset(seed=0)

    fixed_values = torch.tensor([0.17, 0.53], device=device)
    assert fixed_values.shape[0] == NUM_ENVS

    def fake_predict_rew_end(obs, act, next_obs, hx_cx):
        b = next_obs.size(0)
        rew_out = fixed_values[:b].reshape(b, 1)  # matches real continuous_reward=True output shape (b, t=1)
        logits_end = torch.zeros(b, 1, 2, device=next_obs.device)  # always "not end"
        return rew_out, logits_end, hx_cx

    wm_env.rew_end_model.predict_rew_end = fake_predict_rew_end
    next_obs_dummy = torch.zeros(NUM_ENVS, 1, IMG_CHANNELS, IMG_SIZE, IMG_SIZE, device=device)
    rew, end = wm_env.predict_rew_end(next_obs_dummy)

    assert rew.shape == (NUM_ENVS,), f"expected (num_envs,), got {rew.shape}"
    assert torch.allclose(rew, fixed_values), f"expected exact passthrough {fixed_values.tolist()}, got {rew.tolist()}"
    assert not torch.isin(rew, torch.tensor([-1.0, 0.0, 1.0], device=device)).any(), (
        "continuous reward was clipped/converted to {-1,0,1}"
    )
    print(f"  input scalars {fixed_values.tolist()} -> output rew {rew.tolist()} (exact match, no sign/Categorical)  OK")


def check_continuous_end_to_end(device, action_dim: int) -> None:
    print(f"\n--- Continuous end-to-end, action_dim={action_dim} [{device}] ---")
    denoiser = make_denoiser(continuous_action_dim=action_dim).to(device)
    rew_end_model = make_rew_end_model(continuous_action_dim=action_dim, continuous_reward=True).to(device)
    wm_env = make_world_model_env(denoiser, rew_end_model, action_dim=action_dim, num_actions=None, device=device)

    wm_env.reset(seed=0)
    all_rew = []
    for _ in range(10):
        act = sample_action(action_dim, None, device)
        assert act.shape == (NUM_ENVS, action_dim)
        obs, rew, end, trunc, info = wm_env.step(act)
        assert obs.shape == (NUM_ENVS, IMG_CHANNELS, IMG_SIZE, IMG_SIZE), obs.shape
        assert rew.shape == (NUM_ENVS,), f"expected reward shape (num_envs,), got {rew.shape}"
        assert rew.dtype.is_floating_point
        assert torch.isfinite(rew).all()
        assert end.shape == (NUM_ENVS,) and trunc.shape == (NUM_ENVS,)
        all_rew.append(rew.clone())
    all_rew = torch.cat(all_rew)

    # A real (untrained but randomly-initialized) network's raw scalar predictions should not
    # collapse onto exact integers -- if they did, that would indicate accidental
    # rounding/classification sneaking back in.
    frac = (all_rew - all_rew.round()).abs()
    assert frac.max().item() > 1e-4, "continuous reward values look suspiciously integer-valued"
    print(f"  reward shape=(num_envs,)={tuple(rew.shape)}, sample values={all_rew[:4].tolist()}, "
          f"max fractional part={frac.max().item():.4f} (not integer-collapsed)  OK")


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"Devices under test: {[str(d) for d in devices]}")

    for device in devices:
        check_atari_reward_unchanged(device)
        check_exact_continuous_passthrough(device)
        check_continuous_end_to_end(device, action_dim=4)
        check_continuous_end_to_end(device, action_dim=6)

    print("\nAll Stage 5C checks passed.")


if __name__ == "__main__":
    main()
