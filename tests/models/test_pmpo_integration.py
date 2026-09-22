from copy import deepcopy
from types import SimpleNamespace

import pytest
import torch
from torch.utils.data import DataLoader, IterableDataset

from envs import WorldModelEnv, WorldModelEnvConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig
from models.diffusion.inner_model import InnerModelConfig
from models.rew_end_model import RewEndModel, RewEndModelConfig
from models.pmpo_beta import PMPOOptimizers
from test_pmpo_beta import make_controller


class Prompts(IterableDataset):
    def __iter__(self):
        while True:
            # Real prompt rewards deliberately huge: they must never become targets.
            yield SimpleNamespace(obs=torch.rand(2, 3, 8, 8) * 2 - 1,
                                  act=torch.tensor([[-0.5, 3.], [-0.5, 3.]]), rew=torch.full((2,), 1e9))


def collate(samples):
    return SimpleNamespace(obs=torch.stack([s.obs for s in samples]), act=torch.stack([s.act for s in samples]))


def make_system(device="cpu", **kwargs):
    torch.manual_seed(7)
    model = make_controller(**kwargs).to(device)
    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16,
                                    depths=[1], channels=[8], attn_depths=[False], continuous_action_dim=2),
        sigma_data=0.5, sigma_offset_noise=0.3)).to(device)
    reward = RewEndModel(RewEndModelConfig(lstm_dim=16, img_channels=3, img_size=8, cond_channels=16,
                                         depths=[1], channels=[8], attn_depths=[False],
                                         continuous_action_dim=2, continuous_reward=True)).to(device)
    loader = DataLoader(Prompts(), batch_size=2, collate_fn=collate, generator=torch.Generator().manual_seed(0))
    env = WorldModelEnv(denoiser, reward, loader,
                        WorldModelEnvConfig(horizon=2, num_batches_to_preload=1,
                                            diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)))
    model.setup_training(env)
    return model, env


def test_actual_diamond_rollout_and_update():
    torch.set_num_threads(1)
    model, env = make_system()
    opt = PMPOOptimizers(model, warmup_steps=0)
    actor_before = deepcopy(model.actor.state_dict())
    value_before = deepcopy(model.value.state_dict())
    loss, metrics = model()
    assert metrics["rollout_length"] == 3
    assert abs(metrics["return_mean"]) < 1e6
    loss.backward()
    metrics.update(opt.step())
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
    assert any(not torch.equal(p, actor_before[n]) for n, p in model.actor.state_dict().items())
    assert any(not torch.equal(p, value_before[n]) for n, p in model.value.state_dict().items())
    assert all(p.grad is None for p in env.sampler.denoiser.parameters())
    assert all(p.grad is None for p in env.rew_end_model.parameters())
    assert all(p.grad is None for p in model.prior_actor.parameters())


def test_rollout_detached_and_reward_hook():
    model, _ = make_system()
    rollout = model.collect_imagination()
    assert all(not t.requires_grad for t in rollout[:-1])
    received = []
    def hook(infos, rew):
        received.append((infos, rew.clone()))
        return torch.zeros_like(rew) + 2
    model.set_intrinsic_reward_fn(hook)
    _, metrics = model()
    assert len(received) == 1 and len(received[0][0]) == 3
    assert metrics["return_mean"] > 1


def test_prior_refresh_blocks():
    model, _ = make_system(prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 0)
    original = deepcopy(model.prior_actor.state_dict())
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model()
        for key, p in model.prior_actor.state_dict().items():
            torch.testing.assert_close(p, original[key], rtol=0, atol=0)
        loss.backward()
        opt.step()
    expected = deepcopy(model.actor.state_dict())
    model()
    for key, p in model.prior_actor.state_dict().items():
        torch.testing.assert_close(p, expected[key], rtol=0, atol=0)


def test_env_loop_preserves_pre_reset_observation():
    from coroutines.env_loop import make_env_loop
    class InPlaceEnv:
        num_envs = 1
        def reset(self, **kwargs):
            self.obs = torch.ones(1, 3, 8, 8)
            return self.obs, {}
        def step(self, action):
            final = self.obs.clone() * 2
            self.obs.fill_(9)
            return self.obs, torch.zeros(1), torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long), {"final_observation": final}
    model = make_controller()
    with torch.no_grad():
        rollout = make_env_loop(InPlaceEnv(), model).send(1)
    torch.testing.assert_close(rollout[0], torch.ones(1, 1, 3, 8, 8))
    final_stack = torch.cat([torch.ones(1, 9, 8, 8), torch.ones(1, 3, 8, 8) * 2], 1)
    torch.testing.assert_close(rollout[7][:, 0], model.value(final_stack).squeeze(-1))
