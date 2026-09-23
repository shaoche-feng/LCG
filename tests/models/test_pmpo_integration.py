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
    assert "kl_initial" in metrics and float(metrics["kl_initial"]) == pytest.approx(0.0, abs=1e-5)
    assert all(p.grad is None for p in model.initial_actor.parameters())
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


def test_fixed_prior_never_refreshes_while_actor_moves():
    from torch.distributions import kl_divergence

    # Tiny refresh_interval=2 means a moving prior WOULD refresh almost every call;
    # fixed_prior=True must override that entirely, for the whole run.
    model, _ = make_system(fixed_prior=True, prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 0)
    original_prior = deepcopy(model.prior_actor.state_dict())
    original_actor = deepcopy(model.actor.state_dict())

    for name, p in original_prior.items():
        torch.testing.assert_close(p, original_actor[name], rtol=0, atol=0)

    obs = torch.randn(4, 12, 8, 8)
    kl_initial = kl_divergence(model.distribution(model.actor(obs)), model.distribution(model.prior_actor(obs))).sum(-1)
    torch.testing.assert_close(kl_initial, torch.zeros_like(kl_initial), atol=1e-6, rtol=0)

    for _ in range(5):
        opt.zero_grad()
        loss, metrics = model()
        assert torch.isfinite(torch.as_tensor(metrics["kl_prior"])).all()
        loss.backward()
        assert all(p.grad is None for p in model.prior_actor.parameters())
        opt.step()
        for key, p in model.prior_actor.state_dict().items():
            torch.testing.assert_close(p, original_prior[key], rtol=0, atol=0)

    assert all(not p.requires_grad for p in model.prior_actor.parameters())
    assert any(not torch.equal(p, original_actor[n]) for n, p in model.actor.state_dict().items())
    kl_after = kl_divergence(model.distribution(model.actor(obs)), model.distribution(model.prior_actor(obs))).sum(-1)
    assert torch.isfinite(kl_after).all() and (kl_after > 0).any()


def test_slower_prior_refresh_blocks_and_checkpoint_round_trip():
    # prior_refresh_interval=3 exercises EXACTLY the same modular-arithmetic refresh
    # code path as the real 500-update experiment (nothing in forward()/PMPOBetaConfig
    # is specific to the number 500) -- a small interval keeps this test fast while
    # covering every block-boundary behavior requested for refresh=500.
    from torch.distributions import kl_divergence
    from models.pmpo_beta import PMPOBeta

    model, _ = make_system(prior_refresh_interval=3)
    opt = PMPOOptimizers(model, 0)

    # 1. prior equals current actor at initialization.
    initial_actor = deepcopy(model.actor.state_dict())
    for key, p in model.prior_actor.state_dict().items():
        torch.testing.assert_close(p, initial_actor[key], rtol=0, atol=0)
    obs = torch.randn(4, 12, 8, 8)
    # 6. KL immediately after refresh (here: at initialization, block 0) is ~0.
    kl0 = kl_divergence(model.distribution(model.actor(obs)), model.distribution(model.prior_actor(obs))).sum(-1)
    torch.testing.assert_close(kl0, torch.zeros_like(kl0), atol=1e-6, rtol=0)

    def full_cycle():
        # A real training step: only opt.step() advances self.updates (the modular
        # refresh counter forward() reads), so every refresh-boundary check below uses
        # this, never a bare model() call (which would leave self.updates unchanged).
        opt.zero_grad()
        loss, metrics = model()
        loss.backward()
        opt.step()
        return metrics

    # 2 & 3. prior remains frozen (no gradient, bit-identical) through updates 0, 1, 2
    # (the whole first block, refresh_interval=3 -> next refresh at update 3).
    for _ in range(3):
        for key, p in model.prior_actor.state_dict().items():
            torch.testing.assert_close(p, initial_actor[key], rtol=0, atol=0)
        full_cycle()
        assert all(p.grad is None for p in model.prior_actor.parameters())

    # 4. current actor changed over that interval.
    assert any(not torch.equal(p, initial_actor[n]) for n, p in model.actor.state_dict().items())
    assert model.updates.item() == 3

    # 5 & 6. at the refresh update (self.updates==3) the prior becomes an exact copy of
    # the actor as of the START of this cycle, and the KL this SAME forward() computed
    # (dist vs the just-refreshed prior, before this cycle's own gradient step moves the
    # actor further) is ~0 -- read directly from that cycle's own metrics, not
    # recomputed afterward (which would already reflect one more step of drift).
    actor_at_block1_start = deepcopy(model.actor.state_dict())
    metrics_at_refresh = full_cycle()
    for key, p in model.prior_actor.state_dict().items():
        torch.testing.assert_close(p, actor_at_block1_start[key], rtol=0, atol=0)
    torch.testing.assert_close(metrics_at_refresh["kl_prior"], torch.zeros_like(metrics_at_refresh["kl_prior"]), atol=1e-5, rtol=0)

    # 7. prior stays frozen through the whole second block (updates 3, 4, 5).
    prior_block1 = deepcopy(model.prior_actor.state_dict())
    for _ in range(2):
        for key, p in model.prior_actor.state_dict().items():
            torch.testing.assert_close(p, prior_block1[key], rtol=0, atol=0)
        full_cycle()
    assert model.updates.item() == 6

    # 8. second refresh occurs correctly at self.updates==6.
    actor_at_block2_start = deepcopy(model.actor.state_dict())
    full_cycle()
    for key, p in model.prior_actor.state_dict().items():
        torch.testing.assert_close(p, actor_at_block2_start[key], rtol=0, atol=0)

    # 9. checkpoint/resume: state_dict/load_state_dict round-trips the current prior
    # block (prior_actor weights) and the update counter (which block we're in).
    sd = deepcopy(model.state_dict())
    fresh = make_controller(prior_refresh_interval=3)
    fresh.load_state_dict(sd)
    torch.testing.assert_close(fresh.updates, model.updates, rtol=0, atol=0)
    for key, p in fresh.prior_actor.state_dict().items():
        torch.testing.assert_close(p, model.prior_actor.state_dict()[key], rtol=0, atol=0)
    for key, p in fresh.actor.state_dict().items():
        torch.testing.assert_close(p, model.actor.state_dict()[key], rtol=0, atol=0)


def test_resume_mid_block_does_not_restart_prior_schedule_from_zero():
    # Simulates a real resume: train to a NON-multiple-of-interval update count (2 out
    # of a 3-update block, i.e. the equivalent of "250 into the 1500-1999 block" from
    # the 500-update production schedule), checkpoint via state_dict, load into a FRESH
    # model/optimizer/env (matching what a real process restart does), then confirm the
    # very next update does NOT refresh (still mid-block) and the refresh AFTER that
    # lands at the correct absolute boundary (3), not at a schedule restarted from 0.
    model, _ = make_system(prior_refresh_interval=3)
    opt = PMPOOptimizers(model, 0)
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model()
        loss.backward()
        opt.step()
    assert model.updates.item() == 2
    prior_at_save = deepcopy(model.prior_actor.state_dict())
    sd = deepcopy(model.state_dict())

    fresh, _ = make_system(prior_refresh_interval=3)
    fresh_opt = PMPOOptimizers(fresh, 0)
    fresh.load_state_dict(sd)
    for key, p in fresh.prior_actor.state_dict().items():
        torch.testing.assert_close(p, prior_at_save[key], rtol=0, atol=0)

    # Resumed update index 2 (mid-block): must NOT refresh.
    fresh_opt.zero_grad()
    loss, _ = fresh()
    loss.backward()
    fresh_opt.step()
    assert fresh.updates.item() == 3
    for key, p in fresh.prior_actor.state_dict().items():
        torch.testing.assert_close(p, prior_at_save[key], rtol=0, atol=0)

    # Resumed update index 3: this IS the correct absolute block boundary (3, not 0) --
    # refresh must fire here, using the actor as of the start of this exact call.
    actor_before_refresh = deepcopy(fresh.actor.state_dict())
    fresh_opt.zero_grad()
    loss, _ = fresh()
    loss.backward()
    fresh_opt.step()
    for key, p in fresh.prior_actor.state_dict().items():
        torch.testing.assert_close(p, actor_before_refresh[key], rtol=0, atol=0)


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
