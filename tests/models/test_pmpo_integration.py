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


def compose_cleanup_config(overrides=()):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    root = Path(__file__).resolve().parents[2]
    with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
        return compose(config_name="trainer", overrides=list(overrides))


def test_default_hydra_agent_and_checkpoint(tmp_path):
    from agent import Agent, get_action_space_kwargs
    from hydra.utils import instantiate
    cfg = compose_cleanup_config()
    assert (cfg.env.train.domain_name, cfg.env.train.task_name) == ("cheetah", "run")
    assert cfg.agent.actor_critic.prior_refresh_interval == 500
    assert cfg.collection.train.steps_per_epoch == 100  # Recipes override this to 500.
    env = SimpleNamespace(action_dim=6, action_low=torch.full((6,), -1.), action_high=torch.ones(6))
    agent = Agent(instantiate(cfg.agent, **get_action_space_kwargs(env)))
    assert isinstance(agent.rew_end_model.lstm, torch.nn.LSTM)
    assert not hasattr(agent.actor_critic, "lstm_dim")
    states = torch.randn(2, 12, 64, 64)
    with torch.no_grad():
        raw, value = agent.actor_critic.predict_act_value(states)
        actions = agent.actor_critic.sample_action(raw, deterministic=True)
    path = tmp_path / "agent.pt"
    torch.save(agent.state_dict(), path)
    restored = Agent(instantiate(cfg.agent, **get_action_space_kwargs(env)))
    restored.load(path)
    with torch.no_grad():
        actual_raw, actual_value = restored.actor_critic.predict_act_value(states)
        actual_actions = restored.actor_critic.sample_action(actual_raw, deterministic=True)
    for actual, expected in ((actual_raw, raw), (actual_value, value), (actual_actions, actions)):
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)


def test_prior_refresh_interval_defaults_agree_and_explicit_values_are_honored():
    from dataclasses import fields
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from models.pmpo_beta import PMPOBetaConfig
    from utils import normalize_saved_config
    python_default = next(f.default for f in fields(PMPOBetaConfig) if f.name == "prior_refresh_interval")
    cfg = compose_cleanup_config()
    assert python_default == cfg.agent.actor_critic.prior_refresh_interval == 500
    assert make_controller().cfg.prior_refresh_interval == 500
    assert make_controller(prior_refresh_interval=3).cfg.prior_refresh_interval == 3
    overridden = compose_cleanup_config(["agent.actor_critic.prior_refresh_interval=250"])
    assert instantiate(overridden.agent.actor_critic).prior_refresh_interval == 250
    # Saved run configs always state their interval, including the early value of 10.
    saved = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    saved.agent.actor_critic.prior_refresh_interval = 10
    assert instantiate(normalize_saved_config(saved).agent.actor_critic).prior_refresh_interval == 10


def test_saved_pmpo_config_migration_preserves_learning_fields(tmp_path):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf, open_dict
    from utils import normalize_saved_config
    cfg = compose_cleanup_config(["common.seed=23", "collection.train.steps_per_epoch=500",
                                  "agent.actor_critic.entropy_coef=0.01"])
    with open_dict(cfg):
        cfg.training.model_free = False
        cfg.env.train.type = "dm_control"
        cfg.env.test.type = "${..train.type}"
        cfg.collection.train.first_epoch.max = 10000
        cfg.collection.train.first_epoch.threshold_rew = 10
        cfg.actor_critic.actor_critic_loss = {
            "_target_": "models.actor_critic.ActorCriticLossConfig", "backup_every": 15,
            "gamma": .985, "lambda_": .95, "weight_value_loss": 1., "weight_entropy_loss": .001,
        }
        cfg.actor_critic.optimizer = {"lr": 123., "weight_decay": 0., "eps": 1e-8}
        cfg.agent.actor_critic.lstm_dim = 512
        for component in (cfg.agent, cfg.agent.denoiser.inner_model,
                          cfg.agent.rew_end_model, cfg.agent.actor_critic):
            component.num_actions = None
        for component in (cfg.agent, cfg.agent.rew_end_model, cfg.agent.actor_critic):
            component.continuous_reward = True
    path = tmp_path / "historical.yaml"
    OmegaConf.save(cfg, path)
    original = path.read_bytes()
    migrated = normalize_saved_config(OmegaConf.load(path))
    assert path.read_bytes() == original and cfg.agent.actor_critic.lstm_dim == 512
    assert migrated.agent.actor_critic.entropy_coef == 0.01
    assert migrated.common.seed == 23 and migrated.collection.train.steps_per_epoch == 500
    assert "model_free" not in migrated.training and "lstm_dim" not in migrated.agent.actor_critic
    assert "actor_critic_loss" not in migrated.actor_critic and "optimizer" not in migrated.actor_critic
    assert migrated.agent.rew_end_model.lstm_dim == 512
    # Agent subtree migration also works with interpolation resolved by its parent.
    kwargs = dict(continuous_action_dim=2, action_low=[-3., 0.5], action_high=[2., 7.])
    assert instantiate(migrated.agent, **kwargs).actor_critic.entropy_coef == 0.01
    assert instantiate(normalize_saved_config(cfg).agent, **kwargs).actor_critic.gamma == cfg.agent.actor_critic.gamma
    # New unknown fields are not silently consumed by migration.
    with open_dict(cfg):
        cfg.agent.actor_critic.unknown_knob = 42
    untouched = normalize_saved_config(cfg)
    assert untouched.agent.actor_critic.unknown_knob == 42
    with pytest.raises(Exception, match="unknown_knob"):
        instantiate(untouched.agent, **kwargs)


@pytest.mark.parametrize("field,value", [
    ("training.model_free", True), ("env.train.type", "atari"),
    ("agent.num_actions", 6), ("agent.rew_end_model.continuous_reward", False),
    ("agent.actor_critic._target_", "models.actor_critic.ActorCriticConfig"),
    ("training.model_free", None), ("training.model_free", 0), ("training.model_free", ""),
    ("agent.denoiser.inner_model.continuous_reward", True),
    ("agent.actor_critic.lstm_dim", "512"), ("agent.actor_critic.lstm_dim", 0),
    ("collection.train.first_epoch.max", -1), ("collection.train.first_epoch.threshold_rew", "10"),
])
def test_unsupported_saved_paths_fail_explicitly(field, value):
    from omegaconf import OmegaConf
    from utils import normalize_saved_config
    cfg = compose_cleanup_config()
    OmegaConf.update(cfg, field, value, force_add=True)
    with pytest.raises(ValueError):
        normalize_saved_config(cfg)


@pytest.mark.parametrize("section,field,value", [
    ("actor_critic_loss", "_target_", "invalid.CustomLoss"),
    ("actor_critic_loss", "unknown", 42), ("actor_critic_loss", "gamma", float("nan")),
    ("actor_critic_loss", "lambda_", 1.1), ("actor_critic_loss", "backup_every", True),
    ("actor_critic_loss", "weight_entropy_loss", "0.001"),
    ("optimizer", "_target_", "invalid.Optimizer"), ("optimizer", "lr", "0.01"),
    ("optimizer", "eps", 0.), ("optimizer", "weight_decay", -1.),
])
def test_migration_rejects_invalid_fields_inside_discarded_sections(section, field, value):
    from omegaconf import open_dict
    from utils import normalize_saved_config
    cfg = compose_cleanup_config()
    with open_dict(cfg):
        cfg.actor_critic.actor_critic_loss = {
            "_target_": "models.actor_critic.ActorCriticLossConfig", "backup_every": 15,
            "gamma": .985, "lambda_": .95, "weight_value_loss": 1., "weight_entropy_loss": .001,
        }
        cfg.actor_critic.optimizer = {"lr": 1e-4, "weight_decay": 0., "eps": 1e-8}
        cfg.actor_critic[section][field] = value
    with pytest.raises(ValueError):
        normalize_saved_config(cfg)


def test_migration_resolves_references_before_removing_obsolete_fields():
    from omegaconf import OmegaConf, open_dict
    from utils import normalize_saved_config
    cfg = compose_cleanup_config()
    with open_dict(cfg):
        cfg.actor_critic.actor_critic_loss = {
            "_target_": "models.actor_critic.ActorCriticLossConfig", "backup_every": 15,
            "gamma": .93, "lambda_": .95, "weight_value_loss": 1., "weight_entropy_loss": .001,
        }
        cfg.agent.actor_critic.gamma = "${actor_critic.actor_critic_loss.gamma}"
        cfg.collection.train.first_epoch.max = 10000
        cfg.collection.train.first_epoch.threshold_rew = "${.max}"
    original = OmegaConf.to_container(cfg, resolve=False)
    migrated = normalize_saved_config(cfg)
    assert migrated.agent.actor_critic.gamma == .93
    assert "actor_critic_loss" not in migrated.actor_critic
    assert "threshold_rew" not in migrated.collection.train.first_epoch
    assert OmegaConf.to_container(cfg, resolve=False) == original
    assert normalize_saved_config(cfg.agent).actor_critic.gamma == .93


def test_historical_initialization_fingerprints_ignore_tensor_storage_identity(monkeypatch):
    from pathlib import Path
    import runpy
    import sys
    root = Path(__file__).resolve().parents[2]
    # The historical script adds its import paths; isolate that side effect.
    monkeypatch.setattr(sys, "path", list(sys.path))
    fingerprint = runpy.run_path(str(root / "scripts/lcg_diagnostic/pretrained_collectors/create_theta0.py"))["fingerprint"]
    state = make_controller().state_dict()
    copied = deepcopy(state)
    assert fingerprint(state) == fingerprint(copied)
    copied["_extra_state"]["rng"]["cpu"][0] ^= 1
    assert fingerprint(state) != fingerprint(copied)


@pytest.mark.parametrize("diagnostic", [False, True])
def test_static_dataset_trainer_world_model_and_pmpo_resume(tmp_path, monkeypatch, diagnostic):
    """One real optimization step per component, without real-env collection."""
    from pathlib import Path
    import numpy as np
    from omegaconf import OmegaConf
    from data import Dataset, Episode
    from trainer import Trainer
    import trainer as trainer_module
    import pmpo_diagnostic
    root = Path(__file__).resolve().parents[2]
    dataset_root = tmp_path / "static"
    for split in ("train", "test"):
        dataset = Dataset(dataset_root / split, cache_in_ram=True)
        for _ in range(2):
            episode = Episode(torch.rand(10, 3, 8, 8) * 2 - 1, torch.rand(10, 2) * 2 - 1,
                              torch.rand(10), torch.zeros(10, dtype=torch.long),
                              torch.zeros(10, dtype=torch.long), {})
            dataset.add_episode(episode)
        dataset.save_to_default_path()
    cfg = compose_cleanup_config([
        "common.seed=7", "common.devices=cpu", "env.train.size=8",
        "agent.denoiser.inner_model.num_steps_conditioning=2",
        "agent.denoiser.inner_model.cond_channels=16", "agent.denoiser.inner_model.depths=[1]",
        "agent.denoiser.inner_model.channels=[8]", "agent.denoiser.inner_model.attn_depths=[0]",
        "agent.rew_end_model.lstm_dim=16", "agent.rew_end_model.cond_channels=16",
        "agent.rew_end_model.depths=[1]", "agent.rew_end_model.channels=[8]",
        "agent.rew_end_model.attn_depths=[0]", "agent.actor_critic.channels=[8]",
        "agent.actor_critic.down=[0]", "agent.actor_critic.actor_hidden_dims=[16]",
        "agent.actor_critic.value_hidden_dims=[16]", "world_model_env.horizon=2",
        "world_model_env.num_batches_to_preload=1", "world_model_env.diffusion_sampler.num_steps_denoising=2",
        f"static_dataset.path={dataset_root.as_posix()}", "training.num_final_epochs=1",
        "training.compile_wm=false", "evaluation.every=1",
        "denoiser.training.batch_size=2", "rew_end_model.training.batch_size=2",
        "actor_critic.training.batch_size=2", "denoiser.training.steps_first_epoch=1",
        "rew_end_model.training.steps_first_epoch=1", "actor_critic.training.steps_first_epoch=1",
        *(["+pmpo_diagnostic=bounded", "denoiser.training.steps_per_epoch=1",
           "rew_end_model.training.steps_per_epoch=1", "actor_critic.training.steps_per_epoch=1"]
          if diagnostic else []),
    ])
    fake_env = SimpleNamespace(action_dim=2, action_low=torch.full((2,), -1.), action_high=torch.ones(2))
    monkeypatch.setattr(trainer_module, "make_dm_control_env", lambda **kwargs: fake_env)
    class TinyEvalEnv:
        def __init__(self, **kwargs):
            pass
        def reset(self, seed):
            self.t = 0
            return np.zeros((8, 8, 3), dtype=np.uint8), {}
        def step(self, action):
            self.t += 1
            return np.full((8, 8, 3), self.t, dtype=np.uint8), 0., False, self.t == 4, {}
        def close(self):
            pass
    monkeypatch.setattr(pmpo_diagnostic, "DMControlEnv", TinyEvalEnv)
    for method in ("init", "save", "log"):
        monkeypatch.setattr(trainer_module.wandb, method, lambda *args, **kwargs: None)
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hydra").mkdir()
    OmegaConf.save(cfg, tmp_path / ".hydra/config.yaml")
    trainer = Trainer(cfg, root)
    trainer._run()
    assert trainer.epoch == 1 and trainer.num_epochs_collect == 0
    assert trainer.train_dataset.num_steps == 20
    assert trainer.agent.actor_critic.updates.item() == 1
    assert trainer.num_batch_train.denoiser == trainer.num_batch_train.rew_end_model == 1
    cfg.common.resume = True
    resumed = Trainer(cfg, root)
    assert resumed.epoch == 1 and resumed.num_batch_train.actor_critic == 1
    for key, value in trainer.agent.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(resumed.agent.state_dict()[key], value, rtol=0, atol=0)
    if diagnostic:
        fixed_real = trainer._pmpo_diagnostic.fixed_real
        original = trainer._pmpo_diagnostic.state_dict()
        restored = resumed._pmpo_diagnostic.state_dict()
        restored_fixed, original_fixed = restored.pop("fixed_real"), original.pop("fixed_real")
        assert restored_fixed["encoding"] == original_fixed["encoding"] == "uint8_pixels_div"
        assert torch.equal(restored_fixed["values"], original_fixed["values"])
        assert torch.equal(resumed._pmpo_diagnostic.fixed_real, fixed_real)
        assert restored == original and restored["updates_observed"] == 1
        stored = torch.load(tmp_path / "checkpoints/state.pt", weights_only=False)["pmpo_diagnostic"]["fixed_real"]
        assert stored["encoding"] == "uint8_pixels_div" and stored["values"].dtype == torch.uint8
        # state.pt, not fixed_real_stacks.pt, is authoritative for the fixed states.
        path = tmp_path / "pmpo_diagnostic/fixed_real_stacks.pt"
        torch.save(fixed_real + 1, path)
        different = path.read_bytes()
        resumed = Trainer(cfg, root)
        torch.testing.assert_close(resumed._pmpo_diagnostic.fixed_real, fixed_real, rtol=0, atol=0)
        assert path.read_bytes() == different
        path.unlink()
        cfg.training.num_final_epochs = 2
        resumed = Trainer(cfg, root)
        resumed._run()
        assert resumed.epoch == 2 and resumed._pmpo_diagnostic.updates_observed == 2
        torch.testing.assert_close(resumed._pmpo_diagnostic.fixed_real, fixed_real, rtol=0, atol=0)
        torch.testing.assert_close(torch.load(path, weights_only=True), fixed_real, rtol=0, atol=0)


class Prompts(IterableDataset):
    def __iter__(self):
        while True:
            # Real prompt rewards deliberately huge: they must never become targets.
            yield SimpleNamespace(obs=torch.rand(2, 3, 8, 8) * 2 - 1,
                                  act=torch.tensor([[-0.5, 3.], [-0.5, 3.]]), rew=torch.full((2,), 1e9))


def collate(samples):
    obs = torch.stack([s.obs for s in samples])
    # Full-length prompts: no padding before an episode start.
    return SimpleNamespace(obs=obs, act=torch.stack([s.act for s in samples]),
                           mask_padding=torch.ones(obs.shape[:2], dtype=torch.bool))


def make_system(device="cpu", **kwargs):
    torch.manual_seed(7)
    model = make_controller(**kwargs).to(device)
    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16,
                                    depths=[1], channels=[8], attn_depths=[False], continuous_action_dim=2),
        sigma_data=0.5, sigma_offset_noise=0.3)).to(device)
    reward = RewEndModel(RewEndModelConfig(lstm_dim=16, img_channels=3, img_size=8, cond_channels=16,
                                         depths=[1], channels=[8], attn_depths=[False],
                                         continuous_action_dim=2)).to(device)
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


def test_entropy_coef_zero_reproduces_baseline_actor_loss_exactly():
    # entropy_coef defaults to 0.0: the Stage 3 code path must be bit-identical to the
    # frozen 100k baseline when the new coefficient is left at its default.
    torch.manual_seed(11)
    baseline, _ = make_system()
    torch.manual_seed(11)
    zero_entropy, _ = make_system(entropy_coef=0.0)
    loss_baseline, metrics_baseline = baseline()
    loss_zero, metrics_zero = zero_entropy()
    torch.testing.assert_close(loss_baseline, loss_zero, rtol=0, atol=0)
    torch.testing.assert_close(metrics_baseline["loss_actor"], metrics_zero["loss_actor"], rtol=0, atol=0)
    assert float(metrics_zero["loss_entropy"]) == pytest.approx(0.0, abs=0.0)
    torch.testing.assert_close(metrics_zero["loss_actor_total"], metrics_zero["loss_actor"], rtol=0, atol=0)


def test_entropy_coef_adds_signed_bonus_and_logs_components_separately():
    model, _ = make_system(entropy_coef=0.01)
    loss, metrics = model()
    # loss_actor_total must equal the sum of the three logged components exactly.
    expected = metrics["loss_pmpo_raw"] + metrics["loss_kl"] + metrics["loss_entropy"]
    torch.testing.assert_close(metrics["loss_actor_total"], expected, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(metrics["loss_actor"], metrics["loss_actor_total"], rtol=0, atol=0)
    # loss_entropy = -entropy_coef * mean(H(pi)); its sign is tied to the sign of entropy.
    expected_entropy_term = -0.01 * metrics["policy_entropy"]
    torch.testing.assert_close(metrics["loss_entropy"], expected_entropy_term, rtol=1e-4, atol=1e-6)
    # Per-action-dimension entropy is logged and sums (over dims) to ~ the total entropy.
    dims = model.cfg.continuous_action_dim
    per_dim_sum = sum(metrics[f"policy_entropy_dim_{d}"] for d in range(dims))
    torch.testing.assert_close(per_dim_sum, metrics["policy_entropy"], rtol=1e-4, atol=1e-4)


def test_negative_entropy_coef_rejected():
    with pytest.raises(ValueError):
        make_controller(entropy_coef=-0.01)


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
