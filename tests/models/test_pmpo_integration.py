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
        return compose(config_name="exploration", overrides=list(overrides))


def test_default_hydra_agent_and_checkpoint(tmp_path):
    from agent import Agent, get_action_space_kwargs
    from hydra.utils import instantiate
    cfg = compose_cleanup_config()
    assert cfg.env.train.domain_name == "walker"
    assert list(cfg.env.train.task_names) == list(cfg.env.test.task_names) == ["stand", "walk", "run"]
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
    from exploration_trainer import ExplorationTrainer
    import exploration_trainer as trainer_module
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
        # Tiny LCG scorer for the one-level test denoiser.
        "intrinsic_reward.theta_s.include=[conv_out.*]", "intrinsic_reward.precision_reference_size=8",
        "intrinsic_reward.precision_num_mc=1", "intrinsic_reward.candidate_num_mc=2",
        "intrinsic_reward.candidate_chunk_size=4",
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
    trainer = ExplorationTrainer(cfg, root)
    trainer._run()
    assert trainer.epoch == 1 and trainer.num_epochs_collect == 0
    assert trainer.train_dataset.num_steps == 20
    assert trainer.agent.actor_critic.updates.item() == 1
    assert trainer.num_batch_train.denoiser == 1 and trainer.agent.rew_end_model is None
    cfg.common.resume = True
    resumed = ExplorationTrainer(cfg, root)
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
        resumed = ExplorationTrainer(cfg, root)
        torch.testing.assert_close(resumed._pmpo_diagnostic.fixed_real, fixed_real, rtol=0, atol=0)
        assert path.read_bytes() == different
        path.unlink()
        cfg.training.num_final_epochs = 2
        resumed = ExplorationTrainer(cfg, root)
        resumed._run()
        assert resumed.epoch == 2 and resumed._pmpo_diagnostic.updates_observed == 2
        torch.testing.assert_close(resumed._pmpo_diagnostic.fixed_real, fixed_real, rtol=0, atol=0)
        torch.testing.assert_close(torch.load(path, weights_only=True), fixed_real, rtol=0, atol=0)


class Prompts(IterableDataset):
    def __init__(self, action_dim=2):
        self.action = torch.tensor([-0.5, 3.]).repeat(action_dim // 2)

    def __iter__(self):
        while True:
            # Real prompt rewards deliberately huge: they must never become targets.
            yield SimpleNamespace(obs=torch.rand(2, 3, 8, 8) * 2 - 1,
                                  act=self.action.repeat(2, 1), rew=torch.full((2,), 1e9))


def collate(samples):
    obs = torch.stack([s.obs for s in samples])
    # Full-length prompts: no padding before an episode start.
    return SimpleNamespace(obs=obs, act=torch.stack([s.act for s in samples]),
                           mask_padding=torch.ones(obs.shape[:2], dtype=torch.bool))


def make_system(device="cpu", action_dim=2, **kwargs):
    """Controller and a tiny DIAMOND world model; action_dim (even) tiles the 2-d bounds."""
    torch.manual_seed(7)
    if action_dim != 2:
        kwargs.update(continuous_action_dim=action_dim, action_low=[-3., 0.5] * (action_dim // 2),
                      action_high=[2., 7.] * (action_dim // 2))
    model = make_controller(**kwargs).to(device)
    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16,
                                    depths=[1], channels=[8], attn_depths=[False], continuous_action_dim=action_dim),
        sigma_data=0.5, sigma_offset_noise=0.3)).to(device)
    reward = RewEndModel(RewEndModelConfig(lstm_dim=16, img_channels=3, img_size=8, cond_channels=16,
                                         depths=[1], channels=[8], attn_depths=[False],
                                         continuous_action_dim=action_dim)).to(device)
    loader = DataLoader(Prompts(action_dim), batch_size=2, collate_fn=collate, generator=torch.Generator().manual_seed(0))
    env = WorldModelEnv(denoiser, reward, loader,
                        WorldModelEnvConfig(horizon=2, num_batches_to_preload=1,
                                            diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)))
    model.setup_training(env)
    return model, env


def test_actual_diamond_rollout_and_update():
    torch.set_num_threads(1)
    model, env = make_system(debug_metrics=True)
    opt = PMPOOptimizers(model, warmup_steps=0)
    actor_before = deepcopy(model.actor.state_dict())
    value_before = deepcopy(model.value.state_dict())
    loss, metrics = model()
    assert not {"rollout_length", "all_finite", "loss_actor_total"} & set(metrics)  # removed in both modes
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
    baseline, _ = make_system(debug_metrics=True)
    torch.manual_seed(11)
    zero_entropy, _ = make_system(entropy_coef=0.0, debug_metrics=True)
    loss_baseline, metrics_baseline = baseline()
    loss_zero, metrics_zero = zero_entropy()
    torch.testing.assert_close(loss_baseline, loss_zero, rtol=0, atol=0)
    torch.testing.assert_close(metrics_baseline["loss_actor"], metrics_zero["loss_actor"], rtol=0, atol=0)
    assert float(metrics_zero["loss_entropy"]) == pytest.approx(0.0, abs=0.0)


def test_entropy_coef_adds_signed_bonus_and_logs_components_separately():
    model, _ = make_system(entropy_coef=0.01, debug_metrics=True)
    loss, metrics = model()
    # loss_actor must equal the sum of the three logged (debug) components.
    expected = metrics["loss_pmpo_raw"] + metrics["loss_kl"] + metrics["loss_entropy"]
    torch.testing.assert_close(metrics["loss_actor"], expected, rtol=1e-5, atol=1e-6)
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

# ---------------------------------------------------------------- Gaussian H10 EMA-critic controller
# Ported from the reference (kingplus0:/home/bigking/scfeng/LCG-stage3-gaussian-h10-emacritic,
# tests/models/test_tanh_gaussian.py and the untracked tests/models/test_ema_critic.py), plus the
# objective, frame-history and checkpoint-compatibility checks of this migration.

G = dict(actor_distribution="tanh_gaussian")
E = dict(actor_distribution="tanh_gaussian", slow_critic_coef=1.0)
H10 = dict(actor_distribution="tanh_gaussian", slow_critic_coef=1.0, slow_critic_rate=0.02,
           imagination_history="repeat_anchor")


def transformed_reference(model, raw):
    """Independent reference: torch's own transformed distribution, per dimension."""
    from torch.distributions import TransformedDistribution
    from torch.distributions.transforms import AffineTransform, TanhTransform
    dist = model.distribution(raw)
    width = model.action_high - model.action_low
    # cache_size=1: scoring a tensor this distribution just sampled reuses the exact pre-image.
    return TransformedDistribution(dist, [TanhTransform(cache_size=1),
                                          AffineTransform(model.action_low + width / 2, width / 2, cache_size=1)])


def test_gauss_hermite_constants_equal_numpy_hermgauss():
    """The stored rule is numpy's hermgauss(32), computed in a process without torch (on Windows
    LAPACK aborts once torch's OpenMP runtime is loaded)."""
    import json
    import subprocess
    import sys
    from models.pmpo_beta import GAUSS_HERMITE_32
    code = "import json, numpy as np; n, w = np.polynomial.hermite.hermgauss(32); print(json.dumps([n.tolist(), w.tolist()]))"
    nodes, weights = json.loads(subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=True).stdout)
    assert list(GAUSS_HERMITE_32[0]) == nodes and list(GAUSS_HERMITE_32[1]) == weights


def test_gaussian_parameterization_bounds_and_deterministic_rule():
    from models.pmpo_beta import GAUSSIAN_STD_MIN, GAUSSIAN_STD_SPAN
    torch.manual_seed(0)
    model = make_controller(**G)
    raw = torch.randn(64, 4) * 3
    dist = model.distribution(raw)
    mu, raw_std = raw.chunk(2, dim=-1)
    torch.testing.assert_close(dist.mean, mu)
    torch.testing.assert_close(dist.stddev, GAUSSIAN_STD_MIN + GAUSSIAN_STD_SPAN * torch.sigmoid(raw_std))
    assert (dist.stddev > GAUSSIAN_STD_MIN).all() and (dist.stddev < GAUSSIAN_STD_MIN + GAUSSIAN_STD_SPAN).all()
    actions = model.sample_action(raw)
    assert actions.dtype == torch.float32 and actions.shape == (64, 2) and not actions.requires_grad
    assert ((actions >= model.action_low) & (actions <= model.action_high)).all()
    det = model.sample_action(raw, deterministic=True)
    width = model.action_high - model.action_low
    torch.testing.assert_close(det, model.action_low + (torch.tanh(mu) + 1) / 2 * width)
    assert torch.equal(det, model.sample_action(raw, deterministic=True))
    huge = torch.tensor([[50., -50., 0., 0.]])  # saturating means still give in-bounds actions
    sat = model.sample_action(huge, deterministic=True)
    assert ((sat >= model.action_low) & (sat <= model.action_high)).all()


def test_gaussian_log_prob_matches_independent_transformed_distribution():
    torch.manual_seed(1)
    model = make_controller(**G)
    raw = torch.randn(256, 4) * torch.tensor([1.5, 1.5, 2., 2.])
    actions = model.sample_action(raw)
    logp, _ = model.log_prob_and_entropy(raw, actions)
    expected = transformed_reference(model, raw).log_prob(actions).sum(-1)
    canonical = 2 * (actions - model.action_low) / (model.action_high - model.action_low) - 1
    safe = (canonical.abs() < 0.999).all(-1)  # away from float32 inversion limits
    assert safe.sum() > 100
    torch.testing.assert_close(logp[safe], expected[safe], rtol=1e-4, atol=1e-3)


def test_gaussian_log_prob_and_gradients_finite_at_and_near_action_limits():
    model = make_controller(**G)
    low, high = model.action_low, model.action_high
    width = high - low
    actions = torch.stack([low, high, low + 1e-6 * width, high - 1e-6 * width, low + 1e-3 * width, high - 1e-3 * width])
    for scale in (0., 5., -12.):
        raw = torch.zeros(len(actions), 4)
        raw[:, :2] = scale
        raw.requires_grad_(True)
        logp, entropy = model.log_prob_and_entropy(raw, actions)
        assert torch.isfinite(logp).all() and torch.isfinite(entropy).all()
        (logp.sum() + entropy.sum()).backward()
        assert torch.isfinite(raw.grad).all()
    with pytest.raises(ValueError):
        model.log_prob_and_entropy(torch.zeros(1, 4), (high + 1e-3).unsqueeze(0))


def test_gaussian_transformed_entropy_matches_monte_carlo_and_has_gradient():
    torch.manual_seed(2)
    model = make_controller(**G)
    raw = torch.tensor([[0., 0., -2., 2.], [1.5, -0.5, 1., -1.], [3., 0.2, 0., 3.]], requires_grad=True)
    _, entropy = model.log_prob_and_entropy(raw, model.sample_action(raw, deterministic=True))
    ref = transformed_reference(model, raw.detach())
    mc = -ref.log_prob(ref.sample((200_000,))).sum(-1).mean(0)
    torch.testing.assert_close(entropy.detach(), mc, rtol=0, atol=0.02)
    entropy.sum().backward()
    assert torch.isfinite(raw.grad).all() and raw.grad.abs().sum() > 0
    narrow = model.entropy_per_dim(model.distribution(torch.tensor([[0., 0., -3., -3.]])))
    wide = model.entropy_per_dim(model.distribution(torch.tensor([[0., 0., 3., 3.]])))
    assert (wide > narrow).all()


def test_gaussian_kl_is_analytic_between_pre_tanh_normals_and_prior_frozen():
    from torch.distributions import kl_divergence
    model = make_controller(**G)
    obs = torch.randn(4, 12, 8, 8)
    prior = model.distribution(model.prior_actor(obs))
    same = kl_divergence(model.distribution(model.actor(obs)), prior).sum(-1)
    torch.testing.assert_close(same, torch.zeros_like(same), atol=1e-6, rtol=0)
    with torch.no_grad():
        model.actor.head[-1].bias[0] += 1
        model.actor.head[-1].bias[3] -= 1
    p, q = model.distribution(model.actor(obs)), prior
    kl = kl_divergence(p, q).sum(-1)
    closed = (torch.log(q.stddev / p.stddev) + (p.stddev ** 2 + (p.mean - q.mean) ** 2) / (2 * q.stddev ** 2) - 0.5).sum(-1)
    torch.testing.assert_close(kl, closed, rtol=1e-5, atol=1e-6)
    assert (kl > 0).all()
    kl.mean().backward()
    assert all(p.grad is None and not p.requires_grad for p in model.prior_actor.parameters())


@pytest.mark.parametrize("objective", ["sign", "advantage_weighted"])
def test_h10_update_is_finite_with_gaussian_diagnostics_only(objective):
    model, _ = make_system(**H10, actor_objective=objective, prior_refresh_interval=2, debug_metrics=True)
    opt = PMPOOptimizers(model, 0)
    for _ in range(3):
        opt.zero_grad()
        loss, metrics = model()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())
        assert {"pretanh_mean_mean", "pretanh_std_mean", "pretanh_std_near_floor_fraction", "tanh_inverse_limited_fraction",
                "policy_conditional_std_0", "loss_slowreg", "loss_value_target", "slow_value_mean", "slow_value_gap"} <= set(metrics)
        assert not any(k.startswith(("alpha_", "beta_")) for k in metrics)  # no Beta concentrations
        loss.backward()
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        opt.step()


def test_objective_weights_only_the_pmpo_term_kl_entropy_and_critic_unchanged():
    """Same rollout (same seeds) under both objectives: only loss_pmpo_raw differs; the
    advantage-weighted term equals the detached-weight formula on the captured rollout."""
    from models.pmpo_beta import pmpo_loss
    results = {}
    for objective in ("sign", "advantage_weighted"):
        model, _ = make_system(**H10, actor_objective=objective, debug_metrics=True)
        captured = {}
        collect = model.collect_imagination

        def spy(collect=collect, captured=captured):
            captured["rollout"] = collect()
            return captured["rollout"]
        model.collect_imagination = spy
        _, metrics = model()
        obs, act, rew, end, trunc, _, old_values, bootstrap, _ = captured["rollout"]
        from models.actor_critic import compute_lambda_returns
        advantage = (compute_lambda_returns(rew, end, trunc, bootstrap, model.cfg.gamma, model.cfg.lambda_) - old_values).flatten()
        logp, _ = model.log_prob_and_entropy(model.actor(obs.flatten(0, 1)), act.flatten(0, 1))
        expected = pmpo_loss(logp, advantage, model.cfg.alpha_pmpo, objective, 5.0)
        assert float(metrics["loss_pmpo_raw"]) == pytest.approx(float(expected), rel=1e-6, abs=1e-7)
        results[objective] = metrics
    for key in ("loss_kl", "loss_entropy", "kl_prior", "loss_value", "loss_value_target", "loss_slowreg", "policy_entropy"):
        torch.testing.assert_close(torch.as_tensor(results["sign"][key]), torch.as_tensor(results["advantage_weighted"][key]),
                                   rtol=0, atol=0)
    assert float(results["sign"]["weight_std_positive"]) == 0


def test_slow_critic_disabled_creates_nothing_and_settings_are_validated():
    torch.manual_seed(0)
    base = make_controller(**G)
    torch.manual_seed(0)
    off = make_controller(**G, slow_critic_coef=0.0)
    assert not hasattr(off, "slow_value") and list(base.state_dict()) == list(off.state_dict())
    for coef, rate in ((-1.0, 0.02), (1.0, 0.0), (1.0, 1.5)):
        with pytest.raises(ValueError, match="slow critic"):
            make_controller(**G, slow_critic_coef=coef, slow_critic_rate=rate)
    with pytest.raises(ValueError, match="actor distribution"):
        make_controller(actor_distribution="gaussian")


def test_slow_critic_starts_as_an_exact_copy_without_gradients_outside_the_optimizer():
    torch.manual_seed(0)
    model = make_controller(**E)
    for (k, s), (_, v) in zip(model.slow_value.state_dict().items(), model.value.state_dict().items()):
        assert torch.equal(s, v), k
    assert not any(p.requires_grad for p in model.slow_value.parameters())
    opt = PMPOOptimizers(model, warmup_steps=0)
    optimized = {id(p) for g in opt.actor.param_groups + opt.value.param_groups for p in g["params"]}
    assert not optimized & {id(p) for p in model.slow_value.parameters()}


def test_slow_critic_loss_formula_gradient_paths_and_update_after_the_critic_step():
    import torch.nn.functional as F
    torch.set_num_threads(1)
    model, _ = make_system(**E)
    opt = PMPOOptimizers(model, warmup_steps=0)
    captured = {}
    collect = model.collect_imagination

    def spy():
        out = collect()
        captured["obs"] = out[0]
        return out
    model.collect_imagination = spy
    with torch.no_grad():  # make the slow critic differ so the regularizer is non-trivial
        for p in model.slow_value.parameters():
            p.add_(0.05 * torch.randn_like(p))
    slow_before = deepcopy(model.slow_value.state_dict())
    loss, metrics = model()
    flat = captured["obs"].flatten(0, 1)
    with torch.no_grad():
        v, vs = model.value(flat).squeeze(-1), model.slow_value(flat).squeeze(-1)
    assert float(metrics["loss_slowreg"]) == pytest.approx(float(F.mse_loss(v, vs)), rel=1e-6)
    assert float(metrics["loss_value"]) == pytest.approx(float(metrics["loss_value_target"]) + float(metrics["loss_slowreg"]), rel=1e-6)
    loss.backward()
    assert all(p.grad is None for p in model.slow_value.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value.parameters())
    opt.step()  # EMA with the POST-step online critic, rate 0.02
    for k, online in model.value.state_dict().items():
        torch.testing.assert_close(model.slow_value.state_dict()[k], 0.98 * slow_before[k] + 0.02 * online, rtol=0, atol=1e-7)


def test_online_critic_gives_bootstraps_and_advantages_not_the_slow_critic():
    torch.set_num_threads(1)
    model, _ = make_system(**H10)
    with torch.no_grad():
        for p in model.slow_value.parameters():
            p.add_(1.0)  # very different slow critic
    obs, act, rew, end, trunc, _, values, bootstrap, infos = model.collect_imagination()
    with torch.no_grad():
        online = model.value(obs.flatten(0, 1)).squeeze(-1).view_as(values)
        slow = model.slow_value(obs.flatten(0, 1)).squeeze(-1).view_as(values)
    torch.testing.assert_close(values, online, rtol=0, atol=1e-6)  # batched recomputation: float rounding only
    assert not torch.allclose(values, slow)
    alive = ~(end + trunc).bool()[:, :-1]
    torch.testing.assert_close(bootstrap[:, :-1][alive], online[:, 1:][alive], rtol=0, atol=1e-6)


def test_h10_imagination_starts_from_the_repeated_anchor_frame():
    """Every imagined start (the first state, and the state after a sampled ending or the
    horizon cut) is its start frame repeated; otherwise the stack advances one frame."""
    model, env = make_system(**H10)
    obs, _, _, end, trunc = model.collect_imagination()[:5]
    dead = (end + trunc).bool()
    starts = 0
    for row in range(obs.shape[0]):
        for t in range(obs.shape[1]):
            frames = obs[row, t].reshape(4, 3, 8, 8)
            if t == 0 or dead[row, t - 1]:
                starts += 1
                for k in range(1, 4):
                    torch.testing.assert_close(frames[k], frames[0], rtol=0, atol=0)
            else:
                torch.testing.assert_close(obs[row, t, :9], obs[row, t - 1, 3:], rtol=0, atol=0)
    assert starts > obs.shape[0]  # this rollout also restarts within the horizon


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_h10_checkpoint_resume_is_bit_exact_with_slow_critic_and_prior_refresh(tmp_path, device):
    torch.set_num_threads(1)
    torch.backends.cudnn.deterministic = True
    model, env = make_system(device, **H10, actor_objective="advantage_weighted", prior_refresh_interval=2)
    opt = PMPOOptimizers(model, 5)
    loss, _ = model()
    loss.backward()
    opt.step()
    path = tmp_path / "pmpo.pt"
    torch.save(dict(model=model.state_dict(), optimizer=opt.state_dict()), path)
    restored, restored_env = make_system(device, **H10, actor_objective="advantage_weighted", prior_refresh_interval=2)
    torch.manual_seed(99)
    restored_opt = PMPOOptimizers(restored, 5)
    saved = torch.load(path, map_location=device, weights_only=False)
    assert not any(k.startswith("_gh_") for k in saved["model"])  # quadrature buffers not persisted
    assert any(k.startswith("slow_value.") for k in saved["model"])
    assert saved["model"]["_extra_state"]["controller"] == dict(
        actor_distribution="tanh_gaussian", slow_critic=dict(coef=1.0, rate=0.02),
        imagination_history="repeat_anchor", imagination_horizon=3)
    restored.load_state_dict(saved["model"])
    restored_opt.load_state_dict(saved["optimizer"])
    restored_env.sampler.denoiser.load_state_dict(env.sampler.denoiser.state_dict())
    restored_env.rew_end_model.load_state_dict(env.rew_end_model.state_dict())
    for _ in range(2):  # crosses a prior-refresh boundary
        opt.zero_grad(); restored_opt.zero_grad()
        loss, metrics = model()
        actual, actual_metrics = restored()
        torch.testing.assert_close(actual, loss, rtol=0, atol=0)
        for key in metrics:
            torch.testing.assert_close(torch.as_tensor(actual_metrics[key]), torch.as_tensor(metrics[key]), rtol=0, atol=0)
        loss.backward(); actual.backward()
        opt.step(); restored_opt.step()
    for (k, p), q in zip(model.state_dict().items(), restored.state_dict().values()):
        if torch.is_tensor(p):
            torch.testing.assert_close(p, q, rtol=0, atol=0, msg=k)


@pytest.mark.parametrize("change,match", [
    (dict(actor_distribution="beta", slow_critic_coef=0.0), "'actor_distribution': 'beta'"),  # Beta weights into the Gaussian actor
    (dict(imagination_history="replay_prompt"), "imagination_history"),
    (dict(slow_critic_rate=0.05), "slow_critic"),
    (dict(imagination_horizon=5), "imagination_horizon"),
])
def test_incompatible_controller_state_is_refused_before_anything_is_loaded(change, match):
    torch.manual_seed(1)
    source = make_controller(**{**H10, **change})
    torch.manual_seed(2)
    target = make_controller(**H10)
    before = deepcopy(target.state_dict())
    state = {k: v for k, v in source.state_dict().items() if not k.startswith("slow_value.")}
    if "slow_value.head.0.weight" in target.state_dict():
        state.update({k: v for k, v in target.state_dict().items() if k.startswith("slow_value.")})
    with pytest.raises(ValueError, match="different controller") as error:
        target.load_state_dict(state)
    assert match in str(error.value)
    for k, v in target.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, before[k]), k  # nothing was copied


def test_legacy_controller_state_is_never_reinterpreted():
    """State saved before controller settings were recorded is a Beta, replay-prompt controller:
    refused by the Gaussian H10 controller, accepted only by a matching legacy-style config."""
    legacy = make_controller()  # Beta, replay_prompt, no slow critic
    state = legacy.state_dict()
    del state["_extra_state"]["controller"]
    with pytest.raises(ValueError, match="saved before controller settings were recorded"):
        make_controller(**H10).load_state_dict(state)
    make_controller().load_state_dict(state)


def test_trainer_world_model_pmpo_diagnostic_and_resume_with_the_h10_controller(tmp_path, monkeypatch):
    """The reference's trainer-level check, on this branch's default agent config (already the
    Gaussian H10 EMA-critic controller): diagnostic rows carry Gaussian parameters only."""
    import json
    test_static_dataset_trainer_world_model_and_pmpo_resume(tmp_path, monkeypatch, True)
    saved = (tmp_path / "config/exploration.yaml").read_text()
    for line in ("actor_distribution: tanh_gaussian", "slow_critic_coef: 1.0", "imagination_history: repeat_anchor"):
        assert line in saved
    rows = [json.loads(line) for line in (tmp_path / "pmpo_diagnostic/updates.jsonl").read_text().splitlines()]
    assert rows and all("pretanh_std_mean" in r and "alpha_mean" not in r for r in rows)

# ---------------------------------------------------------------- actor_critic metric schema and debug switch

def logged_paths(model, metrics):
    from models.pmpo_beta import metric_path
    return {metric_path(k, model.cfg.actor_distribution) for k in metrics}


@pytest.mark.parametrize("debug", [False, True])
def test_walker_six_dimension_metric_schema(debug, ac_schema):
    """Controller + optimizer metrics of one update, as logged paths, for the Gaussian EMA
    controller with Walker's 6 action dimensions: the agreed schema without the trainer's
    intrinsic keys (43 - 4 by default, 148 - 4 - 12 with debug metrics)."""
    model, _ = make_system(action_dim=6, **H10, actor_objective="advantage_weighted", debug_metrics=debug)
    opt = PMPOOptimizers(model, 0)
    loss, metrics = model()
    loss.backward()
    metrics.update(opt.step())
    metrics["num_batch_train_actor_critic"] = 1
    schema = ac_schema(6, debug)
    assert len(ac_schema(6, False)) == 43 and len(ac_schema(6, True)) == 148
    intrinsic = {k for k in schema if k.startswith("intrinsic/") or k.startswith("debug/intrinsic_")}
    assert logged_paths(model, metrics) == schema - intrinsic
    assert not {"loss_actor_total", "rollout_length", "all_finite"} & set(metrics)  # removed in both modes
    assert all(torch.isfinite(torch.as_tensor(v)).all() for v in metrics.values())


def test_metric_paths_are_invertible_and_beta_is_not_labelled_gaussian():
    from models.pmpo_beta import flat_metric_name, metric_path
    model, _ = make_system(**H10, debug_metrics=True)
    _, metrics = model()
    names = list(metrics) + ["intrinsic_raw_reward_mean", "intrinsic_scaled_reward_max", "intrinsic_raw_reward_p99"]
    for name in names:
        assert flat_metric_name(metric_path(name, "tanh_gaussian")) == name
    beta, _ = make_system(debug_metrics=False)  # Beta actor, no slow critic
    _, beta_metrics = beta()
    paths = logged_paths(beta, beta_metrics)
    assert {"beta/alpha_mean", "beta/beta_max", "beta/alpha_near_floor_fraction", "beta/policy_conditional_std_1"} <= paths
    assert not any(p.startswith(("gaussian/", "EMA/", "debug/")) for p in paths)


def test_debug_off_skips_the_expensive_probes():
    model, _ = make_system(**H10)
    assert model.debug_metrics is False

    def forbidden(*args, **kwargs):
        raise AssertionError("debug-only computation with debug_metrics=false")
    model.temporal_diagnostics = forbidden  # fixed-state policy probes
    model.initial_actor.forward = forbidden  # KL to the initial actor
    model.initial_actor.parameters = forbidden  # parameter distance from the initial actor
    opt = PMPOOptimizers(model, 0)
    for _ in range(2):
        opt.zero_grad()
        loss, metrics = model()
        loss.backward()
        opt.step()
    assert model._fixed_obs is None  # the debug-only probe cache is never filled
    assert not any(k.startswith(("action_", "policy_mean", "fixed_", "advantage_", "kl_initial")) for k in metrics)


def test_debug_metrics_do_not_change_training():
    """Same seeds, debug off vs on: identical losses, gradients, parameter updates, optimizer,
    scheduler, slow critic, prior and update counter, and identical global and controller RNG.
    The only difference is the debug-only fixed-state cache in the extra state."""
    runs = {}
    for debug in (False, True):
        model, env = make_system(**H10, actor_objective="advantage_weighted", prior_refresh_interval=2,
                                 debug_metrics=debug)
        opt = PMPOOptimizers(model, 3)
        torch.manual_seed(123)
        losses, grads, shared = [], [], []
        for _ in range(3):
            opt.zero_grad()
            loss, metrics = model()
            loss.backward()
            grads.append([p.grad.clone() for p in model.parameters() if p.grad is not None])
            metrics.update(opt.step())
            losses.append(loss.detach().clone())
            shared.append({k: torch.as_tensor(v).clone() for k, v in metrics.items()})
        runs[debug] = dict(model=model, opt=opt, losses=losses, grads=grads, metrics=shared,
                           rng=torch.get_rng_state().clone())
    off, on = runs[False], runs[True]
    for a, b in zip(off["losses"], on["losses"]):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    for ga, gb in zip(off["grads"], on["grads"]):
        assert len(ga) == len(gb) and all(torch.equal(x, y) for x, y in zip(ga, gb))
    for ma, mb in zip(off["metrics"], on["metrics"]):  # every default metric has the same value
        assert set(ma) < set(mb)
        for k in ma:
            torch.testing.assert_close(ma[k], mb[k], rtol=0, atol=0, msg=k)
    sa, sb = off["model"].state_dict(), on["model"].state_dict()
    assert sa.keys() == sb.keys()
    for k, v in sa.items():
        if torch.is_tensor(v):
            torch.testing.assert_close(sb[k], v, rtol=0, atol=0, msg=k)  # incl. slow_value, prior, updates
    ea, eb = sa["_extra_state"], sb["_extra_state"]
    assert torch.equal(ea["rng"]["cpu"], eb["rng"]["cpu"]) and ea["rng"]["python"] == eb["rng"]["python"]
    assert (ea["objective"], ea["controller"]) == (eb["objective"], eb["controller"])
    assert ea["fixed_obs"] is None and eb["fixed_obs"] is not None  # the debug-only cache
    oa, ob = off["opt"].state_dict(), on["opt"].state_dict()
    for name in ("actor", "value"):
        for pid, state in oa[name]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(ob[name]["state"][pid][key], value, rtol=0, atol=0)
    assert oa["schedulers"] == ob["schedulers"]
    assert torch.equal(off["rng"], on["rng"])  # global torch RNG after training
