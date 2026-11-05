"""LCG preset: selected reference scaling and policy schedule, shared environment/model
settings, fixed and reference reward integration, and resume compatibility."""
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from exploration_trainer import check_collection_budget
from test_exploration_trainer import COLLECT, compose, run_trainer
from test_test_diagnostics import BASE, Crash, start

SHARED = ["env", "collection", "denoiser", "milestones", "evaluation", "training", "actor_critic", "world_model_env",
          "agent"]


def test_lcg_100k_preset_resolves_the_agreed_settings():
    cfg = compose(["+experiment=lcg_100k"])
    OmegaConf.resolve(cfg)
    c = cfg.collection.train
    assert (cfg.env.train.domain_name, list(cfg.env.train.task_names)) == ("walker", ["stand", "walk", "run"])
    assert (c.first_epoch.mode, c.mode) == ("uniform", "policy")
    check_collection_budget(c)
    lcg = instantiate(cfg.intrinsic_reward)
    assert lcg.enabled and list(lcg.theta_s.include) == ["unet.u_blocks.3.resblocks.0.*"]
    assert (lcg.theta_s.expected_num_tensors, lcg.theta_s.expected_dim) == (10, 217_664)
    assert (lcg.candidate_num_mc, lcg.candidate_chunk_size) == (5, 32)
    assert (lcg.precision_reference_size, lcg.precision_num_mc, lcg.damping, lcg.beta) == (320, 3, 1e-4, 1.0)
    assert lcg.reward_scaling_settings() == dict(mode="reference", reference_num_prompts=32)
    assert (cfg.actor_critic.training.steps_first_epoch, cfg.actor_critic.training.steps_per_epoch) == (200, 200)
    # The environment, replay budget, denoiser and PMPO schedules, milestones and models match
    # uniform (which never uses its PMPO schedule). LCG's exploration objective is the explicit difference.
    uniform = compose(["+experiment=uniform_100k"])
    OmegaConf.resolve(uniform)
    for key in SHARED:
        a, b = OmegaConf.to_container(cfg[key]), OmegaConf.to_container(uniform[key])
        if key == "collection":
            a["train"].pop("mode"), b["train"].pop("mode")
        if key == "agent":  # uniform trains no policy; LCG's exploration objective is advantage-weighted
            assert a["actor_critic"].pop("actor_objective") == "advantage_weighted"
            b["actor_critic"].pop("actor_objective")
        assert a == b, key
    assert (uniform.collection.train.mode, cfg.common.seed) == ("uniform", uniform.common.seed)
    # The Gaussian H10 EMA-critic controller of the reference H10 runs.
    ac = cfg.agent.actor_critic
    assert (ac.actor_distribution, ac.imagination_horizon, ac.slow_critic_coef, ac.slow_critic_rate) == (
        "tanh_gaussian", 10, 1.0, 0.02)
    assert (ac.imagination_history, ac.advantage_weight_cap, ac.entropy_coef, ac.prior_refresh_interval) == (
        "repeat_anchor", 5.0, 0.01, 500)


def test_lcg_preset_rejects_a_mismatched_parameter_block_before_any_work(tmp_path, monkeypatch):
    # The tiny test denoiser has no u_blocks.3: the preset's R1 pattern must fail when the
    # model is built, before any collection, checkpoint or training.
    with pytest.raises(ValueError, match="matched no parameters"):
        run_trainer(tmp_path, monkeypatch, ["+experiment=lcg_100k", "wandb.mode=disabled",
                                            "env.train.task_names=[stand,walk]",
                                            "intrinsic_reward.theta_s.include=[unet.u_blocks.3.resblocks.0.*]",
                                            *COLLECT], build_only=True)
    assert not (tmp_path / "run/checkpoints/state.pt").exists() and not (tmp_path / "run/dataset").exists()
    with pytest.raises(ValueError, match="expected 1 tensors"):
        run_trainer(tmp_path / "b", monkeypatch, [*COLLECT, "intrinsic_reward.theta_s.expected_num_tensors=1"],
                    build_only=True)
    assert not (tmp_path / "b/run/checkpoints/state.pt").exists()


FIXED = ["intrinsic_reward.reward_scaling=fixed", "intrinsic_reward.fixed_divisor=100.0",
         "intrinsic_reward.rms_enabled=false"]


def test_trainer_policy_reward_is_raw_lcg_over_the_fixed_divisor(tmp_path, monkeypatch):
    rewards = []
    import intrinsic_reward.lcg.lifecycle as lifecycle_module
    original = lifecycle_module.LCGLifecycle.scale

    def spy(self, raw):
        reward = original(self, raw)
        rewards.append((raw.clone(), reward.clone()))
        return reward

    monkeypatch.setattr(lifecycle_module.LCGLifecycle, "scale", spy)
    trainer, logs = run_trainer(tmp_path, monkeypatch, [*COLLECT, *FIXED])
    assert len(rewards) == trainer.agent.actor_critic.updates.item() == 3 and trainer._lcg_lifecycle.rms is None
    for raw, reward in rewards:
        assert torch.equal(reward, raw / 100.0)
    rows = [row for row in logs if "actor_critic/train/intrinsic/raw_reward_mean" in row]
    assert len(rows) == 3
    for row, (raw, reward) in zip(rows, rewards):
        assert row["actor_critic/train/intrinsic/raw_reward_max"] == raw.max()
        assert row["actor_critic/train/intrinsic/scaled_reward_mean"] == reward.float().mean()
        assert row["actor_critic/train/intrinsic/scaled_reward_max"] == reward.float().max()
        assert not any("/debug/" in k for k in row)  # distribution detail only with debug metrics
    # Canonical timing only: the per-update PMPO time; LCG's own timers run but are not logged.
    assert any("timing/pmpo_seconds_per_update" in row for row in logs) and trainer._lcg_lifecycle.scoring_seconds > 0
    assert not any(k.startswith(("timing/lcg_", "lcg/")) or k == "timing/pmpo_update_seconds" for row in logs for k in row)


def test_resume_refuses_switched_reward_scaling(tmp_path, monkeypatch):
    from exploration_trainer import ExplorationTrainer
    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + FIXED + ["evaluation.should=false"])
    with pytest.raises(Crash):
        trainer._run()
    monkeypatch.undo()
    for overrides in (["intrinsic_reward.fixed_divisor=50.0"],
                      ["intrinsic_reward.reward_scaling=rms", "intrinsic_reward.fixed_divisor=null",
                       "intrinsic_reward.rms_enabled=true"]):
        with pytest.raises(ValueError, match="a resume must not switch it"):
            start(tmp_path / "run", monkeypatch, overrides, resume=True)
    resumed, _ = start(tmp_path / "run", monkeypatch, [], resume=True)  # unchanged settings resume
    resumed._run()
    assert resumed.epoch == 2 and resumed.agent.actor_critic.updates.item() == 3


REFERENCE = ["intrinsic_reward.reward_scaling=reference", "intrinsic_reward.fixed_divisor=null",
             "intrinsic_reward.reference_num_prompts=3", "intrinsic_reward.rms_enabled=false"]


def test_reference_is_calibrated_once_before_policy_updates_and_frozen_for_evaluation(tmp_path, monkeypatch):
    from copy import deepcopy
    from intrinsic_reward.lcg import LCGLifecycle
    from test_test_diagnostics import same
    from utils import rng_state

    calibrations, updates = {}, []
    calibrate, scale = LCGLifecycle.calibrate, LCGLifecycle.scale

    def calibrated(self, scores):
        assert self.needs_reference and self.round_id not in calibrations
        assert scores.numel() == 3 * 2  # prompts times the configured world-model horizon
        calibrate(self, scores)
        calibrations[self.round_id] = float(scores.double().mean())
        assert self.reward_divisor == calibrations[self.round_id]

    def scaled(self, raw):
        assert self.reward_divisor == calibrations[self.round_id]
        result = scale(self, raw)
        torch.testing.assert_close(result, raw / calibrations[self.round_id], rtol=0, atol=0)
        updates.append(self.round_id)
        return result

    monkeypatch.setattr(LCGLifecycle, "calibrate", calibrated)
    monkeypatch.setattr(LCGLifecycle, "scale", scaled)
    trainer, logs = run_trainer(tmp_path, monkeypatch, [*COLLECT, *REFERENCE])
    assert sorted(calibrations) == [1, 2] and updates == [1, 1, 2]
    lifecycle = trainer._lcg_lifecycle
    assert lifecycle.rms is None
    state, random = deepcopy(lifecycle.state_dict()), rng_state()
    monitoring = lifecycle.last_raw, lifecycle.last_scaled, lifecycle.scoring_seconds
    episode = trainer.train_dataset.load_episode(0)
    raw, reward = trainer._intrinsic.score_episode(episode)
    torch.testing.assert_close(reward, raw / lifecycle.reward_divisor, rtol=0, atol=0)
    assert same(state, lifecycle.state_dict()) and same(random, rng_state())
    assert same(monitoring, (lifecycle.last_raw, lifecycle.last_scaled, lifecycle.scoring_seconds))
    assert sorted(calibrations) == [1, 2]  # evaluation never calibrates or refreshes
    assert {r["lcg/reward_divisor"] for r in logs if "lcg/reward_divisor" in r} == set(calibrations.values())


def test_reference_resume_restores_frozen_divisor_and_refuses_changed_settings(tmp_path, monkeypatch):
    from exploration_trainer import ExplorationTrainer
    from intrinsic_reward.lcg import LCGLifecycle
    from test_test_diagnostics import same
    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + REFERENCE + ["evaluation.should=false"])
    with pytest.raises(Crash):
        trainer._run()
    expected = trainer._lcg_lifecycle.state_dict()
    monkeypatch.undo()
    for overrides in (["intrinsic_reward.reference_num_prompts=4"],
                      ["intrinsic_reward.reward_scaling=fixed", "intrinsic_reward.fixed_divisor=100",
                       "intrinsic_reward.reference_num_prompts=null"]):
        with pytest.raises(ValueError, match="a resume must not switch it"):
            start(tmp_path / "run", monkeypatch, overrides, resume=True)
    with pytest.raises(ValueError, match="controller"):
        start(tmp_path / "run", monkeypatch,
              ["world_model_env.horizon=3", "agent.actor_critic.imagination_horizon=3"], resume=True)
    monkeypatch.setattr(LCGLifecycle, "refresh", lambda *a, **kw: pytest.fail("resume refreshed the scorer"))
    monkeypatch.setattr(LCGLifecycle, "calibrate", lambda *a, **kw: pytest.fail("resume recalibrated the divisor"))
    monkeypatch.setattr(ExplorationTrainer, "lcg_reference_scores", lambda *a: pytest.fail("resume imagined new references"))
    restored, _ = start(tmp_path / "run", monkeypatch, [], resume=True)
    assert same(expected, restored._lcg_lifecycle.state_dict())
    raw = torch.tensor([1., 2.])
    torch.testing.assert_close(restored._lcg_lifecycle.scale_frozen(raw), raw / expected["reward_divisor"], rtol=0, atol=0)
    assert not restored._lcg_lifecycle.needs_reference
