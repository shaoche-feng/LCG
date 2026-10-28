"""Step 6 LCG preset: resolved settings, shared schedules with the uniform preset, fixed
reward scaling in the trainer, and resume refusing a switched scaling."""
import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from exploration_trainer import check_collection_budget
from test_exploration_trainer import COLLECT, compose, run_trainer
from test_test_diagnostics import BASE, Crash, start

SHARED = ["env", "collection", "denoiser", "milestones", "evaluation", "training", "actor_critic", "world_model_env",
          "agent"]


def test_walker_lcg_100k_preset_resolves_the_agreed_settings():
    cfg = compose(["+experiment=walker_lcg_100k"])
    OmegaConf.resolve(cfg)
    c = cfg.collection.train
    assert (cfg.env.train.domain_name, list(cfg.env.train.task_names)) == ("walker", ["stand", "walk", "run"])
    assert (c.first_epoch.mode, c.mode) == ("uniform", "policy")
    check_collection_budget(c)
    lcg = instantiate(cfg.intrinsic_reward)
    assert lcg.enabled and list(lcg.theta_s.include) == ["unet.u_blocks.3.resblocks.0.*"]
    assert (lcg.theta_s.expected_num_tensors, lcg.theta_s.expected_dim) == (10, 217_664)
    assert (lcg.candidate_num_mc, lcg.candidate_chunk_size) == (5, 16)
    assert (lcg.precision_reference_size, lcg.precision_num_mc, lcg.damping, lcg.beta) == (320, 3, 1e-4, 1.0)
    assert lcg.reward_scaling_settings() == dict(mode="fixed", fixed_divisor=100.0)
    # Everything but the method matches the uniform preset: budget, schedules, milestones,
    # diagnostics, models.
    uniform = compose(["+experiment=walker_uniform_100k"])
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
        run_trainer(tmp_path, monkeypatch, ["+experiment=walker_lcg_100k", "wandb.mode=disabled",
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
    import lcg.lifecycle as lifecycle_module
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
    rows = [row for row in logs if "actor_critic/train/lcg_raw_reward_mean" in row]
    assert len(rows) == 3
    for row, (raw, reward) in zip(rows, rewards):
        assert row["actor_critic/train/lcg_raw_reward_max"] == raw.max()
        assert row["actor_critic/train/lcg_policy_reward_mean"] == reward.float().mean()
        assert {f"actor_critic/train/lcg_policy_reward_{k}" for k in ("std", "p10", "median", "p90", "p99")} <= set(row)
    assert any("timing/lcg_scoring_seconds" in row and "timing/pmpo_update_seconds" in row for row in logs)


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
