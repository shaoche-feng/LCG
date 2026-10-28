"""The PMPO actor objective and controller in ExplorationTrainer: resolved config, logged
diagnostics, resume refusing a changed objective, weight cap or controller (Gaussian H10 EMA
critic), and exact controller restoration."""
import math
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from exploration_trainer import ExplorationTrainer
from test_test_diagnostics import BASE, Crash, start

WEIGHTED = ["agent.actor_critic.actor_objective=advantage_weighted", "agent.actor_critic.advantage_weight_cap=5.0"]
OFF = ["evaluation.should=false"]


@pytest.mark.parametrize("objective", ["sign", "advantage_weighted"])
def test_objective_is_resolved_and_its_diagnostics_are_logged(tmp_path, monkeypatch, objective):
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + OFF + [f"agent.actor_critic.actor_objective={objective}"])
    trainer._run()
    saved = OmegaConf.load("config/trainer.yaml").agent.actor_critic
    assert (saved.actor_objective, saved.advantage_weight_cap) == (objective, 5.0)
    rows = [r for _, r in logs if "actor_critic/train/weight_mean_positive" in r]
    assert len(rows) == trainer.agent.actor_critic.updates.item() == 3
    keys = ["positive_fraction", "negative_fraction", "loss_pmpo_raw", "loss_kl", "loss_entropy", "actor_grad_norm",
            "value_grad_norm", "value_bias", "value_target_correlation", "kl_prior", "actor_rel_distance_from_init",
            "policy_entropy_dim_0"] + [f"{k}_{g}" for k in ("advantage_abs_mean", "advantage_abs_max", "weight_mean",
                                                            "weight_max", "weight_std", "weight_clipped_fraction")
                                       for g in ("positive", "negative")]
    for row in rows:
        values = {k: float(row[f"actor_critic/train/{k}"]) for k in keys}
        assert all(math.isfinite(v) for v in values.values())
        if objective == "sign":
            assert values["weight_clipped_fraction_positive"] == 0 and values["weight_std_positive"] == 0
        else:
            assert 0 <= values["weight_max_positive"] <= 5 and 0 <= values["weight_clipped_fraction_negative"] <= 1


def test_resume_rejects_a_changed_objective(tmp_path, monkeypatch):
    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + OFF + WEIGHTED)
    with pytest.raises(Crash):
        trainer._run()
    monkeypatch.undo()
    for overrides in (["agent.actor_critic.actor_objective=sign"], ["agent.actor_critic.advantage_weight_cap=3.0"]):
        with pytest.raises(ValueError, match="cannot be loaded or resumed with a different objective"):
            start(tmp_path / "run", monkeypatch, overrides, resume=True)
    resumed, _ = start(tmp_path / "run", monkeypatch, [], resume=True)
    resumed._run()
    assert resumed.agent.actor_critic.objective_settings() == dict(mode="advantage_weighted", weight_cap=5.0)

# ---------------------------------------------------------------- Gaussian H10 EMA-critic controller

def crash_at_epoch_2_training(monkeypatch):
    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)


def test_exploration_controller_is_resolved_recorded_and_imagines_from_the_repeated_anchor(tmp_path, monkeypatch):
    from models.pmpo_beta import PMPOBeta
    starts = []
    collect = PMPOBeta.collect_imagination

    def spy(self):
        rollout = collect(self)
        starts.append(rollout[0][:, 0].reshape(rollout[0].shape[0], 4, -1))
        return rollout

    monkeypatch.setattr(PMPOBeta, "collect_imagination", spy)
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + OFF + WEIGHTED)
    trainer._run()
    saved = OmegaConf.load("config/trainer.yaml").agent.actor_critic
    assert (saved.actor_distribution, saved.slow_critic_coef, saved.slow_critic_rate, saved.imagination_history,
            saved.actor_objective) == ("tanh_gaussian", 1.0, 0.02, "repeat_anchor", "advantage_weighted")
    controller = trainer.provenance()["controller"]
    assert controller == dict(actor_distribution="tanh_gaussian", slow_critic=dict(coef=1.0, rate=0.02),
                              imagination_history="repeat_anchor", imagination_horizon=2,  # TINY horizon
                              actor_objective=dict(mode="advantage_weighted", weight_cap=5.0), entropy_coef=0.01,
                              beta_kl=0.3, prior_refresh_interval=500)
    assert hasattr(trainer.agent.actor_critic, "slow_value")
    assert len(starts) == 3
    for first in starts:  # the policy's first imagined stack repeats its start frame
        for k in range(1, 4):
            torch.testing.assert_close(first[:, k], first[:, 0], rtol=0, atol=0)
    rows = [r for _, r in logs if "actor_critic/train/loss_slowreg" in r]
    assert len(rows) == 3 and all("actor_critic/train/pretanh_std_mean" in r for r in rows)
    assert not any(k.startswith(("actor_critic/train/alpha_", "actor_critic/train/beta_")) for _, r in logs for k in r)
    for key in ("lcg_raw_reward_mean", "lcg_policy_reward_mean", "loss_actor", "loss_value", "kl_prior", "positive_fraction"):
        assert all(math.isfinite(float(r[f"actor_critic/train/{key}"])) for r in rows), key


def test_exploration_resume_restores_the_controller_and_slow_critic_exactly(tmp_path, monkeypatch):
    """CPU: crash before epoch 2's training, resume; the controller (actor, critic, slow critic,
    prior, optimizers, schedulers, update counter, private RNG) ends equal to an uninterrupted run."""
    uninterrupted, _ = start(tmp_path / "a", monkeypatch, BASE + OFF + WEIGHTED)
    uninterrupted._run()
    expected = deepcopy(uninterrupted.agent.actor_critic.state_dict())
    expected_opt = deepcopy(uninterrupted.opt.get("actor_critic").state_dict())
    monkeypatch.undo()
    crash_at_epoch_2_training(monkeypatch)
    trainer, _ = start(tmp_path / "b", monkeypatch, BASE + OFF + WEIGHTED)
    with pytest.raises(Crash):
        trainer._run()
    monkeypatch.undo()
    resumed, _ = start(tmp_path / "b", monkeypatch, [], resume=True)
    resumed._run()
    actual = resumed.agent.actor_critic.state_dict()
    assert any(k.startswith("slow_value.") for k in actual) and actual.keys() == expected.keys()
    for k, v in expected.items():
        if torch.is_tensor(v):
            torch.testing.assert_close(actual[k], v, rtol=0, atol=0, msg=k)
    rng_a, rng_b = actual["_extra_state"]["rng"], expected["_extra_state"]["rng"]
    assert torch.equal(rng_a["cpu"], rng_b["cpu"]) and rng_a["python"] == rng_b["python"]
    assert actual["_extra_state"]["controller"] == expected["_extra_state"]["controller"]
    opt = resumed.opt.get("actor_critic").state_dict()
    for name in ("actor", "value"):
        for pid, state in expected_opt[name]["state"].items():
            for key, value in state.items():
                torch.testing.assert_close(opt[name]["state"][pid][key], value, rtol=0, atol=0)
    assert opt["schedulers"] == expected_opt["schedulers"]
    assert int(actual["updates"]) == int(expected["updates"]) == 3


@pytest.mark.parametrize("saved,current", [
    (["agent.actor_critic.actor_distribution=beta", "agent.actor_critic.slow_critic_coef=0.0"], []),
    (["agent.actor_critic.imagination_history=replay_prompt"], ["agent.actor_critic.imagination_history=repeat_anchor"]),
    ([], ["agent.actor_critic.slow_critic_coef=0.5"]),
])
def test_exploration_resume_refuses_another_controller_before_loading_weights(tmp_path, monkeypatch, saved, current):
    import exploration_trainer as trainer_module
    crash_at_epoch_2_training(monkeypatch)
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + OFF + WEIGHTED + saved)
    with pytest.raises(Crash):
        trainer._run()
    monkeypatch.undo()
    loaded, original = [], trainer_module.StateDictMixin.load_state_dict

    def spy(self, state):
        if isinstance(self, ExplorationTrainer):
            loaded.append(True)  # the trainer's weights (agent, optimizers) are never reached
        return original(self, state)

    monkeypatch.setattr(trainer_module.StateDictMixin, "load_state_dict", spy)
    overrides = current or ["agent.actor_critic.actor_distribution=tanh_gaussian", "agent.actor_critic.slow_critic_coef=1.0"]
    with pytest.raises(ValueError, match="Checkpoint's exploration controller .* different controller"):
        start(tmp_path / "run", monkeypatch, overrides, resume=True)
    assert not loaded
