"""The PMPO actor objective in ExplorationTrainer: resolved config, logged diagnostics, and
resume refusing a changed objective or weight cap."""
import math

import pytest
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
