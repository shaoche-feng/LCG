"""Step 6: the uniform-random exploration preset and the collection-budget check."""
import pytest
import torch
from omegaconf import OmegaConf

from exploration_trainer import check_collection_budget
from models.pmpo_beta import PMPOBeta
from test_exploration_trainer import COLLECT, compose, run_trainer, TASKS


def test_walker_uniform_100k_preset_resolves_the_intended_budget():
    cfg = compose(["+experiment=walker_uniform_100k"])
    OmegaConf.resolve(cfg)
    c = cfg.collection.train
    assert (cfg.env.train.domain_name, list(cfg.env.train.task_names)) == ("walker", ["stand", "walk", "run"])
    assert (c.first_epoch.mode, c.mode, cfg.intrinsic_reward.enabled) == ("uniform", "uniform", False)
    assert (c.num_envs, c.first_epoch.min, c.steps_per_epoch, c.num_steps_total) == (1, 5000, 500, 100000)
    check_collection_budget(c)
    epochs_collect = (c.num_steps_total - c.first_epoch.min) // c.steps_per_epoch
    assert epochs_collect == 190 and c.first_epoch.min + epochs_collect * c.steps_per_epoch == 100000
    assert cfg.env.train.action_repeat == 2  # 100k environment steps = 200k simulator control steps
    # Denoiser schedule and milestones stay at the repository defaults (shared across methods).
    default = compose([])
    assert cfg.denoiser == default.denoiser and cfg.milestones == default.milestones
    assert cfg.training.num_final_epochs == default.training.num_final_epochs
    # Command-line overrides (e.g. a smoke schedule) replace preset values.
    smoke = compose(["+experiment=walker_uniform_100k", "collection.train.num_steps_total=800",
                     "collection.train.first_epoch.min=500", "collection.train.steps_per_epoch=100"])
    assert (smoke.collection.train.num_steps_total, smoke.collection.train.mode) == (800, "uniform")


@pytest.mark.parametrize("budget,match", [
    (dict(min=5000, per=500, total=100250, envs=1), "num_steps_total=100250 minus first_epoch.min"),
    (dict(min=5100, per=500, total=100000, envs=1), "first_epoch.min=5100 must be a multiple"),
    (dict(min=5000, per=500, total=100000, envs=3), "must be a multiple of steps_per_epoch \\(500\\) and num_envs \\(3\\)"),
    (dict(min=0, per=500, total=100000, envs=1), "first_epoch.min must be in"),
    (dict(min=200000, per=500, total=100000, envs=1), "first_epoch.min must be in"),
])
def test_collection_budget_must_end_exactly(budget, match):
    c = OmegaConf.create(dict(num_envs=budget["envs"], first_epoch=dict(min=budget["min"]),
                              steps_per_epoch=budget["per"], num_steps_total=budget["total"]))
    with pytest.raises(ValueError, match=match):
        check_collection_budget(c)


def test_uniform_preset_runs_without_policy_use_or_updates(tmp_path, monkeypatch):
    calls = []
    original = PMPOBeta.predict_act_value
    monkeypatch.setattr(PMPOBeta, "predict_act_value", lambda self, obs: calls.append(1) or original(self, obs))
    # Preset first; the tiny test schedule and fake env's task labels override it.
    trainer, logs = run_trainer(tmp_path, monkeypatch, ["+experiment=walker_uniform_100k", "wandb.mode=disabled",
                                                        "env.train.task_names=[stand,walk]", *COLLECT])
    assert not calls and trainer.agent.actor_critic.updates.item() == 0
    assert trainer.opt.actor_critic is None and trainer._lcg_lifecycle is None
    assert trainer.train_dataset.num_steps == 24 and trainer.train_dataset.task_names == TASKS
    assert trainer.num_batch_train.denoiser == 3
    assert not any(key.startswith("actor_critic") for row in logs for key in row)
