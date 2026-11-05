"""Step 6: the uniform-random exploration preset, the collection-budget check, and the selected
100k schedule shared by all five method presets for every accepted task group."""
from types import SimpleNamespace

import pytest
import torch
from omegaconf import OmegaConf

from envs.dm_control_env import REWARD_LABEL_TASK_GROUPS
from exploration_trainer import check_collection_budget, ExplorationTrainer
from models.pmpo_beta import PMPOBeta
from test_exploration_trainer import COLLECT, compose, run_trainer, TASKS

PRESETS = {"uniform": 1, "lcg": 1, "ensemble": 5, "plan2explore": 1, "laplace": 1}  # denoisers trained


GROUPS = [(domain, list(group)) for domain, groups in REWARD_LABEL_TASK_GROUPS.items() for group in groups]
SAVED_STEPS = [5500, *range(10000, 100001, 5000)]


@pytest.mark.parametrize("domain,tasks", GROUPS, ids=[f"{d}-{'-'.join(t)}" for d, t in GROUPS])
@pytest.mark.parametrize("method", PRESETS)
def test_100k_presets_resolve_the_selected_schedule_for_every_task_group(method, domain, tasks):
    """Each preset as a launch script composes it (domain, tasks and seed given; no schedule
    overrides), stepped through the trainer's epoch arithmetic: initial collection to
    first_epoch.min before epoch 1 (collect_initial_dataset), steps_per_epoch more in each
    collection epoch, then num_final_epochs training-only epochs; the trainer's own
    mark_due_milestone at every epoch boundary. No preset resets the domain to Walker."""
    cfg = compose([f"+experiment={method}_100k", f"env.train.domain_name={domain}",
                   f"env.train.task_names=[{','.join(tasks)}]", "common.seed=0"])
    assert (cfg.env.train.domain_name, list(cfg.env.train.task_names)) == (domain, tasks)
    assert (cfg.env.test.domain_name, list(cfg.env.test.task_names)) == (domain, tasks)
    assert (cfg.collection.train.record_physical_state, cfg.collection.test.record_physical_state) == (True, False)
    assert cfg.agent.actor_critic.debug_metrics is False
    c, m = cfg.collection.train, cfg.milestones
    assert (c.first_epoch.min, c.steps_per_epoch, c.num_steps_total) == (5000, 500, 100000)
    assert (list(m.at_steps), m.every_steps, m.keep_last) == (SAVED_STEPS, None, None)
    assert cfg.training.num_final_epochs == 0
    d, ac = cfg.denoiser.training, cfg.actor_critic.training
    assert (d.start_after_epochs, d.steps_first_epoch, d.steps_per_epoch) == (0, 10000, 400)
    assert (ac.start_after_epochs, ac.steps_first_epoch, ac.steps_per_epoch) == (0, 200, 200)
    learned = c.mode == "policy"  # ExplorationTrainer._learned_exploration: uniform builds no PMPO optimizer
    assert learned == (method != "uniform") and cfg.intrinsic_reward.enabled == learned
    members = cfg.intrinsic_reward.num_members if method == "ensemble" else 1
    assert members == PRESETS[method]

    check_collection_budget(c)
    epochs_collect = (c.num_steps_total - c.first_epoch.min) // c.steps_per_epoch  # collect_initial_dataset
    num_epochs = epochs_collect + cfg.training.num_final_epochs
    assert epochs_collect == num_epochs == 190
    stub = SimpleNamespace(_cfg=cfg, milestones=[], milestone_targets_done=[], epoch=0,
                           train_dataset=SimpleNamespace(num_steps=c.first_epoch.min))
    denoiser_updates = pmpo_updates = 0
    steps_at = {}
    for epoch in range(1, num_epochs + 1):
        stub.epoch = epoch
        if epoch <= epochs_collect:
            stub.train_dataset.num_steps += c.steps_per_epoch
        steps_at[epoch] = stub.train_dataset.num_steps
        denoiser_updates += members * (d.steps_first_epoch if epoch == 1 else d.steps_per_epoch)
        if learned:
            pmpo_updates += ac.steps_first_epoch if epoch == 1 else ac.steps_per_epoch
        ExplorationTrainer.mark_due_milestone(stub)
    assert (steps_at[1], steps_at[10], steps_at[190]) == (5500, 10000, 100000)
    assert stub.milestone_targets_done == SAVED_STEPS  # 20 saved milestones, all retained
    assert [x["name"] for x in stub.milestones] == [f"steps_{t:06d}" for t in SAVED_STEPS]
    assert [x["epoch"] for x in stub.milestones] == [1, *range(10, 191, 10)]
    assert not any(x["retired"] for x in stub.milestones)
    assert denoiser_updates == members * 85_600 and pmpo_updates == (38_000 if learned else 0)


def test_uniform_100k_preset_resolves_the_intended_budget():
    cfg = compose(["+experiment=uniform_100k"])
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
    smoke = compose(["+experiment=uniform_100k", "collection.train.num_steps_total=800",
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
    trainer, logs = run_trainer(tmp_path, monkeypatch, ["+experiment=uniform_100k", "wandb.mode=disabled",
                                                        "env.train.task_names=[stand,walk]", *COLLECT])
    assert not calls and trainer.agent.actor_critic.updates.item() == 0
    assert trainer.opt.actor_critic is None and trainer._lcg_lifecycle is None
    assert trainer.train_dataset.num_steps == 24 and trainer.train_dataset.task_names == TASKS
    assert trainer.num_batch_train.denoiser == 3
    assert not any(key.startswith("actor_critic") for row in logs for key in row)
