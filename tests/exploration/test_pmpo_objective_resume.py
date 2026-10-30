"""The PMPO actor objective and controller in ExplorationTrainer: resolved config, logged
diagnostics, resume refusing a changed objective, weight cap or controller (Gaussian H10 EMA
critic), and exact controller restoration."""
import math
from copy import deepcopy

import pytest
import torch
from omegaconf import OmegaConf

from exploration_trainer import ExplorationTrainer
from models.pmpo_beta import metric_path
from test_test_diagnostics import BASE, Crash, start

WEIGHTED = ["agent.actor_critic.actor_objective=advantage_weighted", "agent.actor_critic.advantage_weight_cap=5.0"]
OFF = ["evaluation.should=false"]


@pytest.mark.parametrize("objective", ["sign", "advantage_weighted"])
def test_objective_is_resolved_and_its_diagnostics_are_logged(tmp_path, monkeypatch, objective):
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + OFF + [f"agent.actor_critic.actor_objective={objective}",
                                                                       "agent.actor_critic.debug_metrics=true"])
    trainer._run()
    saved = OmegaConf.load("config/exploration.yaml").agent.actor_critic
    assert (saved.actor_objective, saved.advantage_weight_cap) == (objective, 5.0)
    rows = [r for _, r in logs if "actor_critic/train/debug/weight_mean_positive" in r]
    assert len(rows) == trainer.agent.actor_critic.updates.item() == 3
    keys = ["positive_fraction", "negative_fraction", "loss_pmpo_raw", "loss_kl", "loss_entropy", "actor_grad_norm",
            "value_grad_norm", "value_bias", "value_target_correlation", "kl_prior", "actor_rel_distance_from_init",
            "policy_entropy_dim_0"] + [f"{k}_{g}" for k in ("advantage_abs_mean", "advantage_abs_max", "weight_mean",
                                                            "weight_max", "weight_std", "weight_clipped_fraction")
                                       for g in ("positive", "negative")]
    for row in rows:
        values = {k: float(row["actor_critic/train/" + metric_path(k, "tanh_gaussian")]) for k in keys}
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
    saved = OmegaConf.load("config/exploration.yaml").agent.actor_critic
    assert (saved.actor_distribution, saved.slow_critic_coef, saved.slow_critic_rate, saved.imagination_history,
            saved.actor_objective) == ("tanh_gaussian", 1.0, 0.02, "repeat_anchor", "advantage_weighted")
    controller = trainer.provenance()["controller"]
    assert controller == dict(actor_distribution="tanh_gaussian", slow_critic=dict(coef=1.0, rate=0.02),
                              imagination_history="repeat_anchor", imagination_horizon=2,  # TINY horizon
                              actor_objective=dict(mode="advantage_weighted", weight_cap=5.0), entropy_coef=0.01,
                              beta_kl=0.3, prior_refresh_interval=500, debug_metrics=False)
    assert hasattr(trainer.agent.actor_critic, "slow_value")
    assert len(starts) == 3
    for first in starts:  # the policy's first imagined stack repeats its start frame
        for k in range(1, 4):
            torch.testing.assert_close(first[:, k], first[:, 0], rtol=0, atol=0)
    rows = [r for _, r in logs if "actor_critic/train/EMA/loss_slowreg" in r]
    assert len(rows) == 3 and all("actor_critic/train/gaussian/pretanh_std_mean" in r for r in rows)
    assert not any(k.startswith(("actor_critic/train/beta/", "actor_critic/train/debug/")) for _, r in logs for k in r)
    for key in ("intrinsic/raw_reward_mean", "intrinsic/scaled_reward_mean", "core/loss_actor", "core/loss_value",
                "core/kl_prior", "core/positive_fraction"):
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

# ---------------------------------------------------------------- actor_critic metric schema

def actor_critic_keys(logs):
    rows = [r if isinstance(r, dict) else r[1] for r in logs]
    return {k for r in rows for k in r if k.startswith("actor_critic/")}, [r for r in rows if any(k.startswith("actor_critic/") for k in r)]


@pytest.mark.parametrize("method", ["lcg", "ensemble", "laplace", "plan2explore"])
def test_default_schema_is_shared_by_every_learned_method(tmp_path, monkeypatch, method, ac_schema):
    """Debug off: exactly the agreed keys (2 action dims in the test env), the same intrinsic
    names for every method, no legacy flat or method-prefixed alias, every row complete."""
    if method == "lcg":
        trainer, logs = start(tmp_path / "run", monkeypatch, BASE + OFF + WEIGHTED)
        trainer._run()
    else:
        import test_ensemble, test_laplace, test_plan2explore
        runner = dict(ensemble=test_ensemble.run_ens, laplace=test_laplace.run_lap, plan2explore=test_plan2explore.run_p2e)
        trainer, logs = runner[method](tmp_path, monkeypatch, ["evaluation.should=false"])
    keys, rows = actor_critic_keys(logs)
    expected = {f"actor_critic/train/{k}" for k in ac_schema(2, debug=False)}
    assert keys == expected and len(keys) == 39  # 43 with Walker's 6 action dimensions
    assert len(rows) == trainer.agent.actor_critic.updates.item() == 3 and all(set(r) >= expected for r in rows)
    legacy = ("lcg_", "ens_", "laplace_", "p2e_", "loss_actor_total", "rollout_length", "all_finite")
    assert not [k for k in keys if k.rsplit("/", 1)[1].startswith(legacy) or k.count("/") != 3]


def test_debug_schema_and_unchanged_training_through_the_trainer(tmp_path, monkeypatch, ac_schema):
    from test_test_diagnostics import run, same
    runs = {}
    for debug in ("false", "true"):
        trainer, logs = start(tmp_path / debug, monkeypatch, BASE + OFF + WEIGHTED + [f"agent.actor_critic.debug_metrics={debug}"])
        runs[debug] = (run(trainer), logs)
        monkeypatch.undo()
    (off, logs_off), (on, logs_on) = runs["false"], runs["true"]
    keys_on, rows_on = actor_critic_keys(logs_on)
    assert keys_on == {f"actor_critic/train/{k}" for k in ac_schema(2, debug=True)} and len(keys_on) == 112
    _, rows_off = actor_critic_keys(logs_off)
    for a, b in zip(rows_off, rows_on):  # default values unchanged by debug metrics
        assert {k: float(v) for k, v in a.items()} == {k: float(v) for k, v in b.items() if k in a}

    def training_state(trainer):  # the fixed-state probe cache is debug-only, not training state
        state = trainer.agent.state_dict()
        state["actor_critic._extra_state"] = {k: v for k, v in state["actor_critic._extra_state"].items() if k != "fixed_obs"}
        return state
    assert same(training_state(off), training_state(on))
    assert same(off.opt.state_dict(), on.opt.state_dict()) and same(off.lr_sched.state_dict(), on.lr_sched.state_dict())
    for k in ("python", "numpy", "torch"):
        assert repr(off._final_rng[k]) == repr(on._final_rng[k]), k
    assert off.agent.actor_critic._fixed_obs is None and on.agent.actor_critic._fixed_obs is not None
    # Epoch rows (progress counters) are identical in both modes and carry no PMPO aggregates.
    epochs = lambda logs: [{k: v for k, v in r.items() if not k.startswith("timing/")}
                           for _, r in logs if "timing/epoch_seconds" in r]
    assert epochs(logs_off) == epochs(logs_on) and len(epochs(logs_off)) == 2
    assert all(set(row) == {"progress/train_env_steps", "progress/test_env_steps", "progress/denoiser_updates_total",
                            "progress/pmpo_updates"} for row in epochs(logs_off))
