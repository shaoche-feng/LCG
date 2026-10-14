import json
from types import SimpleNamespace

import pytest
import torch

from pmpo_diagnostic import PMPORunDiagnostic, DiagnosticStop
from test_pmpo_beta import make_controller


def diagnostics(tmp_path):
    return PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22]),
                             make_controller(), {})


def healthy():
    return dict(alpha_max=2., beta_max=2., alpha_mean=2., beta_mean=2., kl_prior=0.,
                value_abs_max=1., return_abs_max=1., near_lower_fraction=0., near_upper_fraction=0.,
                fixed_policy_mean_state_std_0=0.1, policy_entropy=1.)


@pytest.mark.parametrize("key,value", [("alpha_max", 10001), ("kl_prior", 11),
                                      ("value_abs_max", 1000001), ("policy_entropy", -100)])
def test_health_stops_are_observation_only(tmp_path, key, value):
    diag = diagnostics(tmp_path)
    before = [p.clone() for p in diag.model.parameters()]
    metrics = healthy()
    metrics[key] = value
    with pytest.raises(DiagnosticStop):
        diag.check_update(metrics, 1, 0)
    for p, q in zip(before, diag.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)


def test_diagnostic_checkpoint_saved_at_milestones_without_touching_model(tmp_path):
    from models.pmpo_beta import PMPOOptimizers

    model = make_controller()
    opt = PMPOOptimizers(model, warmup_steps=0)
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22],
                                             checkpoint_updates=(2, 4)), model, {}, optimizers=opt)
    before = [p.clone() for p in diag.model.parameters()]
    metrics = healthy()
    diag.check_update(metrics, 1, 1)
    assert not (tmp_path / "checkpoint_update_00001.pt").exists()
    diag.check_update(metrics, 1, 2)
    assert (tmp_path / "checkpoint_update_00002.pt").exists()
    diag.check_update(metrics, 1, 3)
    assert not (tmp_path / "checkpoint_update_00003.pt").exists()
    diag.check_update(metrics, 1, 4)
    assert (tmp_path / "checkpoint_update_00004.pt").exists()
    for p, q in zip(before, diag.model.parameters()):
        torch.testing.assert_close(p, q, rtol=0, atol=0)
    saved = torch.load(tmp_path / "checkpoint_update_00002.pt", weights_only=False)
    assert saved["update"] == 2 and set(saved) == {"model_state", "optimizers", "rng", "config", "update"}
    for key, p in saved["model_state"].items():
        if torch.is_tensor(p):  # _extra_state (the model's own RNG dict) isn't a tensor
            torch.testing.assert_close(p, diag.model.state_dict()[key], rtol=0, atol=0)
    torch.testing.assert_close(saved["optimizers"]["actor"]["param_groups"][0]["lr"],
                               opt.state_dict()["actor"]["param_groups"][0]["lr"])
    assert saved["config"].fixed_prior == model.cfg.fixed_prior


def test_diagnostic_checkpoint_saved_before_stop(tmp_path):
    diag = diagnostics(tmp_path)
    metrics = healthy()
    metrics["kl_prior"] = 11
    with pytest.raises(DiagnosticStop):
        diag.check_update(metrics, 1, 7)
    assert (tmp_path / "checkpoint_update_00007_stop.pt").exists()


def test_full_training_checkpoint_saved_every_prior_refresh_interval(tmp_path):
    model = make_controller(prior_refresh_interval=3)
    calls = []
    def trainer_state_fn():
        calls.append(1)
        return {"fake": "trainer_state", "n_calls": len(calls)}
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22]),
                             model, {}, trainer_state_fn=trainer_state_fn)
    metrics = healthy()
    for u in range(1, 7):
        diag.check_update(metrics, epoch=2, update=u)
    # prior_refresh_interval=3 -> full checkpoints at updates 3 and 6 only.
    assert (tmp_path / "agent_pmpo_update_000003.pt").exists()
    assert (tmp_path / "agent_pmpo_update_000006.pt").exists()
    for u in (1, 2, 4, 5):
        assert not (tmp_path / f"agent_pmpo_update_{u:06d}.pt").exists()
    saved = torch.load(tmp_path / "agent_pmpo_update_000003.pt", weights_only=False)
    assert saved["epoch"] == 2 and saved["update"] == 3
    assert saved["trainer_state"] == {"fake": "trainer_state", "n_calls": 1}
    checkpoint_events = [json.loads(l) for l in (tmp_path / "checkpoints.jsonl").read_text().splitlines()]
    assert [e["update"] for e in checkpoint_events] == [3, 6]


def test_full_training_checkpoint_saved_on_stop_and_none_without_callback(tmp_path):
    model = make_controller(prior_refresh_interval=1000)  # never triggers on its own
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22]),
                             model, {}, trainer_state_fn=lambda: {"ok": True})
    metrics = healthy()
    metrics["kl_prior"] = 11
    with pytest.raises(DiagnosticStop):
        diag.check_update(metrics, epoch=1, update=5)
    assert (tmp_path / "agent_pmpo_update_000005_stop.pt").exists()

    # Without a trainer_state_fn, save_full_training_checkpoint is a safe no-op --
    # check_update must not raise or write anything for the full-checkpoint path.
    diag2 = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path / "no_cb"), evaluation_seeds=[11, 22]),
                              make_controller(prior_refresh_interval=1), {})
    diag2.check_update(healthy(), epoch=1, update=1)
    assert not any((tmp_path / "no_cb").glob("agent_pmpo_update_*"))


def test_prior_block_logging_fields_and_real_steps(tmp_path):
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22]),
                             make_controller(prior_refresh_interval=500), {})
    for update, expect_event, expect_position in [(0, True, 0), (1, False, 1), (499, False, 499), (500, True, 0)]:
        diag.check_update(healthy(), epoch=1, update=update, real_steps=12345)
    rows = [json.loads(l) for l in (tmp_path / "updates.jsonl").read_text().splitlines()]
    for row, (update, expect_event, expect_position) in zip(rows, [(0, True, 0), (1, False, 1), (499, False, 499), (500, True, 0)]):
        assert row["update"] == update
        assert row["real_steps"] == 12345
        assert row["prior_refresh_interval"] == 500
        assert row["prior_refresh_event"] is expect_event
        assert row["prior_block_position"] == expect_position


def test_check_update_works_and_logs_correctly_without_wandb(tmp_path):
    import wandb
    assert wandb.run is None  # no active run in the test environment
    diag = diagnostics(tmp_path)
    # Must not raise even though wandb is inactive -- the wandb.run is not None guard
    # makes the mirroring call a no-op, updates.jsonl remains the sole authoritative log.
    diag.check_update(healthy(), epoch=1, update=1, real_steps=100)
    rows = [json.loads(l) for l in (tmp_path / "updates.jsonl").read_text().splitlines()]
    assert len(rows) == 1 and rows[0]["update"] == 1


def test_finish_epoch_reports_world_model_losses_and_saves_full_checkpoint(tmp_path, monkeypatch):
    import numpy as np
    import pmpo_diagnostic

    class TinyEnv:
        def __init__(self, **kwargs):
            pass
        def reset(self, seed):
            self.t = 0
            return np.zeros((8, 8, 3), dtype=np.uint8), {}
        def step(self, action):
            self.t += 1
            return np.full((8, 8, 3), self.t, dtype=np.uint8), float(action.sum()), False, self.t == 4, {}
        def close(self):
            pass
    monkeypatch.setattr(pmpo_diagnostic, "DMControlEnv", TinyEnv)

    model = make_controller()
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22]),
                             model, {}, trainer_state_fn=lambda: {"ok": True})
    metrics = healthy()
    diag.check_update(metrics, epoch=1, update=1)
    diag.check_update(metrics, epoch=1, update=2)
    pmpo_log = dict(metrics, loss_actor=0.5, positive_fraction=0.5, negative_fraction=0.5,
                    alpha_min=1., beta_min=1.)
    logs = [{"actor_critic/train/" + k: v for k, v in pmpo_log.items()},
            {"denoiser/train/loss": 1.5, "denoiser/train/other": 0.2},
            {"denoiser/train/loss": 1.3, "denoiser/train/other": 0.1},
            {"rew_end_model/train/loss": 0.7}]
    row = diag.finish_epoch(1, logs, real_steps=500)
    assert row["denoiser_train_loss_mean"] == pytest.approx(1.4)
    assert row["denoiser_train_steps"] == 2
    assert row["rew_end_model_train_loss_mean"] == pytest.approx(0.7)
    assert row["global_pmpo_update"] == model.updates.item()
    assert (tmp_path / f"agent_pmpo_update_{model.updates.item():06d}_epoch.pt").exists()


def test_sustained_boundary_and_constant_stops(tmp_path):
    diag = diagnostics(tmp_path)
    metrics = healthy()
    metrics["near_lower_fraction"] = 0.99
    for i in range(49):
        diag.check_update(metrics, 1, i)
    with pytest.raises(DiagnosticStop, match="near-bound"):
        diag.check_update(metrics, 1, 49)
    diag = diagnostics(tmp_path)
    metrics = healthy()
    metrics["fixed_policy_mean_state_std_0"] = 0.
    for i in range(99):
        diag.check_update(metrics, 1, i)
    with pytest.raises(DiagnosticStop, match="constant"):
        diag.check_update(metrics, 1, 99)


@pytest.mark.parametrize("episode_length", [4, 9])
def test_fixed_seed_evaluation_and_temporal_probe_preserve_rng(tmp_path, monkeypatch, episode_length):
    import numpy as np
    import pmpo_diagnostic
    seeds = []
    class TinyEnv:
        def __init__(self, **kwargs):
            pass
        def reset(self, seed):
            seeds.append(seed)
            self.t = 0
            return np.zeros((8, 8, 3), dtype=np.uint8), {}
        def step(self, action):
            self.t += 1
            return np.full((8, 8, 3), self.t, dtype=np.uint8), float(action.sum()), False, self.t == episode_length, {}
        def close(self):
            pass
    monkeypatch.setattr(pmpo_diagnostic, "DMControlEnv", TinyEnv)
    diag = diagnostics(tmp_path)
    before = torch.get_rng_state().clone()
    a = diag.evaluate()
    b = diag.evaluate()
    assert a == b
    assert seeds == [11, 22, 11, 22]
    assert torch.equal(before, torch.get_rng_state())
    assert a["real_fixed_temporal_variation_fraction"] > 0
    assert a["real_fixed_ordered_repeated_action_difference"] > 0
