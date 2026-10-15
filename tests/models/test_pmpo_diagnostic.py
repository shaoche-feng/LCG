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


def test_world_model_summary_counts_actual_denoiser_loss_key(tmp_path):
    summary = diagnostics(tmp_path).world_model_loss_summary([
        {"denoiser/train/loss_denoising": 1.5}, {"denoiser/train/loss_denoising": .5},
        {"denoiser/test/loss_denoising": .75}, {"rew_end_model/train/loss_total": .25},
    ])
    assert summary["denoiser_train_steps"] == 2
    assert summary["denoiser_test_steps"] == 1
    assert summary["denoiser_train_loss_denoising_mean"] == 1.
    assert summary["rew_end_model_train_steps"] == 1


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


def row(**changes):
    return dict(healthy(), **changes)


def entropy_row(policy_entropy, concentration=2.):
    return row(policy_entropy=policy_entropy, alpha_mean=concentration)


NEAR_BOUND = row(near_lower_fraction=0.99)
CONSTANT = row(fixed_policy_mean_state_std_0=0.)


def monitored(path, interval):
    """A diagnostic whose full snapshots carry the model and diagnostic state, as Trainer's do."""
    model = make_controller(prior_refresh_interval=interval)
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(path), evaluation_seeds=[11, 22]), model, {},
                             trainer_state_fn=lambda: dict(model=model.state_dict(), pmpo_diagnostic=diag.state_dict()))
    return model, diag


def run_rows(model, diag, rows, first_update):
    """Trainer's order: the optimizer step completes the update, then check_update."""
    outcomes = []
    for update, metrics in enumerate(rows, first_update):
        model.updates.fill_(update + 1)
        try:
            diag.check_update(metrics, 1, update)
            outcomes.append(None)
        except DiagnosticStop as stop:
            outcomes.append(str(stop))
    return outcomes


@pytest.mark.parametrize("interval,prefix,snapshot,continuation,expected", [
    # Refresh snapshots include the row whose update they follow.
    (1, [NEAR_BOUND] * 49, "agent_pmpo_update_000048.pt", [NEAR_BOUND, row()], ["near-bound", None]),
    (1, [CONSTANT] * 99, "agent_pmpo_update_000098.pt", [CONSTANT], ["constant"]),
    # A stale previous entropy (1.0) would hide this collapse; the snapshot row's (2.0) shows it.
    (1, [entropy_row(1.), entropy_row(2.)], "agent_pmpo_update_000001.pt", [entropy_row(-2.5, 200.)], ["Abrupt"]),
    # Stop snapshots include the stopping row.
    (1000, [NEAR_BOUND] * 50, "agent_pmpo_update_000049_stop.pt", [NEAR_BOUND, row()], ["near-bound", None]),
    # A stale previous entropy (2.0) would report a collapse; the stopping row's (-2.5) does not.
    (1000, [entropy_row(2.), entropy_row(-2.5, 200.)], "agent_pmpo_update_000001_stop.pt",
     [entropy_row(-5., 200.)], [None]),
    # A non-finite row is observed but stays out of the history.
    (1000, [entropy_row(1.), entropy_row(2.), row(kl_prior=float("nan"))], "agent_pmpo_update_000002_stop.pt",
     [entropy_row(-2.5, 200.)], ["Abrupt"]),
], ids=["refresh-boundary", "refresh-constant", "refresh-entropy", "stop-boundary", "stop-entropy",
        "stop-nonfinite"])
def test_restored_snapshot_monitors_like_uninterrupted_run(tmp_path, interval, prefix, snapshot, continuation,
                                                           expected):
    model, live = monitored(tmp_path / "live", interval)
    run_rows(model, live, prefix, 0)
    payload = torch.load(tmp_path / "live" / snapshot, weights_only=False)["trainer_state"]
    restored_model, restored = monitored(tmp_path / "restored", interval)
    restored_model.load_state_dict(payload["model"])
    restored.load_state_dict(payload["pmpo_diagnostic"])
    assert restored.state_dict() == live.state_dict()

    live_outcomes = run_rows(model, live, continuation, len(prefix))
    restored_outcomes = run_rows(restored_model, restored, continuation, len(prefix))
    assert restored_outcomes == live_outcomes
    assert all(e is None and o is None or e is not None and o is not None and e in o
               for e, o in zip(expected, live_outcomes))
    assert restored.state_dict() == live.state_dict()


def test_nonfinite_row_is_counted_but_kept_out_of_history(tmp_path):
    model, diag = monitored(tmp_path, 1000)
    run_rows(model, diag, [NEAR_BOUND, entropy_row(2.)], 0)
    before = diag.state_dict()
    assert run_rows(model, diag, [row(kl_prior=float("nan"), near_lower_fraction=0.99)], 2) == [
        "Non-finite controller diagnostic"]
    saved = torch.load(tmp_path / "agent_pmpo_update_000002_stop.pt", weights_only=False)["trainer_state"]
    assert saved["model"]["updates"].item() == 3 == saved["pmpo_diagnostic"]["updates_observed"]
    assert saved["pmpo_diagnostic"]["nonfinite_updates"] == 1
    for key in ("boundary_streak", "constant_streak", "one_sided_epochs", "previous_entropy_per_dim"):
        assert saved["pmpo_diagnostic"][key] == before[key]
    assert not (tmp_path / "updates.jsonl").read_text().count("NaN")


def test_history_must_describe_the_restored_model_update(tmp_path):
    model, live = monitored(tmp_path / "live", 1)
    run_rows(model, live, [row()] * 3, 0)
    payload = torch.load(tmp_path / "live/agent_pmpo_update_000002.pt", weights_only=False)["trainer_state"]
    assert payload["model"]["updates"].item() == 3 == payload["pmpo_diagnostic"]["updates_observed"]
    lagging_model, diag = monitored(tmp_path / "lagging", 1)
    lagging_model.load_state_dict(payload["model"])
    lagging_model.updates.fill_(2)  # Trainer's PMPO batch counter at that save.
    with pytest.raises(ValueError, match="restored model completed 2"):
        diag.load_state_dict(payload["pmpo_diagnostic"])
    # A checkpoint without history restarts the streaks and counts from the model.
    model, diag = monitored(tmp_path / "older", 1)
    model.updates.fill_(7)
    diag.load_state_dict(None)
    assert diag.updates_observed == 7 and diag.boundary_streak == 0 and diag.fixed_real is None
    assert run_rows(model, diag, [NEAR_BOUND], 7) == [None] and diag.state_dict()["updates_observed"] == 8


def frame_env(offset):
    """Evaluation stand-in whose frames shift by offset[0], so a re-collection differs."""
    import numpy as np

    class TinyEnv:
        def __init__(self, **kwargs):
            pass
        def reset(self, seed):
            self.t = 0
            return np.full((8, 8, 3), offset[0], dtype=np.uint8), {}
        def step(self, action):
            self.t += 1
            return np.full((8, 8, 3), offset[0] + self.t, dtype=np.uint8), float(action.sum()), False, self.t == 4, {}
        def close(self):
            pass
    return TinyEnv


def recorded_reference(tmp_path, monkeypatch):
    """Evaluate once, then shift the environment so any re-collection would differ."""
    import pmpo_diagnostic
    offset = [0]
    monkeypatch.setattr(pmpo_diagnostic, "DMControlEnv", frame_env(offset))
    model = make_controller()
    cfg = SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22])
    first = PMPORunDiagnostic(cfg, model, {})
    expected = first.evaluate()
    offset[0] = 100
    return model, cfg, first, expected


def assert_same_fixed_metrics(actual, expected):
    keys = [k for k in expected if k.startswith("real_fixed_")]
    assert keys and all(actual[k] == expected[k] for k in keys)


def test_fresh_diagnostic_records_new_reference_and_keeps_stale_file(tmp_path, monkeypatch):
    import pmpo_diagnostic
    path = tmp_path / "fixed_real_stacks.pt"
    stale = torch.full((3, 12, 8, 8), 0.5)
    torch.save(stale, path)
    stale_bytes = path.read_bytes()
    monkeypatch.setattr(pmpo_diagnostic, "DMControlEnv", frame_env([0]))
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22]), make_controller(), {})
    assert diag.fixed_real is None and path.read_bytes() == stale_bytes  # Not read or touched yet.
    diag.evaluate()
    assert not torch.equal(diag.fixed_real[:3], stale)
    assert torch.equal(torch.load(path, weights_only=True), diag.fixed_real)
    assert (tmp_path / "fixed_real_stacks.preserved_1.pt").read_bytes() == stale_bytes


@pytest.mark.parametrize("file_state", ["missing", "different", "unreadable"])
def test_checkpointed_fixed_real_is_authoritative_over_file(tmp_path, monkeypatch, file_state):
    model, cfg, first, expected = recorded_reference(tmp_path, monkeypatch)
    state = first.state_dict()
    assert state["fixed_real"]["encoding"] == "uint8_pixels_div"
    path = tmp_path / "fixed_real_stacks.pt"
    if file_state == "missing":
        path.unlink()
    elif file_state == "different":
        torch.save(first.fixed_real + 1, path)
    else:
        path.write_bytes(b"not a torch checkpoint")
    kept = path.read_bytes() if path.exists() else None
    resumed = PMPORunDiagnostic(cfg, model, {})
    resumed.load_state_dict(state)
    assert torch.equal(resumed.fixed_real, first.fixed_real)
    if file_state == "missing":
        assert torch.equal(torch.load(path, weights_only=True), first.fixed_real)
    else:
        assert path.read_bytes() == kept  # Possible evidence is left in place.
    assert_same_fixed_metrics(resumed.evaluate(), expected)
    assert not list(tmp_path.glob("*.preserved_*"))  # Evaluation adds no new reference set.


@pytest.mark.parametrize("older_state", ["no_diagnostic_state", "no_fixed_real_entry"])
def test_older_checkpoint_uses_valid_reference_file(tmp_path, monkeypatch, capsys, older_state):
    model, cfg, first, expected = recorded_reference(tmp_path, monkeypatch)
    file_bytes = (tmp_path / "fixed_real_stacks.pt").read_bytes()
    state = None
    if older_state == "no_fixed_real_entry":
        state = first.state_dict()
        del state["fixed_real"]
    resumed = PMPORunDiagnostic(cfg, model, {})
    resumed.load_state_dict(state)
    assert "using fixed_real_stacks.pt" in capsys.readouterr().out
    assert torch.equal(resumed.fixed_real, first.fixed_real)
    assert_same_fixed_metrics(resumed.evaluate(), expected)
    assert (tmp_path / "fixed_real_stacks.pt").read_bytes() == file_bytes


@pytest.mark.parametrize("invalid", ["unreadable", "not_tensor", "uint8", "shape", "non_finite", "out_of_range"])
def test_older_checkpoint_rejects_invalid_reference_file(tmp_path, monkeypatch, capsys, invalid):
    model, cfg, first, _ = recorded_reference(tmp_path, monkeypatch)
    path = tmp_path / "fixed_real_stacks.pt"
    bad = {"not_tensor": {"fixed": first.fixed_real}, "uint8": torch.zeros(2, 12, 8, 8, dtype=torch.uint8),
           "shape": torch.zeros(2, 3, 8, 8), "non_finite": first.fixed_real.clone().fill_(float("nan")),
           "out_of_range": first.fixed_real * 3}
    if invalid == "unreadable":
        path.write_bytes(b"not a torch checkpoint")
    else:
        torch.save(bad[invalid], path)
    bad_bytes = path.read_bytes()
    resumed = PMPORunDiagnostic(cfg, model, {})
    resumed.load_state_dict(None)
    assert "fixed_real_stacks.pt is not usable" in capsys.readouterr().out
    assert resumed.fixed_real is None and path.read_bytes() == bad_bytes
    resumed.evaluate()  # Records a new set from the shifted environment; the rejected file is kept.
    assert not torch.equal(resumed.fixed_real, first.fixed_real)
    assert torch.equal(torch.load(path, weights_only=True), resumed.fixed_real)
    assert (tmp_path / "fixed_real_stacks.preserved_1.pt").read_bytes() == bad_bytes


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.skipif(
    not torch.cuda.is_available(), reason="CUDA unavailable"))])
def test_fixed_real_uint8_checkpoint_round_trip_is_exact(tmp_path, monkeypatch, device):
    import random
    import numpy as np
    import pmpo_diagnostic
    from pmpo_diagnostic import decode_fixed_real, encode_fixed_real, normalize_pixels
    expected_encoding = "uint8_pixels_div" if device == "cpu" else "uint8_pixels_mul_reciprocal"
    # Every pixel value, normalized on the collection device.
    every_value = normalize_pixels(torch.arange(256, dtype=torch.uint8, device=device))
    stored = encode_fixed_real(every_value)
    assert stored["encoding"] == expected_encoding
    assert torch.equal(stored["values"], torch.arange(256, dtype=torch.uint8))
    assert torch.equal(decode_fixed_real(stored), every_value.cpu())

    # Reference stacks from the evaluation preprocessing path; the frames cover all 256 values.
    class TinyEnv:
        def __init__(self, **kwargs):
            pass
        def frame(self):
            return ((np.arange(192) + 64 * self.t) % 256).astype(np.uint8).reshape(8, 8, 3)
        def reset(self, seed):
            self.t = 0
            return self.frame(), {}
        def step(self, action):
            self.t += 1
            return self.frame(), float(action.sum()), False, self.t == 4, {}
        def close(self):
            pass
    monkeypatch.setattr(pmpo_diagnostic, "DMControlEnv", TinyEnv)
    model = make_controller().to(device)
    cfg = SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22])
    first = PMPORunDiagnostic(cfg, model, {})
    expected = first.evaluate()
    assert first.fixed_real.unique().numel() == 256
    (tmp_path / "fixed_real_stacks.pt").unlink()  # Restore from the checkpoint alone.

    def rng_state():
        cuda = [s.clone() for s in torch.cuda.get_rng_state_all()] if torch.cuda.is_available() else []
        return torch.get_rng_state().clone(), cuda, np.random.get_state(), random.getstate()
    before = rng_state()
    state = first.state_dict()
    torch.save(state, tmp_path / "state.pt")
    # Trainer maps its checkpoint onto the training device.
    state = torch.load(tmp_path / "state.pt", map_location=device, weights_only=False)
    resumed = PMPORunDiagnostic(cfg, model, {})
    resumed.load_state_dict(state)
    after = rng_state()
    assert torch.equal(before[0], after[0]) and len(before[1]) == len(after[1])
    assert all(torch.equal(a, b) for a, b in zip(before[1], after[1]))
    assert before[2][0] == after[2][0] and np.array_equal(before[2][1], after[2][1]) and before[2][2:] == after[2][2:]
    assert before[3] == after[3]

    assert state["fixed_real"]["encoding"] == expected_encoding and state["fixed_real"]["values"].dtype == torch.uint8
    assert resumed.fixed_real.dtype == torch.float32 and resumed.fixed_real.device.type == "cpu"
    assert torch.equal(resumed.fixed_real, first.fixed_real)
    assert torch.equal(torch.load(tmp_path / "fixed_real_stacks.pt", weights_only=True), first.fixed_real)
    actual = resumed.evaluate()
    fixed_keys = [k for k in expected if k.startswith("real_fixed_")]
    assert fixed_keys and all(actual[k] == expected[k] for k in fixed_keys)


def test_fixed_real_float32_fallback_and_older_states(tmp_path):
    from pmpo_diagnostic import decode_fixed_real, encode_fixed_real, normalize_pixels
    pixels = torch.randint(0, 256, (2, 12, 8, 8), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    fixed_real = normalize_pixels(pixels)
    # States that are not exactly 8-bit pixels are kept as float32.
    off_grid = fixed_real * 0.5 + 1e-3
    stored = encode_fixed_real(off_grid)
    assert stored["encoding"] == "float32" and torch.equal(decode_fixed_real(stored), off_grid)
    with_nan = fixed_real.clone()
    with_nan[0, 0, 0, 0] = float("nan")
    assert encode_fixed_real(with_nan)["encoding"] == "float32"
    assert encode_fixed_real(None) is None and decode_fixed_real(None) is None
    with pytest.raises(ValueError, match="encoding"):
        decode_fixed_real(dict(encoding="png", values=pixels))

    # An earlier diagnostic state stored the bare float32 tensor.
    history = diagnostics(tmp_path).state_dict()
    restored = diagnostics(tmp_path)
    restored.load_state_dict(dict(history, fixed_real=fixed_real))
    assert torch.equal(restored.fixed_real, fixed_real)
    # A history-only state falls back to fixed_real_stacks.pt (written by the restore above).
    history.pop("fixed_real")
    fallback = diagnostics(tmp_path)
    fallback.load_state_dict(history)
    assert torch.equal(fallback.fixed_real, fixed_real)


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
