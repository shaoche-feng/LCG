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
    diag = PMPORunDiagnostic(SimpleNamespace(output_dir=str(tmp_path), evaluation_seeds=[11, 22],
                                             checkpoint_updates=(2, 4)), make_controller(), {})
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
    assert saved["update"] == 2 and set(saved) == {"actor", "value", "prior_actor", "update"}
    for key, p in saved["actor"].items():
        torch.testing.assert_close(p, diag.model.actor.state_dict()[key], rtol=0, atol=0)


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
