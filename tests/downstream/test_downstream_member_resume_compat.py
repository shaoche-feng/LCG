"""Downstream resume compatibility with ensemble member selection: checkpoints saved before
initialization.member_index existed (meaning member 0) still resume and match uninterrupted
training; a missing index equals an explicit 0; changing the member is still refused."""
from pathlib import Path

import pytest
import torch
from omegaconf import OmegaConf

from downstream_trainer import changed_settings, DownstreamTrainer, normalized_saved_config
from test_downstream_eval_resume import Crash, identical, last_rows, make, run_uninterrupted, training_state
from test_downstream_ensemble_member import build_ensemble_milestone
from test_downstream_trainer import build_milestone, tree_hashes
from utils import rng_state


def crash_in_pmpo_phase(monkeypatch):
    original = DownstreamTrainer.policy_update

    def policy_update(self):
        original(self)
        if self.num_updates["actor_critic"] == 3:  # checkpoint at 2
            raise Crash()

    monkeypatch.setattr(DownstreamTrainer, "policy_update", policy_update)
    return original


def drop_member_index(checkpoint):
    """Make a checkpoint look like one saved before the field existed (a test fixture)."""
    state = torch.load(checkpoint, weights_only=False)
    del state["config"]["initialization"]["member_index"]
    torch.save(state, checkpoint)


@pytest.mark.parametrize("explicit_zero", [False, True])
def test_old_checkpoint_without_member_index_resumes_and_matches(tmp_path, monkeypatch, explicit_zero):
    full, full_rng, full_output = run_uninterrupted(tmp_path, monkeypatch)
    milestone = build_milestone(tmp_path / "split")
    sources = tree_hashes(tmp_path / "split/run/milestones", tmp_path / "split/run/dataset")
    original = crash_in_pmpo_phase(monkeypatch)
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    checkpoint = tmp_path / "split/run/downstream/steps_000026/walk/.seed_0.tmp/checkpoints/state.pt"
    drop_member_index(checkpoint)
    assert "member_index" not in torch.load(checkpoint, weights_only=False)["config"]["initialization"]

    extra = ["initialization.member_index=0"] if explicit_zero else []
    resumed = make(milestone, monkeypatch, extra=extra, resume=True)
    output = resumed.run()
    identical(training_state(full), training_state(resumed), "final training state")
    identical(full_rng, rng_state(), "global RNG progression")
    full_rows, _ = last_rows(full_output / "metrics.jsonl")
    resumed_rows, _ = last_rows(output / "metrics.jsonl")
    assert resumed_rows == full_rows
    assert tree_hashes(tmp_path / "split/run/milestones", tmp_path / "split/run/dataset") == sources


def test_missing_index_equals_zero_and_member_changes_are_detected():
    current = dict(initialization=dict(milestone_path="m", member_index=0), downstream=dict(seed=0))
    old = dict(initialization=dict(milestone_path="m"), downstream=dict(seed=0))
    assert changed_settings(normalized_saved_config(old), current) == []
    assert "member_index" not in old["initialization"]  # the saved config itself is not changed
    assert changed_settings(normalized_saved_config(old), dict(current, initialization=dict(milestone_path="m",
                                                                                         member_index=2))) == [
        "initialization.member_index"]
    explicit = dict(initialization=dict(milestone_path="m", member_index=3), downstream=dict(seed=0))
    assert changed_settings(normalized_saved_config(explicit), current) == ["initialization.member_index"]


def test_resume_refuses_a_different_ensemble_member(tmp_path, monkeypatch):
    path, _ = build_ensemble_milestone(tmp_path)
    original = crash_in_pmpo_phase(monkeypatch)
    with pytest.raises(Crash):
        make(path, monkeypatch, extra=["initialization.member_index=1"]).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    checkpoint = Path(path).parent.parent / "downstream" / path.name / "walk/.seed_0_member_1.tmp/checkpoints/state.pt"
    # The checkpoint records member 1; a saved member 2 under the same output must not resume as member 1.
    state = torch.load(checkpoint, weights_only=False)
    assert state["config"]["initialization"]["member_index"] == 1
    state["config"]["initialization"]["member_index"] = 2
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="different settings: \\['initialization.member_index'\\]"):
        make(path, monkeypatch, extra=["initialization.member_index=1"], resume=True)
    # And an older checkpoint without the field (member 0) cannot be resumed as member 1.
    del state["config"]["initialization"]["member_index"]
    torch.save(state, checkpoint)
    with pytest.raises(ValueError, match="different settings: \\['initialization.member_index'\\]"):
        make(path, monkeypatch, extra=["initialization.member_index=1"], resume=True)
