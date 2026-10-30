"""Downstream training on an ensemble milestone: the configured member (default 0) is loaded
as the single frozen denoiser; its index and identity are recorded; single-model milestones
keep working and only accept member 0."""
from pathlib import Path

import pytest
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

from data.milestone import load_milestone, publish_milestone, read_manifest, state_sha256
from models.diffusion import Denoiser
from test_downstream_trainer import build_milestone, compose, make_trainer, states_equal, TINY_DENOISER


def build_ensemble_milestone(root, members=3):
    """build_milestone's replay and member 0, plus members 1..K-1 (different weights)."""
    single = build_milestone(root)
    manifest = read_manifest(single)
    cfg = compose("exploration", TINY_DENOISER)
    denoiser_cfg = instantiate(cfg.agent.denoiser)
    denoiser_cfg.inner_model.continuous_action_dim = 6
    states = []
    for k in range(1, members):
        torch.manual_seed(100 + k)
        states.append(Denoiser(denoiser_cfg).state_dict())
    keep = {k: manifest[k] for k in ("created_by", "domain", "task_names", "collected_steps", "num_episodes", "epoch",
                                     "counters", "exploration", "model", "run_config", "source")}
    keep["ensemble"] = dict(members=members, bootstrap=True, downstream_default_member=0)
    milestone = load_milestone(single, "walk")
    return publish_milestone(single.parent, "steps_ensemble", milestone.denoiser_state, milestone.dataset,
                             milestone.successor_observations, keep, states), states


def test_configured_member_is_loaded_recorded_and_kept_apart(tmp_path, monkeypatch):
    path, states = build_ensemble_milestone(tmp_path)
    manifest = read_manifest(path)
    assert manifest["ensemble"]["members"] == 3 and len(manifest["identity"]["member_sha256"]) == 3
    default = make_trainer(path, monkeypatch)
    assert default.milestone.member_index == 0
    assert states_equal(torch.load(path / "denoiser.pt"), default.agent.denoiser.state_dict())

    trainer = make_trainer(path, monkeypatch, extra=["initialization.member_index=2"])
    assert trainer.milestone.member_index == 2
    assert states_equal(states[1], trainer.agent.denoiser.state_dict())  # member 2's weights, frozen
    assert not any(p.requires_grad for p in trainer.agent.denoiser.parameters())
    assert trainer.output_dir.name == "seed_0_member_2" and default.output_dir.name == "seed_0"
    out = trainer.run()
    summary = OmegaConf.load(out / "summary.json")
    assert summary.milestone.member_index == 2
    assert summary.milestone.member_sha256 == state_sha256(states[1]) == manifest["identity"]["member_sha256"][2]

    with pytest.raises(ValueError, match="member_index 3 is not one of them"):
        make_trainer(path, monkeypatch, extra=["initialization.member_index=3"])


def test_single_model_milestones_still_load_and_accept_only_member_0(tmp_path, monkeypatch):
    single = build_milestone(tmp_path)
    trainer = make_trainer(single, monkeypatch)
    assert trainer.milestone.member_index == 0 and "ensemble" not in trainer.milestone.manifest
    with pytest.raises(ValueError, match="has 1 denoiser"):
        make_trainer(single, monkeypatch, extra=["initialization.member_index=1"])
