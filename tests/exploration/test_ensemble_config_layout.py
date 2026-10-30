"""Ensemble settings live in intrinsic_reward (num_members, bootstrap). Configs saved with the
older top-level `ensemble: {members, bootstrap}` are read as the new fields in memory, keep
their saved values, and train and resume identically; conflicting layouts and real changes
are refused; uniform and LCG keep one denoiser."""
import pytest
from omegaconf import OmegaConf

import test_exploration_trainer as base
import test_test_diagnostics as diag
from ensemble import migrate_ensemble_layout
from exploration_trainer import ExplorationTrainer
from test_ensemble import ENS, OFF, run_ens
from test_exploration_trainer import COLLECT, compose, run_trainer
from test_test_diagnostics import BASE, Crash, same, start
from utils import rng_state

NEW = ["intrinsic_reward.num_members=3", "intrinsic_reward.bootstrap=false"]  # non-default values


def plain(cfg):
    return OmegaConf.create(OmegaConf.to_container(cfg))


def to_old_layout(cfg):
    """The same settings in the older layout (as saved before the refactor)."""
    cfg = plain(cfg)
    members, bootstrap = cfg.intrinsic_reward.num_members, cfg.intrinsic_reward.bootstrap
    del cfg.intrinsic_reward["num_members"], cfg.intrinsic_reward["bootstrap"]
    cfg.ensemble = dict(members=members, bootstrap=bootstrap)
    return cfg


def test_defaults_and_command_line_overrides(tmp_path, monkeypatch):
    cfg = compose(["+experiment=walker_ensemble_100k"])
    assert (cfg.intrinsic_reward.num_members, cfg.intrinsic_reward.bootstrap) == (5, True) and "ensemble" not in cfg
    cfg = compose(["+experiment=walker_ensemble_100k", "intrinsic_reward.num_members=7", "intrinsic_reward.bootstrap=false"])
    assert (cfg.intrinsic_reward.num_members, cfg.intrinsic_reward.bootstrap) == (7, False)
    trainer, _ = run_ens(tmp_path, monkeypatch, ["intrinsic_reward.num_members=4", "intrinsic_reward.bootstrap=false"],
                         build_only=True)
    assert len(trainer.agent.denoisers) == 4 and trainer.ensemble.settings() == dict(members=4, bootstrap=False)


def test_old_layout_keeps_saved_values_and_conflicts_are_rejected():
    new = compose([*ENS, *NEW])
    old = to_old_layout(new)
    migrated = migrate_ensemble_layout(old)
    assert "ensemble" not in migrated
    assert (migrated.intrinsic_reward.num_members, migrated.intrinsic_reward.bootstrap) == (3, False)  # not defaults
    assert old.ensemble.members == 3 and "num_members" not in old.intrinsic_reward  # input untouched
    assert OmegaConf.to_container(migrated) == OmegaConf.to_container(plain(new))
    agree = plain(new)
    agree.ensemble = dict(members=3, bootstrap=False)
    assert OmegaConf.to_container(migrate_ensemble_layout(agree)) == OmegaConf.to_container(migrated)
    for key, value in (("members", 4), ("bootstrap", True)):
        conflict = plain(new)
        conflict.ensemble = dict(members=3, bootstrap=False)
        conflict.ensemble[key] = value
        with pytest.raises(ValueError, match=f"Conflicting ensemble settings: ensemble.{key}"):
            migrate_ensemble_layout(conflict)
    assert migrate_ensemble_layout(new) is new  # new layout: nothing to do


@pytest.mark.parametrize("extra", [[], ["collection.train.mode=uniform", "intrinsic_reward.enabled=false"]])
def test_lcg_and_uniform_keep_one_denoiser_with_or_without_the_old_section(tmp_path, monkeypatch, extra):
    trainer, _ = run_trainer(tmp_path / "new", monkeypatch, [*COLLECT, *extra], build_only=True)
    assert len(trainer.agent.denoisers) == 1 and trainer.ensemble is None
    old = plain(compose([*base.TINY, *COLLECT, *extra]))
    old.ensemble = dict(members=5, bootstrap=True)  # what older trainer configs carried (unused by LCG/uniform)
    migrated = migrate_ensemble_layout(old)
    assert "ensemble" not in migrated and "num_members" not in migrated.intrinsic_reward
    run_dir = tmp_path / "old"
    (run_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(old, run_dir / ".hydra/config.yaml")
    monkeypatch.chdir(run_dir)
    trainer = ExplorationTrainer(old, base.ROOT)
    assert len(trainer.agent.denoisers) == 1 and trainer.ensemble is None


def test_old_and_new_layouts_resume_identically_and_changes_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", ENS)
    full, _ = start(tmp_path / "full", monkeypatch, BASE + OFF + NEW)
    full._run()
    full_rng = rng_state()

    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    crashed, _ = start(tmp_path / "split", monkeypatch, BASE + OFF + NEW)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.setattr(ExplorationTrainer, "train_agent", original)
    # The saved run config rewritten in the older layout (a test fixture for a pre-refactor run).
    saved = tmp_path / "split/config/exploration.yaml"
    OmegaConf.save(to_old_layout(OmegaConf.load(saved)), saved)
    assert "ensemble" in OmegaConf.load(saved)
    with pytest.raises(ValueError, match="Conflicting ensemble settings"):  # new-field change vs the old file
        start(tmp_path / "split", monkeypatch, ["intrinsic_reward.num_members=4"], resume=True)
    with pytest.raises(ValueError, match="ensemble settings"):  # old-layout change
        start(tmp_path / "split", monkeypatch, ["ensemble.members=4"], resume=True)
    with pytest.raises(ValueError, match="must not switch"):  # scoring change
        start(tmp_path / "split", monkeypatch, ["intrinsic_reward.fixed_divisor=2.0"], resume=True)
    resumed, _ = start(tmp_path / "split", monkeypatch, [], resume=True)
    resumed._run()
    assert resumed.ensemble.settings() == dict(members=3, bootstrap=False)
    assert same(full.agent.state_dict(), resumed.agent.state_dict())
    assert same(full.ensemble.state_dict(), resumed.ensemble.state_dict())
    assert same(full.opt.state_dict(), resumed.opt.state_dict())
    for k in ("python", "numpy", "torch"):
        assert same(full_rng[k], rng_state()[k])


def test_a_fresh_run_from_an_old_layout_config_equals_the_new_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", ENS)
    new, _ = start(tmp_path / "new", monkeypatch, BASE + OFF + NEW)
    new._run()
    cfg = to_old_layout(compose([*ENS, *BASE, *OFF, *NEW]))  # same settings, older layout
    run_dir = tmp_path / "old"
    (run_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / ".hydra/config.yaml")
    monkeypatch.chdir(run_dir)
    old = ExplorationTrainer(cfg, base.ROOT)
    old._run()
    assert same(new.agent.state_dict(), old.agent.state_dict())
    assert same(new.ensemble.state_dict(), old.ensemble.state_dict())
