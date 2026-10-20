"""Exploration test collection and diagnostics: evaluation.should / every / collect_test,
per-task return summaries, test-interaction accounting, isolation and resume."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import exploration_trainer as trainer_module
from data import Dataset, Episode
from data.milestone import referenced_files
from exploration_trainer import check_evaluation_config, ExplorationTrainer, task_return_summary
from test_exploration_trainer import COLLECT, compose, FakeEnv, ROOT, TASKS, TINY, UNIFORM
from utils import rng_state

BASE = [o for o in COLLECT if not o.startswith("evaluation.")]  # 12 + 2 x 6 train steps
OFF = ["evaluation.should=false"]
# Fresh test data every epoch: 2 episodes of 6 steps; final: 3 episodes.
FRESH = ["evaluation.should=true", "evaluation.every=1", "evaluation.collect_test=true",
         "collection.test.num_episodes=2", "collection.test.num_final_episodes=3"]
REUSE = ["evaluation.collect_test=false"]


def start(run_dir, monkeypatch, overrides, resume=False):
    """ExplorationTrainer on CPU with the fake env; a resume reloads the run's saved config
    with the given overrides on top. Logs are captured as (epoch, row)."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer_module, "make_dm_control_env", lambda num_envs, device, **kwargs: FakeEnv(device))
    for method in ("init", "save"):
        monkeypatch.setattr(trainer_module.wandb, method, lambda *args, **kwargs: None)
    logs = []
    monkeypatch.setattr(trainer_module, "wandb_log", lambda rows, epoch: logs.extend((epoch, r) for r in rows))
    if resume:
        monkeypatch.chdir(run_dir)
        cfg = OmegaConf.load(run_dir / "config/trainer.yaml")
        cfg.common.resume = True
        for override in overrides:
            key, value = override.split("=")
            OmegaConf.update(cfg, key, yaml_value(value))
    else:
        cfg = compose(TINY + list(overrides))
        (run_dir / ".hydra").mkdir(parents=True)
        OmegaConf.save(cfg, run_dir / ".hydra/config.yaml")
        monkeypatch.chdir(run_dir)
    return ExplorationTrainer(cfg, ROOT), logs


def yaml_value(text):
    return OmegaConf.create(f"v: {text}").v


def run(trainer):
    trainer._run()
    trainer._final_rng = rng_state()
    return trainer


def rows_with(logs, prefix):
    return [(epoch, row) for epoch, row in logs if any(k.startswith(prefix) for k in row)]


def spy_calls(monkeypatch):
    """Record collect_test and test_agent calls in order."""
    calls = []
    collect, test = ExplorationTrainer.collect_test, ExplorationTrainer.test_agent

    def collect_spy(self, final=False):
        calls.append(("collect", self.epoch, final))
        return collect(self, final)

    def test_spy(self):
        calls.append(("test_agent", self.epoch))
        return test(self)

    monkeypatch.setattr(ExplorationTrainer, "collect_test", collect_spy)
    monkeypatch.setattr(ExplorationTrainer, "test_agent", test_spy)
    return calls


def training_rows(logs):
    """Learning metrics only: losses, PMPO metrics (incl. sampled actions), train collection."""
    out = []
    for epoch, row in logs:
        values = {k: float(v) for k, v in row.items() if np.ndim(v) == 0
                  and k.split("/")[0] in ("denoiser", "actor_critic", "train_dataset") and "/test/" not in k}
        if values:
            out.append((epoch, values))
    return out


def same(x, y):
    if torch.is_tensor(x):
        return torch.is_tensor(y) and torch.equal(x, y)
    if isinstance(x, np.ndarray):
        return isinstance(y, np.ndarray) and np.array_equal(x, y)
    if isinstance(x, dict):
        return isinstance(y, dict) and x.keys() == y.keys() and all(same(x[k], y[k]) for k in x)
    if isinstance(x, (list, tuple)):
        return len(x) == len(y) and all(same(u, v) for u, v in zip(x, y))
    return x == y


def assert_same_training(a, b):
    assert same(a.agent.state_dict(), b.agent.state_dict())
    assert same(a.opt.state_dict(), b.opt.state_dict())
    assert same(a.lr_sched.state_dict(), b.lr_sched.state_dict())
    assert a.num_batch_train.state_dict() == b.num_batch_train.state_dict()
    assert a.train_dataset.lengths.tolist() == b.train_dataset.lengths.tolist()
    for i in range(a.train_dataset.num_episodes):
        x, y = a.train_dataset.load_episode(i), b.train_dataset.load_episode(i)
        for field in ("obs", "act", "rew", "task_rew"):
            assert torch.equal(getattr(x, field), getattr(y, field))
    for k in ("python", "numpy", "torch"):
        assert repr(a._final_rng[k]) == repr(b._final_rng[k]), k


# ---------------------------------------------------------------- flags

def test_master_switch_off_collects_and_diagnoses_nothing(tmp_path, monkeypatch):
    calls = spy_calls(monkeypatch)
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + UNIFORM + OFF + ["evaluation.collect_test=true"])
    run(trainer)
    assert calls == []
    assert not rows_with(logs, "test") and not rows_with(logs, "final_test") and not rows_with(logs, "denoiser/test")
    assert trainer.num_test_steps == 0 and trainer.test_dataset.num_episodes == 0 and not trainer.final_test_done
    assert not list(Path("dataset/test").rglob("*_*.pt"))
    # should=false needs no test data even without collection (no reuse check).
    trainer, _ = start(tmp_path / "run2", monkeypatch, BASE + UNIFORM + OFF + REUSE)
    run(trainer)
    assert calls == []


def test_fresh_mode_collects_then_diagnoses_and_counts_test_interactions(tmp_path, monkeypatch):
    calls = spy_calls(monkeypatch)
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + UNIFORM + FRESH)
    run(trainer)
    # Each scheduled point and the final: collection, then diagnostics.
    assert calls == [("collect", 1, False), ("test_agent", 1), ("collect", 2, False), ("test_agent", 2),
                     ("collect", 2, True), ("test_agent", 2)]
    summaries = [(epoch, row) for epoch, row in logs if "test/fresh" in row or "final_test/fresh" in row]
    got = [(e, r.get("test/new_env_steps", r.get("final_test/new_env_steps")),
            r.get("test/total_new_env_steps", r.get("final_test/total_new_env_steps"))) for e, r in summaries]
    assert got == [(1, 12, 12), (2, 12, 24), (2, 18, 42)]
    assert all(r.get("test/fresh", r.get("final_test/fresh")) == 1 for _, r in summaries)
    assert summaries[-1][1]["final_test/num_episodes"] == 3 and summaries[-1][1]["final_test/num_steps"] == 18
    assert trainer.num_test_steps == 42 and trainer.num_episodes_test == 7 and trainer.final_test_done
    assert len(rows_with(logs, "denoiser/test")) > 0
    # The training budget is unaffected; the final state is committed.
    assert trainer.train_dataset.num_steps == 24
    saved = torch.load("checkpoints/state.pt", weights_only=False)
    assert saved["num_test_steps"] == 42 and saved["final_test_done"] and saved["test_dataset"]["num_episodes"] == 3
    # Only the committed test collection's files remain on disk.
    on_disk = {p.resolve() for p in Path("dataset/test").rglob("*.pt") if p.name != "info.pt"}
    assert on_disk == {p.resolve() for p in trainer.test_dataset.episode_files()} and len(on_disk) == 3


@pytest.mark.parametrize("evaluation,match", [
    (dict(should=True, every=0), "every must be a positive integer"),
    (dict(should=True, every=2.5), "every must be a positive integer"),
    (dict(should=True, every=True), "every must be a positive integer"),
    (dict(should=True, every=1, collect_test="yes"), "collect_test must be true or false"),
    (dict(should="no", every=1), "should must be true or false"),
])
def test_invalid_evaluation_settings_are_rejected(evaluation, match):
    with pytest.raises(ValueError, match=match):
        check_evaluation_config(OmegaConf.create(evaluation))


def test_evaluation_defaults_and_old_configs():
    assert check_evaluation_config(compose([]).evaluation) == (True, True)
    assert check_evaluation_config(OmegaConf.create(dict(should=True, every=10))) == (True, True)  # no collect_test
    assert check_evaluation_config(OmegaConf.create(dict(should=False, every=0))) == (False, True)  # interval unused
    preset = compose(["+experiment=walker_uniform_100k"]).evaluation
    assert (preset.should, preset.every, preset.collect_test) == (True, 10, True)


# ---------------------------------------------------------------- reuse-only

def test_reuse_only_without_test_data_fails_before_training(tmp_path, monkeypatch):
    with pytest.raises(ValueError, match="a new run has none"):
        start(tmp_path / "run", monkeypatch, BASE + UNIFORM + FRESH + REUSE)
    assert not (tmp_path / "run/checkpoints").exists()
    # A static dataset whose test split is empty: never replaced by training replay.
    static = tmp_path / "static"
    train = Dataset(static / "train")
    for _ in range(2):
        train.add_episode(Episode(torch.zeros(10, 3, 8, 8), torch.zeros(10, 2), torch.zeros(10),
                                  torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long), {},
                                  task_rew=torch.zeros(10, 2), task_names=TASKS))
    train.save_to_default_path()
    for flag in ("true", "false"):  # static datasets never collect, whatever collect_test says
        with pytest.raises(ValueError, match="needs an existing test dataset"):
            start(tmp_path / f"static_run_{flag}", monkeypatch, UNIFORM + FRESH + [
                f"static_dataset.path={static.as_posix()}", f"evaluation.collect_test={flag}",
                "training.num_final_epochs=1"])
        assert not (tmp_path / f"static_run_{flag}/checkpoints/state.pt").exists()


def test_reuse_only_after_resume_diagnoses_without_collecting(tmp_path, monkeypatch):
    full, logs_full = start(tmp_path / "full", monkeypatch, BASE + UNIFORM + FRESH)
    run(full)

    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, _ = start(tmp_path / "split", monkeypatch, BASE + UNIFORM + FRESH)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.undo()

    calls = spy_calls(monkeypatch)
    resumed, logs = start(tmp_path / "split", monkeypatch, REUSE, resume=True)
    committed = [p.resolve() for p in resumed.test_dataset.episode_files()]
    expected = task_return_summary(resumed.test_dataset)
    run(resumed)
    assert [c[0] for c in calls] == ["test_agent", "test_agent"]  # epoch 2 and final, no collection
    summaries = [row for _, row in logs if "test/fresh" in row or "final_test/fresh" in row]
    assert len(summaries) == 2
    for row in summaries:
        prefix = "test" if "test/fresh" in row else "final_test"
        assert row[f"{prefix}/fresh"] == 0 and row[f"{prefix}/new_env_steps"] == 0
        assert row[f"{prefix}/total_new_env_steps"] == 12  # epoch 1's collection only
        assert row[f"{prefix}/num_episodes"] == 2
        assert {k.split("/", 1)[1]: v for k, v in row.items() if "/return_" in k} == expected
    assert [p.resolve() for p in resumed.test_dataset.episode_files()] == committed
    assert all(p.is_file() for p in committed)
    # Diagnostics are isolated: training continues exactly as in the fresh uninterrupted run.
    assert_same_training(full, resumed)
    assert training_rows([x for x in logs_full if x[0] == 2]) == training_rows([x for x in logs if x[0] == 2])

    # A missing committed test file is reported, never silently replaced.
    monkeypatch.undo()
    done, _ = start(tmp_path / "split2", monkeypatch, BASE + UNIFORM + FRESH)
    done._run()
    done.test_dataset.episode_files()[0].unlink()
    with pytest.raises(FileNotFoundError, match="is missing"):
        start(tmp_path / "split2", monkeypatch, REUSE, resume=True)


# ---------------------------------------------------------------- summaries and isolation

def test_task_return_summary_uses_each_recorded_column():
    dataset = Dataset(Path("unused"), cache_in_ram=True, save_on_disk=False)
    returns = []
    for scale in (1.0, 2.0, 4.0):
        rew = scale * torch.arange(5.)
        task_rew = torch.stack([rew, -rew + 1], 1)
        dataset.add_episode(Episode(torch.zeros(5, 3, 8, 8), torch.zeros(5, 2), rew, torch.zeros(5, dtype=torch.long),
                                    torch.zeros(5, dtype=torch.long), {}, task_rew=task_rew, task_names=TASKS))
        returns.append(task_rew.sum(0).tolist())
    stand, walk = np.array(returns).T  # stand: 10, 20, 40; walk: -5, -15, -35
    summary = task_return_summary(dataset)
    assert summary == {"return_stand/mean": stand.mean(), "return_stand/std": stand.std(), "return_stand/min": 10.0,
                       "return_stand/max": 40.0, "return_walk/mean": walk.mean(), "return_walk/std": walk.std(),
                       "return_walk/min": -35.0, "return_walk/max": -5.0}
    assert task_return_summary(Dataset(Path("unused"), cache_in_ram=True, save_on_disk=False)) == {}


def test_trainer_return_summaries_match_the_test_episodes(tmp_path, monkeypatch):
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + UNIFORM + FRESH)
    run(trainer)
    final = [row for _, row in logs if "final_test/fresh" in row][-1]
    episodes = [trainer.test_dataset.load_episode(i) for i in range(3)]
    walk = np.array([ep.task_rew[:, 1].double().sum().item() for ep in episodes])
    stand = np.array([ep.rew.double().sum().item() for ep in episodes])  # the first task drives collection
    assert final["final_test/return_walk/mean"] == pytest.approx(walk.mean(), abs=1e-12)
    assert final["final_test/return_stand/max"] == pytest.approx(stand.max(), abs=1e-6)
    # The fake env labels walk = 3 * stand + 1 per step (6 steps): a real per-column sum.
    assert walk == pytest.approx(3 * stand + 6, abs=1e-4)


def test_test_data_stays_out_of_training_replay_and_milestones(tmp_path, monkeypatch):
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + UNIFORM + FRESH + ["milestones.at_steps=[12,24]"])
    run(trainer)
    test_dir = Path("dataset/test").resolve()
    train_files = [p.resolve() for p in trainer.train_dataset.episode_files()]
    milestone_files = [p.resolve() for m in Path("milestones").glob("steps_*") for p in referenced_files(m)]
    assert len(milestone_files) > 0 and trainer.train_dataset.num_steps == 24
    assert not any(test_dir in p.parents for p in train_files + milestone_files)
    assert trainer._data_loader_train.denoiser.dataset is trainer.train_dataset
    assert trainer._data_loader_test.denoiser.dataset is trainer.test_dataset


def test_diagnostics_do_not_change_training_and_restore_modes_and_rng(tmp_path, monkeypatch):
    # Learned exploration (LCG + PMPO): with and without fresh diagnostics every epoch.
    off, logs_off = start(tmp_path / "off", monkeypatch, BASE + OFF)
    run(off)
    on, logs_on = start(tmp_path / "on", monkeypatch, BASE + FRESH)
    run(on)
    assert on.agent.actor_critic.updates.item() == off.agent.actor_critic.updates.item() == 3
    assert_same_training(off, on)
    assert training_rows(logs_off) == training_rows(logs_on)

    # A direct call: mixed submodule modes, global RNG and weights are left as they were.
    on.agent.train()
    on.agent.denoiser.inner_model.eval()
    modes = {name: m.training for name, m in on.agent.named_modules()}
    weights = {k: v.clone() for k, v in on.agent.state_dict().items() if torch.is_tensor(v)}
    before = rng_state()
    test_rng = on.test_rng
    on.test_diagnostics()
    assert {name: m.training for name, m in on.agent.named_modules()} == modes
    after = rng_state()
    assert all(repr(before[k]) == repr(after[k]) for k in ("python", "numpy", "torch"))
    assert all(torch.equal(v, on.agent.state_dict()[k]) for k, v in weights.items())
    assert repr(on.test_rng) != repr(test_rng)  # the test stream advanced instead


# ---------------------------------------------------------------- resume

class Crash(Exception):
    pass


def crash_at_test_agent(monkeypatch, epoch):
    original = ExplorationTrainer.test_agent

    def crash(self):
        if self.epoch == epoch:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "test_agent", crash)


def test_fresh_resume_matches_uninterrupted_and_keeps_committed_test_files(tmp_path, monkeypatch):
    full, logs_full = start(tmp_path / "full", monkeypatch, BASE + UNIFORM + FRESH)
    run(full)

    # Crash after epoch 2's fresh test collection wrote new files, before its commit.
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, _ = start(tmp_path / "split", monkeypatch, BASE + UNIFORM + FRESH)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.undo()
    saved = torch.load(tmp_path / "split/checkpoints/state.pt", weights_only=False)
    assert saved["epoch"] == 1 and saved["num_test_steps"] == 12
    committed = Dataset(tmp_path / "split/dataset/test")
    committed.load_state_dict(saved["test_dataset"])
    assert committed.generation == 1 and all(p.is_file() for p in committed.episode_files())
    assert crashed.test_dataset.generation == 2  # the uncommitted collection went elsewhere

    resumed, logs_resumed = start(tmp_path / "split", monkeypatch, [], resume=True)
    run(resumed)
    assert_same_training(full, resumed)
    assert (resumed.num_test_steps, resumed.num_episodes_test, resumed.final_test_done) == (42, 7, True)
    assert repr(resumed.test_rng) == repr(full.test_rng)
    for i in range(3):
        x, y = full.test_dataset.load_episode(i), resumed.test_dataset.load_episode(i)
        assert torch.equal(x.obs, y.obs) and torch.equal(x.task_rew, y.task_rew)

    def after_1(logs):
        return [(e, {k: v for k, v in r.items() if k != "duration" and not k.startswith("timing/") and np.ndim(v) == 0})
                for e, r in logs
                if e > 1 and not any(k.startswith("resume/") for k in r)]

    assert repr(after_1(logs_full)) == repr(after_1(logs_resumed))
    # A finished run resumed again repeats no final diagnostics.
    calls = spy_calls(monkeypatch)
    again, _ = start(tmp_path / "split", monkeypatch, [], resume=True)
    run(again)
    assert calls == [] and again.num_test_steps == 42
