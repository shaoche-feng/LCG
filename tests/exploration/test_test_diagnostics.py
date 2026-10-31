"""Exploration test collection and diagnostics: evaluation.should / every / collect_test,
no task-return logging, test-interaction accounting, isolation and resume."""
import json
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import exploration_trainer as trainer_module
from data import Dataset, Episode
from data.milestone import referenced_files
from exploration_trainer import check_evaluation_config, ExplorationTrainer
from test_exploration_trainer import COLLECT, compose, FakeEnv, ROOT, TASKS, TINY, UNIFORM
from utils import rng_state


def denoiser_padding_example():
    from hydra.utils import instantiate
    from models.diffusion import Denoiser
    cfg = compose(TINY + ["agent.denoiser.inner_model.num_steps_conditioning=4"])
    denoiser_cfg = instantiate(cfg.agent.denoiser)
    denoiser_cfg.inner_model.continuous_action_dim = 2
    model = Denoiser(denoiser_cfg)
    model.setup_training(instantiate(cfg.denoiser.sigma_distribution))
    obs = torch.linspace(-1, 1, 8)[:, None, None, None].expand(8, 3, 8, 8).clone()
    ep = Episode(obs, torch.zeros(8, 2), torch.zeros(8), torch.zeros(8, dtype=torch.long),
                 torch.zeros(8, dtype=torch.long), {})
    return model, ep


def test_denoiser_eight_frame_traversal_keeps_only_valid_targets(tmp_path):
    from data import DatasetTraverser, collate_segments_without_rewards
    model, ep = denoiser_padding_example()
    dataset = Dataset(tmp_path / "replay", cache_in_ram=True, save_on_disk=False)
    dataset.add_episode(ep)
    traversal = DatasetTraverser(dataset, 1, 6, collate_segments_without_rewards, num_steps_conditioning=4)
    batches = list(traversal)
    assert len(traversal) == len(batches) == 1
    assert batches[0].segment_ids[0].start == 0 and batches[0].segment_ids[0].stop == 6
    assert batches[0].mask_padding[:, 4:].sum() == 2
    assert torch.equal(batches[0].obs[0, 4:], ep.obs[4:6])
    loss, _ = model(batches[0])
    assert torch.isfinite(loss)
    # A dataset whose only chunk has conditioning frames but no targets yields no batches.
    short = Dataset(tmp_path / "short", cache_in_ram=True, save_on_disk=False)
    short.add_episode(Episode(ep.obs[:2], ep.act[:2], ep.rew[:2], ep.end[:2], ep.trunc[:2], {}))
    empty = DatasetTraverser(short, 1, 6, collate_segments_without_rewards, num_steps_conditioning=4)
    assert len(empty) == 0 and list(empty) == []


def test_denoiser_left_padded_single_target_has_finite_undiluted_loss(monkeypatch):
    from data import collate_segments_without_rewards
    from data.segment import SegmentId
    from data.utils import make_segment
    import models.diffusion.denoiser as denoiser_module
    model, ep = denoiser_padding_example()
    batch = collate_segments_without_rewards([make_segment(ep, SegmentId(0, -5, 1))])
    assert batch.mask_padding.tolist() == [[False, False, False, False, False, True]]
    terms = []
    mse = denoiser_module.F.mse_loss
    def capture(prediction, target):
        terms.append((prediction.detach().clone(), target.detach().clone()))
        return mse(prediction, target)
    monkeypatch.setattr(denoiser_module.F, "mse_loss", capture)
    loss, _ = model(batch)
    assert torch.isfinite(loss) and len(terms) == 1 and terms[0][0].shape[0] == 1
    torch.testing.assert_close(loss, mse(*terms[0]), rtol=0, atol=0)
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)


def test_denoiser_rejects_completely_unusable_batch_before_rng_or_forward(monkeypatch):
    from data import collate_segments_without_rewards
    from data.segment import SegmentId
    from data.utils import make_segment
    model, ep = denoiser_padding_example()
    batch = collate_segments_without_rewards([make_segment(ep, SegmentId(0, 6, 12))])
    before = torch.get_rng_state().clone()
    monkeypatch.setattr(model, "compute_model_output", lambda *a: pytest.fail("unusable batch reached the model"))
    with pytest.raises(ValueError, match="no valid prediction targets"):
        model(batch)
    assert torch.equal(torch.get_rng_state(), before)


@pytest.mark.parametrize("split", ["train", "test"])
@pytest.mark.parametrize("damage", ["altered", "missing", "unreadable"])
def test_resume_verifies_checkpoint_replay_once(tmp_path, monkeypatch, split, damage):
    import data.dataset as dataset_module
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + FRESH + UNIFORM + ["training.should=false"])
    trainer._run()
    dataset = getattr(trainer, f"{split}_dataset")
    path = dataset.episode_files()[0].resolve()
    if damage == "altered":
        stored = torch.load(path, weights_only=False)
        shape = stored["obs"].shape
        stored["obs"] = stored["obs"].roll(1, dims=0)
        assert stored["obs"].shape == shape
        assert not torch.equal(stored["obs"], torch.load(path, weights_only=False)["obs"])
        torch.save(stored, path)
    elif damage == "missing":
        path.unlink()
    else:
        original = dataset_module.file_sha256
        def unreadable(file):
            if Path(file).resolve() == path:
                raise PermissionError("test: cannot read episode")
            return original(file)
        monkeypatch.setattr(dataset_module, "file_sha256", unreadable)
    error_type = FileNotFoundError if damage == "missing" else ValueError
    with pytest.raises(error_type, match=f"{split}_dataset:.*(checksum|missing|unreadable)"):
        start(tmp_path / "run", monkeypatch, [], resume=True)


def test_replay_verification_runs_once_per_resume_not_per_update(tmp_path, monkeypatch):
    crash_at_test_agent(monkeypatch, epoch=2)
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + FRESH + UNIFORM)
    with pytest.raises(Crash):
        trainer._run()
    monkeypatch.undo()
    calls, verify = [], Dataset.verify_episode_files
    def checked(dataset):
        calls.append(dataset.name)
        return verify(dataset)
    monkeypatch.setattr(Dataset, "verify_episode_files", checked)
    resumed, _ = start(tmp_path / "run", monkeypatch, [], resume=True)
    assert calls == ["train_dataset", "test_dataset"]
    resumed._run()
    assert calls == ["train_dataset", "test_dataset"]


def test_resume_refuses_unreadable_episode_payload_even_with_matching_digest(tmp_path, monkeypatch):
    from utils import file_sha256
    trainer, _ = start(tmp_path / "run", monkeypatch, BASE + OFF + UNIFORM + ["training.should=false"])
    trainer._run()
    path = trainer.train_dataset.episode_files()[0]
    path.write_bytes(b"unreadable episode payload")
    state = torch.load(trainer._path_state_ckpt, weights_only=False)
    state["train_dataset"]["file_sha256"][0] = file_sha256(path)
    torch.save(state, trainer._path_state_ckpt)
    with pytest.raises(ValueError, match="train_dataset: unreadable checkpoint episode"):
        start(tmp_path / "run", monkeypatch, [], resume=True)

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
        cfg = OmegaConf.load(run_dir / "config/exploration.yaml")
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


def load_evaluation(name):
    return json.loads(Path(f"evaluation/{name}.json").read_text())


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
    """Learning metrics only: training losses and PMPO metrics (incl. sampled actions)."""
    out = []
    for epoch, row in logs:
        values = {k: float(v) for k, v in row.items() if np.ndim(v) == 0
                  and k.split("/")[0] in ("denoiser", "actor_critic") and k.split("/")[1] not in ("test", "final_test")}
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
    # W&B: cumulative test steps on the epoch rows and the final row (the final collection included).
    totals = [(e, r["progress/test_env_steps"]) for e, r in logs
              if "timing/epoch_seconds" in r or "timing/final_test_diagnostic_seconds" in r]
    assert totals == [(1, 12), (2, 24), (2, 42)]
    # Per diagnostic: interactions and fresh status in evaluation/*.json.
    saved = [load_evaluation(name) for name in ("epoch_00001", "epoch_00002", "final_epoch_00002")]
    assert [(s["new_test_env_steps"], s["total_test_env_steps"], s["fresh"]) for s in saved] == \
        [(12, 12, True), (12, 24, True), (18, 42, True)]
    assert (saved[-1]["num_episodes"], saved[-1]["num_steps"]) == (3, 18)
    assert trainer.num_test_steps == 42 and trainer.num_episodes_test == 7 and trainer.final_test_done
    assert len(rows_with(logs, "denoiser/test")) == 2 and len(rows_with(logs, "denoiser/final_test")) == 1
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
    run(resumed)
    assert [c[0] for c in calls] == ["test_agent", "test_agent"]  # epoch 2 and final, no collection
    for name in ("epoch_00002", "final_epoch_00002"):
        saved = load_evaluation(name)
        assert not saved["fresh"] and saved["new_test_env_steps"] == 0 and saved["kind"].startswith("dataset_diagnostic")
        assert saved["total_test_env_steps"] == 12  # epoch 1's collection only
        assert saved["num_episodes"] == 2
    final = [r for _, r in logs if "timing/final_test_diagnostic_seconds" in r]
    assert len(final) == 1 and final[0]["progress/test_env_steps"] == 12
    assert not any(is_task_return(k) for _, row in logs for k in row)
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

def is_task_return(key):
    """Downstream task-return keys (the PMPO imagined return actor_critic/train/return_* is not one)."""
    return key == "return" or key.startswith("final_return") or "/return_" in key and not key.startswith("actor_critic/")


def test_task_returns_are_not_logged_but_test_episodes_keep_task_rewards(tmp_path, monkeypatch, capsys):
    trainer, logs = start(tmp_path / "run", monkeypatch, BASE + UNIFORM + FRESH)
    run(trainer)
    assert logs and not [k for _, row in logs for k in row if is_task_return(k)]
    out = capsys.readouterr().out
    assert "return =" not in out and "final_return" not in out and "task returns" not in out
    # The episodes still carry every task-reward column (downstream adaptation needs them).
    episodes = [trainer.test_dataset.load_episode(i) for i in range(3)]
    for ep in episodes:
        assert ep.task_names == TASKS and ep.task_rew.shape == (len(ep), len(TASKS))
        torch.testing.assert_close(ep.task_rew[:, 0], ep.rew)  # the first task drives collection
    # The fake env labels walk = 3 * stand + 1 per step (6 steps): a real per-column record.
    walk = np.array([ep.task_rew[:, 1].double().sum().item() for ep in episodes])
    stand = np.array([ep.rew.double().sum().item() for ep in episodes])
    assert walk == pytest.approx(3 * stand + 6, abs=1e-4)
    train = trainer.train_dataset.load_episode(0)
    assert train.task_names == TASKS and train.task_rew.shape == (len(train), len(TASKS))


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
