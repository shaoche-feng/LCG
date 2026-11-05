"""Step 5 (part 2): real-task evaluation and faithful downstream checkpoint/resume."""
from copy import deepcopy
import json
import os
import re
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import downstream_trainer as downstream_module
from coroutines.frame_history import FrameHistory
from downstream_trainer import (DownstreamTrainer, evaluate_policy, evaluate_saved_output, frame_tensor,
                                recover_metrics_log)
from envs import DMControlEnv
from test_downstream_trainer import FakeWandb, TINY_DOWNSTREAM, build_milestone, compose, tree_hashes
from utils import configure_progress, rng_state

RESUMABLE = [o for o in TINY_DOWNSTREAM if not o.startswith(("rew_end_model.training.steps", "actor_critic.training.steps"))]
RESUMABLE += ["rew_end_model.training.steps=6", "actor_critic.training.steps=4", "checkpointing.every=2"]


class Crash(Exception):
    pass


def make(milestone, monkeypatch, extra=(), resume=False, task="walk"):
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    cfg = compose("downstream", [f"initialization.milestone_path={Path(milestone).as_posix()}",
                                 f"downstream.task_name={task}", *RESUMABLE, *extra])
    return DownstreamTrainer(cfg, resume=resume)


def identical(x, y, where="state"):
    if torch.is_tensor(x):
        assert torch.is_tensor(y) and x.dtype == y.dtype and torch.equal(x, y), where
    elif isinstance(x, np.ndarray):
        assert np.array_equal(x, y), where
    elif isinstance(x, dict):
        assert x.keys() == y.keys(), where
        for k in x:
            identical(x[k], y[k], f"{where}.{k}")
    elif isinstance(x, (list, tuple)):
        assert len(x) == len(y), where
        for i, (u, v) in enumerate(zip(x, y)):
            identical(u, v, f"{where}[{i}]")
    else:
        assert x == y, where


def training_state(t):
    """Everything training depends on, including global RNG and every submodule's mode. An
    independent copy: later in-place updates cannot change a snapshot."""
    return deepcopy(dict(phase=t.phase, num_updates=dict(t.num_updates),
                rew_end_model=t.agent.rew_end_model.state_dict(), actor_critic=t.agent.actor_critic.state_dict(),
                denoiser=t.agent.denoiser.state_dict(), rew_end_opt=t.rew_end_opt.state_dict(),
                rew_end_lr_sched=t.rew_end_lr_sched.state_dict(), actor_critic_opt=t.actor_critic_opt.state_dict(),
                rng=rng_state(), modes={name: m.training for name, m in t.agent.named_modules()},
                requires_grad=[p.requires_grad for p in t.agent.parameters()]))


def last_rows(path):
    rows = [json.loads(line) for line in Path(path).read_text().splitlines()]
    return {(r["phase"], r["update"]): r for r in rows if "update" in r}, rows


# ---------------------------------------------------------------- evaluation

@pytest.mark.parametrize("actor_distribution", ["tanh_gaussian", "beta"])
def test_evaluation_protocol_task_actions_history_and_returns(tmp_path, monkeypatch, actor_distribution):
    trainer = make(build_milestone(tmp_path), monkeypatch,
                   [f"agent.actor_critic.actor_distribution={actor_distribution}"])
    model, env_cfg = trainer.agent.actor_critic, trainer.cfg.env.train
    inputs = []
    handle = model.actor.register_forward_pre_hook(lambda m, args: inputs.append(args[0].clone()))
    result = evaluate_policy(model, env_cfg, "walk", [5, 6], max_steps=7)
    assert result["policy"] == ("deterministic (tanh(mu))" if actor_distribution == "tanh_gaussian"
                                else "deterministic (Beta mean)")
    handle.remove()
    assert [e["seed"] for e in result["episodes"]] == [5, 6] and result["task_name"] == "walk"
    for i, episode in enumerate(result["episodes"]):
        assert episode["length"] == 7 and episode["step_limit"] and not episode["terminated"] and not episode["truncated"]
        first = inputs[7 * i]  # a new episode's stack repeats its first frame
        assert all(torch.equal(first[:, :3], first[:, 3 * k:3 * k + 3]) for k in range(4))

    # Independent replay: Walk rewards (action-repeat sums) with the same deterministic actions.
    for episode in result["episodes"]:
        env = DMControlEnv("walker", ["walk"], size=8, camera_id=0, action_repeat=2, seed=episode["seed"])
        image, _ = env.reset(seed=episode["seed"])
        history, total = FrameHistory(frame_tensor(image, model.device), 4), 0.0
        for _ in range(7):
            with torch.no_grad():
                action = model.sample_action(model.actor(history.state), deterministic=True)
            image, reward, *_ = env.step(action[0].numpy())
            total += reward
            history.advance(frame_tensor(image, model.device), torch.tensor([False]))
        env.close()
        assert episode["episode_return"] == total
    returns = np.array([e["episode_return"] for e in result["episodes"]])
    assert (result["return_mean"], result["return_median"], result["return_std"], result["return_min"],
            result["return_max"]) == (returns.mean(), np.median(returns), returns.std(), returns.min(), returns.max())
    stand = evaluate_policy(model, env_cfg, "stand", [5, 6], max_steps=7)
    assert stand["return_mean"] != result["return_mean"]  # the requested task's reward
    assert evaluate_policy(model, env_cfg, "walk", [5, 6], max_steps=7) == result  # deterministic

    long = evaluate_policy(model, env_cfg, "walk", [7], max_steps=600)  # Walker: 500 env steps
    assert long["episodes"][0]["length"] == 500 and long["episodes"][0]["truncated"]


def test_evaluation_changes_no_training_state_and_not_the_next_update(tmp_path, monkeypatch):
    extra = ["rew_end_model.training.steps=3", "actor_critic.training.steps=2", "evaluation.seeds=[5,6]",
             "evaluation.max_steps=5"]

    def run(name, probe):
        """Two reward updates and two PMPO updates; with probe, evaluate before each second one.
        Runs one after the other: each trainer re-seeds the process-wide RNG when built."""
        trainer = make(build_milestone(tmp_path / name), monkeypatch, extra)
        sources = tree_hashes(tmp_path / name / "run/milestones", tmp_path / name / "run/dataset")
        snapshots = []
        trainer.agent.rew_end_model.train()
        trainer.reward_model_update()
        if probe:
            before = training_state(trainer)
            trainer.evaluate()
            identical(before, training_state(trainer), "after evaluation in the reward phase")
        trainer.reward_model_update()
        snapshots.append(training_state(trainer))
        trainer.reward_model_update()
        trainer.agent.rew_end_model.requires_grad_(False).eval()
        trainer.agent.actor_critic.train()
        trainer.policy_update()
        if probe:
            before = training_state(trainer)
            trainer.evaluate()
            identical(before, training_state(trainer), "after evaluation in the PMPO phase")
        trainer.policy_update()
        snapshots.append(training_state(trainer))
        assert trainer.milestone.dataset.num_steps == 26  # no data added
        assert tree_hashes(tmp_path / name / "run/milestones", tmp_path / name / "run/dataset") == sources
        return snapshots

    plain, probed = run("plain", probe=False), run("probed", probe=True)
    identical(plain[0], probed[0], "next reward update")
    identical(plain[1], probed[1], "next PMPO update")


def test_saved_policy_evaluation_matches_in_memory(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    protocol = ["evaluation.after_training=true", "evaluation.seeds=[5,6]", "evaluation.max_steps=6"]
    trainer = make(milestone, monkeypatch, protocol)
    output = trainer.run()
    assert not (output / "checkpoints").exists()
    (saved,) = (output / "evaluations").iterdir()
    in_memory = trainer.evaluate()
    stored = OmegaConf.to_container(OmegaConf.load(saved))
    assert {k: v for k, v in stored.items() if k not in ("policy_sha256_prefix", "protocol_id", "evaluation_kind")} == in_memory
    assert stored["evaluation_kind"] == "after_training"
    assert stored["protocol_id"] == downstream_module.protocol_id(in_memory["seeds"], in_memory["max_steps"])
    assert not any(row["phase"] == "evaluation" for row in last_rows(output / "metrics.jsonl")[1])

    cfg = compose("downstream", [f"initialization.milestone_path={milestone.as_posix()}", "downstream.task_name=walk",
                                 "downstream.mode=evaluate", *RESUMABLE, *protocol])
    with pytest.raises(FileExistsError, match="already exists; not overwriting"):
        evaluate_saved_output(cfg)  # same policy and protocol
    cfg.evaluation.seeds = [8]
    other = OmegaConf.to_container(OmegaConf.load(evaluate_saved_output(cfg)))
    trainer.cfg.evaluation.seeds = [8]
    assert other["episodes"] == trainer.evaluate()["episodes"]


# ---------------------------------------------------------------- resume

def run_uninterrupted(tmp_path, monkeypatch):
    trainer = make(build_milestone(tmp_path / "full"), monkeypatch)
    output = trainer.run()
    return trainer, rng_state(), output


@pytest.mark.parametrize("crash", ["reward_phase", "phase_transition", "pmpo_phase"])
def test_resume_matches_uninterrupted_training(tmp_path, monkeypatch, capsys, crash):
    full, full_rng, full_output = run_uninterrupted(tmp_path, monkeypatch)
    milestone = build_milestone(tmp_path / "split")
    sources = tree_hashes(tmp_path / "split/run/milestones", tmp_path / "split/run/dataset")
    original_reward, original_policy = DownstreamTrainer.reward_model_update, DownstreamTrainer.policy_update

    def reward_update(self):
        original_reward(self)
        if crash == "reward_phase" and self.num_updates["rew_end_model"] == 3:  # checkpoint at 2
            raise Crash()

    def policy_update(self):
        if crash == "phase_transition":
            raise Crash()  # after the transition checkpoint, before any PMPO update
        original_policy(self)
        if self.num_updates["actor_critic"] == 3:  # checkpoint at 2
            raise Crash()

    monkeypatch.setattr(DownstreamTrainer, "reward_model_update", reward_update)
    monkeypatch.setattr(DownstreamTrainer, "policy_update", policy_update)
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original_policy)
    reward_calls = []
    monkeypatch.setattr(DownstreamTrainer, "reward_model_update",
                        lambda self: reward_calls.append(1) or original_reward(self))

    resumed = make(milestone, monkeypatch, resume=True)
    expected_phase = {"reward_phase": "rew_end_model", "phase_transition": "actor_critic", "pmpo_phase": "actor_critic"}
    assert resumed.phase == expected_phase[crash]
    capsys.readouterr()
    output = resumed.run()
    resumed_rng = rng_state()
    # Bars start at the restored counts; a phase completed before the crash gets none.
    out = capsys.readouterr().out
    starts = {desc: re.search(rf"{desc}: +\d+%\|[^|]*\| (\d+/\d+) ", out) for desc in ("Reward-predictor training", "Task-policy training")}
    expected = {"reward_phase": ("2/6", "0/4"), "phase_transition": (None, "0/4"), "pmpo_phase": (None, "2/4")}[crash]
    assert tuple(m and m[1] for m in starts.values()) == expected
    assert out.rstrip("\n").rsplit("\n", 1)[-1] == f"Downstream outputs saved: > {output.resolve()}"
    if crash != "reward_phase":
        assert not reward_calls  # a resumed PMPO phase does not retrain the reward model
    identical(training_state(full), training_state(resumed), "final training state")
    identical(full_rng, resumed_rng, "global RNG progression")
    for name in ("rew_end_model.pt", "actor_critic.pt"):
        identical(torch.load(full_output / name, weights_only=False), torch.load(output / name, weights_only=False), name)
    full_rows, _ = last_rows(full_output / "metrics.jsonl")
    resumed_rows, all_rows = last_rows(output / "metrics.jsonl")
    assert resumed_rows == full_rows  # every update's batch losses and PMPO metrics
    replayed = [(r["phase"], r["update"]) for r in all_rows if "update" in r]
    if crash != "phase_transition":
        assert len(replayed) == len(set(replayed)) + 1  # update 3 logged before the crash and again
    assert any(r["phase"] == "resume" for r in all_rows)
    assert tree_hashes(tmp_path / "split/run/milestones", tmp_path / "split/run/dataset") == sources


def test_terminal_output_changes_no_training_state_metrics_or_evaluation(tmp_path, monkeypatch, capsys):
    """Presentation only: with progress output off, the same run ends in the same training state and
    global RNG, writes the same metrics log and the same evaluation file."""
    protocol = ["evaluation.after_training=true", "evaluation.seeds=[5,6]", "evaluation.max_steps=4"]
    runs = {}
    for mode in ("bar", "none"):
        configure_progress(mode)
        try:
            trainer = make(build_milestone(tmp_path / mode), monkeypatch, protocol)
            output = trainer.run()
        finally:
            configure_progress("bar")
        (evaluation,) = (output / "evaluations").iterdir()
        runs[mode] = (training_state(trainer), rng_state(), (output / "metrics.jsonl").read_bytes(), evaluation.name,
                      evaluation.read_bytes())
        out = capsys.readouterr().out
        assert ("Task-policy training: 100%" in out) == (mode == "bar") and "Evaluation summary: max " in out
        assert "episode_return" not in out and "seed 5" not in out  # no per-seed rows on screen
    identical(runs["bar"][0], runs["none"][0], "training state")
    identical(runs["bar"][1], runs["none"][1], "global RNG")
    assert runs["bar"][2:] == runs["none"][2:]  # metrics.jsonl, evaluation file name and contents


def test_interrupted_checkpoint_write_and_moved_run_folder(tmp_path, monkeypatch):
    full, full_rng, full_output = run_uninterrupted(tmp_path, monkeypatch)
    build_milestone(tmp_path / "split")
    original = downstream_module.save_atomically

    def interrupted(obj, path):
        if obj["num_updates"]["rew_end_model"] == 4:
            Path(path).with_suffix(".tmp").write_bytes(b"partial")  # torch.save cut short
            raise Crash()
        return original(obj, path)

    monkeypatch.setattr(downstream_module, "save_atomically", interrupted)
    with pytest.raises(Crash):
        make(tmp_path / "split/run/milestones/steps_000026", monkeypatch).run()
    monkeypatch.setattr(downstream_module, "save_atomically", original)
    moved = Path(shutil.move(str(tmp_path / "split"), str(tmp_path / "moved")))
    resumed = make(moved / "run/milestones/steps_000026", monkeypatch, resume=True)
    assert resumed.num_updates == {"rew_end_model": 2, "actor_critic": 0}
    output = resumed.run()
    identical(training_state(full), training_state(resumed), "final training state")
    identical(full_rng, rng_state(), "global RNG progression")
    assert last_rows(output / "metrics.jsonl")[0] == last_rows(full_output / "metrics.jsonl")[0]


def test_resume_rejects_missing_weights_only_and_incompatible_inputs(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    with pytest.raises(FileNotFoundError, match="No resumable downstream checkpoint"):
        make(milestone, monkeypatch, resume=True)
    make(milestone, monkeypatch)  # an attempt that never reached a checkpoint
    with pytest.raises(FileNotFoundError, match="No resumable downstream checkpoint"):
        make(milestone, monkeypatch, resume=True)
    with pytest.raises(FileExistsError, match="resume it with downstream.mode=resume"):
        make(milestone, monkeypatch)
    shutil.rmtree(tmp_path / "run/downstream")

    original = DownstreamTrainer.policy_update
    monkeypatch.setattr(DownstreamTrainer, "policy_update", lambda self: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    with pytest.raises(ValueError, match="different settings: \\['rew_end_model.optimizer.lr'\\]"):
        make(milestone, monkeypatch, ["rew_end_model.optimizer.lr=0.5"], resume=True)
    path = tmp_path / "run/downstream/steps_000026/walk/.seed_0.tmp/checkpoints/state.pt"
    saved = torch.load(path, weights_only=False)
    for change, match in ((lambda s: s.pop("format"), "not a downstream training checkpoint"),
                          (lambda s: s["milestone"]["identity"].update(epoch=99), "different milestone"),
                          (lambda s: s["source"]["packages"].update(torch="0.0"), "package versions changed")):
        state = dict(saved, milestone=dict(saved["milestone"], identity=dict(saved["milestone"]["identity"])),
                     source=dict(saved["source"], packages=dict(saved["source"]["packages"])))
        change(state)
        torch.save(state, path)
        with pytest.raises(ValueError, match=match):
            make(milestone, monkeypatch, resume=True)
    torch.save(saved, path)
    make(milestone, monkeypatch, resume=True).run()

    # A completed (or earlier weights-only) output is never resumed or overwritten, but evaluates.
    with pytest.raises(FileExistsError, match="already exists; not overwriting"):
        make(milestone, monkeypatch, resume=True)
    output = tmp_path / "run/downstream/steps_000026/walk/seed_0"
    for name in ("config.yaml",):
        assert (output / name).is_file()
    cfg = compose("downstream", [f"initialization.milestone_path={milestone.as_posix()}", "downstream.task_name=walk",
                                 "downstream.mode=evaluate", "evaluation.seeds=[9]", "evaluation.max_steps=3", *RESUMABLE])
    assert evaluate_saved_output(cfg).is_file()


# ---------------------------------------------------------------- review fixes

def test_evaluation_restores_every_submodule_mode_even_on_error(tmp_path, monkeypatch):
    trainer = make(build_milestone(tmp_path), monkeypatch, ["evaluation.seeds=[5]", "evaluation.max_steps=3"])
    model = trainer.agent.actor_critic
    model.train()
    model.prior_actor.eval()  # frozen snapshots in eval inside a training-mode PMPO model
    model.initial_actor.eval()
    modes = {name: m.training for name, m in model.named_modules()}
    assert True in modes.values() and False in modes.values()
    grads, rng = [p.requires_grad for p in model.parameters()], deepcopy(rng_state())
    trainer.evaluate()
    assert {name: m.training for name, m in model.named_modules()} == modes
    assert [p.requires_grad for p in model.parameters()] == grads

    def broken_step(self, action):
        raise RuntimeError("environment failed")

    monkeypatch.setattr(DMControlEnv, "step", broken_step)
    with pytest.raises(RuntimeError, match="environment failed"):
        trainer.evaluate()
    assert {name: m.training for name, m in model.named_modules()} == modes
    identical(rng, rng_state(), "global RNG after a failed evaluation")


def test_resume_rejects_a_valid_replacement_milestone_with_other_rewards(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path / "split")
    monkeypatch.setattr(DownstreamTrainer, "policy_update", lambda self: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.undo()
    # Same lengths, denoiser and counters; different walk rewards; correct checksums.
    replacement = build_milestone(tmp_path / "other", column_offsets=(0.0, 0.5, 0.0))
    for name in ("milestones", "dataset"):
        shutil.rmtree(tmp_path / "split/run" / name)
        shutil.copytree(tmp_path / "other/run" / name, tmp_path / "split/run" / name)
    from data.milestone import load_milestone
    loaded = load_milestone(milestone, "walk")  # ordinary integrity validation passes
    saved = torch.load(tmp_path / "split/run/downstream/steps_000026/walk/.seed_0.tmp/checkpoints/state.pt",
                       weights_only=False)
    assert loaded.manifest["identity"] == saved["milestone"]["identity"]
    with pytest.raises(ValueError, match="different milestone replay"):
        make(milestone, monkeypatch, resume=True)


def test_format_1_checkpoints_are_rejected(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    monkeypatch.setattr(DownstreamTrainer, "policy_update", lambda self: (_ for _ in ()).throw(Crash()))
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.undo()
    path = tmp_path / "run/downstream/steps_000026/walk/.seed_0.tmp/checkpoints/state.pt"
    state = torch.load(path, weights_only=False)
    state["format"] = 1
    del state["milestone"]["dataset_sha256"]
    torch.save(state, path)
    with pytest.raises(ValueError, match="predates the replay content checksum"):
        make(milestone, monkeypatch, resume=True)


def test_metrics_log_recovery_cases(tmp_path):
    good = [json.dumps(dict(phase="rew_end_model", update=i, loss=0.5 * i)) for i in (1, 2)]
    path = tmp_path / "metrics.jsonl"
    path.write_text("".join(row + "\n" for row in good), encoding="utf-8")
    original = path.read_bytes()
    fragment = b'{"phase": "rew_end_model", "upd'
    path.write_bytes(original + fragment)
    recovery = recover_metrics_log(path)
    assert recovery.name == "metrics.jsonl.interrupted-1" and recovery.read_bytes() == fragment
    assert path.read_bytes() == original  # complete rows untouched
    assert recover_metrics_log(path) is None and path.read_bytes() == original  # repeatable
    path.write_bytes(original + fragment)
    assert recover_metrics_log(path).name == "metrics.jsonl.interrupted-2"  # never overwrites a recovery

    complete = json.dumps(dict(phase="rew_end_model", update=3, loss=1.0)).encode()
    path.write_bytes(original + complete)  # a complete record that lost only its newline
    assert recover_metrics_log(path) is None
    assert [json.loads(line) for line in path.read_text().splitlines()][-1]["update"] == 3

    path.write_bytes(original + b"{oops\n" + complete + b"\n")  # corruption inside the log
    with pytest.raises(ValueError, match="row 3 is corrupt"):
        recover_metrics_log(path)


@pytest.mark.parametrize("damage", ["missing", "cut_tail", "corrupt_row", "readable_change", "non_boundary"])
def test_resume_refuses_damaged_committed_metrics(tmp_path, monkeypatch, damage):
    milestone = build_milestone(tmp_path)
    trainer = make(milestone, monkeypatch)
    trainer.reward_model_update()
    trainer.save_checkpoint()
    path = trainer._work_dir / "metrics.jsonl"
    data = path.read_bytes()
    state = torch.load(trainer._path_checkpoint, weights_only=False)
    assert state["metrics_bytes"] == len(data) and data.endswith(b"\n")
    if damage == "missing":
        path.unlink()
    elif damage == "cut_tail":
        path.write_bytes(data[:len(data) // 2])
    elif damage == "corrupt_row":
        path.write_bytes(b"!" + data[1:])
    elif damage == "readable_change":
        altered = data.replace(b'"update": 1', b'"update": 9', 1)
        assert altered != data and len(altered) == len(data)
        assert json.loads(altered)["update"] == 9
        path.write_bytes(altered)
    else:
        state["metrics_bytes"] -= 1
        torch.save(state, trainer._path_checkpoint)
    before = path.read_bytes() if path.exists() else None
    with pytest.raises(ValueError, match="committed.*(history|corrupt)"):
        make(milestone, monkeypatch, resume=True)
    assert (path.read_bytes() if path.exists() else None) == before
    assert not list(path.parent.glob("metrics.jsonl.interrupted-*"))


def test_metrics_boundary_fsync_precedes_checkpoint_and_tail_repair_is_idempotent(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    trainer = make(milestone, monkeypatch)
    trainer.reward_model_update()
    events = []
    fsync, save = downstream_module.os.fsync, downstream_module.save_atomically
    def synced(fd):
        events.append("fsync")
        fsync(fd)
    def publish(state, path):
        events.append("checkpoint")
        assert state["metrics_bytes"] == (trainer._work_dir / "metrics.jsonl").stat().st_size
        save(state, path)
    with monkeypatch.context() as patch:
        patch.setattr(downstream_module.os, "fsync", synced)
        patch.setattr(downstream_module, "save_atomically", publish)
        trainer.save_checkpoint()
    assert events.index("fsync") < events.index("checkpoint")
    path = trainer._work_dir / "metrics.jsonl"
    committed = path.read_bytes()
    fragment = b'{"phase": "rew_end_model", "update": 2'
    path.write_bytes(committed + fragment)
    resumed = make(milestone, monkeypatch, resume=True)
    assert resumed.num_updates == dict(rew_end_model=1, actor_critic=0)
    assert path.read_bytes().startswith(committed)
    assert path.with_name("metrics.jsonl.interrupted-1").read_bytes() == fragment
    repaired = path.read_bytes()
    assert recover_metrics_log(path, len(committed)) is None and path.read_bytes() == repaired


@pytest.mark.parametrize("legacy_format", [2, 3])
def test_resume_explicitly_refuses_checkpoints_without_metrics_boundary(tmp_path, monkeypatch, legacy_format):
    milestone = build_milestone(tmp_path)
    trainer = make(milestone, monkeypatch)
    trainer.save_checkpoint()
    state = torch.load(trainer._path_checkpoint, weights_only=False)
    state["format"] = legacy_format
    state.pop("metrics_bytes")
    torch.save(state, trainer._path_checkpoint)
    with pytest.raises(ValueError, match="predates the committed metrics boundary"):
        make(milestone, monkeypatch, resume=True)


def test_resume_after_an_interrupted_metrics_row(tmp_path, monkeypatch):
    full, full_rng, full_output = run_uninterrupted(tmp_path, monkeypatch)
    milestone = build_milestone(tmp_path / "split")
    original_policy = DownstreamTrainer.policy_update

    def policy_update(self):
        original_policy(self)
        if self.num_updates["actor_critic"] == 3:  # checkpoint at 2; row 3 is newer than it
            raise Crash()

    monkeypatch.setattr(DownstreamTrainer, "policy_update", policy_update)
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original_policy)
    log = tmp_path / "split/run/downstream/steps_000026/walk/.seed_0.tmp/metrics.jsonl"
    history = log.read_bytes()
    fragment = b'{"phase": "actor_critic", "update": 4, "loss_ac'  # the write a crash cut short
    log.write_bytes(history + fragment)

    resumed = make(milestone, monkeypatch, resume=True)
    output = resumed.run()
    recovered = output / "metrics.jsonl.interrupted-1"
    assert recovered.read_bytes() == fragment
    repaired = (output / "metrics.jsonl").read_bytes()
    assert repaired.startswith(history)  # every complete historical row, incl. update 3
    rows = [json.loads(line) for line in repaired.decode().splitlines()]  # every row parses
    assert any(r["phase"] == "resume" for r in rows)
    identical(training_state(full), training_state(resumed), "final training state")
    identical(full_rng, rng_state(), "global RNG progression")
    assert last_rows(output / "metrics.jsonl")[0] == last_rows(full_output / "metrics.jsonl")[0]

# ---------------------------------------------------------------- Gaussian H10 EMA-critic controller

def test_resume_restores_the_slow_critic_and_refuses_another_controller_before_any_weights(tmp_path, monkeypatch):
    milestone = build_milestone(tmp_path)
    original_policy = DownstreamTrainer.policy_update

    def policy_update(self):
        original_policy(self)
        if self.num_updates["actor_critic"] == 3:  # checkpoint at 2
            raise Crash()

    monkeypatch.setattr(DownstreamTrainer, "policy_update", policy_update)
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original_policy)
    path = tmp_path / "run/downstream/steps_000026/walk/.seed_0.tmp/checkpoints/state.pt"
    saved = torch.load(path, weights_only=False)
    assert saved["actor_critic"]["_extra_state"]["controller"] == dict(
        actor_distribution="tanh_gaussian", slow_critic=dict(coef=1.0, rate=0.02),
        imagination_history="repeat_anchor", imagination_horizon=2)
    resumed = make(milestone, monkeypatch, resume=True)
    restored = resumed.agent.actor_critic.state_dict()
    for k, v in saved["actor_critic"].items():
        if torch.is_tensor(v):
            assert torch.equal(restored[k].cpu(), v.cpu()), k
    assert any(k.startswith("slow_value.") for k in saved["actor_critic"])
    assert not states_equal_modules(resumed.agent.actor_critic.slow_value, resumed.agent.actor_critic.value)
    # A state saved by an older (Beta, unrecorded) controller under the same settings is refused
    # before the reward model or the controller is loaded.
    legacy = deepcopy(saved)
    del legacy["actor_critic"]["_extra_state"]["controller"]
    torch.save(legacy, path)
    loaded = []
    monkeypatch.setattr(torch.nn.Module, "load_state_dict", lambda self, *a, **k: loaded.append(type(self).__name__))
    with pytest.raises(ValueError, match="saved before controller settings were recorded"):
        make(milestone, monkeypatch, resume=True)
    assert "RewEndModel" not in loaded and "PMPOBeta" not in loaded


def states_equal_modules(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


# ---------------------------------------------------------------- W&B logging and resume

def wandb_run(milestone, monkeypatch, fake, extra=(), resume=False):
    monkeypatch.setattr(downstream_module, "wandb", fake)
    return make(milestone, monkeypatch, ["wandb.mode=online", *extra], resume=resume)


def test_logging_and_debug_flags_change_no_training(tmp_path, monkeypatch):
    """W&B on with both debug flags versus W&B off: the same training state, global RNG, weights,
    evaluation and metrics.jsonl values (debug adds keys only)."""
    protocol = ["evaluation.after_training=true", "evaluation.seeds=[5,6]", "evaluation.max_steps=4"]
    plain = make(build_milestone(tmp_path / "plain"), monkeypatch, protocol)
    plain_output = plain.run()
    plain_rng = rng_state()
    logged = wandb_run(build_milestone(tmp_path / "logged"), monkeypatch, FakeWandb(),
                       [*protocol, "rew_end_model.debug_metrics=true", "agent.actor_critic.debug_metrics=true"])
    logged_output = logged.run()
    logged_rng = rng_state()

    a, b = training_state(plain), training_state(logged)
    assert a["actor_critic"]["_extra_state"]["fixed_obs"] is None and b["actor_critic"]["_extra_state"]["fixed_obs"] is not None
    for state in (a, b):
        state["actor_critic"]["_extra_state"].pop("fixed_obs")
    identical(a, b, "training state")
    identical(plain_rng, logged_rng, "global RNG")
    for name in ("rew_end_model.pt", "actor_critic.pt"):
        x, y = (torch.load(o / name, weights_only=False) for o in (plain_output, logged_output))
        for state in (x, y):
            state.get("_extra_state", {}).pop("fixed_obs", None)
        identical(x, y, name)
    (a,), (b,) = (plain_output / "evaluations").iterdir(), (logged_output / "evaluations").iterdir()
    assert a.name == b.name and a.read_bytes() == b.read_bytes()
    plain_rows, logged_rows = (last_rows(o / "metrics.jsonl")[0] for o in (plain_output, logged_output))
    assert plain_rows.keys() == logged_rows.keys()
    for key, row in plain_rows.items():
        assert all(logged_rows[key][k] == v for k, v in row.items()), key  # debug adds keys, changes none
        assert set(logged_rows[key]) > set(row)


@pytest.mark.parametrize("server", ["equal", "lost_rows", "beyond_checkpoint", "beyond_mismatch", "beyond_unverified",
                                    "offline"])
def test_resume_continues_the_wandb_run_without_duplicate_updates(tmp_path, monkeypatch, capsys, server):
    """A crash at PMPO update 3 (checkpoint at 2: 6 + 2 = 8 committed rows) and a resume. Whatever the
    server kept, every update appears once in the run, with the uninterrupted run's values."""
    reference = FakeWandb()
    wandb_run(build_milestone(tmp_path / "full"), monkeypatch, reference).run()
    milestone = build_milestone(tmp_path / "split")
    original = DownstreamTrainer.policy_update

    def crash(self):
        original(self)
        if self.num_updates["actor_critic"] == 3:
            raise Crash()

    first = FakeWandb()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", crash)
    with pytest.raises(Crash):
        wandb_run(milestone, monkeypatch, first).run()
    assert [s for s, _ in first.logged] == list(range(9)) and first.summary["status"] == "failed" and first.finished == [1]
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    if server in ("beyond_mismatch", "beyond_unverified"):
        # The interrupted segment's local record of PMPO update 3 (W&B step 8, on the server): changed (as if
        # the replay came out differently) or missing (cut by the crash).
        log = tmp_path / "split/run/downstream/steps_000026/walk/.seed_0.tmp/metrics.jsonl"
        rows = [json.loads(line) for line in log.read_text().splitlines()]
        assert (rows[-1]["phase"], rows[-1]["update"]) == ("actor_critic", 3)
        if server == "beyond_mismatch":
            rows[-1]["loss_actor"] += 1.0
        else:
            rows.pop()
        log.write_text("".join(json.dumps(r) + "\n" for r in rows))
    capsys.readouterr()

    held = dict(equal=8, lost_rows=5, beyond_checkpoint=9, beyond_mismatch=9, beyond_unverified=9, offline=0)[server]
    second = FakeWandb(starting_step=held, offline=server == "offline", summary=dict(status="running"))
    output = wandb_run(milestone, monkeypatch, second, resume=True).run()
    run_id = json.loads((tmp_path / "split/run/downstream/wandb_walk_seed_0.json").read_text())["identity"]["id"]
    assert second.inits[0]["id"] == run_id and second.inits[0]["resume"] == "allow"  # the same run
    event = second.summary["resume"]
    assert (event["committed_rows"], event["previous_status"], event["count"]) == (8, "running", 1)

    server_rows = {s: r for s, r in first.logged if s < held} if server != "offline" else {}
    combined = dict(server_rows)
    for step, row in second.logged:
        if server != "offline":
            assert step not in combined, f"step {step} logged twice"
        combined[step] = row
    expected = {s: r for s, r in reference.logged if server != "offline" or s >= 8}  # offline: this segment's rows
    assert combined == expected  # every update once, with the uninterrupted run's values
    if server == "lost_rows":
        assert event["backfilled_rows"] == 3 and [s for s, _ in second.logged[:3]] == [5, 6, 7]
    out = capsys.readouterr().out
    if server.startswith("beyond"):  # the replayed update is compared, never logged twice
        outcome = dict(beyond_checkpoint="matched", beyond_mismatch="mismatched", beyond_unverified="unverified")[server]
        assert event["server_rows_beyond_checkpoint"] == 1
        assert {k: v for k, v in event.items() if k.startswith("replayed_")} == {f"replayed_{outcome}": 1}
        assert 8 not in [s for s, _ in second.logged]
    warning = "W&B warning: replayed actor_critic update 3 (W&B step 8) differs from the value the interrupted segment sent"
    assert (warning in out) == (server == "beyond_mismatch")
    if server == "offline":
        assert "nothing backfilled" in event["note"] and [s for s, _ in second.logged][0] == 8
    assert second.summary["status"] == "completed" and json.loads((output / "wandb_run.json").read_text())["id"] == run_id


def test_resume_of_a_run_without_wandb_starts_one_and_backfills_its_committed_updates(tmp_path, monkeypatch):
    """A fenced checkpoint without W&B and with older optional metric columns absent."""
    milestone = build_milestone(tmp_path)
    original = DownstreamTrainer.policy_update
    original_log_row = DownstreamTrainer.log_row
    old_keys = ("predicted_reward_mean", "target_reward_mean", "task_reward_mean", "task_reward_max")

    def older_columns(self, row):
        # Construct the older row schema before checkpointing it. Rewriting committed
        # bytes afterwards is now correctly refused as corruption.
        original_log_row(self, {k: v for k, v in row.items() if k not in old_keys})

    def crash(self):
        original(self)
        if self.num_updates["actor_critic"] == 3:
            raise Crash()

    monkeypatch.setattr(DownstreamTrainer, "policy_update", crash)
    monkeypatch.setattr(DownstreamTrainer, "log_row", older_columns)
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    monkeypatch.setattr(DownstreamTrainer, "log_row", original_log_row)
    work = tmp_path / "run/downstream/steps_000026/walk/.seed_0.tmp"
    path = work / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert all(not set(row).intersection(old_keys) for row in rows)
    fake = FakeWandb()
    wandb_run(milestone, monkeypatch, fake, resume=True).run()
    event = fake.summary["resume"]
    assert event["previous_status"] == "no earlier W&B run" and event["backfilled_rows"] == 8
    assert [s for s, _ in fake.logged] == list(range(10))  # 8 committed rows backfilled, then PMPO updates 3 and 4


def test_resume_needs_the_milestone_manifest_json(tmp_path, monkeypatch):
    """A resume reads the milestone's manifest.json like a fresh run; without it the resume stops with a
    clear error and the work folder is kept."""
    milestone = build_milestone(tmp_path)
    original = DownstreamTrainer.policy_update

    def crash(self):
        original(self)
        if self.num_updates["actor_critic"] == 3:
            raise Crash()

    monkeypatch.setattr(DownstreamTrainer, "policy_update", crash)
    with pytest.raises(Crash):
        make(milestone, monkeypatch).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    manifest = milestone / "manifest.json"
    saved = manifest.read_bytes()
    manifest.unlink()
    work = tmp_path / "run/downstream/steps_000026/walk/.seed_0.tmp"
    with pytest.raises(FileNotFoundError, match="No milestone manifest at .*manifest.json"):
        make(milestone, monkeypatch, resume=True)
    assert (work / "checkpoints/state.pt").is_file()
    manifest.write_bytes(saved)
    output = make(milestone, monkeypatch, resume=True).run()
    assert json.loads((output / "summary.json").read_text())["num_updates"] == {"rew_end_model": 6, "actor_critic": 4}


# ---------------------------------------------------------------- the shared W&B run: resume and publication

EVALUATED = ["evaluation.after_training=true", "evaluation.seeds=[3001]", "evaluation.max_steps=3"]


def crash_at_policy_update(monkeypatch, n):
    original = DownstreamTrainer.policy_update

    def crash(self):
        original(self)
        if self.num_updates["actor_critic"] == n:
            raise Crash()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", crash)
    return original


def shared_rows(root, server):
    record = json.loads((Path(root) / "run/downstream/wandb_walk_seed_0.json").read_text())
    return record, server[record["identity"]["id"]]["rows"]


def test_resume_in_a_shared_run_continues_the_milestone_block(tmp_path, monkeypatch):
    """Milestone 20 completes; milestone 26 crashes at PMPO update 3 (checkpoint at 2) and is resumed. Every row
    is in the one shared run once, in increasing steps, equal to an uninterrupted reference."""
    def adapt(root, server, crash=False):
        milestones = build_milestone(root, early=(2,)).parent
        wandb_run(milestones / "steps_000020", monkeypatch, FakeWandb(server=server), EVALUATED).run()
        if not crash:
            return wandb_run(milestones / "steps_000026", monkeypatch, FakeWandb(server=server), EVALUATED).run()
        original = crash_at_policy_update(monkeypatch, 3)
        with pytest.raises(Crash):
            wandb_run(milestones / "steps_000026", monkeypatch, FakeWandb(server=server), EVALUATED).run()
        monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
        second = FakeWandb(server=server)
        output = wandb_run(milestones / "steps_000026", monkeypatch, second, EVALUATED, resume=True).run()
        event = second.summary["resume"]
        # 6 + 2 committed rows from step 11; the server holds update 3 too (replayed, not logged again).
        assert (event["block_first_step"], event["committed_rows"], event["server_rows_beyond_checkpoint"]) == (11, 8, 1)
        assert event["replayed_matched"] == 1
        return output

    reference_server, server = {}, {}
    adapt(tmp_path / "reference", reference_server)
    output = adapt(tmp_path / "split", server, crash=True)
    (_, reference), (record, rows) = shared_rows(tmp_path / "reference", reference_server), shared_rows(tmp_path / "split", server)
    assert [s for s, _ in rows] == list(range(22))  # 20: steps 0-9 + evaluation 10; 26: steps 11-20 + evaluation 21
    for (step, row), (ref_step, ref_row) in zip(rows, reference):
        assert step == ref_step and set(row) == set(ref_row)
        assert all(row[k] == ref_row[k] for k in row if not k.startswith(("eval/", "eval_protocols/"))), step  # same values as uninterrupted
    block = record["milestones"]["steps_000026"]
    assert (block["first_step"], block["reward_offset"], block["policy_offset"], block["attempt"]) == (11, 6, 4, 1)
    reward = [r for _, r in rows if downstream_module.REWARD_AXIS in r]
    assert [r[downstream_module.REWARD_CUMULATIVE_AXIS] for r in reward] == list(range(1, 13))
    assert [r[downstream_module.REWARD_AXIS] for r in reward] == list(range(1, 7)) * 2  # fresh model per milestone
    assert [r["exploration/train_env_steps"] for _, r in rows if any(k.startswith(("eval/", "eval_protocols/")) for k in r)] == [20, 26]
    assert json.loads((output / "summary.json").read_text())["num_updates"] == {"rew_end_model": 6, "actor_critic": 4}


def evaluate_only(milestone, monkeypatch, fake, seeds, *extra):
    """An evaluate-only command (another protocol) attaching to the shared run through `fake`."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(downstream_module, "wandb", fake)
    cfg = compose("downstream", [f"initialization.milestone_path={Path(milestone).as_posix()}", "downstream.task_name=walk",
                                 "downstream.mode=evaluate", *RESUMABLE, "wandb.mode=online", f"evaluation.seeds={seeds}",
                                 "evaluation.max_steps=3", *extra])
    return evaluate_saved_output(cfg)


def points(rows, report):
    """Steps of the server's points of one evaluation file (its protocol and training env steps)."""
    result = json.loads(Path(report).read_text())
    record = json.loads((Path(report).parents[5] / "downstream/wandb_walk_seed_0.json").read_text())
    namespace = downstream_module.evaluation_namespace(report, (record.get("primary_protocol") or {}).get("id"))
    return [s for s, r in rows if r.get(f"{namespace}/return_mean") == result["return_mean"]]


def entry_of(root, report):
    record = json.loads((Path(root) / "run/downstream/wandb_walk_seed_0.json").read_text())
    return record["evaluations"][Path(os.path.relpath(report, Path(root) / "run/downstream")).as_posix()]


def is_evaluation(step, row):
    return any(k.startswith(("eval/", "eval_protocols/")) for k in row)


@pytest.mark.parametrize("interrupted", ["before_record", "after_record", "after_log"])
def test_interrupted_evaluation_publication_is_completed_exactly_once(tmp_path, monkeypatch, interrupted):
    """The after-training evaluation's publication is interrupted: before its step is recorded, after that but
    before logging, or after logging but before the local record is updated. The next process attaching to the
    shared run checks the server's history: a point that is there is confirmed, a missing one is logged again;
    the point is in the run exactly once. A second protocol is a separate series."""
    milestone = build_milestone(tmp_path)
    server = {}
    first = FakeWandb(server=server)
    if interrupted == "before_record":
        def interrupt(*args, **kwargs):
            raise Crash()
        monkeypatch.setattr(downstream_module, "publish_evaluation", interrupt)
    elif interrupted == "after_record":
        def log(row, step=None):
            if any(k.startswith(("eval/", "eval_protocols/")) for k in row):
                raise Crash()
            FakeWandb.log(first, row, step)
        first.log = log
    else:
        original_write = downstream_module.write_json

        def write(path, data):
            if Path(path).name.startswith("wandb_") and any(e.get("status") == "logged" for e in data["evaluations"].values()):
                raise Crash()
            original_write(path, data)
        monkeypatch.setattr(downstream_module, "write_json", write)
    with pytest.raises(Crash):
        wandb_run(milestone, monkeypatch, first, EVALUATED).run()
    monkeypatch.undo()
    output = tmp_path / "run/downstream/steps_000026/walk/seed_0"
    (after_training,) = (output / "evaluations").iterdir()
    record = json.loads((tmp_path / "run/downstream/wandb_walk_seed_0.json").read_text())
    entry = record["evaluations"].get(f"steps_000026/walk/seed_0/evaluations/{after_training.name}")
    assert entry == dict(step=10, original_step=10, kind="after_training", status="assigned",
                         attempted=interrupted != "before_record", namespace="eval", protocol=record["primary_protocol"])

    second = FakeWandb(server=server)
    path = evaluate_only(milestone, monkeypatch, second, "[5,6]")
    record, rows = shared_rows(tmp_path, server)
    # 6 + 4 updates at steps 0-9, then the reserved step 10; without a recorded step: the next free one (11).
    expected = 10  # the intent and reserved step now survive even a crash before publication registration
    assert points(rows, after_training) == [expected] and points(rows, path) == [expected + 1]
    assert second.summary["evaluation_publication"]["confirmed"] == (1 if interrupted == "after_log" else 0)
    assert second.summary["evaluation_publication"]["recovered"] == (0 if interrupted == "after_log" else 1)
    assert {evaluation_status(e) for e in record["evaluations"].values()} == {"confirmed"}  # both seen on the server
    assert entry_of(tmp_path, after_training)["kind"] == "after_training"


def evaluation_status(entry):
    return downstream_module.evaluation_status(entry)


def test_a_logged_evaluation_lost_before_upload_is_recovered_once(tmp_path, monkeypatch, capsys):
    """Regression (1, 3): wandb.log returns and the logged state is saved, but the row never reaches the server
    (the process died before upload). It is not confirmed; the next attachment finds it missing in the server's
    history and logs it again, once; further attachments find it there and change nothing."""
    milestone = build_milestone(tmp_path)
    server = {}
    lost = FakeWandb(server=server, lose=is_evaluation)
    wandb_run(milestone, monkeypatch, lost, EVALUATED).run()
    (report,) = (tmp_path / "run/downstream/steps_000026/walk/seed_0/evaluations").iterdir()
    assert [s for s, r in lost.logged if is_evaluation(s, r)] == [10]  # wandb.log returned
    record, rows = shared_rows(tmp_path, server)
    assert points(rows, report) == [] and entry_of(tmp_path, report)["status"] == "unverified"

    capsys.readouterr()
    evaluate_only(milestone, monkeypatch, FakeWandb(server=server), "[5]")
    assert "logged the evaluation" in capsys.readouterr().out
    record, rows = shared_rows(tmp_path, server)
    assert points(rows, report) == [10] and entry_of(tmp_path, report)["status"] == "confirmed"
    shared = dict(path=tmp_path / "run/downstream/wandb_walk_seed_0.json", record=record)
    for _ in range(2):  # repeated recovery: idempotent
        again = FakeWandb(server=server)
        monkeypatch.setattr(downstream_module, "wandb", again)
        again.init(id=record["identity"]["id"])
        assert downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False) == dict(
            confirmed=0, recovered=0, unverified=0, history="nothing to verify")
        assert not again.logged
    evaluate_only(milestone, monkeypatch, FakeWandb(server=server), "[7]")
    record, rows = shared_rows(tmp_path, server)
    assert points(rows, report) == [10] and len([s for s, r in rows if is_evaluation(s, r)]) == 3


def test_an_evaluation_already_on_the_server_is_confirmed_not_duplicated(tmp_path, monkeypatch):
    """Regression (2): the point reached the server but its confirmation could not be read (history unavailable
    after the run); the next attachment finds it and confirms it without logging it again."""
    milestone = build_milestone(tmp_path)
    server = {}
    wandb_run(milestone, monkeypatch, FakeWandb(server=server, api_error=ConnectionError("no API")), EVALUATED).run()
    (report,) = (tmp_path / "run/downstream/steps_000026/walk/seed_0/evaluations").iterdir()
    assert entry_of(tmp_path, report)["status"] == "unverified"
    second = FakeWandb(server=server)
    evaluate_only(milestone, monkeypatch, second, "[5]")
    record, rows = shared_rows(tmp_path, server)
    assert points(rows, report) == [10] and entry_of(tmp_path, report)["status"] == "confirmed"
    assert second.summary["evaluation_publication"] == dict(confirmed=1, recovered=0, unverified=0, history="server")


def test_a_missing_evaluation_behind_later_server_rows_is_logged_at_the_next_free_step(tmp_path, monkeypatch):
    """Regression (4): the evaluation of milestone 20 never reached the server, and while it could not be
    verified, milestone 26 logged later rows. A later server step is no proof: once the history is readable the
    point is found missing and logged again at the next free step (W&B steps only increase), exactly once."""
    milestones = build_milestone(tmp_path, early=(2,)).parent
    server = {}
    wandb_run(milestones / "steps_000020", monkeypatch, FakeWandb(server=server, lose=is_evaluation), EVALUATED).run()
    unavailable = FakeWandb(server=server, api_error=ConnectionError("no API"))
    wandb_run(milestones / "steps_000026", monkeypatch, unavailable, EVALUATED).run()
    assert unavailable.summary["evaluation_publication"]["unverified"] == 1  # not confirmed, not logged again
    assert unavailable.summary["evaluation_publication"]["history"].startswith("unavailable (ConnectionError")
    (report,) = (tmp_path / "run/downstream/steps_000020/walk/seed_0/evaluations").iterdir()
    record, rows = shared_rows(tmp_path, server)
    assert points(rows, report) == [] and max(s for s, _ in rows) == 21  # 26's block (11-21) is on the server
    evaluate_only(milestones / "steps_000026", monkeypatch, FakeWandb(server=server), "[5]")
    record, rows = shared_rows(tmp_path, server)
    entry = entry_of(tmp_path, report)
    assert points(rows, report) == [22] and (entry["step"], entry["recovered_from_step"], entry["status"]) == (22, 10, "confirmed")
    assert [r["exploration/train_env_steps"] for s, r in rows if is_evaluation(s, r)] == [26, 20, 26]


def test_offline_evaluations_are_never_claimed_confirmed(tmp_path, monkeypatch):
    """Offline the server's history cannot be read: logged evaluations stay unconfirmed and are not logged
    again by later offline attachments; the summary says why."""
    milestone = build_milestone(tmp_path)
    offline = FakeWandb(offline=True)
    wandb_run(milestone, monkeypatch, offline, [*EVALUATED, "wandb.mode=offline"]).run()
    (report,) = (tmp_path / "run/downstream/steps_000026/walk/seed_0/evaluations").iterdir()
    assert entry_of(tmp_path, report)["status"] == "unverified"
    again = FakeWandb(offline=True)
    evaluate_only(milestone, monkeypatch, again, "[5]", "wandb.mode=offline")
    assert entry_of(tmp_path, report)["status"] == "unverified" and len(again.logged) == 1  # only its own point
    assert again.summary["evaluation_publication"]["unverified"] == 1
    assert again.summary["evaluation_publication"]["history"].startswith("offline")


def test_resume_after_later_steps_in_the_shared_run_keeps_its_rows_local(tmp_path, monkeypatch, capsys):
    """Milestone 26 crashes; before it is resumed, evaluate-only adds a point for milestone 20 after the block of
    26. W&B steps only increase, so the remaining updates of 26 are not logged (they stay in metrics.jsonl); its
    evaluation is logged at the next free step. Nothing is logged out of order or twice."""
    milestones = build_milestone(tmp_path, early=(2,)).parent
    server = {}
    wandb_run(milestones / "steps_000020", monkeypatch, FakeWandb(server=server), EVALUATED).run()
    original = crash_at_policy_update(monkeypatch, 3)
    with pytest.raises(Crash):
        wandb_run(milestones / "steps_000026", monkeypatch, FakeWandb(server=server), EVALUATED).run()
    monkeypatch.setattr(DownstreamTrainer, "policy_update", original)
    cfg = compose("downstream", [f"initialization.milestone_path={(milestones / 'steps_000020').as_posix()}",
                                 "downstream.task_name=walk", "downstream.mode=evaluate", *RESUMABLE, "wandb.mode=online",
                                 "evaluation.seeds=[8]", "evaluation.max_steps=3"])
    monkeypatch.setattr(downstream_module, "wandb", FakeWandb(server=server))
    evaluate_saved_output(cfg)  # at step 22, after the reserved block of 26 (steps 11-21)
    capsys.readouterr()
    second = FakeWandb(server=server)
    output = wandb_run(milestones / "steps_000026", monkeypatch, second, EVALUATED, resume=True).run()
    assert "the shared run already holds later steps" in capsys.readouterr().out
    assert "later steps" in second.summary["resume"]["note"]
    record, rows = shared_rows(tmp_path, server)
    steps = [s for s, _ in rows]
    assert steps == sorted(set(steps)) and steps[-2:] == [22, 23]  # nothing below the server's step; no duplicates
    assert [r["exploration/train_env_steps"] for _, r in rows if any(k.startswith(("eval/", "eval_protocols/")) for k in r)] == [20, 20, 26]
    # PMPO update 4 of 26 is local only; the server has updates 1-3 from before the crash.
    assert [r[downstream_module.POLICY_AXIS] for _, r in rows if r.get("exploration/train_env_steps") == 26
            and downstream_module.POLICY_AXIS in r] == [1, 2, 3]
    assert json.loads((output / "summary.json").read_text())["num_updates"] == {"rew_end_model": 6, "actor_critic": 4}


# Independent review regressions (CPU, fake W&B only).
def publication_case(tmp_path, monkeypatch):
    cfg = OmegaConf.create(dict(wandb=dict(project='review', entity='offline-test', name=None, group=None),
                                downstream=dict(task_name='walk',seed=0),
                                evaluation=dict(after_training=False,seeds=[3001,3002],max_steps=3)))
    path = tmp_path/'run/downstream/wandb_walk_seed_0.json'
    shared = dict(path=path,record=downstream_module.open_shared_run(cfg,path,'review','review'))
    server = {}
    attach_publication_case(shared,server,monkeypatch)
    milestone = SimpleNamespace(path=tmp_path/'run/milestones/steps_000020',
                                manifest=dict(identity={'review':'fixed'},collected_steps=20))
    downstream_module.register_milestone(shared,milestone,6,4,True,0)
    return cfg,shared,server,milestone


def attach_publication_case(shared,server,monkeypatch):
    fake=FakeWandb(server=server)
    monkeypatch.setattr(downstream_module,'wandb',fake)
    fake.init(**shared['record']['identity'])
    return fake


def publication_report(milestone,seeds=(3001,3002),mean=0.25):
    output=downstream_module.downstream_output_dir(milestone.path,'walk',0)
    output.mkdir(parents=True,exist_ok=True)
    result=dict(task_name='walk',seeds=list(seeds),max_steps=3,
                episodes=[dict(seed=s,episode_return=mean) for s in seeds],
                return_mean=mean,return_max=mean,return_median=mean,return_min=mean,return_std=0.)
    return downstream_module.save_evaluation(output,result,{}),result


def publication_server_rows(shared,server):
    return server[shared['record']['identity']['id']]['rows']


def test_late_primary_selection_is_durable_before_wandb_log(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    path,result=publication_report(m)
    original=downstream_module.wandb.log
    def interrupted(row,step=None):
        original(row,step)
        raise Crash('uploaded before local logged-state write')
    monkeypatch.setattr(downstream_module.wandb,'log',interrupted)
    with pytest.raises(Crash):
        downstream_module.publish_evaluation(shared,path,result,20,'after_training',reserved=10)
    resumed=dict(path=shared['path'],record=downstream_module.read_json(shared['path'],'shared record'))
    attach_publication_case(resumed,server,monkeypatch)
    event=downstream_module.reconcile_evaluations(resumed,downstream_module.server_next_step(),False)
    assert len(publication_server_rows(resumed,server)) == 1, (event,publication_server_rows(resumed,server))


def test_recovering_file_before_publication_establishes_late_primary(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    cfg.evaluation.after_training=True
    output=downstream_module.downstream_output_dir(m.path,'walk',0)
    output.mkdir(parents=True)
    OmegaConf.save(cfg,output/'config.yaml')
    path,result=publication_report(m)  # after-training result saved; crash before publish_evaluation
    downstream_module.reconcile_evaluations(shared,downstream_module.server_next_step(),False)
    expected=downstream_module.protocol_id(result['seeds'],result['max_steps'])
    assert shared['record']['primary_protocol'] == dict(id=expected,seeds=result['seeds'],max_steps=3)


def test_older_record_preserves_first_after_training_primary(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    first,result=publication_report(m)
    downstream_module.publish_evaluation(shared,first,result,20,'after_training',reserved=10)
    for entry in shared['record']['evaluations'].values():
        entry['status']='confirmed'
    expected=shared['record'].pop('primary_protocol')  # older shared schema-1 record
    downstream_module.write_json(shared['path'],shared['record'])
    resumed=dict(path=shared['path'],record=downstream_module.open_shared_run(cfg,shared['path'],'review','review'))
    attach_publication_case(resumed,server,monkeypatch)
    downstream_module.reconcile_evaluations(resumed,downstream_module.server_next_step(),False)
    second=SimpleNamespace(path=m.path.with_name('steps_000030'),manifest=dict(identity={'review':'second'},collected_steps=30))
    downstream_module.register_milestone(resumed,second,6,4,True,downstream_module.server_next_step())
    other,result2=publication_report(second,seeds=(5,6))
    downstream_module.publish_evaluation(resumed,other,result2,30,'after_training')
    assert resumed['record']['primary_protocol'] == expected


def test_pre_primary_evaluate_only_namespace_survives_primary_selection(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    path,result=publication_report(m)
    downstream_module.publish_evaluation(shared,path,result,20,'evaluate_only')
    unavailable=attach_publication_case(shared,server,monkeypatch)
    unavailable.api_error=ConnectionError('history unavailable during the next training attachment')
    event=downstream_module.reconcile_evaluations(shared,downstream_module.server_next_step(),False)
    assert event['unverified']==1
    second=SimpleNamespace(path=m.path.with_name('steps_000030'),manifest=dict(identity={'review':'second'},collected_steps=30))
    downstream_module.register_milestone(shared,second,6,4,True,downstream_module.server_next_step())
    path2,result2=publication_report(second)
    downstream_module.publish_evaluation(shared,path2,result2,30,'after_training')
    attach_publication_case(shared,server,monkeypatch)
    event=downstream_module.reconcile_evaluations(shared,downstream_module.server_next_step(),False)
    first_rows=[row for _,row in publication_server_rows(shared,server) if row[downstream_module.EVAL_AXIS]==20]
    assert len(first_rows)==1,(event,publication_server_rows(shared,server))


def test_recovery_does_not_log_updates_below_its_new_live_step(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    cfg.evaluation.after_training=True
    shared['record']['primary_protocol']=downstream_module.configured_protocol(cfg)
    path,result=publication_report(m)
    downstream_module.publish_evaluation(shared,path,result,20,'after_training',reserved=10)
    publication_server_rows(shared,server).clear()  # that evaluation was queued, never uploaded
    second=SimpleNamespace(path=m.path.with_name('steps_000030'),manifest=dict(identity={'review':'second'},collected_steps=30))
    block=downstream_module.register_milestone(shared,second,6,4,True,11)
    publication_server_rows(shared,server).append((18,{'training':1.}))  # server holds next block through PMPO update 2
    fake=attach_publication_case(shared,server,monkeypatch)
    downstream_module.reconcile_evaluations(shared,downstream_module.server_next_step(),False)  # older evaluation recovered at step 22
    trainer=object.__new__(downstream_module.DownstreamTrainer)
    trainer.cfg=OmegaConf.create(dict(wandb=dict(mode='online'),rew_end_model=dict(training=dict(steps=6)),
                                    actor_critic=dict(training=dict(steps=4)),
                                    agent=dict(actor_critic=dict(actor_distribution='tanh_gaussian'))))
    trainer._block=block; trainer._blocked=False; trainer._replayed=None
    trainer.num_updates=dict(rew_end_model=6,actor_critic=2); trainer.phase='actor_critic'
    trainer._work_dir=tmp_path/'work'; trainer._work_dir.mkdir()
    (trainer._work_dir/'metrics.jsonl').write_text('')
    trainer.reconcile_wandb({'id':fake.run.id})
    trainer.publish(dict(phase='actor_critic',update=3,loss_actor=0.50))
    assert publication_server_rows(shared,server)[-1][0]==22


def test_confirmed_recovery_restores_missing_protocol_summary(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    shared['record']['primary_protocol']=downstream_module.configured_protocol(cfg)
    downstream_module.write_json(shared['path'],shared['record'])
    path,result=publication_report(m)
    original=downstream_module.wandb.log
    def interrupted(row,step=None):
        original(row,step)
        raise Crash('point uploaded; summary not yet published')
    monkeypatch.setattr(downstream_module.wandb,'log',interrupted)
    with pytest.raises(Crash):
        downstream_module.publish_evaluation(shared,path,result,20,'after_training',reserved=10)
    resumed=dict(path=shared['path'],record=downstream_module.read_json(shared['path'],'record'))
    fake=attach_publication_case(resumed,server,monkeypatch)
    event=downstream_module.reconcile_evaluations(resumed,downstream_module.server_next_step(),False)
    assert event['confirmed']==1 and len(publication_server_rows(resumed,server))==1
    assert 'evaluation_protocols/'+downstream_module.protocol_label(path) in fake.summary


def test_protocol_identifier_normalizes_supported_callers():
    seeds=[3001,3002]
    cfg=OmegaConf.create(dict(evaluation=dict(seeds=seeds,max_steps=500,after_training=True)))
    expected=downstream_module.protocol_id(seeds,500)
    assert downstream_module.protocol_id(tuple(seeds),500)==expected
    assert downstream_module.configured_protocol(cfg)['id']==expected
    assert downstream_module.evaluation_protocol(cfg.evaluation)['protocol']==expected
    path=downstream_module.evaluation_path(Path('.'),'walk',cfg.evaluation.seeds,cfg.evaluation.max_steps,{})
    assert downstream_module.protocol_label(path)=='protocol-'+expected
    assert downstream_module.protocol_id(list(reversed(seeds)),500)!=expected  # ordered protocol, consistently
    assert downstream_module.protocol_id(seeds,499)!=expected


def test_namespace_axes_both_use_exploration_steps(monkeypatch):
    fake=FakeWandb();monkeypatch.setattr(downstream_module,'wandb',fake)
    downstream_module.define_axes()
    assert ('eval/*','exploration/train_env_steps') in fake.metrics
    assert ('eval_protocols/*','exploration/train_env_steps') in fake.metrics


def test_matching_protocols_at_same_budget_and_mean_stay_separate(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    shared['record']['primary_protocol']=downstream_module.configured_protocol(cfg)
    first,a=publication_report(m)
    downstream_module.publish_evaluation(shared,first,a,20,'after_training',reserved=10)
    second,b=publication_report(m,seeds=(5,6))
    downstream_module.publish_evaluation(shared,second,b,20,'evaluate_only')
    fake=attach_publication_case(shared,server,monkeypatch)
    result=downstream_module.reconcile_evaluations(shared,downstream_module.server_next_step(),False)
    assert result['confirmed']==2 and result['recovered']==0 and not fake.logged
    assert len(publication_server_rows(shared,server))==2


def test_legacy_missing_protocol_metadata_uses_mtime_then_filename(tmp_path,monkeypatch):
    cfg,shared,server,m=publication_case(tmp_path,monkeypatch)
    first,a=publication_report(m)
    second,b=publication_report(m,seeds=(5,6))
    for path in (first,second):
        r=downstream_module.read_json(path,'report');r.pop('protocol_id');downstream_module.write_json(path,r)
    os.utime(first,ns=(1_000_000_000,1_000_000_000))
    os.utime(second,ns=(2_000_000_000,2_000_000_000))
    downstream_module.log_evaluation(a,first,20,'after_training',10,0)
    attach_publication_case(shared,server,monkeypatch)  # evaluate-only is a new process/run attachment
    downstream_module.log_evaluation(b,second,20,'evaluate_only',10,0)
    held=publication_server_rows(shared,server)
    assert 'eval/return_mean' in held[0][1]
    assert 'eval_protocols/'+downstream_module.protocol_label(second)+'/return_mean' in held[1][1]


@pytest.mark.parametrize("hidden_reads", [1, 2, 10])
def test_delayed_history_is_retried_without_duplicate_publication(tmp_path, monkeypatch, hidden_reads, capsys):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    path, result = publication_report(milestone)
    downstream_module.publish_evaluation(shared, path, result, 20, "after_training", reserved=10)
    fake = attach_publication_case(shared, server, monkeypatch)
    read_history = downstream_module.server_history
    calls = []

    def delayed(*args):
        calls.append(1)
        return ({}, "server") if len(calls) <= hidden_reads else read_history(*args)

    monkeypatch.setattr(downstream_module, "server_history", delayed)
    monkeypatch.setattr(downstream_module, "HISTORY_RETRY_SECONDS", 0)
    before = rng_state()
    event = downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    identical(before, rng_state())
    assert len(calls) == min(hidden_reads + 1, downstream_module.HISTORY_ATTEMPTS)
    assert not fake.logged and len(publication_server_rows(shared, server)) == 1
    entry = next(iter(shared["record"]["evaluations"].values()))
    if hidden_reads < downstream_module.HISTORY_ATTEMPTS:
        assert event["confirmed"] == 1 and entry["status"] == "confirmed"
    else:
        assert event["unverified"] == 1 and entry["status"] == "unverified"
        assert "unverified" in capsys.readouterr().out
        monkeypatch.setattr(downstream_module, "server_history", read_history)
        event = downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
        assert event["confirmed"] == 1 and not fake.logged


def test_delayed_evaluation_with_visible_later_rows_is_not_republished(tmp_path, monkeypatch):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    path, result = publication_report(milestone)
    downstream_module.publish_evaluation(shared, path, result, 20, "after_training", reserved=10)
    publication_server_rows(shared, server).append((18, {"later_training": 1.0}))
    fake = attach_publication_case(shared, server, monkeypatch)
    read_history = downstream_module.server_history
    calls = []

    def delayed(*args):
        history, source = read_history(*args)
        calls.append(1)
        if len(calls) < downstream_module.HISTORY_ATTEMPTS:
            history.pop(10)
        return history, source

    monkeypatch.setattr(downstream_module, "server_history", delayed)
    monkeypatch.setattr(downstream_module, "HISTORY_RETRY_SECONDS", 0)
    event = downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    assert event["confirmed"] == 1 and event["recovered"] == 0
    assert len(calls) == downstream_module.HISTORY_ATTEMPTS and not fake.logged
    assert [s for s, _ in publication_server_rows(shared, server)] == [10, 18]


def test_confirmed_summaries_are_rebuilt_on_every_attachment(tmp_path, monkeypatch):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    path, result = publication_report(milestone)
    downstream_module.publish_evaluation(shared, path, result, 20, "after_training", reserved=10)
    downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    fake = attach_publication_case(shared, server, monkeypatch)
    fake.summary.clear()
    event = downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    assert event["history"] == "nothing to verify" and not fake.logged
    assert fake.summary["evaluation_protocols/primary"] == shared["record"]["primary_protocol"]
    assert fake.summary["evaluation_protocols/" + downstream_module.protocol_label(path)]["namespace"] == "eval"
    assert fake.summary["evaluations/" + path.stem]["step"] == 10


def test_primary_migration_uses_original_step_and_published_namespace(tmp_path, monkeypatch):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    first, result = publication_report(milestone)
    downstream_module.publish_evaluation(shared, first, result, 20, "after_training", reserved=10)
    second, other = publication_report(milestone, seeds=(5, 6))
    downstream_module.publish_evaluation(shared, second, other, 20, "after_training")
    entries = list(shared["record"]["evaluations"].values())
    entries[0].update(step=25, recovered_from_step=10, status="confirmed")
    entries[1]["status"] = "confirmed"
    old_namespace = entries[1]["namespace"]
    expected = shared["record"].pop("primary_protocol")
    downstream_module.write_json(shared["path"], shared["record"])
    migrated = downstream_module.open_shared_run(cfg, shared["path"], "review", "review")
    assert migrated["primary_protocol"] == expected
    assert list(migrated["evaluations"].values())[1]["namespace"] == old_namespace


@pytest.mark.parametrize("unavailable_reads", [2, 10])
def test_history_api_failures_have_bounded_retries(tmp_path, monkeypatch, unavailable_reads):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    path, result = publication_report(milestone)
    downstream_module.publish_evaluation(shared, path, result, 20, "after_training", reserved=10)
    fake = attach_publication_case(shared, server, monkeypatch)
    read_history = downstream_module.server_history
    calls = []

    def unavailable(*args):
        calls.append(1)
        return (None, "unavailable (ConnectionError: fixture)") if len(calls) <= unavailable_reads else read_history(*args)

    monkeypatch.setattr(downstream_module, "server_history", unavailable)
    monkeypatch.setattr(downstream_module, "HISTORY_RETRY_SECONDS", 0)
    event = downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    assert len(calls) == downstream_module.HISTORY_ATTEMPTS
    assert not fake.logged and len(publication_server_rows(shared, server)) == 1
    assert event["confirmed" if unavailable_reads == 2 else "unverified"] == 1


def test_historical_confirmation_is_not_revoked_by_lagging_namespace_lookup(tmp_path, monkeypatch):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    path, result = publication_report(milestone)
    downstream_module.publish_evaluation(shared, path, result, 20, "after_training", reserved=10)
    downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    entry = next(iter(shared["record"]["evaluations"].values()))
    entry.pop("namespace")
    publication_server_rows(shared, server).append((18, {"later_training": 1.0}))
    fake = attach_publication_case(shared, server, monkeypatch)
    monkeypatch.setattr(downstream_module, "server_history", lambda *args: ({18: {"later_training": 1.0}}, "server"))
    monkeypatch.setattr(downstream_module, "HISTORY_RETRY_SECONDS", 0)
    event = downstream_module.reconcile_evaluations(shared, downstream_module.server_next_step(), False)
    assert entry["status"] == "confirmed" and entry["namespace_inferred"]
    assert event["unverified"] == 1 and not fake.logged
    assert len(publication_server_rows(shared, server)) == 2


def test_after_training_intent_is_saved_before_evaluation_or_report(tmp_path, monkeypatch):
    cfg, shared, server, milestone = publication_case(tmp_path, monkeypatch)
    target = downstream_module.evaluation_path(tmp_path / "output", "walk", cfg.evaluation.seeds, 3, {})
    metadata = dict(seeds=list(cfg.evaluation.seeds), max_steps=3)
    downstream_module.prepare_evaluation(shared, target, metadata, "after_training", 10)
    saved = downstream_module.read_json(shared["path"], "record")
    entry = next(iter(saved["evaluations"].values()))
    assert not target.exists() and not publication_server_rows(shared, server)
    assert entry == dict(step=10, original_step=10, kind="after_training", status="assigned", attempted=False,
                         protocol=saved["primary_protocol"], namespace="eval")
