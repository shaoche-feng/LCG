"""Step 5 (part 2): real-task evaluation and faithful downstream checkpoint/resume."""
from copy import deepcopy
import json
import re
import shutil
from pathlib import Path

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

def test_evaluation_protocol_task_actions_history_and_returns(tmp_path, monkeypatch):
    trainer = make(build_milestone(tmp_path), monkeypatch)  # milestone env: walker, first task stand
    model, env_cfg = trainer.agent.actor_critic, trainer.cfg.env.train
    inputs = []
    handle = model.actor.register_forward_pre_hook(lambda m, args: inputs.append(args[0].clone()))
    result = evaluate_policy(model, env_cfg, "walk", [5, 6], max_steps=7)
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
    assert {k: v for k, v in stored.items() if k != "policy_sha256_prefix"} == in_memory
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
    assert second.inits[0]["id"] == "dsrun1" and second.inits[0]["resume"] == "allow"  # the same run
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
    assert second.summary["status"] == "completed" and json.loads((output / "wandb_run.json").read_text())["id"] == "dsrun1"


def test_resume_of_a_run_without_wandb_starts_one_and_backfills_its_committed_updates(tmp_path, monkeypatch):
    """Legacy: a checkpoint written without W&B (and metrics rows without the new reward keys)."""
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
    work = tmp_path / "run/downstream/steps_000026/walk/.seed_0.tmp"
    path = work / "metrics.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    for row in rows:  # as written before these keys existed
        for key in ("predicted_reward_mean", "target_reward_mean", "task_reward_mean", "task_reward_max"):
            row.pop(key, None)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    fake = FakeWandb()
    wandb_run(milestone, monkeypatch, fake, resume=True).run()
    event = fake.summary["resume"]
    assert event["previous_status"] == "no earlier W&B run" and event["backfilled_rows"] == 8
    assert [s for s, _ in fake.logged] == list(range(10))  # 8 committed rows backfilled, then PMPO updates 3 and 4
