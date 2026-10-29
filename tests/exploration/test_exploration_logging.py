"""Exploration-stage logging (terminal, W&B rows and axes) and the real-environment intrinsic
evaluation: seeded test episodes, alignment with each method's imagination scoring path,
isolation from training and reward-normalization state, uniform omissions, logging-on/off
equivalence, counters and resume."""
from copy import deepcopy
import json
from pathlib import Path
import re
import time
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import exploration_trainer as trainer_module
import utils
from data.segment import SegmentId
from data.utils import make_segment
from envs.world_model_env import ImaginedCandidate, WorldModelEnv
from exploration_trainer import ExplorationTrainer
from intrinsic_eval import sampler_windows
from test_ensemble import ENS
from test_exploration_trainer import compose, FakeEnv, ROOT, TINY, UNIFORM
from test_laplace import LAP
from test_plan2explore import P2E
from test_test_diagnostics import assert_same_training, BASE, Crash, crash_at_test_agent, FRESH, REUSE, run, \
    same, training_rows, yaml_value
from laplace import LaplaceEnsemble
from lcg.lifecycle import LCGLifecycle
from models.pmpo_beta import PMPOBeta, PMPOOptimizers
from plan2explore import FeatureExtractor, Plan2Explore
from utils import history_records, RegionTimer, repair_history, rng_state

SEEDS = [3001, 3002, 3003, 3004]
SEEDED = ["evaluation.should=true", "evaluation.every=1", "evaluation.collect_test=true",
          "collection.test.num_episodes=4", f"evaluation.reset_seeds=[{','.join(map(str, SEEDS))}]",
          "collection.test.num_final_episodes=2", "evaluation.final_reset_seeds=[11,12]"]


class SeedEnv(FakeEnv):
    """FakeEnv whose frames depend on the reset seed; records every seeded reset."""
    resets = []

    def reset(self, **kwargs):
        seed = kwargs.get("seed")
        if seed is not None:
            self.offset = (seed[0] % 5) * 0.1
            SeedEnv.resets.append(seed[0])
        return super().reset()

    def frame(self, action=None):
        return (super().frame(action) + getattr(self, "offset", 0.0)).clamp(-1, 1)

    def get_env_states(self):  # the offset is part of the env state a resume restores
        return [dict(t=self.t, offset=getattr(self, "offset", 0.0))]

    def set_env_states(self, states):
        self.t, self.offset = states[0]["t"], states[0]["offset"]


class FakeWandb:
    """Stands in for the wandb module: records init kwargs, axes, config, summary and logs."""

    def __init__(self, run_id="fakerun1", starting_step=0, offline=False):
        self.run, self.run_id, self.starting_step, self.offline = None, run_id, starting_step, offline
        self.inits, self.metrics, self.hidden, self.config_updates, self.logged = [], [], [], [], []
        self.summary = {}
        self.config = SimpleNamespace(update=lambda d, **kw: self.config_updates.append(d))

    def init(self, **kwargs):
        self.inits.append(kwargs)
        self.run = SimpleNamespace(id=kwargs.get("id") or self.run_id, project=kwargs.get("project"), entity=None,
                                   name=kwargs.get("name"), group=kwargs.get("group"), url="https://wandb.test/run",
                                   starting_step=self.starting_step, offline=self.offline)

    def save(self, *args, **kwargs):
        pass

    def define_metric(self, name, step_metric=None, hidden=None):
        self.metrics.append((name, step_metric))
        if hidden:
            self.hidden.append(name)

    def log(self, row):
        self.logged.append(row)


def launch(run_dir, monkeypatch, tiny, overrides, resume=False, fake_wandb=None):
    """ExplorationTrainer on CPU with SeedEnv; logs captured as (epoch, row)."""
    SeedEnv.resets = []
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer_module, "make_dm_control_env", lambda num_envs, device, **kwargs: SeedEnv(device))
    if fake_wandb is None:
        for method in ("init", "save"):
            monkeypatch.setattr(trainer_module.wandb, method, lambda *args, **kwargs: None)
    else:
        monkeypatch.setattr(trainer_module, "wandb", fake_wandb)
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
        cfg = compose(list(tiny) + list(overrides))
        (run_dir / ".hydra").mkdir(parents=True)
        OmegaConf.save(cfg, run_dir / ".hydra/config.yaml")
        monkeypatch.chdir(run_dir)
    return ExplorationTrainer(cfg, ROOT), logs


def rows_of(logs, prefix):
    return [(e, r) for e, r in logs if any(k.startswith(prefix) for k in r)]


def merged(logs, prefix, epoch=None):
    out = {}
    for e, r in logs:
        if epoch is None or e == epoch:
            out.update({k: v for k, v in r.items() if k.startswith(prefix)})
    return out


def scalars(row):
    """Scalar values without wall-clock monitoring (timing/*, duration)."""
    return {k: float(v) for k, v in row.items() if np.ndim(v) == 0 and not k.startswith("timing/") and k != "duration"}


# W&B categories that are no longer logged (old runs keep them).
RETIRED_CATEGORIES = ("duration", "length", "train_dataset", "test_dataset", "epoch_summary", "test", "final_test",
                      "ensemble", "plan2explore", "laplace", "lcg", "classification_metrics")


def eval_keys(block, seeds):
    """The W&B keys of a seeded evaluation block: per-seed raw and scaled returns, their mean and std."""
    kinds = ("intrinsic_raw_return", "intrinsic_scaled_return")
    return ({f"{block}/seed_{s}/{k}" for s in seeds for k in kinds}
            | {f"{block}/{k}_{stat}" for k in kinds for stat in ("mean", "std")})


def pmpo_rows(logs):
    return [(e, r) for e, r in logs if "actor_critic/train/core/num_batch_train_actor_critic" in r]


def terminal_lines(out):
    """What stays on screen: tqdm redraws a bar in place with carriage returns, so only the text
    after a line's last carriage return remains."""
    return [line.rsplit("\r", 1)[-1] for line in out.split("\n")]


# A closed bar: description, 100%, completed/total, [elapsed<remaining, rate], the rate with a
# space before its unit and inverted below one per second ("3.85 s/update").
BAR = re.compile(r"^(?P<desc>.+): 100%\|[^|]*\| (?P<n>\d+)/(?P=n) \[\d+:\d\d<(?:\d+:\d\d|\?), "
                 r"(?:\d+\.\d\d (?P<u1>[a-z]+)/s|\d+\.\d\d s/(?P<u2>[a-z]+)|\? (?P<u3>[a-z]+)/s)\]$")
PLURAL = dict(step="steps", update="updates", trans="trans", ep="ep")


def bar(lines, desc):
    """Final counts ("n/total unit") of every closed progress bar with this description."""
    found = []
    for line in lines:
        match = BAR.match(line)
        if match and match["desc"] == desc:
            unit = match["u1"] or match["u3"] or PLURAL[match["u2"]]
            found.append(f"{match['n']}/{match['n']} {unit}")
    return found


# Blank lines: one before each epoch heading and before the blocks outside epochs (initial
# collection, final evaluation); none between the phases of an epoch.
BLOCK_STARTS = ("=== Epoch ", "Initial collection", "Final evaluation episodes")


def assert_epoch_spacing(lines):
    body = lines[lines.index("Reward-free exploration") + 1:]
    while body and body[-1] == "":
        body.pop()
    for i, line in enumerate(body):
        if line == "" and i + 1 < len(body):
            assert body[i + 1].startswith(BLOCK_STARTS), (body[i - 1], body[i + 1])
    assert follows_blank(lines, "=== Epoch ")


def follows_blank(lines, start):
    """Every line starting with `start` comes right after a blank line (a phase boundary)."""
    found = [i for i, line in enumerate(lines) if line.startswith(start)]
    return bool(found) and all(i > 0 and lines[i - 1] == "" for i in found)


# Never on the terminal: losses, timing summaries, memory, PMPO diagnostics, task returns, the
# epoch summary line, the old long labels and the old test-diagnostic messages.
FORBIDDEN = ("loss", "[TIMING]", "seconds", "GPU", "memory", "done in", "Epoch 1 done", "PMPO_DIAGNOSTICS", "actor=",
             "kl=", "value=", "entropy", "return =", "task returns", "final_return", "return_", "Test diagnostics",
             "Summary of", "test_eval:", "fresh test data", "[LCG]", "[P2E]", "[LAPLACE]", "assert_setup_valid",
             "parameters in", "Starting on", "Intrinsic reward [", "Intrinsic return [", "per imagined step]",
             "episode total", "frozen", "real test episodes", "length ", "(PMPO)", "collection epoch")


def assert_clean_terminal(lines):
    text = "\n".join(lines)
    for word in FORBIDDEN:
        assert word not in text, word
    # Every bar is closed complete on its own line with tqdm's timing and rate fields (nothing
    # else written into it), and phases are separated by single blank lines.
    bars = [line for line in lines if "%|" in line]
    assert bars and all(BAR.match(line) for line in bars), [line for line in bars if not BAR.match(line)]
    assert not any(a == b == "" for a, b in zip(lines, lines[1:]))


# ---------------------------------------------------------------- seeded episodes, alignment

def test_seeded_test_episodes_are_labeled_counted_scored_and_saved(tmp_path, monkeypatch):
    trainer, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED)
    run(trainer)
    # Each scheduled point resets the test env once per configured seed, in order; final likewise.
    assert SeedEnv.resets[1:] == SEEDS + SEEDS + [11, 12]  # [0]: the training env's own reset
    assert trainer.test_episode_seeds == [11, 12]
    # Test interactions: 4 x 6 steps per point, 2 x 6 at the end (on the final row); training budget unchanged.
    totals = [r["progress/test_env_steps"] for _, r in logs if "timing/epoch_seconds" in r
              or "timing/final_test_diagnostic_seconds" in r]
    assert totals == [24, 48, 60] and trainer.num_test_steps == 60 and trainer.train_dataset.num_steps == 24
    for epoch in (1, 2):
        # W&B: exactly the 12 return keys (4 seeds x raw/scaled, raw/scaled mean and std).
        block = merged(logs, "test_eval/", epoch)
        assert set(block) == eval_keys("test_eval", SEEDS)
        # Lengths, scored steps and per-step values: in evaluation/*.json only.
        saved = json.loads(Path(f"evaluation/epoch_{epoch:05d}.json").read_text())
        assert saved["intrinsic"]["applicable"] and saved["intrinsic"]["scored"] and saved["new_test_env_steps"] == 24
        for seed, record in zip(SEEDS, saved["episodes"]):
            assert record["seed"] == seed and record["length"] == 6
            assert record["intrinsic_scored_steps"] == 6  # final observation completes step 5
            raw, scaled = block[f"test_eval/seed_{seed}/intrinsic_raw_return"], block[f"test_eval/seed_{seed}/intrinsic_scaled_return"]
            assert (record["intrinsic_raw_return"], record["intrinsic_scaled_return"]) == (raw, scaled)
            assert np.isfinite(raw) and np.isfinite(scaled) and raw != scaled
            assert record["intrinsic_raw_per_step"] == pytest.approx(raw / 6)
        for kind in ("raw", "scaled"):
            returns = [block[f"test_eval/seed_{s}/intrinsic_{kind}_return"] for s in SEEDS]
            assert block[f"test_eval/intrinsic_{kind}_return_mean"] == pytest.approx(np.mean(returns))
            assert block[f"test_eval/intrinsic_{kind}_return_std"] == pytest.approx(np.std(returns))
    assert set(merged(logs, "final_test_eval/")) == eval_keys("final_test_eval", (11, 12))
    # Saved locally with the protocol.
    saved = json.loads(Path("evaluation/epoch_00002.json").read_text())
    assert saved["kind"] == "current_policy_evaluation" and saved["protocol"]["reset_seeds"] == SEEDS
    assert [e["seed"] for e in saved["episodes"]] == SEEDS and saved["intrinsic"]["scorer"]["method"] == "lcg"
    assert "units" in saved["protocol"]["intrinsic_note"]
    assert json.loads(Path("evaluation/final_epoch_00002.json").read_text())["protocol"]["final_reset_seeds"] == [11, 12]

    # Alignment: the real windows are the imagination prompts' (replay segments, zero padding
    # before the episode start), and the LCG score equals the PMPO hook on the same candidates.
    episode = trainer.test_dataset.load_episode(0)
    n = 2
    obs, act, successors = sampler_windows(episode, n)
    for t in range(len(episode)):
        segment = make_segment(episode, SegmentId(0, t - n + 1, t + 1))
        assert torch.equal(obs[t], segment.obs) and torch.equal(act[t], segment.act)
    final = episode.info["final_observation"]
    assert torch.equal(successors[-1], final.reshape(successors[-1].shape)) and torch.equal(successors[:-1], episode.obs[1:])
    raw, scaled = trainer._intrinsic.score_episode(episode)
    lcg = trainer._lcg_lifecycle
    s2 = lcg.rms.s2.clone()
    infos = [{"imagined_candidate": ImaginedCandidate(obs[t:t + 1], act[t:t + 1], successors[t:t + 1])}
             for t in range(len(episode))]
    lcg.intrinsic_reward_fn(infos, torch.zeros(1, len(episode)))
    assert torch.equal(lcg.last_raw[0], raw)
    assert torch.allclose(scaled, raw * (1.0 / (s2.sqrt() + lcg.cfg.rms_eps)))


@pytest.mark.parametrize("extra,match", [
    (["evaluation.reset_seeds=[1,2,3]"], "has 3 seeds but collection.test.num_episodes=4"),
    (["evaluation.reset_seeds=[1,1,2,3]"], "distinct non-negative integers"),
    (["evaluation.reset_seeds=[1,2,3,4]", "collection.test.num_envs=2"], "num_envs=1"),
    (["evaluation.final_reset_seeds=[5]"], "num_final_episodes"),
    (["evaluation.intrinsic=maybe"], "evaluation.intrinsic must be true or false"),
])
def test_invalid_evaluation_seeds_are_rejected_before_any_file(tmp_path, monkeypatch, extra, match):
    with pytest.raises(ValueError, match=match):
        launch(tmp_path / "run", monkeypatch, TINY, BASE + FRESH + ["collection.test.num_episodes=4"] + extra)
    assert not (tmp_path / "run/checkpoints").exists()


def scorer_state(trainer):
    """Everything evaluation must leave unchanged: scorer and normalization state, monitoring
    counters, banks/heads, weights, optimizers, PMPO counters and the global RNG."""
    state = dict(agent=deepcopy(trainer.agent.state_dict()),
                 opt=trainer.opt.state_dict(), updates=trainer.agent.actor_critic.updates.item(),
                 rng={k: repr(v) for k, v in rng_state().items() if k != "cuda"})
    lcg, ens, lap, p2e = trainer._lcg_lifecycle, trainer._ensemble_scorer, trainer.laplace, trainer.plan2explore
    if lcg is not None:
        state["lcg"] = (lcg.rms.s2.clone(), lcg.last_raw.clone(), lcg.last_scaled.clone(), lcg.scoring_seconds, lcg.round_id)
    if ens is not None:
        state["scorer"] = (ens.last_raw.clone(), ens.last_scaled.clone(), None if ens.rms is None else ens.rms.s2.clone())
    if lap is not None:
        state["laplace"] = (lap.perturbations.clone(), lap.precision.clone(), lap.refreshes, lap.refresh_epoch,
                            lap.prediction_seconds, lap.prediction_calls, lap.prefix_evaluations)
    if p2e is not None:
        state["p2e"] = ({k: v.clone() for k, v in p2e.heads.state_dict().items()}, p2e.mean.clone(), p2e.std.clone(),
                        p2e.refreshes, list(p2e.fit_updates), p2e.extractor.seconds, p2e.scoring_seconds,
                        p2e.last_raw.clone(), p2e.cache_identity)
    return state


@pytest.mark.parametrize("method", ["ensemble", "laplace", "plan2explore", "lcg"])
def test_evaluation_matches_imagination_scoring_and_leaves_scorer_state_unchanged(tmp_path, monkeypatch, method):
    tiny, extra = dict(ensemble=(ENS, ["intrinsic_reward.reward_scaling=rms"]), laplace=(LAP, []),
                       plan2explore=(P2E, []), lcg=(TINY, []))[method]
    trainer, logs = launch(tmp_path / "run", monkeypatch, tiny, BASE + SEEDED + extra)
    run(trainer)
    block = merged(logs, "final_test_eval/")
    assert json.loads(Path("evaluation/final_epoch_00002.json").read_text())["intrinsic"]["scored"]
    assert set(block) == eval_keys("final_test_eval", (11, 12))
    assert np.isfinite(block["final_test_eval/intrinsic_raw_return_mean"])
    assert np.isfinite(block["final_test_eval/intrinsic_scaled_return_mean"])

    # Isolation: a full diagnostic (collection + scoring) changes no scorer, normalization,
    # bank/head, monitoring, weight, optimizer or global-RNG state.
    before = scorer_state(trainer)
    trainer.test_diagnostics()
    assert same(before, scorer_state(trainer))

    # Alignment with the imagination scoring path on the same real windows.
    episode = trainer.test_dataset.load_episode(1)
    obs, act, _ = sampler_windows(episode, 2)
    rl_env = trainer._rl_env
    if method in ("ensemble", "laplace"):
        rl_env.obs_buffer, rl_env.act_buffer = obs.clone(), act.clone()
        torch.manual_seed(5)
        rl_env.predict_next_obs_ensemble()
        torch.manual_seed(5)
        raw, scaled = trainer._intrinsic.score_episode(episode)
        assert torch.equal(raw, rl_env.last_disagreement)
        if method == "ensemble":  # RunningRMS applied with its current scale, not updated
            ens = trainer._ensemble_scorer
            assert torch.allclose(scaled, raw / (ens.rms.s2.sqrt() + ens.cfg.rms_eps))
        else:
            assert torch.equal(scaled, raw / trainer._laplace_cfg.fixed_divisor)
    elif method == "plan2explore":
        # WorldModelEnv.step's feature: current frame, the frame/action before the window, and
        # the window's earlier frames/actions; the taken action goes to the heads.
        p2e = trainer.plan2explore
        steps = range(len(episode))  # window o_{t-1}, o_t (n = 2); the step before it is t - 2
        prev_obs = torch.stack([episode.obs[t - 2] if t >= 2 else torch.zeros_like(episode.obs[0]) for t in steps])
        prev_act = torch.stack([episode.act[t - 2] if t >= 2 else torch.zeros_like(episode.act[0]) for t in steps])
        context = torch.cat((prev_obs.unsqueeze(1), obs[:, :-1]), dim=1)
        context_act = torch.cat((prev_act.unsqueeze(1), act[:, :-1]), dim=1)
        feature = p2e.extractor(obs[:, -1], context, context_act)
        reward = p2e([{"plan2explore_feature": feature, "plan2explore_action": act[:, -1]}], torch.zeros(len(episode), 1))
        raw, scaled = trainer._intrinsic.score_episode(episode)
        assert torch.equal(scaled, reward[:, 0]) and torch.equal(raw, p2e.last_raw[:, 0])
    else:
        raw, _ = trainer._intrinsic.score_episode(episode)
        assert raw.shape == (len(episode),)


# ---------------------------------------------------------------- uniform

def test_uniform_omits_scorer_and_pmpo_and_marks_intrinsic_not_applicable(tmp_path, monkeypatch, capsys):
    trainer, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + UNIFORM + SEEDED)
    run(trainer)
    lines = terminal_lines(capsys.readouterr().out)
    out = "\n".join(lines)
    assert out.count("Uniform random: no scorer refresh, no exploration-policy training, no intrinsic scores") == 1
    # No bars or messages for the skipped phases, and no fabricated intrinsic numbers.
    assert "Scorer refresh" not in out and "Exploration-policy training" not in out and "Historical precision" not in out
    assert "Imagined intrinsic reward" not in out
    assert bar(lines, "Initial collection (uniform actions)") == ["12/12 steps"]
    assert bar(lines, "Denoiser training") == ["2/2 updates", "1/1 updates"]
    assert lines.count("Intrinsic evaluation:") == 2 and lines.count("Final intrinsic evaluation:") == 1
    assert "  seed 3001 | 6 steps | intrinsic N/A" in lines and "  seed 11 | 6 steps | intrinsic N/A" in lines
    assert not any(" raw " in line or "scaled" in line for line in lines)
    assert_epoch_spacing(lines)
    assert_clean_terminal(lines)
    assert not any("intrinsic_raw" in k or "intrinsic_scaled" in k or "_raw_reward" in k for _, r in logs for k in r)
    # No evaluation keys at all; the not-applicable status and lengths are in evaluation/*.json.
    assert not merged(logs, "test_eval/") and not merged(logs, "final_test_eval/")
    saved = json.loads(Path("evaluation/epoch_00001.json").read_text())
    assert saved["intrinsic"] == dict(applicable=False, scored=False, scorer=None,
                                      reason="not applicable: uniform random has no intrinsic scorer")
    assert [e["length"] for e in saved["episodes"]] == [6] * 4 and saved["fresh"]
    # Epoch rows: no PMPO counter and no PMPO or scorer timing (nothing fabricated).
    epochs = [r for _, r in logs if "timing/epoch_seconds" in r]
    assert len(epochs) == 2
    assert not any(k in r for r in epochs for k in ("progress/pmpo_updates", "timing/epoch_pmpo_seconds",
                                                      "timing/epoch_method_refresh_seconds", "timing/pmpo_seconds_per_update",
                                                      "timing/intrinsic_scoring_seconds_per_update"))


# ---------------------------------------------------------------- rows, axes, counters

def test_epoch_rows_counters_axes_and_training_only_epochs(tmp_path, monkeypatch, capsys):
    trainer, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED + ["training.num_final_epochs=1"])
    run(trainer)
    lines = terminal_lines(capsys.readouterr().out)
    epochs = {e: r for e, r in logs if "timing/epoch_seconds" in r}
    assert sorted(epochs) == [1, 2, 3]
    # Epoch 3 trains without collecting: steps stay, updates advance, still one row per epoch.
    expected = {1: (18, 2, 2, 24), 2: (24, 3, 3, 48), 3: (24, 4, 4, 72)}
    for epoch, (steps, denoiser, pmpo, test) in expected.items():
        r = epochs[epoch]
        progress = {k: v for k, v in r.items() if k.startswith("progress/")}
        assert progress == {"progress/train_env_steps": steps, "progress/denoiser_updates_total": denoiser,
                            "progress/pmpo_updates": pmpo, "progress/test_env_steps": test}
        timing = {"timing/epoch_seconds", "timing/epoch_denoiser_seconds", "timing/epoch_method_refresh_seconds",
                  "timing/epoch_pmpo_seconds", "timing/pmpo_seconds_per_update", "timing/intrinsic_scoring_seconds_per_update",
                  "timing/epoch_test_seconds"}
        if epoch < 3:
            timing.add("timing/epoch_collection_seconds")
        assert set(r) == set(progress) | timing and r["timing/epoch_seconds"] > 0  # CPU: no GPU peak
        # Every row carries the env-step axis at the time it was logged.
        assert all(row["progress/train_env_steps"] == steps for e, row in logs if e == epoch)
        # One mean denoiser test loss per diagnostic.
        assert len([row for e, row in logs if e == epoch and "denoiser/test/loss_denoising" in row]) == 1
    assert trainer.agent.actor_critic.updates.item() == 4 and trainer.num_batch_train.denoiser == 4
    # PMPO rows keep their own update counter (W&B step metric), raw and policy reward separate.
    pmpo = [r for _, r in logs if "actor_critic/train/core/num_batch_train_actor_critic" in r]
    assert [r["actor_critic/train/core/num_batch_train_actor_critic"] for r in pmpo] == [1, 2, 3, 4]
    assert all(r["actor_critic/train/intrinsic/raw_reward_mean"] != r["actor_critic/train/intrinsic/scaled_reward_mean"]
               for r in pmpo)  # RunningRMS: different values
    # Retired categories are gone (the LCG refresh identity is in the W&B summary).
    assert not [k for _, r in logs for k in r if k.split("/")[0] in RETIRED_CATEGORIES]
    assert not [k for _, r in logs for k in r if k == "return" or "/return_" in k and not k.startswith("actor_critic/")]

    # Terminal: the compact banner, one closed bar per phase, labelled intrinsic statistics.
    out = "\n".join(lines)
    for text in ("Reward-free exploration", "  Domain: walker; downstream tasks: stand, walk", "  Exploration method: LCG",
                 "  Seed: 7", "  Training-data budget: 24 env steps", "  Output directory: ",
                 "=== Epoch 1/3 | exploration phase | training env steps 12/24 ===",
                 "=== Epoch 3/3 | training-only phase (no new collection) | training env steps 24/24 ===",
                 "  Compute device: CPU", "  Rendering device: ",
                 "Scorer refresh (LCG): historical precision and candidate bank on this epoch's denoiser"):
        assert any(line.startswith(text) for line in lines), text
    assert "W&B" not in out
    assert bar(lines, "Initial collection (uniform actions)") == ["12/12 steps"]
    assert bar(lines, "Collection (policy actions)") == ["6/6 steps"] * 2  # epoch 3 collects nothing
    assert bar(lines, "Denoiser training") == ["2/2 updates", "1/1 updates", "1/1 updates"]
    assert len(bar(lines, "Historical precision")) == 3
    assert bar(lines, "Exploration-policy training") == ["2/2 updates", "1/1 updates", "1/1 updates"]
    assert bar(lines, "Evaluation episodes (seeds 3001-3004)") == ["4/4 ep"] * 3
    assert bar(lines, "Final evaluation episodes (seeds 11-12)") == ["2/2 ep"]
    assert all(c.endswith(" trans") for c in bar(lines, "Historical precision"))
    assert not any(line.startswith("Collect test_dataset") for line in lines)  # no per-seed duplicates
    # Training line: mean of the per-update means, max of the per-update maxima, per epoch.
    expected = []
    for epoch in (1, 2, 3):
        rows = [r for e, r in pmpo_rows(logs) if e == epoch]
        stat = {k: [float(r[f"actor_critic/train/intrinsic/{k}"]) for r in rows]
                for k in ("raw_reward_mean", "raw_reward_max", "scaled_reward_mean", "scaled_reward_max")}
        expected.append(f"Imagined intrinsic reward/step: raw mean {np.mean(stat['raw_reward_mean']):.4g}, "
                        f"max {max(stat['raw_reward_max']):.4g} | scaled mean {np.mean(stat['scaled_reward_mean']):.4g}, "
                        f"max {max(stat['scaled_reward_max']):.4g}")
    assert [line for line in lines if line.startswith("Imagined intrinsic reward/step: ")] == expected
    assert lines.count("Intrinsic evaluation:") == 3 and lines.count("Final intrinsic evaluation:") == 1
    for prefix, epoch, seeds in (("test_eval/", 1, SEEDS), ("test_eval/", 3, SEEDS), ("final_test_eval/", None, (11, 12))):
        block = merged(logs, prefix, epoch)
        for seed in seeds:
            raw, scaled = block[f"{prefix}seed_{seed}/intrinsic_raw_return"], block[f"{prefix}seed_{seed}/intrinsic_scaled_return"]
            assert f"  seed {seed} | 6 steps | raw return {raw:.4g} | scaled return {scaled:.4g}" in lines
    block = merged(logs, "final_test_eval/")
    # Spacing: blank lines only before epochs and the blocks outside them; within an epoch the
    # scorer refresh follows the denoiser bar directly.
    assert_epoch_spacing(lines)
    for start in ("Initial collection", "Final evaluation episodes"):
        assert follows_blank(lines, start), start
    refresh = lines.index("Scorer refresh (LCG): historical precision and candidate bank on this epoch's denoiser")
    assert bar(lines[refresh - 1:refresh], "Denoiser training")
    # Epoch 1's layout: one line per phase, in this order, then the four seed lines and one blank
    # line before epoch 2 (no milestone in this run).
    start = lines.index("=== Epoch 1/3 | exploration phase | training env steps 12/24 ===")
    epoch1 = lines[start:lines.index("=== Epoch 2/3 | exploration phase | training env steps 18/24 ===")]
    order = ["=== Epoch 1/3", "Collection (policy actions): ", "Denoiser training: ", "Scorer refresh (LCG): ",
             "Historical precision: ", "Exploration-policy training: ", "Imagined intrinsic reward/step: ",
             "Evaluation episodes (seeds 3001-3004): ", "Intrinsic evaluation:"]
    assert [line.startswith(s) for line, s in zip(epoch1, order)] == [True] * len(order)
    rest = epoch1[len(order):]
    assert [line.split(" | ")[0] for line in rest[:-1]] == [f"  seed {s}" for s in SEEDS] and rest[-1] == ""
    # Four-seed aggregates: W&B only.
    assert "final_test_eval/intrinsic_raw_return_mean" in block and "mean" not in "\n".join(
        line for line in lines if line.startswith("  seed"))
    assert_clean_terminal(lines)


@pytest.mark.parametrize("method", ["ensemble", "plan2explore", "laplace"])
def test_method_progress_bars_and_intrinsic_labels(tmp_path, monkeypatch, capsys, method):
    tiny = dict(ensemble=ENS, plan2explore=P2E, laplace=LAP)[method]
    trainer, logs = launch(tmp_path / "run", monkeypatch, tiny, BASE + SEEDED)
    run(trainer)
    lines = terminal_lines(capsys.readouterr().out)
    name = trainer_module.METHOD_NAMES[trainer._method]
    assert f"  Exploration method: {name}" in lines
    assert sum(line.startswith(f"Scorer refresh ({name}): ") for line in lines) == 2
    if method == "ensemble":
        k = trainer.ensemble.members
        for m in range(1, k + 1):
            assert len(bar(lines, f"Denoiser training (member {m}/{k})")) == 2
        assert not bar(lines, "Denoiser training") and not bar(lines, "Historical precision")
    else:
        assert len(bar(lines, "Denoiser training")) == 2
    if method == "plan2explore":
        heads = trainer._p2e_cfg.num_members
        assert len(bar(lines, "P2E replay encoding")) == 2 and len(bar(lines, f"P2E head fitting ({heads} heads)")) == 2
    if method == "laplace":
        assert len(bar(lines, "Historical precision")) == 2
    assert len(bar(lines, "Exploration-policy training")) == 2
    training = [line for line in lines if line.startswith("Imagined intrinsic reward/step: ")]
    assert len(training) == 2 and all(re.fullmatch(r"Imagined intrinsic reward/step: raw mean \S+, max \S+ \| "
                                                   r"scaled mean \S+, max \S+", line) for line in training)
    assert lines.count("Intrinsic evaluation:") == 2 and lines.count("Final intrinsic evaluation:") == 1
    seed_lines = [line for line in lines if line.startswith("  seed ")]
    assert len(seed_lines) == 4 + 4 + 2 and all(
        re.fullmatch(r"  seed \d+ \| 6 steps \| raw return \S+ \| scaled return \S+", line) for line in seed_lines)
    assert_epoch_spacing(lines)
    assert_clean_terminal(lines)
    # The method's refresh bookkeeping is not charted (checkpoint, evaluation JSON and W&B summary keep it).
    assert not [k for _, r in logs for k in r if k.split("/")[0] in RETIRED_CATEGORIES]


def test_wandb_axes_provenance_milestones_and_unchanged_training(tmp_path, monkeypatch, capsys):
    """Logging on (bars, epoch summaries, an active W&B run) vs off: identical training."""
    quiet = ["logging.progress=none", "logging.epoch_summary=false"]
    off, logs_off = launch(tmp_path / "off", monkeypatch, TINY, BASE + FRESH + quiet)
    run(off)
    assert "%|" not in capsys.readouterr().out
    assert not [r for _, r in logs_off if "timing/epoch_seconds" in r]
    monkeypatch.undo()
    fake = FakeWandb()
    on, logs_on = launch(tmp_path / "on", monkeypatch, TINY, BASE + FRESH + [
        "logging.progress=bar", "wandb.mode=offline", "wandb.project=p", "milestones.at_steps=[24]"], fake_wandb=fake)
    run(on)
    assert_same_training(off, on)
    assert training_rows(logs_off) == training_rows(logs_on)
    # Axes: per-update rows on their own counters, everything else on epoch; no logged key matches
    # two globs (so W&B's glob ordering is irrelevant), and every logged key has an axis.
    axes = dict(fake.metrics)
    assert axes["actor_critic/train/*"] == "actor_critic/train/core/num_batch_train_actor_critic"
    assert axes["denoiser/train/*"] == "denoiser/train/num_batch_train_denoiser"
    assert axes["test_eval/*"] == axes["progress/*"] == axes["denoiser/final_test/*"] == "epoch"
    assert fake.hidden == ["epoch"]  # an x-axis only
    assert not [n for n in axes if n.split("/")[0] in RETIRED_CATEGORIES]
    globs = [n for n in axes if n.endswith("*")]
    for _, row in logs_on:
        for key in row:
            matches = [g for g in globs if key.startswith(g[:-1])] + [n for n in axes if n == key]
            assert len([g for g in globs if key.startswith(g[:-1])]) <= 1 and matches, key
    provenance = fake.config_updates[0]["provenance"]
    assert provenance["seed"] == 7 and provenance["method"] == "lcg" and provenance["source"] == on.source
    assert provenance["method_settings"]["reward_scaling"] == on._lcg_reward_scaling
    assert provenance["method_settings"]["lcg"]["theta_s_dim"] == on._lcg_lifecycle._theta_s_dim > 0  # a run constant
    assert provenance["evaluation"]["every"] == 1
    # Unseeded episodes: only the aggregate returns; the episodes are in evaluation/*.json.
    assert set(merged(logs_on, "test_eval/")) == {f"test_eval/intrinsic_{k}_return_{s}" for k in ("raw", "scaled")
                                                  for s in ("mean", "std")}
    # Status and scorer identity: W&B summary metadata (latest), not history.
    status = fake.summary["evaluation/final_test_eval"]
    assert (status["fresh"], status["num_episodes"], status["new_test_env_steps"], status["total_test_env_steps"]) == \
        (True, 3, 18, 42)
    assert status["intrinsic"]["scored"] and status["file"].endswith("final_epoch_00002.json")
    assert fake.summary["evaluation/test_eval"]["epoch"] == 2
    scorer = fake.summary["scorer"]
    assert scorer["method"] == "lcg" and scorer["round"] == 2 and np.isfinite(scorer["h_D_fingerprint"])
    # The run id is kept in the run folder; the milestone identity goes to the summary.
    assert json.loads(Path("wandb_run.json").read_text())["id"] == "fakerun1"
    milestone = fake.summary["milestones/steps_000024"]
    assert milestone["collected_steps"] == 24 and len(milestone["denoiser_sha256"]) == 64
    # Every logged row is mirrored locally with its W&B step (no active-run rows outside it).
    mirror = [json.loads(line) for line in Path("wandb_history.jsonl").read_text().splitlines()]
    assert [m["step"] for m in mirror] == list(range(len(logs_on))) == list(range(on.wandb_steps))
    assert all(m["values"] == {k: float(v) for k, v in row.items() if np.ndim(v) == 0}
               for m, (_, row) in zip(mirror, logs_on))
    lines = terminal_lines(capsys.readouterr().out)
    assert f"Milestone steps_000024 published: > {(Path('milestones') / 'steps_000024').resolve()}" in lines
    milestone = [i for i, line in enumerate(lines) if line.startswith("Milestone ")]
    assert len(milestone) == 1 and lines[milestone[0] - 1] != ""
    assert_epoch_spacing(lines)
    assert_clean_terminal(lines)


FINAL_SEEDS = [11, 12, 13, 14]
SEEDED_4 = [o for o in SEEDED if "final" not in o] + ["collection.test.num_final_episodes=4",
                                                      f"evaluation.final_reset_seeds=[{','.join(map(str, FINAL_SEEDS))}]"]


def expected_keys(trainer, ac_schema):
    """Every W&B key the trainer may publish for this method (CPU: no GPU peak), written out."""
    learned = trainer._method is not None
    keys = {"progress/train_env_steps", "progress/test_env_steps", "progress/denoiser_updates_total",
            "timing/epoch_collection_seconds", "timing/epoch_denoiser_seconds", "timing/epoch_test_seconds",
            "timing/epoch_seconds", "timing/final_test_diagnostic_seconds",
            "denoiser/test/loss_denoising", "denoiser/final_test/loss_denoising"}
    if trainer.ensemble is not None:
        keys |= {f"denoiser/member_{k}/train/{name}" for k in range(trainer.ensemble.members)
                 for name in ("loss_denoising", "num_batch_train_member")}
    else:
        keys |= {"denoiser/train/loss_denoising", "denoiser/train/num_batch_train_denoiser"}
    if learned:
        keys |= {"progress/pmpo_updates", "timing/epoch_method_refresh_seconds", "timing/epoch_pmpo_seconds",
                 "timing/pmpo_seconds_per_update", "timing/intrinsic_scoring_seconds_per_update"}
        keys |= {f"actor_critic/train/{k}" for k in ac_schema(2, debug=False)}
        keys |= eval_keys("test_eval", SEEDS) | eval_keys("final_test_eval", FINAL_SEEDS)
    return keys


@pytest.mark.parametrize("method", ["lcg", "ensemble", "laplace", "plan2explore", "uniform"])
def test_published_keys_axes_mirror_and_summary_by_method(tmp_path, monkeypatch, method, ac_schema):
    """The complete W&B key set of each method (4 fixed seeds per evaluation), one axis per key,
    the mirror equal to what is sent, and the status/identity kept as summary metadata."""
    tiny, extra = METHOD_SETUPS[method]
    fake = FakeWandb()
    trainer, logs = launch(tmp_path / "run", monkeypatch, tiny, BASE + SEEDED_4 + extra + [
        "wandb.mode=offline", "wandb.project=p"], fake_wandb=fake)
    run(trainer)
    keys = {k for _, r in logs for k in r}
    assert keys == expected_keys(trainer, ac_schema)
    assert not [k for k in keys if k.split("/")[0] in RETIRED_CATEGORIES or k.endswith(("/lr", "grad_norm_before_clip"))]
    for block, seeds in (("test_eval", SEEDS), ("final_test_eval", FINAL_SEEDS)):
        assert len({k for k in keys if k.startswith(f"{block}/")}) == (12 if trainer._method else 0)
    # Final test steps are counted on the final row, separately from training steps.
    final = [r for _, r in logs if "timing/final_test_diagnostic_seconds" in r]
    assert len(final) == 1 and final[0]["progress/test_env_steps"] == 72 and final[0]["progress/train_env_steps"] == 24
    assert "denoiser/final_test/loss_denoising" in final[0]
    if trainer.ensemble is not None:  # actual updates over every member
        assert final[0]["progress/denoiser_updates_total"] == sum(trainer.ensemble.updates) == 3 * trainer.num_batch_train.denoiser
    else:
        assert final[0]["progress/denoiser_updates_total"] == trainer.num_batch_train.denoiser == 3
    # Axes: each key matches exactly one definition; member losses on their own counters.
    axes = dict(fake.metrics)
    globs = [n for n in axes if n.endswith("*")]
    for key in keys:
        assert len([g for g in globs if key.startswith(g[:-1])]) == 1, key
    for k in range(trainer.ensemble.members if trainer.ensemble is not None else 0):
        assert axes[f"denoiser/member_{k}/train/*"] == f"denoiser/member_{k}/train/num_batch_train_member"
    assert fake.hidden == ["epoch"]
    # The local mirror holds exactly the rows sent to W&B.
    mirror = [json.loads(line) for line in Path("wandb_history.jsonl").read_text().splitlines()]
    assert [m["values"] for m in mirror] == [{k: float(v) for k, v in r.items() if np.ndim(v) == 0} for _, r in logs]
    # Summary metadata: evaluation status (uniform: not applicable, no scorer) and scorer identity.
    status = fake.summary["evaluation/final_test_eval"]
    assert status["fresh"] and status["num_episodes"] == 4 and status["new_test_env_steps"] == 24
    if trainer._method is None:
        assert status["intrinsic"]["reason"] == "not applicable: uniform random has no intrinsic scorer"
        assert "scorer" not in fake.summary
    else:
        assert status["intrinsic"]["scored"] and fake.summary["scorer"]["method"] == trainer._method
    settings = fake.config_updates[0]["provenance"]["method_settings"]
    if trainer.plan2explore is not None:  # run constants: configuration/provenance
        assert settings["plan2explore"]["head_parameters"] == trainer.plan2explore.num_head_parameters
        assert settings["plan2explore"]["extractor"]["feature_dim"] == trainer.plan2explore.feature_dim
        assert fake.summary["scorer"]["fit_updates"] == trainer.plan2explore.fit_updates
    if trainer.laplace is not None:
        assert settings["laplace"]["dim"] == trainer.laplace.dim and fake.summary["scorer"]["refreshes"] == 2
    if trainer.ensemble is not None:
        assert fake.summary["scorer"]["member_updates"] == trainer.ensemble.updates


def test_denoiser_test_loss_is_one_mean_per_diagnostic_and_unpublished_values_stay_internal(tmp_path, monkeypatch):
    batches, internal = [], []
    test_component, train_agent = ExplorationTrainer.test_component, ExplorationTrainer.train_agent

    def spy_test(self, name):
        rows = test_component(self, name)
        batches.append([float(r["denoiser/test/loss_denoising"]) for r in rows])
        return rows

    def spy_train(self):
        rows = train_agent(self)
        internal.extend(rows)
        return rows

    monkeypatch.setattr(ExplorationTrainer, "test_component", spy_test)
    monkeypatch.setattr(ExplorationTrainer, "train_agent", spy_train)
    trainer, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED + ["collection.test.num_episodes=4",
                                                                                 "denoiser.training.batch_size=2"])
    run(trainer)
    # Several test batches per diagnostic; W&B gets their unweighted mean (periodic, then final).
    assert len(batches) == 3 and all(len(b) > 1 and np.isfinite(b).all() for b in batches)
    logged = [r.get("denoiser/test/loss_denoising", r.get("denoiser/final_test/loss_denoising")) for _, r in logs
              if "denoiser/test/loss_denoising" in r or "denoiser/final_test/loss_denoising" in r]
    assert logged == [pytest.approx(float(np.mean(b)), nan_ok=True) for b in batches]
    assert trainer.num_batch_test.denoiser == sum(map(len, batches))  # the checkpointed counter still counts
    # The trainer's rows keep the learning rate and gradient norm (the PMPO diagnostic reads them);
    # W&B and the mirror do not.
    assert any("denoiser/train/lr" in r and "denoiser/train/grad_norm_before_clip" in r for r in internal)
    assert not any(k.endswith(("/lr", "/grad_norm_before_clip")) for _, r in logs for k in r)
    assert trainer_module.published_row({"denoiser/train/lr": 1.0, "denoiser/train/loss_denoising": 2.0,
                                         "actor_critic/train/core/loss_actor": 3.0}) == \
        {"denoiser/train/loss_denoising": 2.0, "actor_critic/train/core/loss_actor": 3.0}


def test_message_during_an_active_bar_gets_its_own_line(capsys):
    """say() clears an active bar and redraws it below the message; phase() never doubles blanks."""
    from utils import phase, Progress, say
    say("before")
    phase()
    phase()
    pbar = Progress(total=3, desc="Phase A", unit="steps")
    pbar.update(1)
    say("a warning")
    pbar.update(2)
    pbar.close()
    lines = terminal_lines(capsys.readouterr().out)
    assert lines[:2] == ["before", ""] and "a warning" in lines
    assert bar(lines, "Phase A") == ["3/3 steps"]
    assert_clean_terminal(lines)


# ---------------------------------------------------------------- resume

def test_resume_keeps_wandb_run_counters_and_seed_labels(tmp_path, monkeypatch):
    full, logs_full = launch(tmp_path / "full", monkeypatch, TINY, BASE + SEEDED)
    run(full)
    monkeypatch.undo()

    crash_at_test_agent(monkeypatch, epoch=2)
    fake = FakeWandb()
    crashed, logs_crashed = launch(tmp_path / "split", monkeypatch, TINY, BASE + SEEDED + ["wandb.mode=offline"],
                                   fake_wandb=fake)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.undo()
    fake_resumed = FakeWandb(run_id="different")
    resumed, logs = launch(tmp_path / "split", monkeypatch, None, [], resume=True, fake_wandb=fake_resumed)
    run(resumed)
    # The same W&B run continues (id from the run folder), and the resume is labeled.
    assert fake_resumed.inits[0]["id"] == "fakerun1" and fake_resumed.inits[0]["resume"] == "allow"
    assert fake_resumed.summary["resume_count"] == 1
    # The resume event is summary metadata, not a history row.
    resume = fake_resumed.summary["resume"]
    assert resume["from_epoch"] == 1 and resume["count"] == 1 and not rows_of(logs, "resume/")
    # This fake server holds no rows, so every committed row was backfilled (see the backfill test).
    assert resume["backfilled_rows"] == resume["wandb_committed_rows"] > 0
    # Replayed epoch 2 logs exactly what the uninterrupted run logged (wall-clock timing aside).
    def epoch2(rows):
        return [scalars(r) for e, r in rows if e == 2 and not any(k.startswith("resume/") for k in r)]
    assert epoch2(logs_full) == epoch2(logs)
    assert_same_training(full, resumed)
    assert resumed.test_episode_seeds == full.test_episode_seeds == [11, 12]
    assert json.loads(Path("evaluation/epoch_00002.json").read_text())["episodes"] == \
        json.loads((tmp_path / "full/evaluation/epoch_00002.json").read_text())["episodes"]

    # Reuse-only after resume: a dataset diagnostic keeping the saved seed labels.
    monkeypatch.undo()
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, _ = launch(tmp_path / "reuse", monkeypatch, TINY, BASE + SEEDED)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.undo()
    resumed, logs = launch(tmp_path / "reuse", monkeypatch, None, REUSE, resume=True)
    run(resumed)
    assert SeedEnv.resets == []  # no test collection (and the training env continues its episode)
    block = merged(logs, "test_dataset_diagnostic/", 2)
    assert {k.split("/")[1] for k in block if "/seed_" in k} == {f"seed_{s}" for s in SEEDS}
    assert not merged(logs, "test_eval/") and merged(logs, "final_test_dataset_diagnostic/")
    assert json.loads(Path("evaluation/epoch_00002.json").read_text())["kind"].startswith("dataset_diagnostic")

    # A checkpoint from before seed labels: its episodes are never given the configured seeds.
    monkeypatch.undo()
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, _ = launch(tmp_path / "old", monkeypatch, TINY, BASE + SEEDED)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.undo()
    path = tmp_path / "old/checkpoints/state.pt"
    state = torch.load(path, weights_only=False)
    del state["test_episode_seeds"], state["resume_count"]
    torch.save(state, path)
    resumed, logs = launch(tmp_path / "old", monkeypatch, None, REUSE, resume=True)
    run(resumed)
    block = merged(logs, "test_dataset_diagnostic/", 2)
    assert set(block) == {f"test_dataset_diagnostic/intrinsic_{k}_return_{s}" for k in ("raw", "scaled")
                          for s in ("mean", "std")}  # unlabeled episodes: aggregates only
    assert [e["seed"] for e in json.loads(Path("evaluation/epoch_00002.json").read_text())["episodes"]] == [None] * 4
    assert resumed.resume_count == 1


@pytest.mark.parametrize("server", ["lost", "ahead", "offline"])
def test_resume_backfills_rows_the_wandb_server_lost_and_counts_replayed_ones(tmp_path, monkeypatch, capsys, server):
    """A hard crash can kill W&B's upload process: the server then holds fewer rows than the
    checkpoint committed (backfilled from the local mirror), or more (an uncommitted epoch that is
    replayed: counted). Offline W&B reports no server rows: nothing is backfilled (a new offline
    file per segment; re-logging would duplicate rows when synced)."""
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED + ["wandb.mode=offline"], fake_wandb=FakeWandb())
    with pytest.raises(Crash):
        crashed._run()
    committed = torch.load(tmp_path / "run/checkpoints/state.pt", weights_only=False)["wandb_steps"]
    epoch1 = [row for e, row in logs if e == 1]
    assert committed == len(epoch1) > 0
    monkeypatch.undo()
    starting = committed + 7 if server == "ahead" else 3 if server == "lost" else 0
    fake = FakeWandb(starting_step=starting, offline=server == "offline")
    capsys.readouterr()
    resumed, logs = launch(tmp_path / "run", monkeypatch, None, [], resume=True, fake_wandb=fake)
    resume_row = resumed._resume_event  # W&B summary "resume"
    lines = terminal_lines(capsys.readouterr().out)  # essential resume messages stay visible
    assert "  Resumed at epoch 1 (resume #1)" in lines
    message = dict(lost=f"W&B: backfilled {committed - 3} committed rows the server did not receive before the crash",
                   ahead="W&B: 7 rows logged after the checkpoint are on the server; their epochs are replayed",
                   offline="W&B offline: the server's row count is unknown, so no rows are backfilled "
                           "(wandb_history.jsonl keeps every committed row)")[server]
    assert message in lines
    assert resume_row["wandb_committed_rows"] == committed and fake.summary["resume"] is resume_row
    assert resume_row.get("wandb_server_rows") == (None if server == "offline" else starting)
    backfilled = [r for r in fake.logged if r.get("resume/backfilled") == 1]
    if server == "offline":
        assert not backfilled and resumed.wandb_steps == committed
    elif server == "lost":
        # Exactly the committed rows the server lacks, in order, with their epoch and values.
        assert resume_row["backfilled_rows"] == committed - 3 and resumed.wandb_steps == committed
        expected = [{"epoch": 1, **{k: float(v) for k, v in row.items() if np.ndim(v) == 0}, "resume/backfilled": 1}
                    for row in epoch1[3:]]
        assert backfilled == expected
    else:
        assert not backfilled and resume_row["server_rows_beyond_checkpoint"] == 7
        assert resumed.wandb_steps == committed + 7
    run(resumed)
    # The mirror's latest entry per step continues from the reconciled step.
    steps = [json.loads(line)["step"] for line in Path("wandb_history.jsonl").read_text().splitlines()]
    assert steps[-1] == resumed.wandb_steps - 1


# ---------------------------------------------------------------- interrupted W&B mirror

def history_line(step, value=0.0, segment=0):
    return (json.dumps(dict(step=step, epoch=1, segment=segment, values={"x": value})) + "\n").encode()


def test_history_repair_keeps_committed_rows_and_moves_an_interrupted_tail(tmp_path):
    path = tmp_path / "wandb_history.jsonl"
    committed = history_line(0) + history_line(1)
    uncommitted = history_line(2, segment=1)  # a complete row of the interrupted segment: kept
    tail = b'{"step": 3, "epoch": 1, "segment": 1, "val'
    path.write_bytes(committed + uncommitted + tail)
    action = repair_history(path, len(committed))
    assert "interrupted-1" in action
    assert (tmp_path / "wandb_history.jsonl.interrupted-1").read_bytes() == tail  # byte-for-byte
    assert path.read_bytes() == committed + uncommitted
    # Idempotent; a later interruption gets its own recovery file, the first is never overwritten.
    assert repair_history(path, len(committed)) is None and path.read_bytes() == committed + uncommitted
    with open(path, "ab") as f:
        f.write(b'{"step": 3')
    assert "interrupted-2" in repair_history(path, len(committed))
    assert (tmp_path / "wandb_history.jsonl.interrupted-1").read_bytes() == tail
    assert (tmp_path / "wandb_history.jsonl.interrupted-2").read_bytes() == b'{"step": 3'
    # Appending after the repair keeps every line a whole record.
    with open(path, "ab") as f:
        f.write(history_line(3, segment=1))
    assert [r["step"] for r in history_records(path)] == [0, 1, 2, 3]
    assert [r["step"] for r in history_records(path, end=len(committed))] == [0, 1]


def test_history_repair_completes_a_record_that_only_lost_its_newline(tmp_path):
    path = tmp_path / "wandb_history.jsonl"
    data = history_line(0) + history_line(1)[:-1]
    path.write_bytes(data)
    assert "newline" in repair_history(path, len(history_line(0)))
    assert path.read_bytes() == data + b"\n" and not list(tmp_path.glob("*.interrupted-*"))
    assert repair_history(path, len(history_line(0))) is None and path.read_bytes() == data + b"\n"


@pytest.mark.parametrize("case", ["interior", "short", "boundary", "missing", "committed_tail"])
def test_history_repair_rejects_damaged_or_missing_committed_rows(tmp_path, case):
    path = tmp_path / "wandb_history.jsonl"
    rows = history_line(0) + history_line(1)
    committed, data = len(rows), rows
    if case == "interior":  # an earlier cut row that was appended to without repair
        data = history_line(0) + history_line(1)[:20] + history_line(2)
        match = rf"record 2 \(byte offset {len(history_line(0))}\) is not valid JSON"
    elif case == "short":  # committed rows missing from the end
        data, match = history_line(0), "committed rows are missing"
    elif case == "boundary":
        committed, match = committed - 3, "no record boundary"
    elif case == "committed_tail":  # the cut record lies inside the committed bytes
        data, match = rows[:-5], "committed rows are missing"
    else:
        data, match = None, "is missing"
    if data is not None:
        path.write_bytes(data)
    with pytest.raises(ValueError, match=match):
        repair_history(path, committed)
    assert (path.read_bytes() if data is not None else None) == data  # nothing changed
    assert not list(tmp_path.glob("*.interrupted-*"))


def test_resume_repairs_the_mirror_and_backfills_only_committed_rows(tmp_path, monkeypatch, capsys):
    """A hard kill: committed rows, then rows of the interrupted segment (one rewriting a committed
    step) and a cut record. The resume keeps the cut bytes, backfills the rows the server lost from
    the committed records only, and the journal stays whole through a second crash and resume."""
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED + ["wandb.mode=offline"], fake_wandb=FakeWandb())
    with pytest.raises(Crash):
        crashed._run()
    state = torch.load(tmp_path / "run/checkpoints/state.pt", weights_only=False)
    committed, committed_bytes = state["wandb_steps"], state["wandb_history_bytes"]
    path = tmp_path / "run/wandb_history.jsonl"
    assert path.stat().st_size == committed_bytes
    epoch1 = [row for e, row in logs if e == 1]
    tail = b'{"step": 99, "epoch": 2, "segm'
    with open(path, "ab") as f:  # the interrupted segment
        f.write(history_line(committed - 1, value=-1.0, segment=0) + history_line(committed, value=-2.0) + tail)
    monkeypatch.undo()
    crash_at_test_agent(monkeypatch, epoch=2)
    fake = FakeWandb(starting_step=3)
    capsys.readouterr()
    resumed, _ = launch(tmp_path / "run", monkeypatch, None, [], resume=True, fake_wandb=fake)
    out = capsys.readouterr().out
    assert "W&B mirror: kept the interrupted last record of wandb_history.jsonl" in out
    assert (tmp_path / "run/wandb_history.jsonl.interrupted-1").read_bytes() == tail
    backfilled = [r for r in fake.logged if r.get("resume/backfilled") == 1]
    expected = [{"epoch": 1, **{k: float(v) for k, v in row.items() if np.ndim(v) == 0}, "resume/backfilled": 1}
                for row in epoch1[3:]]
    assert backfilled == expected  # the committed values, not the interrupted segment's -1.0 / -2.0
    assert resumed._resume_event["backfilled_rows"] == committed - 3
    with pytest.raises(Crash):  # a second crash in the same epoch
        resumed._run()
    monkeypatch.undo()
    fake = FakeWandb(starting_step=committed)  # nothing lost this time: no backfill, no replay
    resumed, _ = launch(tmp_path / "run", monkeypatch, None, [], resume=True, fake_wandb=fake)
    assert "backfilled_rows" not in resumed._resume_event
    run(resumed)
    records = list(history_records(path))  # every line a whole record after appends and repairs
    assert records[-1]["step"] == resumed.wandb_steps - 1
    assert list(tmp_path.glob("run/wandb_history.jsonl.interrupted-*")) == [tmp_path / "run/wandb_history.jsonl.interrupted-1"]


# ---------------------------------------------------------------- final-only resume

def crash_before_final_evaluation(monkeypatch):
    original = ExplorationTrainer.test_diagnostics

    def crash(self, final=False):
        if final:
            raise Crash()
        return original(self, final)

    monkeypatch.setattr(ExplorationTrainer, "test_diagnostics", crash)


def forbid_training_and_refresh(monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("training, scorer refresh, posterior redraw or head fitting in a final-only resume")
    for owner, name in ((ExplorationTrainer, "train_agent"), (ExplorationTrainer, "train_component"),
                        (ExplorationTrainer, "collect_initial_dataset"), (LCGLifecycle, "refresh"),
                        (LaplaceEnsemble, "refresh"), (Plan2Explore, "refresh")):
        monkeypatch.setattr(owner, name, fail)


def scale_state(trainer):
    """RunningRMS s2 of the method's scorer (None: no running scale)."""
    rms = (trainer._lcg_lifecycle or trainer._ensemble_scorer or SimpleNamespace(rms=None)).rms
    return None if rms is None else rms.s2.clone()


METHOD_SETUPS = dict(lcg=(TINY, []), ensemble=(ENS, ["intrinsic_reward.reward_scaling=rms"]), laplace=(LAP, []),
                     plan2explore=(P2E, []), uniform=(TINY, UNIFORM))


@pytest.mark.parametrize("method", list(METHOD_SETUPS))
def test_final_only_resume_evaluates_with_the_restored_frozen_scorer(tmp_path, monkeypatch, capsys, method):
    tiny, extra = METHOD_SETUPS[method]
    overrides = BASE + SEEDED + extra
    full, logs_full = launch(tmp_path / "full", monkeypatch, tiny, overrides)
    run(full)
    monkeypatch.undo()
    crash_before_final_evaluation(monkeypatch)
    crashed, _ = launch(tmp_path / "split", monkeypatch, tiny, overrides)
    with pytest.raises(Crash):
        crashed._run()
    s2 = scale_state(crashed)
    monkeypatch.undo()
    forbid_training_and_refresh(monkeypatch)
    resumed, logs = launch(tmp_path / "split", monkeypatch, None, [], resume=True)
    if method in ("lcg", "ensemble"):  # a non-default running scale, restored exactly
        assert s2 is not None and torch.equal(scale_state(resumed), s2)
    if method == "lcg":
        lcg, ref = resumed._lcg_lifecycle, full._lcg_lifecycle
        assert lcg.ready and lcg.round_id == ref.round_id == 2 and torch.equal(lcg.h_D, ref.h_D)
        assert all(torch.equal(a, b) for name in ("sigmas", "epsilons", "epsilons_offset", "etas")
                   for a, b in zip(getattr(lcg.bank, name), getattr(ref.bank, name)))
        assert resumed.agent.actor_critic.intrinsic_reward_fn is lcg.intrinsic_reward_fn  # wired as after a refresh
    run(resumed)
    # Only the final evaluation ran: per-seed raw and scaled returns equal the uninterrupted run's.
    assert {e for e, _ in logs} == {2}
    final, final_full = merged(logs, "final_test_eval/"), merged(logs_full, "final_test_eval/")
    assert final == final_full
    if method != "uniform":
        for seed in (11, 12):
            assert f"final_test_eval/seed_{seed}/intrinsic_raw_return" in final
            assert f"final_test_eval/seed_{seed}/intrinsic_scaled_return" in final
    assert_same_training(full, resumed)  # weights, optimizers, counters, replay, training RNG
    assert scale_state(resumed) is None if s2 is None else torch.equal(scale_state(resumed), s2)  # not updated
    assert resumed.final_test_done
    # Another resume does not repeat the completed final evaluation.
    monkeypatch.undo()
    forbid_training_and_refresh(monkeypatch)
    again, logs_again = launch(tmp_path / "split", monkeypatch, None, [], resume=True)
    run(again)
    assert not merged(logs_again, "final_test_eval/")


@pytest.mark.parametrize("method", ["lcg", "ensemble"])
def test_older_checkpoint_without_scorer_state(tmp_path, monkeypatch, capsys, method):
    """Before the final evaluation: the missing frozen state is reported, never re-created.
    Mid-run: the next epoch's refresh rebuilds it, so the run matches an uninterrupted one."""
    tiny, extra = METHOD_SETUPS[method]
    overrides = BASE + SEEDED + extra

    def strip(path):
        state = torch.load(path, weights_only=False)
        del state["lcg_scorer"], state["ensemble_scorer"], state["wandb_history_bytes"]
        torch.save(state, path)

    crash_before_final_evaluation(monkeypatch)
    crashed, _ = launch(tmp_path / "final", monkeypatch, tiny, overrides)
    with pytest.raises(Crash):
        crashed._run()
    strip(tmp_path / "final/checkpoints/state.pt")
    monkeypatch.undo()
    forbid_training_and_refresh(monkeypatch)
    capsys.readouterr()
    resumed, logs = launch(tmp_path / "final", monkeypatch, None, [], resume=True)
    run(resumed)
    out = "\n".join(terminal_lines(capsys.readouterr().out))
    block = merged(logs, "final_test_eval/")
    saved = json.loads(Path("evaluation/final_epoch_00002.json").read_text())
    if method == "lcg":
        assert not block and not saved["intrinsic"]["scored"]  # nothing logged, nothing invented
        assert "Final intrinsic evaluation: N/A (the checkpoint predates saved LCG scorer state" in out
        assert "predates saved LCG scorer state" in saved["intrinsic"]["reason"]
    else:  # raw disagreement from the restored members; the lost RunningRMS scale is not invented
        assert "final_test_eval/seed_11/intrinsic_raw_return" in block
        assert not any("intrinsic_scaled" in k for k in block)
        assert "Warning: no frozen reward scale; the checkpoint predates the saved RunningRMS scale" in out

    # Mid-run with an older checkpoint: training refreshes the scorer before any evaluation.
    monkeypatch.undo()
    full, logs_full = launch(tmp_path / "full", monkeypatch, tiny, overrides)
    run(full)
    monkeypatch.undo()
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, _ = launch(tmp_path / "mid", monkeypatch, tiny, overrides)
    with pytest.raises(Crash):
        crashed._run()
    strip(tmp_path / "mid/checkpoints/state.pt")
    monkeypatch.undo()
    resumed, logs = launch(tmp_path / "mid", monkeypatch, None, [], resume=True)
    run(resumed)
    assert merged(logs, "final_test_eval/") == merged(logs_full, "final_test_eval/")
    assert_same_training(full, resumed)


# ---------------------------------------------------------------- timing definitions

SLEEP, COMMON_SLEEP = 0.03, 0.6


class Calls(list):
    """Slowed-down calls as (tag, epoch, inside a PMPO update)."""
    trainer, in_pmpo = None, False

    def add(self, tag):
        self.append((tag, self.trainer.epoch if self.trainer else 0, self.in_pmpo))


def slow_down(monkeypatch, cls, name, calls, tag, seconds=SLEEP, when=None):
    """cls.name takes `seconds` longer (only where `when` holds); every such call is recorded."""
    original = getattr(cls, name)

    def wrapper(*args, **kwargs):
        if when is None or when(*args, **kwargs):
            calls.add(tag)
            time.sleep(seconds)
        return original(*args, **kwargs)
    monkeypatch.setattr(cls, name, wrapper)


def instrument(monkeypatch, method):
    """Known extra time in each method's refresh work, its intrinsic-scoring work, the common
    denoiser training and the non-scoring part of a PMPO update."""
    calls = Calls()
    forward = PMPOBeta.forward

    def pmpo_forward(self):
        calls.in_pmpo = True
        try:
            return forward(self)
        finally:
            calls.in_pmpo = False
    monkeypatch.setattr(PMPOBeta, "forward", pmpo_forward)
    slow_down(monkeypatch, PMPOOptimizers, "step", calls, "update")
    if method == "ensemble":  # member 0 is the common denoiser; members 1..K-1 the method's refresh cost
        slow_down(monkeypatch, ExplorationTrainer, "train_component", calls, "common", COMMON_SLEEP,
                  when=lambda self, name, steps, member=None: name == "denoiser" and member == 0)
        slow_down(monkeypatch, ExplorationTrainer, "train_component", calls, "refresh",
                  when=lambda self, name, steps, member=None: name == "denoiser" and member is not None and member > 0)
        slow_down(monkeypatch, WorldModelEnv, "predict_next_obs_ensemble", calls, "scoring")
        return calls
    slow_down(monkeypatch, ExplorationTrainer, "train_component", calls, "common", COMMON_SLEEP,
              when=lambda self, name, steps, member=None: name == "denoiser")
    if method == "lcg":
        slow_down(monkeypatch, LCGLifecycle, "refresh", calls, "refresh")
        make_hook = LCGLifecycle._make_hook

        def slow_hook(self, denoiser):
            hook = make_hook(self, denoiser)

            def slowed(infos, rew):
                calls.add("scoring")
                time.sleep(SLEEP)
                return hook(infos, rew)
            return slowed
        monkeypatch.setattr(LCGLifecycle, "_make_hook", slow_hook)
    elif method == "plan2explore":
        slow_down(monkeypatch, Plan2Explore, "refresh", calls, "refresh")
        slow_down(monkeypatch, FeatureExtractor, "__call__", calls, "scoring")  # imagination features (also refresh, eval)
        slow_down(monkeypatch, Plan2Explore, "__call__", calls, "scoring")  # head disagreement
    else:
        slow_down(monkeypatch, LaplaceEnsemble, "refresh", calls, "refresh")
        slow_down(monkeypatch, LaplaceEnsemble, "predict_members", calls, "scoring")
    return calls


@pytest.mark.parametrize("method", ["lcg", "ensemble", "laplace", "plan2explore"])
def test_refresh_and_scoring_timers_cover_their_method_paths(tmp_path, monkeypatch, method):
    """timing/epoch_method_refresh_seconds: the method-specific refresh, for the full ensemble the
    training of members 1..K-1, never the common denoiser. timing/intrinsic_scoring_seconds_per_update:
    the scoring work inside PMPO updates, part of timing/pmpo_seconds_per_update, never the rest of
    an update, the refresh or the real-test intrinsic evaluation."""
    calls = instrument(monkeypatch, method)
    tiny, extra = METHOD_SETUPS[method]
    trainer, logs = launch(tmp_path / "run", monkeypatch, tiny, BASE + SEEDED + extra)
    calls.trainer = trainer
    run(trainer)

    def count(tag, epoch, in_pmpo=None):
        return sum(1 for t, e, p in calls if t == tag and e == epoch and (in_pmpo is None or p == in_pmpo))

    epochs = dict(sorted((e, r) for e, r in logs if "timing/epoch_seconds" in r))
    assert len(epochs) == 2
    previous = 0
    for epoch, r in epochs.items():
        steps, previous = r["progress/pmpo_updates"] - previous, r["progress/pmpo_updates"]
        refresh, denoiser = r["timing/epoch_method_refresh_seconds"], r["timing/epoch_denoiser_seconds"]
        expected = trainer.ensemble.members - 1 if method == "ensemble" else 1
        assert count("refresh", epoch) == expected and count("common", epoch) == 1
        assert SLEEP * expected <= refresh < SLEEP * expected + COMMON_SLEEP  # never the common denoiser
        if method == "ensemble":  # the denoiser total overlaps the additional members: never add both
            assert denoiser >= refresh + COMMON_SLEEP
        scoring = r["timing/intrinsic_scoring_seconds_per_update"] * steps
        pmpo = r["timing/pmpo_seconds_per_update"] * steps
        in_updates = count("scoring", epoch, in_pmpo=True)
        assert in_updates >= steps and scoring >= SLEEP * in_updates
        assert count("update", epoch) == steps and pmpo - scoring >= SLEEP * steps  # inside PMPO time, not the rest
        assert pmpo == pytest.approx(r["timing/epoch_pmpo_seconds"])
    # The final evaluation (scored after the last PMPO phase) did not add to the training-scoring time.
    (_, first), (_, last) = epochs.items()
    last_steps = last["progress/pmpo_updates"] - first["progress/pmpo_updates"]
    assert trainer._scoring_timer.seconds() == pytest.approx(last["timing/intrinsic_scoring_seconds_per_update"] * last_steps)
    if method in ("laplace", "plan2explore"):  # evaluation really went through the slowed scoring code
        assert any(tag == "scoring" and not in_pmpo for tag, _, in_pmpo in calls)


@pytest.mark.parametrize("method", ["lcg", "ensemble", "laplace", "plan2explore"])
def test_timers_leave_training_rewards_and_rng_unchanged(tmp_path, monkeypatch, method):
    """The scoring timer only records clock times (CUDA: events): weights, optimizers, schedules,
    RNG, replay, rewards and every non-timing logged value are identical with it removed."""
    tiny, extra = METHOD_SETUPS[method]
    on, logs_on = launch(tmp_path / "on", monkeypatch, tiny, BASE + SEEDED + extra)
    assert isinstance(on._scoring_timer, RegionTimer)
    assert on._rl_env.scoring_timer is on.agent.actor_critic.scoring_timer is on._scoring_timer
    run(on)
    monkeypatch.undo()
    off, logs_off = launch(tmp_path / "off", monkeypatch, tiny, BASE + SEEDED + extra)
    off._rl_env.scoring_timer = off.agent.actor_critic.scoring_timer = None
    off._scoring_timer = RegionTimer(torch.device("cpu"))  # read, never entered
    run(off)
    assert_same_training(on, off)
    assert training_rows(logs_on) == training_rows(logs_off)
    assert [scalars(r) for _, r in logs_on] == [scalars(r) for _, r in logs_off]
    assert off._scoring_timer.seconds() == 0 < on._scoring_timer.seconds()
    assert not [k for k in on.state_dict() if "timer" in k or "seconds" in k]  # nothing stale to restore on resume


def test_region_timer_counts_nested_regions_once(monkeypatch):
    clock = iter([1.0, 3.0, 10.0, 14.0])  # outer region 1-3 (inner region not timed), then 10-14
    monkeypatch.setattr(utils.time, "perf_counter", lambda: next(clock))
    timer = RegionTimer(torch.device("cpu"))
    with timer.region():
        with timer.region():
            pass
    with timer.region():
        pass
    assert timer.seconds() == 6.0
    timer.reset()
    assert timer.seconds() == 0
