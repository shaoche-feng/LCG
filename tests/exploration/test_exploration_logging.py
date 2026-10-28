"""Exploration-stage logging (terminal, W&B rows and axes) and the real-environment intrinsic
evaluation: seeded test episodes, alignment with each method's imagination scoring path,
isolation from training and reward-normalization state, uniform omissions, logging-on/off
equivalence, counters and resume."""
from copy import deepcopy
import json
from pathlib import Path
import re
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import exploration_trainer as trainer_module
from data.segment import SegmentId
from data.utils import make_segment
from envs.world_model_env import ImaginedCandidate
from exploration_trainer import ExplorationTrainer
from intrinsic_eval import sampler_windows
from test_ensemble import ENS
from test_exploration_trainer import compose, FakeEnv, ROOT, TINY, UNIFORM
from test_laplace import LAP
from test_plan2explore import P2E
from test_test_diagnostics import assert_same_training, BASE, Crash, crash_at_test_agent, FRESH, REUSE, run, \
    same, training_rows, yaml_value
from utils import rng_state

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

    def __init__(self, run_id="fakerun1", starting_step=0):
        self.run, self.run_id, self.starting_step = None, run_id, starting_step
        self.inits, self.metrics, self.config_updates, self.logged = [], [], [], []
        self.summary = {}
        self.config = SimpleNamespace(update=lambda d, **kw: self.config_updates.append(d))

    def init(self, **kwargs):
        self.inits.append(kwargs)
        self.run = SimpleNamespace(id=kwargs.get("id") or self.run_id, project=kwargs.get("project"), entity=None,
                                   name=kwargs.get("name"), group=kwargs.get("group"), url="https://wandb.test/run",
                                   starting_step=self.starting_step)

    def save(self, *args, **kwargs):
        pass

    def define_metric(self, name, step_metric=None):
        self.metrics.append((name, step_metric))

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


def pmpo_rows(logs):
    return [(e, r) for e, r in logs if "actor_critic/train/num_batch_train_actor_critic" in r]


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
    # Test interactions: 4 x 6 steps per point, 2 x 6 at the end; training budget unchanged.
    totals = [r[k] for _, r in logs for k in r if k.endswith("test/total_new_env_steps")]
    assert totals == [24, 48, 60] and trainer.num_test_steps == 60 and trainer.train_dataset.num_steps == 24
    for epoch in (1, 2):
        block = merged(logs, "test_eval/", epoch)
        assert block["test_eval/intrinsic_applicable"] == 1 and block["test_eval/intrinsic_scored"] == 1
        for seed in SEEDS:
            assert block[f"test_eval/seed_{seed}/length"] == 6
            assert block[f"test_eval/seed_{seed}/intrinsic_scored_steps"] == 6  # final observation completes step 5
            raw, scaled = block[f"test_eval/seed_{seed}/intrinsic_raw_return"], block[f"test_eval/seed_{seed}/intrinsic_scaled_return"]
            assert np.isfinite(raw) and np.isfinite(scaled) and raw != scaled
            assert block[f"test_eval/seed_{seed}/intrinsic_raw_per_step"] == pytest.approx(raw / 6)
        returns = [block[f"test_eval/seed_{s}/intrinsic_raw_return"] for s in SEEDS]
        assert block["test_eval/intrinsic_raw_return_mean"] == pytest.approx(np.mean(returns))
        assert block["test_eval/intrinsic_raw_return_std"] == pytest.approx(np.std(returns))
    final = merged(logs, "final_test_eval/")
    assert {k.split("/")[1] for k in final if "/seed_" in k} == {"seed_11", "seed_12"}
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
    assert block["final_test_eval/intrinsic_scored"] == 1
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
    block = merged(logs, "test_eval/", 1)
    assert block["test_eval/intrinsic_applicable"] == 0 and block["test_eval/intrinsic_scored"] == 0
    assert block["test_eval/seed_3001/length"] == 6 and not any("return" in k for k in block)
    saved = json.loads(Path("evaluation/epoch_00001.json").read_text())
    assert saved["intrinsic"] == dict(applicable=False, scored=False, scorer=None,
                                      reason="not applicable: uniform random has no intrinsic scorer")
    summaries = [r for _, r in logs if "epoch_summary/denoiser_loss_mean" in r]
    assert len(summaries) == 2 and all(r["progress/pmpo_updates"] == 0 for r in summaries)
    assert not any(k.startswith("epoch_summary/pmpo_") or k.startswith("epoch_summary/intrinsic") for r in summaries for k in r)


# ---------------------------------------------------------------- rows, axes, counters

def test_epoch_rows_counters_axes_and_training_only_epochs(tmp_path, monkeypatch, capsys):
    trainer, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED + ["training.num_final_epochs=1"])
    run(trainer)
    lines = terminal_lines(capsys.readouterr().out)
    summaries = {e: r for e, r in logs if "epoch_summary/denoiser_loss_mean" in r}
    assert sorted(summaries) == [1, 2, 3]
    # Epoch 3 trains without collecting: steps stay, updates advance, still one row per epoch.
    expected = {1: (18, 6, 2, 2, 24), 2: (24, 6, 3, 3, 48), 3: (24, 0, 4, 4, 72)}
    for epoch, (steps, new, denoiser, pmpo, test) in expected.items():
        r = summaries[epoch]
        assert (r["progress/train_env_steps"], r["progress/new_train_env_steps"], r["progress/denoiser_updates"],
                r["progress/pmpo_updates"], r["progress/test_env_steps"]) == (steps, new, denoiser, pmpo, test)
        assert r["timing/epoch_seconds"] > 0 and "timing/epoch_denoiser_seconds" in r and "timing/epoch_pmpo_seconds" in r
        assert "timing/epoch_scorer_refresh_seconds" in r and "timing/epoch_test_seconds" in r
        assert ("timing/epoch_collection_seconds" in r) == (epoch < 3)
        # Raw and scaled intrinsic rewards stay separate (RunningRMS: different values).
        assert r["epoch_summary/intrinsic_raw_reward_mean"] != r["epoch_summary/intrinsic_scaled_reward_mean"]
        for name in ("loss_actor", "loss_value", "policy_entropy", "kl_prior", "positive_fraction", "actor_grad_norm"):
            assert np.isfinite(r[f"epoch_summary/pmpo_{name}_mean"])
        # Every row carries the env-step axis at the time it was logged.
        assert all(row["progress/train_env_steps"] == steps for e, row in logs if e == epoch)
    assert trainer.agent.actor_critic.updates.item() == 4 and trainer.num_batch_train.denoiser == 4
    # PMPO rows keep their own update counter (W&B step metric), raw and policy reward separate.
    pmpo = [r for _, r in logs if "actor_critic/train/num_batch_train_actor_critic" in r]
    assert [r["actor_critic/train/num_batch_train_actor_critic"] for r in pmpo] == [1, 2, 3, 4]
    assert all("actor_critic/train/lcg_raw_reward_mean" in r and "actor_critic/train/lcg_policy_reward_mean" in r for r in pmpo)
    # The denoiser test loss and the LCG refresh identity stay in W&B only.
    assert any("denoiser/test/loss_denoising" in r for _, r in logs)
    refresh = [r for _, r in logs if "lcg/h_D_fingerprint" in r]
    assert [r["lcg/round"] for r in refresh] == [1, 2, 3] and all(np.isfinite(r["lcg/banks_fingerprint"]) for r in refresh)
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
        stat = {k: [float(r[f"actor_critic/train/lcg_{k}"]) for r in rows]
                for k in ("raw_reward_mean", "raw_reward_max", "policy_reward_mean", "policy_reward_max")}
        expected.append(f"Imagined intrinsic reward/step: raw mean {np.mean(stat['raw_reward_mean']):.4g}, "
                        f"max {max(stat['raw_reward_max']):.4g} | scaled mean {np.mean(stat['policy_reward_mean']):.4g}, "
                        f"max {max(stat['policy_reward_max']):.4g}")
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
    # The method's refresh diagnostics stay in W&B.
    prefix = dict(ensemble="ensemble/", plan2explore="plan2explore/", laplace="laplace/")[method]
    assert any(k.startswith(prefix) for _, r in logs for k in r)


def test_wandb_axes_provenance_milestones_and_unchanged_training(tmp_path, monkeypatch, capsys):
    """Logging on (bars, epoch summaries, an active W&B run) vs off: identical training."""
    quiet = ["logging.progress=none", "logging.epoch_summary=false"]
    off, logs_off = launch(tmp_path / "off", monkeypatch, TINY, BASE + FRESH + quiet)
    run(off)
    assert "%|" not in capsys.readouterr().out
    assert not [r for _, r in logs_off if "epoch_summary/denoiser_loss_mean" in r]
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
    assert axes["actor_critic/train/*"] == "actor_critic/train/num_batch_train_actor_critic"
    assert axes["denoiser/train/*"] == "denoiser/train/num_batch_train_denoiser"
    assert axes["epoch_summary/*"] == axes["test_eval/*"] == axes["progress/*"] == "epoch"
    globs = [n for n in axes if n.endswith("*")]
    for _, row in logs_on:
        for key in row:
            matches = [g for g in globs if key.startswith(g[:-1])] + [n for n in axes if n == key]
            assert len([g for g in globs if key.startswith(g[:-1])]) <= 1 and matches, key
    provenance = fake.config_updates[0]["provenance"]
    assert provenance["seed"] == 7 and provenance["method"] == "lcg" and provenance["source"] == on.source
    assert provenance["method_settings"]["reward_scaling"] == on._lcg_reward_scaling
    assert provenance["evaluation"]["every"] == 1
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
    resume_rows = [r for _, r in logs if "resume/from_epoch" in r]
    assert len(resume_rows) == 1 and resume_rows[0]["resume/from_epoch"] == 1 and resume_rows[0]["resume/count"] == 1
    # This fake server holds no rows, so every committed row was backfilled (see the backfill test).
    assert resume_rows[0]["resume/backfilled_rows"] == resume_rows[0]["resume/wandb_committed_rows"] > 0
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
    assert not [k for k in block if "/seed_" in k]
    assert {k.split("/")[1] for k in block if "/episode_" in k} == {f"episode_{i}" for i in range(4)}
    assert resumed.resume_count == 1


@pytest.mark.parametrize("server", ["lost", "ahead"])
def test_resume_backfills_rows_the_wandb_server_lost_and_counts_replayed_ones(tmp_path, monkeypatch, capsys, server):
    """A hard crash can kill W&B's upload process: the server then holds fewer rows than the
    checkpoint committed (backfilled from the local mirror), or more (an uncommitted epoch that is
    replayed: counted)."""
    crash_at_test_agent(monkeypatch, epoch=2)
    crashed, logs = launch(tmp_path / "run", monkeypatch, TINY, BASE + SEEDED + ["wandb.mode=offline"], fake_wandb=FakeWandb())
    with pytest.raises(Crash):
        crashed._run()
    committed = torch.load(tmp_path / "run/checkpoints/state.pt", weights_only=False)["wandb_steps"]
    epoch1 = [row for e, row in logs if e == 1]
    assert committed == len(epoch1) > 0
    monkeypatch.undo()
    starting = 3 if server == "lost" else committed + 7
    fake = FakeWandb(starting_step=starting)
    capsys.readouterr()
    resumed, logs = launch(tmp_path / "run", monkeypatch, None, [], resume=True, fake_wandb=fake)
    resume_row = resumed._start_logs[-1]
    lines = terminal_lines(capsys.readouterr().out)  # essential resume messages stay visible
    assert "  Resumed at epoch 1 (resume #1)" in lines
    message = (f"W&B: backfilled {committed - 3} committed rows the server did not receive before the crash"
               if server == "lost" else
               "W&B: 7 rows logged after the checkpoint are on the server; their epochs are replayed")
    assert message in lines
    assert resume_row["resume/wandb_server_rows"] == starting and resume_row["resume/wandb_committed_rows"] == committed
    backfilled = [r for r in fake.logged if r.get("resume/backfilled") == 1]
    if server == "lost":
        # Exactly the committed rows the server lacks, in order, with their epoch and values.
        assert resume_row["resume/backfilled_rows"] == committed - 3 and resumed.wandb_steps == committed
        expected = [{"epoch": 1, **{k: float(v) for k, v in row.items() if np.ndim(v) == 0}, "resume/backfilled": 1}
                    for row in epoch1[3:]]
        assert backfilled == expected
    else:
        assert not backfilled and resume_row["resume/server_rows_beyond_checkpoint"] == 7
        assert resumed.wandb_steps == committed + 7
    run(resumed)
    # The mirror's latest entry per step continues from the reconciled step.
    steps = [json.loads(line)["step"] for line in Path("wandb_history.jsonl").read_text().splitlines()]
    assert steps[-1] == resumed.wandb_steps - 1
