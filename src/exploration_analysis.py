"""Analysis of the collected exploration dataset at milestones. Nothing is trained, collected or
evaluated; milestones are only read, with the downstream loader data.milestone.load_milestone: every manifest
file is checked against its checksum, and the replay is the exact episode versions the milestone refers to,
each checked against its recorded checksum.

Tests (docs/analysis/reward_free_experiment_analysis.md, "Collected exploration dataset"); --test is repeatable:
- task-reward-coverage: per task, the fraction of the replay's transitions whose reward exceeds the task's
  threshold and the fraction of episodes with at least one such transition.
- task-reward-lasting: bouts of consecutive above-threshold transitions, kept within episode boundaries;
  per-bout lengths, per-episode counts (episodes with zero bouts included) and a summary.
- follow-intrinsic-reward (run on its own): the frozen scorer saved at one milestone of a learned method
  (docs/analysis/restore_scorer.md) scores the next recorded collection window of that run and the same step
  range of a separate Uniform run; see "Follow intrinsic reward" below.

The coverage and lasting tests read only the replay (CPU); the denoiser weights load_milestone also loads are
not used.

Rewards are the recorded native task rewards (Episode.task_rew) in their stored units: each environment step
sums the reward of its executed action-repeat substeps (data.milestone.RETURN_CONVENTION). A transition is
above the threshold when reward > threshold (strict, as the downstream return thresholds). One threshold
per task serves both tests and every milestone of the invocation, so all milestones must record the same
domain and task labels.

    python src/exploration_analysis.py --test task-reward-coverage [--test task-reward-lasting] \
        --milestone <run>/milestones/steps_100000 [--milestone ...] \
        --threshold stand=1.0 [--threshold hop=1.0] --output <folder>

Writes into --output (never overwriting): coverage.csv; lasting_summary.csv, lasting_episodes.csv and
lasting_bouts.csv; analysis.json (sources, thresholds, tests). Episodes unfinished at the milestone cut are
marked, and a bout reaching such an episode's last stored step is marked censored_at_cut.

Follow intrinsic reward
-----------------------
    python src/exploration_analysis.py --test follow-intrinsic-reward --method lcg \
        --milestone <lcg run>/milestones/steps_N --uniform-run <uniform run> --output <folder> \
        [--scoring-seed 0] [--device cpu|cuda]

- Scorer: rebuilt from the milestone alone -- its denoiser file(s) and scorer checkpoint files (LCG:
  historical_precision.pt, jvp_probe_bank.pt, lcg_scaling.pt; Plan2Explore: MLP_member_<k>.pt,
  plan2explore_normalization.pt and the action bounds of exploration_policy.pt; Laplace: parameter_bank.pt;
  full ensemble: denoiser_member_<k>.pt) -- with the run's saved config (config/exploration.yaml) and the
  existing scorer classes. Nothing is refreshed, redrawn or refitted.
- Window: the next recorded collection window, global steps [N, N + collection.train.steps_per_epoch) in
  collection order, read from the run's replay (<run>/dataset/train, versions checked against their recorded
  checksums), which must extend the milestone's replay exactly. The Uniform batch is the same step range of
  the Uniform run's replay. One collection environment per run is required (episode order = collection order).
- Each transition is scored as intrinsic_reward/intrinsic_eval.py scores real steps: the sampler window
  o_{t-n+1..t}, a_{t-n+1..t} (earlier rows of the same episode; zero padding only before the episode start),
  LCG with the real successor o_{t+1} (next row, the final observation, or the successor recorded by a later
  milestone at that cut; a step without one is not scored), Plan2Explore with the feature of o_t in its context
  and a_t. Randomness: the full ensemble's and Laplace's shared initial noise is one tensor drawn once from a
  CPU generator seeded with --scoring-seed, so transition j of both batches gets the same draw; the global RNG
  is seeded with it as well before each batch.
- Writes follow_batches.csv (one row per batch), follow_summary.csv (difference learned - Uniform of the
  mean raw and scaled score), follow_transitions.csv (every transition) and analysis.json. Scaled scores are
  the scorer's own policy reward of the raw score, when the milestone holds what that needs.

Noisy TV
--------
    python src/exploration_analysis.py --test noisy-tv --condition clean=<clean run>/milestones/steps_100000 \
        --condition tv16=<TV16 run>/milestones/steps_100000 --condition tv24=<TV24 run>/milestones/steps_100000 \
        --output <folder> [--seeds 3001-3030] [--action-seed 0] [--intrinsic [--scoring-seed 0]] [--no-videos] \
        [--device cpu|cuda]

- One method (docs/implement/AlEATRORISTIC_EXPERIMENT_DESIGN.md): each condition's milestone comes from that
  condition's run (saved env.train.noisy_tv: clean, tv16, tv24); method, task group, trigger, panel and milestone
  step must agree.
- Trajectories: <output>/trajectories/<condition>/ (evaluation.json and the episodes, with per-frame qpos/qvel and
  TV membership/activation). Read when present and consistent; otherwise evaluated first by
  evaluate_noisy_tv_milestone: the existing exploration test collector, one episode per reset seed in the run's
  own env.test (its evaluation TV seed), with the milestone's exploration policy (sampled actions) or uniform
  actions. Nothing is trained, refreshed or refitted; the policy weights are hashed before and after.
- Writes noisy_tv_episodes.csv/.json, noisy_tv_summary.csv, noisy_tv_paired.csv and noisy_tv_paired_summary.csv
  (TV - Clean per reset seed), with --intrinsic noisy_tv_intrinsic.csv and noisy_tv_transitions.csv (each
  milestone's frozen scorer, as follow-intrinsic-reward restores it), occupancy/dwell/coverage plots (when
  matplotlib is available), videos/ and analysis.json (metric and bin definitions, settings, provenance).

The coverage and lasting tests import neither trainer nor envs (with MUJOCO_GL=egl, importing dm_control opens
the GPU devices). follow-intrinsic-reward imports the scorer modules (which import envs) only when it runs,
and never the trainers.
"""
import argparse
import json
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data.milestone import (RETURN_CONVENTION, discover_milestones, exploration_identity, load_milestone, method_label,
                            read_manifest, state_sha256, write_json)
from downstream_analysis import write_csv
from utils import file_sha256

FOLLOW = "follow-intrinsic-reward"
NOISY_TV = "noisy-tv"
TESTS = ("task-reward-coverage", "task-reward-lasting", FOLLOW, NOISY_TV)
ALONE = (FOLLOW, NOISY_TV)  # tests that run on their own
# Output label -> intrinsic_reward.method of the learned methods (data.milestone.OUTPUT_LABELS).
LEARNED = {"lcg": "lcg", "full-ensemble": "ensemble_disagreement", "laplace-ensemble": "laplace_disagreement",
           "plan2explore": "plan2explore_disagreement", "lcg-predictor": "lcg_predictor"}
FOLLOW_KEY = ["method", "domain", "milestone", "collected_steps", "window_start", "window_end"]
FOLLOW_BATCH_COLUMNS = FOLLOW_KEY + ["batch", "run", "segments", "num_transitions", "num_scored", "raw_mean",
                                     "scaled_mean"]
FOLLOW_SUMMARY_COLUMNS = FOLLOW_KEY + ["scoring_seed", "num_transitions_learned", "num_transitions_uniform",
                                       "num_scored_learned", "num_scored_uniform", "raw_mean_learned",
                                       "raw_mean_uniform", "raw_difference", "scaled_mean_learned",
                                       "scaled_mean_uniform", "scaled_difference"]
FOLLOW_TRANSITION_COLUMNS = FOLLOW_KEY + ["batch", "position", "global_step", "episode", "row", "raw", "scaled"]
FOLLOW_OUTPUTS = ["follow_batches.csv", "follow_summary.csv", "follow_transitions.csv"]
KEY = ["method", "domain", "task", "milestone", "collected_steps", "threshold"]
COVERAGE_COLUMNS = KEY + ["num_episodes", "num_transitions", "transitions_above", "fraction_transitions_above",
                          "episodes_with_above", "fraction_episodes_with_above"]
SUMMARY_COLUMNS = KEY + ["num_episodes", "episodes_with_bouts", "num_bouts", "bouts_per_episode",
                         "bout_length_mean", "bout_length_median", "bout_length_max", "transitions_in_bouts",
                         "bouts_censored_at_cut"]
EPISODE_COLUMNS = KEY + ["episode", "episode_length", "unfinished_at_cut", "num_bouts", "longest_bout",
                         "transitions_above"]
BOUT_COLUMNS = KEY + ["episode", "bout", "start", "length", "censored_at_cut"]
OUTPUTS = {"task-reward-coverage": ["coverage.csv"],
           "task-reward-lasting": ["lasting_summary.csv", "lasting_episodes.csv", "lasting_bouts.csv"]}


def parse_thresholds(items: Sequence[str]) -> Dict[str, float]:
    """task=value pairs, each task once, finite values."""
    thresholds = {}
    for item in items:
        task, sep, value = item.partition("=")
        task = task.strip()
        if not sep or not task:
            raise ValueError(f"--threshold takes task=value, got {item!r}")
        try:
            number = float(value)
        except ValueError:
            raise ValueError(f"--threshold {item!r}: {value!r} is not a number") from None
        if not math.isfinite(number):
            raise ValueError(f"--threshold {item!r}: the threshold must be finite")
        if task in thresholds:
            raise ValueError(f"--threshold gives task {task!r} more than once")
        thresholds[task] = number
    if not thresholds:
        raise ValueError("Give at least one --threshold task=value")
    return thresholds


def bouts(above: np.ndarray) -> List[Tuple[int, int]]:
    """(start, length) of each maximal run of True in one episode's above-threshold flags."""
    flags = np.concatenate(([False], np.asarray(above, dtype=bool), [False]))
    edges = np.flatnonzero(flags[1:] != flags[:-1])
    starts, ends = edges[0::2], edges[1::2]
    return [(int(s), int(e - s)) for s, e in zip(starts, ends)]


def read_replay(milestone_path: Path) -> Dict[str, Any]:
    """Identity and per-episode task rewards of one milestone's replay (read in place)."""
    path = Path(milestone_path)
    manifest = read_manifest(path)
    if not manifest.get("task_names"):
        raise ValueError(f"Milestone {path} records no task reward labels")
    # The downstream loader validates the milestone for one task; its replay holds every recorded task.
    milestone = load_milestone(path, manifest["task_names"][0], map_location=torch.device("cpu"))
    dataset, manifest = milestone.dataset, milestone.manifest
    exploration = manifest.get("exploration", {})
    rewards = []
    for episode_id in range(dataset.num_episodes):
        episode = dataset.load_episode(episode_id)
        if episode.task_rew is None or list(episode.task_names) != list(dataset.task_names):
            raise ValueError(f"Milestone {path}: episode {episode_id} has task labels {episode.task_names}, "
                             f"the replay {dataset.task_names}")
        rewards.append(episode.task_rew.double().numpy())
    return dict(path=path, name=path.name, manifest=manifest, task_names=list(dataset.task_names),
                method=method_label(bool(exploration.get("intrinsic_reward_enabled")), exploration.get("method")),
                domain=manifest["domain"], collected_steps=int(manifest["collected_steps"]),
                unfinished=set(int(k) for k in milestone.successor_observations), rewards=rewards,
                dataset_sha256=manifest["files"]["dataset"]["sha256"])


def key_row(replay: Dict[str, Any], task: str, threshold: float) -> Dict[str, Any]:
    return dict(method=replay["method"], domain=replay["domain"], task=task, milestone=replay["name"],
                collected_steps=replay["collected_steps"], threshold=threshold)


def task_reward_coverage(replay: Dict[str, Any], task: str, threshold: float) -> Dict[str, Any]:
    column = replay["task_names"].index(task)
    above = [r[:, column] > threshold for r in replay["rewards"]]
    transitions = sum(len(a) for a in above)
    transitions_above = int(sum(int(a.sum()) for a in above))
    episodes_with = sum(bool(a.any()) for a in above)
    return dict(**key_row(replay, task, threshold), num_episodes=len(above), num_transitions=transitions,
                transitions_above=transitions_above,
                fraction_transitions_above=transitions_above / transitions if transitions else float("nan"),
                episodes_with_above=episodes_with,
                fraction_episodes_with_above=episodes_with / len(above) if above else float("nan"))


def task_reward_lasting(replay: Dict[str, Any], task: str, threshold: float
                        ) -> Tuple[Dict[str, Any], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """(summary row, one row per episode, one row per bout). Bouts never cross episode boundaries."""
    column, key = replay["task_names"].index(task), key_row(replay, task, threshold)
    episode_rows, bout_rows, lengths, censored = [], [], [], 0
    for episode_id, rewards in enumerate(replay["rewards"]):
        above = rewards[:, column] > threshold
        unfinished = episode_id in replay["unfinished"]
        found = bouts(above)
        for k, (start, length) in enumerate(found):
            cut = unfinished and start + length == len(above)
            censored += cut
            lengths.append(length)
            bout_rows.append(dict(**key, episode=episode_id, bout=k, start=start, length=length, censored_at_cut=int(cut)))
        episode_rows.append(dict(**key, episode=episode_id, episode_length=len(above), unfinished_at_cut=int(unfinished),
                                 num_bouts=len(found), longest_bout=max((l for _, l in found), default=0),
                                 transitions_above=int(above.sum())))
    n = len(episode_rows)
    summary = dict(**key, num_episodes=n, episodes_with_bouts=sum(r["num_bouts"] > 0 for r in episode_rows),
                   num_bouts=len(lengths), bouts_per_episode=len(lengths) / n if n else float("nan"),
                   bout_length_mean=float(np.mean(lengths)) if lengths else float("nan"),
                   bout_length_median=float(np.median(lengths)) if lengths else float("nan"),
                   bout_length_max=max(lengths, default=0), transitions_in_bouts=int(sum(lengths)),
                   bouts_censored_at_cut=int(censored))
    return summary, episode_rows, bout_rows


def check_replays(replays: List[Dict[str, Any]], thresholds: Dict[str, float]) -> None:
    """One domain and task-label set for the whole invocation, every thresholded task recorded."""
    domains = {r["domain"] for r in replays}
    labels = {tuple(r["task_names"]) for r in replays}
    if len(domains) != 1 or len(labels) != 1:
        raise ValueError("All milestones of one invocation must record the same domain and task labels (one threshold "
                         f"per task across methods); got domains {sorted(domains)}, labels {sorted(labels)}")
    missing = sorted(set(thresholds) - set(replays[0]["task_names"]))
    if missing:
        raise ValueError(f"No recorded reward for task(s) {missing}; the replays record {replays[0]['task_names']}")
    names = [str(r["path"].resolve()) for r in replays]
    if len(set(names)) != len(names):
        raise ValueError("A milestone is given more than once")


def run(tests: Sequence[str], milestones: Sequence[Path], thresholds: Dict[str, float], output: Path,
        argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Run the selected replay tests (coverage, lasting) on every milestone; returns the written files by name."""
    tests = list(dict.fromkeys(tests))
    unknown = [t for t in tests if t not in TESTS]
    if unknown or not tests:
        raise ValueError(f"--test takes {', '.join(TESTS)}; got {tests}")
    if FOLLOW in tests:
        raise ValueError(f"{FOLLOW} runs on its own (one milestone and a Uniform run); use follow_intrinsic_reward()")
    if NOISY_TV in tests:
        raise ValueError(f"{NOISY_TV} runs on its own (one milestone per condition); use noisy_tv_analysis()")
    output = Path(output)
    files = [output / f for t in tests for f in OUTPUTS[t]] + [output / "analysis.json"]
    existing = [str(f) for f in files if f.exists()]
    if existing:
        raise FileExistsError(f"Not overwriting existing results: {existing}")
    replays = [read_replay(Path(m)) for m in milestones]
    if not replays:
        raise ValueError("Give at least one --milestone")
    check_replays(replays, thresholds)
    tasks = [t for t in replays[0]["task_names"] if t in thresholds]  # recorded order
    written = {}
    if "task-reward-coverage" in tests:
        rows = [task_reward_coverage(r, t, thresholds[t]) for r in replays for t in tasks]
        write_csv(output / "coverage.csv", rows, COVERAGE_COLUMNS)
        written["coverage.csv"] = output / "coverage.csv"
    if "task-reward-lasting" in tests:
        summaries, episodes, bout_rows = [], [], []
        for r in replays:
            for t in tasks:
                s, e, b = task_reward_lasting(r, t, thresholds[t])
                summaries.append(s), episodes.extend(e), bout_rows.extend(b)
        for name, rows, columns in (("lasting_summary.csv", summaries, SUMMARY_COLUMNS),
                                    ("lasting_episodes.csv", episodes, EPISODE_COLUMNS),
                                    ("lasting_bouts.csv", bout_rows, BOUT_COLUMNS)):
            write_csv(output / name, rows, columns)
            written[name] = output / name
    write_json(output / "analysis.json", dict(
        tests=tests, thresholds=thresholds, comparison="reward > threshold (strict), per environment step",
        reward_units=RETURN_CONVENTION, domain=replays[0]["domain"], task_names=replays[0]["task_names"],
        sources=[dict(method=r["method"], milestone=r["name"], path=str(r["path"].resolve()),
                      collected_steps=r["collected_steps"], num_episodes=len(r["rewards"]),
                      unfinished_at_cut=sorted(r["unfinished"]), dataset_sha256=r["dataset_sha256"],
                      epoch=r["manifest"].get("epoch"), source=r["manifest"].get("source"))
                 for r in replays],
        command=list(argv) if argv is not None else None))
    written["analysis.json"] = output / "analysis.json"
    return written


# ---------------------------------------------------------------------------------------- follow intrinsic reward

def run_config(run_dir: Path):
    """The run's saved, resolved exploration config, normalized as the trainer does on load."""
    from omegaconf import OmegaConf
    from intrinsic_reward.ensemble import migrate_ensemble_layout
    from utils import normalize_saved_config
    path = Path(run_dir) / "config" / "exploration.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: {run_dir} is not an exploration run folder")
    cfg = migrate_ensemble_layout(normalize_saved_config(OmegaConf.load(path)))
    OmegaConf.resolve(cfg)
    return cfg


def run_replay(run_dir: Path):
    """The run's training replay as last saved (<run>/dataset/train/info.pt), read only."""
    from data import Dataset
    directory = Path(run_dir) / "dataset" / "train"
    if not (directory / "info.pt").is_file():
        raise FileNotFoundError(f"{directory / 'info.pt'} is missing: no saved replay in run {run_dir}")
    dataset = Dataset(directory, "run_replay", save_on_disk=False)
    dataset.load_from_default_path()
    dataset.is_static = True
    if not dataset.versioned or dataset.file_sha256 is None or None in dataset.file_sha256:
        raise ValueError(f"{directory}: the replay has no versioned episode files with checksums")
    return dataset


def window_segments(start_idx: Sequence[int], lengths: Sequence[int], start: int, end: int) -> List[Tuple[int, int, int]]:
    """(episode, first row, end row) of the replay rows at global steps [start, end), in collection order."""
    segments = []
    for k, (s, n) in enumerate(zip(start_idx, lengths)):
        a, b = max(start, int(s)), min(end, int(s) + int(n))
        if a < b:
            segments.append((k, a - int(s), b - int(s)))
    return segments


def load_checked_episode(dataset, episode_id: int):
    """One episode version of a run replay, checked against the checksum recorded when it was written."""
    path = dataset.episode_files()[episode_id]
    if not path.is_file():
        raise FileNotFoundError(f"{dataset.name}: episode file {path} is missing")
    if file_sha256(path) != dataset.file_sha256[episode_id]:
        raise ValueError(f"{dataset.name}: episode file {path} does not match its checksum")
    return dataset.load_episode(episode_id)


def recorded_successor(run_dir: Path, episode_id: int, length: int):
    """The observation after row length-1 of an episode cut at a milestone (dataset.pt successor_observations
    of a milestone of this run whose index holds the episode at exactly that length), else None."""
    milestones = Path(run_dir) / "milestones"
    if not milestones.is_dir():
        return None, None
    for path, manifest in discover_milestones(milestones):
        entry = manifest["files"]["dataset"]
        if file_sha256(path / entry["path"]) != entry["sha256"]:
            raise ValueError(f"Milestone {path}: dataset file does not match its checksum")
        saved = torch.load(path / entry["path"], weights_only=False)
        lengths = saved["index"]["lengths"]
        if episode_id < len(lengths) and int(lengths[episode_id]) == length and episode_id in saved["successor_observations"]:
            return saved["successor_observations"][episode_id], f"successor_observations of {path.name}"
    return None, None


def as_stored(frame: torch.Tensor) -> torch.Tensor:
    """A collector frame ([-1, 1] float) as the replay holds it: Episode.save's uint8 levels, read back by
    Episode.load. Milestone successor observations are the collector's float frames, one float ulp from that."""
    return frame.float().add(1).div(2).mul(255).round().byte().div(255).mul(2).sub(1)


class _Rows:
    """Rows [c, b) of an episode, shaped like an Episode for intrinsic_eval.sampler_windows."""

    def __init__(self, obs, act, successor) -> None:
        self.obs, self.act, self.info = obs, act, {"final_observation": successor}

    def __len__(self) -> int:
        return self.obs.shape[0]


def window_batch(run_dir: Path, dataset, start: int, end: int, n: int) -> Dict[str, Any]:
    """The transitions at global steps [start, end) of a run's replay, with everything every scorer reads:
    sampler windows (obs, act), Plan2Explore context, frame and action, and the real successor (or None)."""
    from intrinsic_reward.intrinsic_eval import sampler_windows
    from intrinsic_reward.plan2explore import context_window
    if int(dataset.num_steps) < end:
        raise ValueError(f"The next window [{start}, {end}) is unavailable in {run_dir}: its replay ends at step "
                         f"{int(dataset.num_steps)}")
    segments = window_segments(dataset.start_idx, dataset.lengths, start, end)
    fields = {k: [] for k in ("obs", "act", "frame", "action", "context", "context_act", "successor", "position")}
    successor_sources = {}
    for k, a, b in segments:
        episode = load_checked_episode(dataset, k)
        c = max(0, a - n)  # n earlier rows of the same episode give every window row its exact history
        final = episode.info.get("final_observation") if isinstance(episode.info, dict) else None
        if b < len(episode):
            successor, source = episode.obs[b], "next row"
        elif final is not None:
            successor, source = final, "final_observation"
        else:
            successor, source = recorded_successor(run_dir, k, b)
            successor = None if successor is None else as_stored(successor)
        successor_sources[k] = source
        rows = _Rows(episode.obs[c:b], episode.act[c:b],
                     None if successor is None else successor.reshape(episode.obs.shape[1:]).to(episode.obs))
        obs_w, act_w, nexts = sampler_windows(rows, n)
        for i in range(a - c, b - c):
            ctx, ctx_act, _ = context_window(rows.obs, rows.act, i, n)
            fields["obs"].append(obs_w[i]), fields["act"].append(act_w[i])
            fields["frame"].append(rows.obs[i]), fields["action"].append(rows.act[i])
            fields["context"].append(ctx), fields["context_act"].append(ctx_act)
            fields["successor"].append(nexts[i] if i < nexts.shape[0] else None)
            fields["position"].append((int(dataset.start_idx[k]) + c + i, k, c + i))
    batch = {k: torch.stack(v) for k, v in fields.items() if k not in ("successor", "position")}
    batch.update(successor=fields["successor"], position=fields["position"], segments=segments,
                 successor_sources=successor_sources)
    return batch


def check_extends(milestone, replay) -> None:
    """The run's replay continues the milestone's exactly: same earlier episodes, and every episode cut at the
    milestone continues from its recorded successor observation."""
    m = milestone.dataset
    if replay.task_names != m.task_names or int(replay.num_episodes) < int(m.num_episodes):
        raise ValueError(f"The run's replay ({replay.num_episodes} episodes, labels {replay.task_names}) does not "
                         f"extend milestone {milestone.path.name} ({m.num_episodes}, {m.task_names})")
    for k in range(int(m.num_episodes)):
        lm, lr = int(m.lengths[k]), int(replay.lengths[k])
        cut = k in milestone.successor_observations
        if lr < lm or (not cut and lr != lm):
            raise ValueError(f"Episode {k}: {lm} rows at milestone {milestone.path.name}, {lr} in the run's replay "
                             f"({'cut' if cut else 'complete'} at the milestone)")
        if cut and lr > lm:
            row = load_checked_episode(replay, k).obs[lm]
            if not torch.equal(row, as_stored(milestone.successor_observations[k].reshape(row.shape))):
                raise ValueError(f"Episode {k}: row {lm} of the run's replay differs from the successor observation "
                                 f"recorded at milestone {milestone.path.name}")


def scorer_requirements(method: str, manifest: Dict[str, Any], members: int) -> List[str]:
    if method == "lcg-predictor":  # imported only here: the method's package imports the feature extractor
        from intrinsic_reward.lcg_predictor import MILESTONE_FILES
        return list(MILESTONE_FILES) + ["exploration_policy.pt"]
    files = {"lcg": ["historical_precision.pt", "jvp_probe_bank.pt", "lcg_scaling.pt"],
             "plan2explore": [f"MLP_member_{k}.pt" for k in range(1, members + 1)]
                             + ["plan2explore_normalization.pt", "exploration_policy.pt"],
             "laplace-ensemble": ["parameter_bank.pt"],
             "full-ensemble": [f"denoiser_member_{k}.pt" for k in range(1, members)]}[method]
    return files


class FrozenScorer:
    """One method's frozen scorer, rebuilt from saved state with the existing scorer classes (no trainer, no
    environment). Construct, load the denoiser weights and the scorer state (load_milestone_files, or a
    caller restoring a training checkpoint through the same objects), then score window batches."""

    def __init__(self, method: str, cfg, action_low: Sequence[float], action_high: Sequence[float],
                 device: torch.device) -> None:
        from hydra.utils import instantiate
        from omegaconf import OmegaConf
        from models.diffusion import Denoiser, DiffusionSampler
        if method not in LEARNED:
            raise ValueError(f"{FOLLOW} needs a learned method ({', '.join(LEARNED)}), got {method!r}")
        self.method, self.cfg, self.device = method, cfg, torch.device(device)
        self.n = int(cfg.agent.denoiser.inner_model.num_steps_conditioning)
        self.seed = cfg.common.seed
        fields = {k: v for k, v in OmegaConf.to_container(cfg.intrinsic_reward).items() if k != "_target_"}
        sigma_cfg = instantiate(cfg.denoiser.sigma_distribution)
        denoiser_cfg = instantiate(cfg.agent.denoiser)
        denoiser_cfg.inner_model.continuous_action_dim = len(action_low)  # as AgentConfig sets it
        members = 1
        if method == "full-ensemble":
            from intrinsic_reward.ensemble import EnsembleDisagreementConfig, EnsembleScorer
            self.method_cfg = EnsembleDisagreementConfig(**fields)
            members = self.method_cfg.num_members
            self.scale_scorer = EnsembleScorer(self.method_cfg)
        self.denoisers = [Denoiser(denoiser_cfg).to(self.device).eval() for _ in range(members)]
        for d in self.denoisers:
            d.setup_training(sigma_cfg)
        sampler_cfg = instantiate(cfg.world_model_env).diffusion_sampler
        self.samplers = [DiffusionSampler(d, sampler_cfg) for d in self.denoisers]
        self.lcg = self.laplace = self.plan2explore = self.lcg_predictor = None
        if method == "lcg":
            from intrinsic_reward.lcg import LCGLifecycle
            self.method_cfg = instantiate(cfg.intrinsic_reward)
            self.lcg = LCGLifecycle(self.method_cfg, sigma_cfg, img_channels=cfg.agent.denoiser.inner_model.img_channels,
                                    img_size=cfg.env.train.size, device=self.device)
            self.lcg.verbose = False
        elif method == "laplace-ensemble":
            from intrinsic_reward.laplace import LaplaceConfig, LaplaceEnsemble, LaplaceScorer
            self.method_cfg = LaplaceConfig(**fields)
            self.laplace = LaplaceEnsemble(self.method_cfg, self.denoisers[0], sigma_cfg,
                                           0 if self.seed is None else self.seed, self.device)
            self.scale_scorer = LaplaceScorer(self.method_cfg)
        elif method == "plan2explore":
            from intrinsic_reward.plan2explore import Plan2Explore, Plan2ExploreConfig
            self.method_cfg = Plan2ExploreConfig(**fields)
            self.plan2explore = Plan2Explore(self.method_cfg, self.denoisers[0], cfg.env.train.size, list(action_low),
                                             list(action_high), 0 if self.seed is None else self.seed, self.device)
        elif method == "lcg-predictor":
            from intrinsic_reward.lcg_predictor import LCGPredictor, LCGPredictorConfig
            self.method_cfg = LCGPredictorConfig(**fields)
            self.lcg_predictor = LCGPredictor(self.method_cfg, self.denoisers[0], cfg.env.train.size, list(action_low),
                                              list(action_high), 0 if self.seed is None else self.seed, self.device)
        self.noise_samples = int(getattr(self.method_cfg, "num_noise_samples", 1))

    # ------------------------------------------------------------------ restore from a milestone

    def load_milestone_files(self, milestone_path: Path, manifest: Dict[str, Any], denoiser_state) -> None:
        """Denoiser weights (member 0 already loaded and checked by load_milestone; members 1.. against the
        manifest identity) and the method's scorer checkpoint files (checksums checked by load_milestone)."""
        path = Path(milestone_path)
        load = lambda name: torch.load(path / name, map_location="cpu", weights_only=False)
        states = [denoiser_state]
        for k in range(1, len(self.denoisers)):
            state = torch.load(path / f"denoiser_member_{k}.pt", map_location="cpu", weights_only=True)
            if state_sha256(state) != manifest["identity"]["member_sha256"][k]:
                raise ValueError(f"Milestone {path}: member {k} weights do not match the recorded identity")
            states.append(state)
        for d, state in zip(self.denoisers, states):
            d.load_state_dict(state)
        if self.method == "lcg":
            hp, bank, sc = load("historical_precision.pt"), load("jvp_probe_bank.pt"), load("lcg_scaling.pt")
            if not hp["round_id"] == bank["round_id"] == sc["round_id"]:
                raise ValueError(f"Milestone {path}: LCG scorer files of different rounds "
                                 f"({hp['round_id']}, {bank['round_id']}, {sc['round_id']})")
            self.lcg.load_state_dict(dict(
                settings=sc["settings"], round_id=hp["round_id"], theta_s_names=hp["theta_s_names"],
                theta_s_dim=hp["theta_s_dim"], h_D=hp["h_D"],
                bank={k: bank[k] for k in ("sigmas", "epsilons", "epsilons_offset", "etas")},
                rms_s2=sc["rms_s2"], reward_divisor=sc["reward_divisor"], reference_raw_mean=sc["reference_raw_mean"]),
                self.denoisers[0])
        elif self.method == "laplace-ensemble":
            bank, lap = load("parameter_bank.pt"), self.laplace
            if bank["names"] != lap.names or [tuple(s) for s in bank["shapes"]] != lap.shapes:
                raise ValueError(f"Milestone {path}: parameter_bank.pt selects other parameters than the config")
            if float(bank["posterior_temperature"]) != float(lap.cfg.posterior_temperature):
                raise ValueError(f"Milestone {path}: parameter_bank.pt temperature differs from the config")
            if self.seed is not None and bank["bank_seed"] != lap.bank_seed(bank["refresh_epoch"]):
                raise ValueError(f"Milestone {path}: parameter_bank.pt bank seed differs from the run's seed")
            delta = torch.cat([bank["perturbations"][name].reshape(lap.cfg.num_members, -1) for name in lap.names], 1)
            lap.perturbations, lap.refresh_epoch = delta.to(self.device), int(bank["refresh_epoch"])
        elif self.method == "plan2explore":
            p2e, norm = self.plan2explore, load("plan2explore_normalization.pt")
            if float(norm["feature_std_floor"]) != float(p2e.cfg.feature_std_floor):
                raise ValueError(f"Milestone {path}: plan2explore_normalization.pt floor differs from the config")
            recorded = (manifest.get("plan2explore") or {}).get("probe_sha256")
            if recorded is not None and recorded != p2e.extractor.probe_sha256:
                raise ValueError(f"Milestone {path}: the feature probe differs from the one the run recorded")
            heads = {f"{k - 1}.{name}": v for k in range(1, p2e.cfg.num_members + 1)
                     for name, v in load(f"MLP_member_{k}.pt").items()}
            p2e.load_state_dict(dict(settings=p2e.settings(), seed=p2e.seed, heads=heads, mean=norm["mean"],
                                     std=norm["std"], refreshes=None, last_refresh_epoch=norm["refresh_epoch"],
                                     fit_updates=[None] * p2e.cfg.num_members,
                                     empty_bootstrap_fallbacks=[None] * p2e.cfg.num_members, cache_identity=None))
        elif self.method == "lcg-predictor":
            recorded = (manifest.get("lcg_predictor") or {}).get("probe_sha256")
            if recorded is not None and recorded != self.lcg_predictor.extractor.probe_sha256:
                raise ValueError(f"Milestone {path}: the feature probe differs from the one the run recorded")
            self.lcg_predictor.load_milestone_files(path)  # also checks the denoiser identity

    # ------------------------------------------------------------------ scoring

    @torch.no_grad()
    def score(self, batch: Dict[str, Any], seed: int, chunk: int = 64) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """(raw (W,) with NaN for unscored steps, scaled (W,) or None when the saved state cannot give the
        policy reward), float64 on CPU."""
        from envs.world_model_env import ensemble_disagreement
        from intrinsic_reward.intrinsic_eval import preserved
        W = batch["obs"].shape[0]
        raw = torch.full((W,), float("nan"), dtype=torch.float64)
        with torch.random.fork_rng(devices=[self.device] if self.device.type == "cuda" else []):
            torch.manual_seed(seed)
            if self.method == "lcg":
                scored = [j for j in range(W) if batch["successor"][j] is not None]
                candidates = []
                for j in scored:
                    o = batch["obs"][j:j + 1].to(self.device)
                    candidates.append((o.reshape(1, -1, *o.shape[-2:]), batch["act"][j:j + 1].to(self.device),
                                       batch["successor"][j].unsqueeze(0).to(self.device)))
                if candidates:
                    raw[scored] = self.lcg.score_candidates(candidates).double().cpu()
            elif self.method in ("plan2explore", "lcg-predictor"):
                model = self.plan2explore or self.lcg_predictor
                score = model.disagreement if self.plan2explore is not None else model.score_features
                with preserved(model.extractor, ("seconds",)):
                    for i in range(0, W, chunk):
                        s = slice(i, min(i + chunk, W))
                        features = model.extractor(batch["frame"][s], batch["context"][s], batch["context_act"][s])
                        raw[s] = score(features, batch["action"][s].to(features.device)).double().cpu()
            else:
                c, h, w = batch["obs"].shape[2:]
                # One draw for the whole window: transition j gets the same noise in every batch of this seed.
                noise = torch.randn(self.noise_samples, W, c, h, w, generator=torch.Generator().manual_seed(seed))
                for i in range(0, W, chunk):
                    s = slice(i, min(i + chunk, W))
                    o, a = batch["obs"][s].to(self.device), batch["act"][s].to(self.device)
                    per_noise = []
                    for z in noise[:, s].to(self.device):
                        if self.laplace is not None:
                            with preserved(self.laplace, ("prediction_seconds", "prediction_calls", "prefix_evaluations")):
                                predictions = self.laplace.predict_members(self.samplers[0], o, a, z)
                        else:
                            predictions = torch.stack([sm.sample(o, a, z)[0] for sm in self.samplers])
                        per_noise.append(ensemble_disagreement(predictions))
                    raw[s] = torch.stack(per_noise).mean(dim=0).double().cpu()
        return raw, self.scale(raw)

    def scale(self, raw: torch.Tensor) -> Optional[torch.Tensor]:
        """The policy reward of raw scores with the frozen scale (no running statistics updated)."""
        x = raw.float().to(self.device)
        if self.method == "lcg":
            scaled = self.lcg.scale_frozen(x)
        elif self.method == "plan2explore":
            scaled = self.plan2explore.scale(x)
        elif self.method == "lcg-predictor":
            scaled = self.lcg_predictor.scale(x)
        else:
            scaled = self.scale_scorer.scale_frozen(x)
        return None if scaled is None else scaled.double().cpu()

    def describe(self) -> Dict[str, Any]:
        d = dict(method=self.method, num_conditioning=self.n, members=len(self.denoisers), noise_samples=self.noise_samples,
                 reward_scaling=getattr(self.method_cfg, "reward_scaling", None))
        if self.lcg is not None:
            d.update(round=self.lcg.round_id, reward_divisor=self.lcg.reward_divisor,
                     candidate_num_mc=self.lcg.cfg.candidate_num_mc)
        if self.laplace is not None:
            d.update(refresh_epoch=self.laplace.refresh_epoch, laplace_members=self.laplace.cfg.num_members)
        if self.plan2explore is not None:
            d.update(refresh_epoch=self.plan2explore.last_refresh_epoch, heads=self.plan2explore.cfg.num_members)
        if self.lcg_predictor is not None:
            lp = self.lcg_predictor
            d.update(round=lp.round_id, reward_divisor=lp.reward_divisor, scorer_backend=lp.cfg.scorer_backend,
                     num_probes=lp.cfg.num_probes, predictor_parameters=lp.num_parameters)
        return d


def restore_milestone_scorer(method: str, milestone_path: Path, device: torch.device) -> Tuple[FrozenScorer, Any, Any]:
    """(scorer, loaded milestone, run config) of a learned method's milestone; refuses a milestone of another
    method or one without the required scorer files."""
    path = Path(milestone_path)
    manifest = read_manifest(path)
    exploration = manifest.get("exploration", {})
    label = method_label(bool(exploration.get("intrinsic_reward_enabled")), exploration.get("method"))
    if label != method:
        raise ValueError(f"Milestone {path} belongs to a {label} run, not {method}")
    from omegaconf import OmegaConf
    run_dir = path.parent.parent
    cfg = run_config(run_dir)
    members = int(OmegaConf.select(cfg, "intrinsic_reward.num_members", default=1) or 1)
    required = scorer_requirements(method, manifest, members)
    listed = {entry["path"] for entry in manifest["files"].values()}
    missing = [f for f in required if f not in listed or not (path / f).is_file()]
    if missing:
        raise FileNotFoundError(f"Milestone {path} lacks the {method} scorer file(s) {missing} "
                                "(docs/analysis/restore_scorer.md; milestones saved before scorer checkpoints have none)")
    milestone = load_milestone(path, manifest["task_names"][0], map_location=torch.device("cpu"))
    if method in ("plan2explore", "lcg-predictor"):  # canonical actions need the policy's bounds
        policy = torch.load(path / "exploration_policy.pt", map_location="cpu", weights_only=False)
        low, high = policy["action_low"].tolist(), policy["action_high"].tolist()
    else:  # bounds are not read by these scorers; the action dimension is the replay's
        dim = milestone.dataset.load_episode(0).act.shape[-1]
        low, high = [-1.0] * dim, [1.0] * dim
    scorer = FrozenScorer(method, cfg, low, high, device)
    scorer.load_milestone_files(path, milestone.manifest, milestone.denoiser_state)
    return scorer, milestone, cfg


def check_uniform_run(uniform_run: Path, cfg, ucfg) -> None:
    """A Uniform run of the same environment (domain, tasks, image size, action repeat)."""
    from omegaconf import OmegaConf
    identity = exploration_identity(uniform_run)
    if identity["method"] != "uniform":
        raise ValueError(f"--uniform-run {uniform_run} is a {identity['method']} run, not Uniform")
    env, uenv = OmegaConf.to_container(cfg.env.train), OmegaConf.to_container(ucfg.env.train)
    for key in ("domain_name", "task_names", "size", "action_repeat"):
        if env.get(key) != uenv.get(key):
            raise ValueError(f"--uniform-run {uniform_run} has env.train.{key}={uenv.get(key)}, the learned run "
                             f"{env.get(key)}")


def follow_intrinsic_reward(method: str, milestone_path: Path, uniform_run: Path, output: Path, scoring_seed: int = 0,
                            device: Optional[str] = None, argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """Score the next recorded collection window of the learned run and the same step range of the Uniform run
    with the milestone's frozen scorer; returns the written files by name."""
    output, milestone_path, uniform_run = Path(output), Path(milestone_path), Path(uniform_run)
    files = [output / f for f in FOLLOW_OUTPUTS] + [output / "analysis.json"]
    existing = [str(f) for f in files if f.exists()]
    if existing:
        raise FileExistsError(f"Not overwriting existing results: {existing}")
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True  # as during exploration training
    run_dir = milestone_path.parent.parent
    cfg = run_config(run_dir)
    ucfg = run_config(uniform_run)
    check_uniform_run(uniform_run, cfg, ucfg)
    for name, c in (("learned", cfg), ("Uniform", ucfg)):
        if int(c.collection.train.num_envs) != 1:
            raise ValueError(f"The {name} run collected with {c.collection.train.num_envs} environments; the window "
                             "is defined in collection order, which needs one")
    # Checked before the scorer is built: a missing window fails fast.
    manifest = read_manifest(milestone_path)
    start = int(manifest["collected_steps"])
    end = start + int(cfg.collection.train.steps_per_epoch)
    replay, ureplay = run_replay(run_dir), run_replay(uniform_run)
    for name, r, where in (("learned", replay, run_dir), ("Uniform", ureplay, uniform_run)):
        if int(r.num_steps) < end:
            raise ValueError(f"The next window [{start}, {end}) after {milestone_path.name} is unavailable in the "
                             f"{name} run {where}: its replay ends at step {int(r.num_steps)}")
    scorer, milestone, _ = restore_milestone_scorer(method, milestone_path, device)
    check_extends(milestone, replay)
    batches = {"learned": window_batch(run_dir, replay, start, end, scorer.n),
               "uniform": window_batch(uniform_run, ureplay, start, end, scorer.n)}
    key = dict(method=method, domain=manifest["domain"], milestone=milestone_path.name, collected_steps=start,
               window_start=start, window_end=end)
    batch_rows, transition_rows, means = [], [], {}
    for name, batch in batches.items():
        raw, scaled = scorer.score(batch, scoring_seed)
        ok = ~torch.isnan(raw)
        means[name] = dict(n=int(raw.numel()), scored=int(ok.sum()),
                           raw=float(raw[ok].mean()) if ok.any() else float("nan"),
                           scaled=None if scaled is None else (float(scaled[ok].mean()) if ok.any() else float("nan")))
        batch_rows.append(dict(**key, batch=name, run=str((run_dir if name == "learned" else uniform_run).resolve()),
                               segments=" ".join(f"{k}:{a}-{b}" for k, a, b in batch["segments"]),
                               num_transitions=means[name]["n"], num_scored=means[name]["scored"],
                               raw_mean=means[name]["raw"], scaled_mean="" if scaled is None else means[name]["scaled"]))
        for j, (step, episode, row) in enumerate(batch["position"]):
            transition_rows.append(dict(**key, batch=name, position=j, global_step=step, episode=episode, row=row,
                                        raw="" if not ok[j] else float(raw[j]),
                                        scaled="" if scaled is None or not ok[j] else float(scaled[j])))
    learned, uniform = means["learned"], means["uniform"]
    scaled_known = learned["scaled"] is not None and uniform["scaled"] is not None
    summary = dict(**key, scoring_seed=scoring_seed, num_transitions_learned=learned["n"],
                   num_transitions_uniform=uniform["n"], num_scored_learned=learned["scored"],
                   num_scored_uniform=uniform["scored"], raw_mean_learned=learned["raw"], raw_mean_uniform=uniform["raw"],
                   raw_difference=learned["raw"] - uniform["raw"],
                   scaled_mean_learned=learned["scaled"] if scaled_known else "",
                   scaled_mean_uniform=uniform["scaled"] if scaled_known else "",
                   scaled_difference=learned["scaled"] - uniform["scaled"] if scaled_known else "")
    write_csv(output / "follow_batches.csv", batch_rows, FOLLOW_BATCH_COLUMNS)
    write_csv(output / "follow_summary.csv", [summary], FOLLOW_SUMMARY_COLUMNS)
    write_csv(output / "follow_transitions.csv", transition_rows, FOLLOW_TRANSITION_COLUMNS)
    finite = lambda v: v if isinstance(v, (int, float)) and math.isfinite(v) else None
    write_json(output / "analysis.json", dict(
        tests=[FOLLOW], method=method, domain=manifest["domain"], milestone=str(milestone_path.resolve()),
        epoch=manifest.get("epoch"), window=[start, end], window_rule="global steps [collected_steps, collected_steps "
        "+ collection.train.steps_per_epoch) in collection order (one environment)", scoring_seed=scoring_seed,
        device=str(device), scorer=scorer.describe(),
        batches={name: dict(run=str((run_dir if name == "learned" else uniform_run).resolve()),
                            segments=[list(s) for s in b["segments"]], num_transitions=means[name]["n"],
                            num_scored=means[name]["scored"], raw_mean=finite(means[name]["raw"]),
                            scaled_mean=finite(means[name]["scaled"]),
                            successor_of_last_rows={str(k): v for k, v in b["successor_sources"].items()})
                 for name, b in batches.items()},
        raw_difference=finite(summary["raw_difference"]),
        scaled_difference=finite(summary["scaled_difference"]) if scaled_known else None,
        scaled_note=None if scaled_known else "the milestone does not hold the scale state needed for the policy reward",
        milestone_files={k: v["sha256"] for k, v in manifest["files"].items()},
        source=manifest.get("source"), command=list(argv) if argv is not None else None))
    return {name: output / name for name in [*FOLLOW_OUTPUTS, "analysis.json"]}


# ---------------------------------------------------------------------------------------- noisy TV

# Cartpole noisy-TV analysis (docs/implement/AlEATRORISTIC_EXPERIMENT_DESIGN.md). Conditions are named by their
# saved env.train.noisy_tv settings: clean (enabled false), tv16, tv24 (enabled, panel size).
NOISY_TV_CONDITIONS = ("clean", "tv16", "tv24")
# Physical-state coverage bins, shared by every method and condition (fixed from the clean calibration
# trajectories, REWARD_FREE_EXPERIMENT_NOTES.md "Cartpole noisy TV"): cart position in 0.2 m bins over the slider
# range [-1.8, 1.8] m plus one underflow and one overflow bin (the soft joint limit lets the cart pass +-1.8 m;
# calibration reached +-2.05 m); pole angle wrapped to [-pi, pi) in 24 bins of 15 degrees (periodic).
POSITION_EDGES = tuple(round(-1.8 + 0.2 * k, 10) for k in range(19))
ANGLE_BINS = 24
NOISY_TV_EPISODE_COLUMNS = [
    "condition", "method", "reset_seed", "episode", "length", "num_observations", "final_observation_included",
    "task_return", "occupancy", "tv_active_fraction", "initial_in_region", "region_entries", "visits", "reached",
    "mean_dwell", "max_dwell", "position_angle_bins_visited", "position_angle_coverage", "position_bins_visited",
    "angle_bins_visited", "cart_velocity_mean_abs", "cart_velocity_std", "cart_velocity_max_abs",
    "pole_angular_velocity_mean_abs", "pole_angular_velocity_std", "pole_angular_velocity_max_abs"]
NOISY_TV_SUMMARY_COLUMNS = [
    "condition", "method", "num_episodes", "occupancy_mean", "occupancy_std", "occupancy_ci95_low",
    "occupancy_ci95_high", "reach_rate", "initial_in_region_episodes", "episodes_without_visits",
    "region_entries_mean", "mean_dwell_mean_given_visit", "max_dwell_mean_given_visit", "max_dwell_max",
    "position_angle_coverage_mean", "position_angle_coverage_std", "union_position_angle_bins",
    "union_position_angle_coverage", "cart_velocity_mean_abs_mean", "pole_angular_velocity_mean_abs_mean",
    "task_return_mean"]
NOISY_TV_PAIRED_COLUMNS = ["condition", "method", "reset_seed", "occupancy_clean", "occupancy_tv", "delta_occupancy",
                           "max_dwell_clean", "max_dwell_tv", "delta_max_dwell", "coverage_clean", "coverage_tv",
                           "delta_coverage"]
NOISY_TV_PAIRED_SUMMARY_COLUMNS = ["condition", "method", "num_pairs", "delta_occupancy_mean", "delta_occupancy_std",
                                   "delta_occupancy_ci95_low", "delta_occupancy_ci95_high", "delta_max_dwell_mean",
                                   "delta_coverage_mean"]
NOISY_TV_INTRINSIC_COLUMNS = ["condition", "method", "subset", "num_transitions", "raw_mean", "raw_std", "scaled_mean"]
NOISY_TV_TRANSITION_COLUMNS = ["condition", "method", "reset_seed", "episode", "row", "in_region", "tv_active",
                               "subset", "raw", "scaled"]
NOISY_TV_EVALUATION = "evaluation.json"


def condition_name(noisy_tv: Optional[Dict[str, Any]]) -> Optional[str]:
    """clean / tv<size> from a noisy-TV section; None for a run without one."""
    if noisy_tv is None:
        return None
    return f"tv{int(noisy_tv['size'])}" if noisy_tv["enabled"] else "clean"


def wrap_angle(angle: np.ndarray) -> np.ndarray:
    """Pole angle wrapped to [-pi, pi)."""
    return (np.asarray(angle, dtype=np.float64) + np.pi) % (2 * np.pi) - np.pi


def coverage_bins(cart_position: np.ndarray, pole_angle: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """(position bin, angle bin) of each state: position bins 0 (below -1.8 m), 1..18 (POSITION_EDGES), 19 (above
    1.8 m; 1.8 itself is the last regular bin); angle bins 0..23 of the wrapped angle."""
    edges = np.asarray(POSITION_EDGES)
    x = np.asarray(cart_position, dtype=np.float64)
    pos = np.searchsorted(edges, x, side="right")  # 0 below edges[0]; len(edges) above edges[-1]
    pos = np.where(x == edges[-1], len(edges) - 1, pos)
    ang = np.floor((wrap_angle(pole_angle) + np.pi) / (2 * np.pi / ANGLE_BINS)).astype(int)
    return pos, np.clip(ang, 0, ANGLE_BINS - 1)


NUM_POSITION_BINS = len(POSITION_EDGES) + 1  # 18 regular + underflow + overflow
NUM_COVERAGE_BINS = NUM_POSITION_BINS * ANGLE_BINS


def runs_of_ones(flags: np.ndarray) -> List[Tuple[int, int]]:
    return bouts(np.asarray(flags, dtype=bool))


def noisy_tv_episode_metrics(qpos: np.ndarray, qvel: np.ndarray, in_region: np.ndarray, tv_active: np.ndarray
                             ) -> Dict[str, Any]:
    """Metrics of one episode from its observations o_0..o_T in order (rows and, when recorded, the final
    observation): qpos/qvel (N, 2) as recorded (cart position, pole angle / their velocities), the recorded region
    membership I_t and TV activation. Durations count observations (agent steps)."""
    inside, active = np.asarray(in_region, dtype=bool), np.asarray(tv_active, dtype=bool)
    if (active & ~inside).any():
        raise ValueError("The TV is recorded active outside the trigger region")
    visits = runs_of_ones(inside)
    dwell = [length for _, length in visits]
    pos, ang = coverage_bins(qpos[:, 0], qpos[:, 1])
    cells = set(zip(pos.tolist(), ang.tolist()))
    v, w = np.abs(qvel[:, 0]), np.abs(qvel[:, 1])
    return dict(num_observations=int(inside.size), occupancy=float(inside.mean()), tv_active_fraction=float(active.mean()),
                initial_in_region=int(inside[0]), region_entries=int((~inside[:-1] & inside[1:]).sum()),
                visits=len(visits), reached=int(inside.any()),
                mean_dwell=float(np.mean(dwell)) if dwell else "", max_dwell=max(dwell) if dwell else "",
                position_angle_bins_visited=len(cells), position_angle_coverage=len(cells) / NUM_COVERAGE_BINS,
                position_bins_visited=len(set(pos.tolist())), angle_bins_visited=len(set(ang.tolist())),
                cart_velocity_mean_abs=float(v.mean()), cart_velocity_std=float(np.std(qvel[:, 0])),
                cart_velocity_max_abs=float(v.max()), pole_angular_velocity_mean_abs=float(w.mean()),
                pole_angular_velocity_std=float(np.std(qvel[:, 1])), pole_angular_velocity_max_abs=float(w.max()),
                cells=cells)


def episode_observation_states(episode) -> Dict[str, np.ndarray]:
    """qpos, qvel, in_region, tv_active of o_0..o_T of one recorded noisy-TV episode (the final observation
    included when its state was recorded). Every row must be recorded."""
    ps = episode.physical_state
    if ps is None or "tv_in_region" not in ps:
        raise ValueError("The episode has no noisy-TV diagnostics (collected without the wrapper or without "
                         "record_physical_state)")
    if not bool(ps["recorded"].all()):
        raise ValueError("The episode has unrecorded frames")
    parts = {k: [ps[k].numpy()] for k in ("qpos", "qvel", "tv_in_region", "tv_active")}
    final = "final_qpos" in ps
    if final:
        for k in parts:
            parts[k].append(ps[f"final_{k}"].numpy()[None])
    out = {k: np.concatenate(v) for k, v in parts.items()}
    out["final_included"] = final
    return out


def mean_std_ci(values: Sequence[float]) -> Tuple[float, float, float, float]:
    """Mean, population std and a normal-approximation 95% interval of the mean over episodes (episode-level
    evaluation variability of one trained policy, not variability across training runs)."""
    x = np.asarray([v for v in values if v != ""], dtype=np.float64)
    if x.size == 0:
        return (float("nan"),) * 4
    m, s = float(x.mean()), float(x.std())
    half = 1.96 * float(x.std(ddof=1)) / math.sqrt(x.size) if x.size > 1 else float("nan")
    return m, s, m - half, m + half


def noisy_tv_condition_rows(condition: str, method: str, episodes: List[Any], seeds: List[int]
                            ) -> Tuple[List[Dict[str, Any]], Dict[str, Any], List[Dict[str, Any]]]:
    """(per-episode rows, summary row, per-episode states for intrinsic diagnostics) of one condition."""
    rows, states, union = [], [], set()
    for k, (episode, seed) in enumerate(zip(episodes, seeds)):
        st = episode_observation_states(episode)
        m = noisy_tv_episode_metrics(st["qpos"], st["qvel"], st["tv_in_region"], st["tv_active"])
        if condition == "clean" and m["tv_active_fraction"] > 0:
            raise ValueError("A Clean episode records an active TV")
        if condition != "clean" and not np.array_equal(st["tv_active"], st["tv_in_region"]):
            raise ValueError(f"A {condition} episode's TV activation differs from its region membership")
        union |= m.pop("cells")
        rows.append(dict(condition=condition, method=method, reset_seed=seed, episode=k, length=len(episode),
                         final_observation_included=int(st["final_included"]),
                         task_return=float(episode.rew.sum()), **m))
        states.append(st)
    occ = mean_std_ci([r["occupancy"] for r in rows])
    visited = [r for r in rows if r["reached"]]
    cov = mean_std_ci([r["position_angle_coverage"] for r in rows])
    summary = dict(condition=condition, method=method, num_episodes=len(rows), occupancy_mean=occ[0],
                   occupancy_std=occ[1], occupancy_ci95_low=occ[2], occupancy_ci95_high=occ[3],
                   reach_rate=float(np.mean([r["reached"] for r in rows])),
                   initial_in_region_episodes=int(sum(r["initial_in_region"] for r in rows)),
                   episodes_without_visits=len(rows) - len(visited),
                   region_entries_mean=float(np.mean([r["region_entries"] for r in rows])),
                   mean_dwell_mean_given_visit=float(np.mean([r["mean_dwell"] for r in visited])) if visited else "",
                   max_dwell_mean_given_visit=float(np.mean([r["max_dwell"] for r in visited])) if visited else "",
                   max_dwell_max=max((r["max_dwell"] for r in visited), default=""),
                   position_angle_coverage_mean=cov[0], position_angle_coverage_std=cov[1],
                   union_position_angle_bins=len(union), union_position_angle_coverage=len(union) / NUM_COVERAGE_BINS,
                   cart_velocity_mean_abs_mean=float(np.mean([r["cart_velocity_mean_abs"] for r in rows])),
                   pole_angular_velocity_mean_abs_mean=float(np.mean([r["pole_angular_velocity_mean_abs"] for r in rows])),
                   task_return_mean=float(np.mean([r["task_return"] for r in rows])))
    return rows, summary, states


def noisy_tv_paired(rows: Dict[str, List[Dict[str, Any]]]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Per reset seed, each TV condition against Clean: delta = TV - Clean (occupancy, max dwell with 0 for no
    visit, position-angle coverage)."""
    paired, summaries = [], []
    clean = {r["reset_seed"]: r for r in rows.get("clean", [])}
    for condition in ("tv16", "tv24"):
        if condition not in rows or not clean:
            continue
        found = []
        for r in rows[condition]:
            c = clean.get(r["reset_seed"])
            if c is None:
                raise ValueError(f"{condition} reset seed {r['reset_seed']} has no Clean episode to pair with")
            dwell = lambda x: 0 if x["max_dwell"] == "" else x["max_dwell"]
            found.append(dict(condition=condition, method=r["method"], reset_seed=r["reset_seed"],
                              occupancy_clean=c["occupancy"], occupancy_tv=r["occupancy"],
                              delta_occupancy=r["occupancy"] - c["occupancy"], max_dwell_clean=dwell(c),
                              max_dwell_tv=dwell(r), delta_max_dwell=dwell(r) - dwell(c),
                              coverage_clean=c["position_angle_coverage"], coverage_tv=r["position_angle_coverage"],
                              delta_coverage=r["position_angle_coverage"] - c["position_angle_coverage"]))
        d = mean_std_ci([f["delta_occupancy"] for f in found])
        summaries.append(dict(condition=condition, method=found[0]["method"] if found else "", num_pairs=len(found),
                              delta_occupancy_mean=d[0], delta_occupancy_std=d[1], delta_occupancy_ci95_low=d[2],
                              delta_occupancy_ci95_high=d[3],
                              delta_max_dwell_mean=float(np.mean([f["delta_max_dwell"] for f in found])),
                              delta_coverage_mean=float(np.mean([f["delta_coverage"] for f in found]))))
        paired += found
    return paired, summaries


def tensor_sha256(state: Dict[str, Any]) -> str:
    """state_sha256 of the tensor entries of a state dict (PMPO's _extra_state is not a tensor)."""
    return state_sha256({k: v for k, v in state.items() if torch.is_tensor(v)})


def evaluate_noisy_tv_milestone(milestone_path: Path, out_dir: Path, seeds: Sequence[int], device: torch.device,
                                action_seed: int = 0) -> Dict[str, Any]:
    """30-seed (seeds) real evaluation of one noisy-TV milestone in its own condition, with the existing exploration
    test collector (coroutines.collector.make_collector, reset every collect, one episode per reset seed): the
    run's env.test settings (its evaluation TV seed), collection.test.epsilon, the milestone's exploration policy
    (sampled actions, as the run's own test collection) or, for Uniform, uniform actions. Nothing is trained,
    refreshed or refitted; the policy's weights are hashed before and after. The global RNGs are seeded per
    episode from SeedSequence([action_seed, reset_seed]) and restored afterwards. Writes out_dir/dataset (the
    episodes with their diagnostics) and out_dir/evaluation.json; returns the evaluation record."""
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import get_action_space_kwargs
    from coroutines.collector import make_collector, NumToCollect
    from data import Dataset
    from envs import make_dm_control_env
    from models.pmpo_beta import PMPOBeta
    from utils import rng_state, set_rng_state, set_seed
    milestone_path, out_dir = Path(milestone_path), Path(out_dir)
    if (out_dir / NOISY_TV_EVALUATION).exists() or (out_dir / "dataset").exists():
        raise FileExistsError(f"Not overwriting the existing evaluation in {out_dir}")
    manifest = read_manifest(milestone_path)
    run_dir = milestone_path.parent.parent
    cfg = run_config(run_dir)
    test_env = OmegaConf.to_container(cfg.env.test, resolve=True)
    if test_env.get("noisy_tv") is None:
        raise ValueError(f"Run {run_dir} has no env.test.noisy_tv: not a noisy-TV run")
    learned = cfg.collection.train.mode == "policy"
    env = make_dm_control_env(num_envs=1, device=device, **test_env, record_physical_state=True)
    saved_rng = rng_state()
    try:
        action_kwargs = get_action_space_kwargs(env)
        ac_cfg = instantiate(cfg.agent.actor_critic)
        ac_cfg.continuous_action_dim = action_kwargs["continuous_action_dim"]
        ac_cfg.action_low, ac_cfg.action_high = action_kwargs["action_low"], action_kwargs["action_high"]
        policy = PMPOBeta(ac_cfg)
        policy_file = None
        if learned:
            policy_file = milestone_path / "exploration_policy.pt"
            listed = {e["path"] for e in manifest["files"].values()}
            if policy_file.name not in listed or not policy_file.is_file():
                raise FileNotFoundError(f"Milestone {milestone_path} has no exploration_policy.pt (a learned method "
                                        "saves it from its first refresh on)")
            entry = next(e for e in manifest["files"].values() if e["path"] == policy_file.name)
            if file_sha256(policy_file) != entry["sha256"]:
                raise ValueError(f"{policy_file} does not match its manifest checksum")
            policy.load_state_dict(torch.load(policy_file, map_location="cpu", weights_only=False))
        policy.to(device).eval()
        before = tensor_sha256(policy.state_dict())
        dataset = Dataset(out_dir / "dataset", "noisy_tv_evaluation", cache_in_ram=True)
        collector = make_collector(env, policy, dataset, float(cfg.collection.test.epsilon), reset_every_collect=True,
                                   verbose=False)
        mode = "policy" if learned else "uniform"
        for seed in seeds:
            set_seed(int(np.random.SeedSequence([int(action_seed), int(seed)]).generate_state(1)[0]))
            collector.send(NumToCollect(episodes=1, mode=mode, reset_seed=int(seed), progress=False))
        after = tensor_sha256(policy.state_dict())
        if before != after:
            raise RuntimeError("The exploration policy's weights changed during evaluation")
        if dataset.num_episodes != len(seeds):
            raise RuntimeError(f"Evaluation gave {dataset.num_episodes} episodes for {len(seeds)} reset seeds")
        dataset.save_to_default_path()
    finally:
        set_rng_state(saved_rng)
        env.close()
    record = dict(
        milestone=str(milestone_path.resolve()), run=str(run_dir.resolve()), domain=manifest["domain"],
        task_names=manifest["task_names"], collected_steps=int(manifest["collected_steps"]),
        method=method_label(learned, OmegaConf.select(cfg, "intrinsic_reward.method", default=None)),
        condition=condition_name(test_env["noisy_tv"]), noisy_tv_test=test_env["noisy_tv"],
        noisy_tv_train=OmegaConf.to_container(cfg.env.train.noisy_tv, resolve=True),
        env_test={k: v for k, v in test_env.items() if k != "noisy_tv"}, reset_seeds=[int(s) for s in seeds],
        action_selection=("policy samples (the run's test collection rule)" if learned else "uniform in the action bounds")
        + f", epsilon {float(cfg.collection.test.epsilon)}", action_seed=int(action_seed),
        action_rng="global RNGs seeded per episode from SeedSequence([action_seed, reset_seed]); restored afterwards",
        collector="coroutines.collector.make_collector(reset_every_collect=True), one episode per reset seed",
        policy_file=None if policy_file is None else policy_file.name,
        policy_sha256=before, policy_sha256_after=after, weights_unchanged=True,
        milestone_files={k: v["sha256"] for k, v in manifest["files"].items()}, source=manifest.get("source"),
        episode_lengths=[int(x) for x in dataset.lengths])
    write_json(out_dir / NOISY_TV_EVALUATION, record)
    return record


def read_noisy_tv_evaluation(out_dir: Path) -> Tuple[Dict[str, Any], List[Any]]:
    """A saved evaluation (evaluation.json and its dataset), every episode file checked against its checksum."""
    from data import Dataset
    out_dir = Path(out_dir)
    record = json.loads((out_dir / NOISY_TV_EVALUATION).read_text(encoding="utf-8"))
    dataset = Dataset(out_dir / "dataset", "noisy_tv_evaluation", save_on_disk=False)
    dataset.load_from_default_path()
    dataset.is_static = True
    episodes = [load_checked_episode(dataset, k) for k in range(int(dataset.num_episodes))]
    return record, episodes


def noisy_tv_batch(episodes: List[Any], n: int) -> Dict[str, Any]:
    """Every transition of the evaluation episodes, with what every scorer reads (as window_batch): sampler
    windows, the real successor (next row or the final observation), the Plan2Explore context, frame and action."""
    from intrinsic_reward.intrinsic_eval import sampler_windows
    from intrinsic_reward.plan2explore import context_window
    fields = {k: [] for k in ("obs", "act", "frame", "action", "context", "context_act", "successor", "position")}
    for k, episode in enumerate(episodes):
        obs_w, act_w, nexts = sampler_windows(episode, n)
        for i in range(len(episode)):
            ctx, ctx_act, _ = context_window(episode.obs, episode.act, i, n)
            fields["obs"].append(obs_w[i]), fields["act"].append(act_w[i])
            fields["frame"].append(episode.obs[i]), fields["action"].append(episode.act[i])
            fields["context"].append(ctx), fields["context_act"].append(ctx_act)
            fields["successor"].append(nexts[i] if i < nexts.shape[0] else None)
            fields["position"].append((k, i))
    batch = {k: torch.stack(v) for k, v in fields.items() if k not in ("successor", "position")}
    batch.update(successor=fields["successor"], position=fields["position"])
    return batch


def noisy_tv_intrinsic(condition: str, method: str, milestone_path: Path, episodes: List[Any],
                       states: List[Dict[str, Any]], seeds: List[int], scoring_seed: int, device: torch.device
                       ) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Frozen-scorer intrinsic rewards (restore_milestone_scorer: nothing refreshed or refitted) of the evaluation
    transitions (o_t, a_t), split by the region membership of o_t: inside / outside, entry (o_t inside, o_{t-1}
    outside or t = 0) / sustained (o_t and o_{t-1} inside)."""
    scorer, _, _ = restore_milestone_scorer(method, milestone_path, device)
    batch = noisy_tv_batch(episodes, scorer.n)
    raw, scaled = scorer.score(batch, scoring_seed)
    transitions, subsets = [], {s: [] for s in ("all", "inside", "outside", "entry", "sustained")}
    for j, (k, i) in enumerate(batch["position"]):
        inside = bool(states[k]["tv_in_region"][i])
        previous = bool(states[k]["tv_in_region"][i - 1]) if i > 0 else False
        subset = ("sustained" if previous else "entry") if inside else "outside"
        ok = not math.isnan(float(raw[j]))
        transitions.append(dict(condition=condition, method=method, reset_seed=seeds[k], episode=k, row=i,
                                in_region=int(inside), tv_active=int(bool(states[k]["tv_active"][i])), subset=subset,
                                raw=float(raw[j]) if ok else "", scaled="" if scaled is None or not ok else float(scaled[j])))
        if ok:
            for s in ("all", "inside" if inside else "outside", subset if inside else None):
                if s is not None:
                    subsets[s].append(j)
    rows = []
    for s, idx in subsets.items():
        r = raw[idx].numpy() if idx else np.array([])
        rows.append(dict(condition=condition, method=method, subset=s, num_transitions=len(idx),
                         raw_mean=float(r.mean()) if idx else "", raw_std=float(r.std()) if idx else "",
                         scaled_mean="" if scaled is None or not idx else float(scaled[idx].mean())))
    return rows, transitions, scorer.describe()


def export_noisy_tv_videos(episodes: Dict[str, List[Any]], rows: Dict[str, List[Dict[str, Any]]], out: Path,
                           scale: int = 4, fps: int = 25) -> List[str]:
    """Representative videos from the saved (replay-format) evaluation observations: per condition, the episode
    with the highest occupancy (ties: the lowest reset seed), as MP4 (OpenCV) and a PNG strip of frames around its
    first region entry. Returns the written paths; none when OpenCV is unavailable."""
    try:
        import cv2
    except ImportError:
        return []
    written = []
    out.mkdir(parents=True, exist_ok=True)
    for condition, eps in episodes.items():
        best = max(range(len(eps)), key=lambda k: (rows[condition][k]["occupancy"], -rows[condition][k]["reset_seed"]))
        frames = eps[best].obs.add(1).div(2).mul(255).round().clamp(0, 255).byte().permute(0, 2, 3, 1).numpy()
        seed = rows[condition][best]["reset_seed"]
        path = out / f"{condition}_seed{seed}.mp4"
        h, w = frames.shape[1] * scale, frames.shape[2] * scale
        writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
        for f in frames:
            writer.write(cv2.resize(np.ascontiguousarray(f[..., ::-1]), (w, h), interpolation=cv2.INTER_NEAREST))
        writer.release()
        written.append(str(path))
        inside = eps[best].physical_state["tv_in_region"].numpy()
        start = int(np.argmax(inside)) if inside.any() else 0
        picks = list(range(max(0, start - 3), min(len(frames), start + 5)))
        strip = np.concatenate([frames[t] for t in picks], axis=1)
        png = out / f"{condition}_seed{seed}_entry_strip.png"
        cv2.imwrite(str(png), cv2.resize(strip[..., ::-1], (strip.shape[1] * scale, strip.shape[0] * scale),
                                         interpolation=cv2.INTER_NEAREST))
        written.append(str(png))
    return written


NOISY_TV_PLOT_PROGRAM = r"""
import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
spec = json.loads(sys.stdin.read())
out, conditions = spec["out"], spec["conditions"]
for key, label, name in (("occupancy", "Trigger-region occupancy (fraction of observations)", "occupancy"),
                         ("max_dwell", "Maximum dwell (observations; episodes with a visit)", "dwell"),
                         ("position_angle_coverage", "Physical coverage (fraction of position x angle bins)", "coverage")):
    fig, ax = plt.subplots(figsize=(6, 4))
    data = [[v for v in spec["values"][c][key] if v is not None] for c in conditions]
    ax.boxplot(data, labels=conditions, showmeans=True)
    for i, d in enumerate(data):
        ax.scatter([i + 1] * len(d), d, s=8, alpha=0.5)
    ax.set_ylabel(label)
    ax.set_title(spec["title"], fontsize=9)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    for suffix in ("png", "pdf"):
        fig.savefig(f"{out}/{name}.{suffix}", dpi=150)
    plt.close(fig)
"""


def plot_noisy_tv(rows: Dict[str, List[Dict[str, Any]]], out: Path, title: str) -> Optional[str]:
    """Occupancy, dwell and coverage plots in a separate process (as downstream_analysis' figures). Returns None
    when drawn, else the reason they were skipped (no matplotlib)."""
    import subprocess
    spec = dict(out=str(out), title=title, conditions=list(rows),
                values={c: {k: [None if r[k] == "" else r[k] for r in rs]
                            for k in ("occupancy", "max_dwell", "position_angle_coverage")} for c, rs in rows.items()})
    result = subprocess.run([sys.executable, "-c", NOISY_TV_PLOT_PROGRAM], input=json.dumps(spec), text=True,
                            capture_output=True)
    if result.returncode != 0:
        return f"plots skipped: {result.stderr.strip().splitlines()[-1] if result.stderr.strip() else 'plot process failed'}"
    return None


def noisy_tv_analysis(conditions: Dict[str, Path], output: Path, seeds: Optional[Sequence[int]] = None,
                      device: Optional[str] = None, intrinsic: bool = False, scoring_seed: int = 0, action_seed: int = 0,
                      videos: bool = True, argv: Optional[Sequence[str]] = None) -> Dict[str, Path]:
    """The noisy-TV comparison of one method: conditions maps clean/tv16/tv24 to a milestone of that condition's run.
    Saved trajectories (output/trajectories/<condition>/) are read when present and consistent (same milestone,
    policy hash and seeds); missing ones are evaluated first (evaluate_noisy_tv_milestone). Writes per-episode,
    summary, paired and (with intrinsic) intrinsic-reward CSV/JSON, plots, videos and analysis.json."""
    from omegaconf import OmegaConf
    output = Path(output)
    names = ["noisy_tv_episodes.csv", "noisy_tv_summary.csv", "noisy_tv_paired.csv", "noisy_tv_paired_summary.csv",
             "noisy_tv_episodes.json", "analysis.json"] + (["noisy_tv_intrinsic.csv", "noisy_tv_transitions.csv"]
                                                           if intrinsic else [])
    existing = [str(output / f) for f in names if (output / f).exists()]
    if existing:
        raise FileExistsError(f"Not overwriting existing results: {existing}")
    unknown = sorted(set(conditions) - set(NOISY_TV_CONDITIONS))
    if unknown or not conditions:
        raise ValueError(f"--condition takes {', '.join(NOISY_TV_CONDITIONS)}=<milestone>; got {sorted(conditions)}")
    torch_device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    # Every condition: a noisy-TV run of that condition; one method, domain, task group, trigger and panel.
    info = {}
    for condition, milestone in conditions.items():
        milestone = Path(milestone)
        manifest = read_manifest(milestone)
        cfg = run_config(milestone.parent.parent)
        train_tv = OmegaConf.select(cfg, "env.train.noisy_tv", default=None)
        train_tv = None if train_tv is None else OmegaConf.to_container(train_tv, resolve=True)
        if condition_name(train_tv) != condition:
            raise ValueError(f"--condition {condition}={milestone}: that run's condition is {condition_name(train_tv)}")
        exploration = manifest.get("exploration", {})
        info[condition] = dict(milestone=milestone, manifest=manifest, cfg=cfg, tv=train_tv,
                               method=method_label(bool(exploration.get("intrinsic_reward_enabled")),
                                                   exploration.get("method")),
                               final_seeds=list(OmegaConf.select(cfg, "evaluation.final_reset_seeds", default=None) or []))
    shared = lambda c: (c["method"], c["manifest"]["domain"], tuple(c["manifest"]["task_names"]),
                        tuple(c["tv"]["position_interval"]), c["tv"]["panel_location"], c["tv"]["margin"],
                        int(c["manifest"]["collected_steps"]))
    if len({shared(c) for c in info.values()}) != 1:
        raise ValueError("All conditions must share method, domain, task group, trigger interval, panel location, "
                         "margin and milestone step: " + str({k: shared(c) for k, c in info.items()}))
    method = next(iter(info.values()))["method"]
    if seeds is None:
        options = {tuple(c["final_seeds"]) for c in info.values()}
        if len(options) != 1 or not next(iter(options)):
            raise ValueError("Give --seeds: the runs record different or no evaluation.final_reset_seeds")
        seeds = list(next(iter(options)))
    seeds = [int(s) for s in seeds]
    rows, summaries, episodes, states, evaluations = {}, [], {}, {}, {}
    for condition in NOISY_TV_CONDITIONS:
        if condition not in info:
            continue
        c = info[condition]
        folder = output / "trajectories" / condition
        if not (folder / NOISY_TV_EVALUATION).is_file():
            say_line(f"Evaluating {condition}: {c['milestone']} ({len(seeds)} reset seeds)")
            evaluate_noisy_tv_milestone(c["milestone"], folder, seeds, torch_device, action_seed)
        record, eps = read_noisy_tv_evaluation(folder)
        if record["milestone"] != str(c["milestone"].resolve()) or record["reset_seeds"] != seeds \
                or record["action_seed"] != action_seed or record["condition"] != condition:
            raise ValueError(f"Saved trajectories {folder} belong to another milestone, condition or seed list")
        if record["milestone_files"] != {k: v["sha256"] for k, v in c["manifest"]["files"].items()}:
            raise ValueError(f"Saved trajectories {folder}: the milestone's files changed since they were evaluated")
        r, s, st = noisy_tv_condition_rows(condition, method, eps, seeds)
        rows[condition], episodes[condition], states[condition], evaluations[condition] = r, eps, st, record
        summaries.append(s)
    paired, paired_summary = noisy_tv_paired(rows)
    all_rows = [r for c in rows.values() for r in c]
    write_csv(output / "noisy_tv_episodes.csv", all_rows, NOISY_TV_EPISODE_COLUMNS)
    write_csv(output / "noisy_tv_summary.csv", summaries, NOISY_TV_SUMMARY_COLUMNS)
    write_csv(output / "noisy_tv_paired.csv", paired, NOISY_TV_PAIRED_COLUMNS)
    write_csv(output / "noisy_tv_paired_summary.csv", paired_summary, NOISY_TV_PAIRED_SUMMARY_COLUMNS)
    write_json(output / "noisy_tv_episodes.json", dict(episodes=all_rows))
    intrinsic_info = None
    if intrinsic:
        if method not in LEARNED:
            raise ValueError(f"--intrinsic needs a learned method; {method} has no intrinsic scorer")
        irows, trows, described = [], [], {}
        for condition in rows:
            ir, tr, d = noisy_tv_intrinsic(condition, method, info[condition]["milestone"], episodes[condition],
                                           states[condition], seeds, scoring_seed, torch_device)
            irows += ir
            trows += tr
            described[condition] = d
        write_csv(output / "noisy_tv_intrinsic.csv", irows, NOISY_TV_INTRINSIC_COLUMNS)
        write_csv(output / "noisy_tv_transitions.csv", trows, NOISY_TV_TRANSITION_COLUMNS)
        intrinsic_info = dict(scoring_seed=scoring_seed, scorers=described,
                              note="raw scores are comparable within this method only; each condition uses its own "
                                   "milestone's frozen scorer")
    video_files = export_noisy_tv_videos(episodes, rows, output / "videos") if videos else []
    plot_note = plot_noisy_tv(rows, output, f"{method}, {next(iter(info.values()))['manifest']['domain']}, "
                                            f"milestone {next(iter(info.values()))['milestone'].name}")
    finite = lambda v: v if isinstance(v, (int, float)) and math.isfinite(v) else None
    write_json(output / "analysis.json", dict(
        tests=[NOISY_TV], method=method, conditions={k: str(c["milestone"].resolve()) for k, c in info.items()},
        reset_seeds=seeds, action_seed=action_seed,
        noisy_tv={k: dict(train=c["tv"], test=evaluations[k]["noisy_tv_test"]) for k, c in info.items()},
        evaluations={k: {kk: v for kk, v in e.items() if kk != "episode_lengths"} for k, e in evaluations.items()},
        metrics=dict(
            observations="o_0..o_T of each episode: the stored rows and, when recorded, the final observation",
            occupancy="fraction of observations with I_t = 1 (cart position in the interval, inclusive)",
            region_entries="outside-to-inside transitions between consecutive observations; an episode starting inside "
                           "is counted in initial_in_region, not as an entry",
            dwell="lengths of maximal runs of I_t = 1, in observations; mean/max over the episode's visits; empty "
                  "without a visit (the summary conditions on visits); runs reaching the episode end are not censored",
            reached="any observation inside", coverage=dict(
                position_edges_m=list(POSITION_EDGES), position_bins=NUM_POSITION_BINS,
                position_out_of_range="bin 0 below -1.8 m, bin 19 above 1.8 m (1.8 itself in the last regular bin)",
                angle="pole angle wrapped to [-pi, pi), 24 bins of 15 degrees", total_bins=NUM_COVERAGE_BINS,
                note="simulator state only; rendered images are never counted"),
            velocities="per episode: mean |v|, std v, max |v| of cart velocity (qvel[0]) and pole angular velocity "
                       "(qvel[1])",
            paired="per reset seed: TV - Clean (occupancy, max dwell with 0 when no visit, coverage)",
            intervals="normal-approximation 95% intervals of episode means: evaluation variability of one trained "
                      "policy per condition, not variability across independent training runs"),
        paired_summary=[{k: finite(v) if isinstance(v, float) else v for k, v in s.items()} for s in paired_summary],
        intrinsic=intrinsic_info, videos=video_files, plots=plot_note or "occupancy, dwell, coverage (.png, .pdf)",
        command=list(argv) if argv is not None else None))
    return {name: output / name for name in names}


def say_line(text: str) -> None:
    print(text, flush=True)


def parse_conditions(items: Sequence[str]) -> Dict[str, Path]:
    """condition=milestone pairs, each condition once."""
    found = {}
    for item in items:
        name, sep, path = item.partition("=")
        if not sep or not name.strip() or not path.strip():
            raise ValueError(f"--condition takes condition=<milestone folder>, got {item!r}")
        if name.strip() in found:
            raise ValueError(f"--condition gives {name.strip()!r} more than once")
        found[name.strip()] = Path(path.strip())
    return found


def parse_seeds(text: Optional[str]) -> Optional[List[int]]:
    """'3001-3030' or '3001,3002,...'."""
    if text is None:
        return None
    if "-" in text and "," not in text:
        a, b = (int(v) for v in text.split("-"))
        return list(range(a, b + 1))
    return [int(v) for v in text.split(",")]


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="append", required=True, choices=TESTS,
                        help="analysis to run; repeat to combine")
    parser.add_argument("--milestone", action="append", default=[], type=Path,
                        help="milestone folder <run>/milestones/steps_N; repeat for several methods (coverage, "
                             f"lasting); exactly one for {FOLLOW}; not used by {NOISY_TV}")
    parser.add_argument("--condition", action="append", default=[], metavar="CONDITION=MILESTONE",
                        help=f"{NOISY_TV}: a milestone of the run of one condition ({', '.join(NOISY_TV_CONDITIONS)}); "
                             "repeat per condition")
    parser.add_argument("--seeds", default=None, help=f"{NOISY_TV}: evaluation reset seeds, e.g. 3001-3030 (default: "
                                                      "the runs' evaluation.final_reset_seeds)")
    parser.add_argument("--action-seed", type=int, default=0, help=f"{NOISY_TV}: seed of the evaluation action RNG")
    parser.add_argument("--intrinsic", action="store_true",
                        help=f"{NOISY_TV}: also score the evaluation transitions with each milestone's frozen scorer")
    parser.add_argument("--no-videos", action="store_true", help=f"{NOISY_TV}: skip the representative videos")
    parser.add_argument("--threshold", action="append", default=[], metavar="TASK=VALUE",
                        help="coverage, lasting: reward threshold of one task in stored units; repeat per task")
    parser.add_argument("--method", choices=sorted(LEARNED), help=f"{FOLLOW}: the milestone's learned method")
    parser.add_argument("--uniform-run", type=Path, help=f"{FOLLOW}: the Uniform run folder to compare with")
    parser.add_argument("--scoring-seed", type=int, default=0,
                        help=f"{FOLLOW}, {NOISY_TV} --intrinsic: seed of the scoring randomness (default 0)")
    parser.add_argument("--device", default=None,
                        help=f"{FOLLOW}, {NOISY_TV}: torch device (default: cuda if available, else cpu)")
    parser.add_argument("--output", required=True, type=Path, help="folder for the CSV results and analysis.json")
    args = parser.parse_args(argv)
    command = sys.argv if argv is None else ["exploration_analysis.py", *argv]
    tests = list(dict.fromkeys(args.test))
    noisy_options = args.condition or args.seeds or args.intrinsic or args.no_videos or args.action_seed
    if NOISY_TV in tests:
        if len(tests) > 1:
            parser.error(f"{NOISY_TV} runs on its own; run the other tests in a separate invocation")
        if args.milestone or args.threshold or args.method or args.uniform_run:
            parser.error(f"{NOISY_TV} takes --condition (not --milestone, --threshold, --method or --uniform-run)")
        written = noisy_tv_analysis(parse_conditions(args.condition), args.output, parse_seeds(args.seeds), args.device,
                                    args.intrinsic, args.scoring_seed, args.action_seed, not args.no_videos,
                                    argv=command)
        for name, path in written.items():
            print(f"{name}: {path.resolve()}")
        return
    if noisy_options:
        parser.error(f"--condition, --seeds, --action-seed, --intrinsic and --no-videos belong to {NOISY_TV}")
    if not args.milestone:
        parser.error("Give at least one --milestone")
    if FOLLOW in tests:
        if len(tests) > 1:
            parser.error(f"{FOLLOW} runs on its own; run the replay tests in a separate invocation")
        if args.method is None or args.uniform_run is None or len(args.milestone) != 1:
            parser.error(f"{FOLLOW} needs --method, exactly one --milestone and --uniform-run")
        if args.threshold:
            parser.error(f"{FOLLOW} takes no --threshold")
        written = follow_intrinsic_reward(args.method, args.milestone[0], args.uniform_run, args.output,
                                          args.scoring_seed, args.device, argv=command)
    else:
        if args.method or args.uniform_run:
            parser.error(f"--method and --uniform-run belong to {FOLLOW}")
        written = run(tests, args.milestone, parse_thresholds(args.threshold), args.output, argv=command)
    for name, path in written.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
