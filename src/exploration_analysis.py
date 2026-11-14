"""Analysis of the collected exploration dataset at milestones (CPU only). Nothing is trained, collected or
evaluated; milestones are only read, with the downstream loader data.milestone.load_milestone: every manifest
file is checked against its checksum, and the replay is the exact episode versions the milestone refers to,
each checked against its recorded checksum. Only the replay is analysed; the denoiser weights it also loads
(on CPU) are not used.

Tests (docs/analysis/reward_free_experiment_analysis.md, "Collected exploration dataset"); --test is repeatable:
- task-reward-coverage: per task, the fraction of the replay's transitions whose reward exceeds the task's
  threshold and the fraction of episodes with at least one such transition.
- task-reward-lasting: bouts of consecutive above-threshold transitions, kept within episode boundaries;
  per-bout lengths, per-episode counts (episodes with zero bouts included) and a summary.
- follow-intrinsic-reward: not implemented here; it needs the restored scorer checkpoints.

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

This module imports neither trainer nor envs: with MUJOCO_GL=egl, importing dm_control opens the GPU devices.
"""
import argparse
import math
from pathlib import Path
import sys
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from data.milestone import RETURN_CONVENTION, load_milestone, method_label, read_manifest, write_json
from downstream_analysis import write_csv

TESTS = ("task-reward-coverage", "task-reward-lasting", "follow-intrinsic-reward")
IMPLEMENTED = ("task-reward-coverage", "task-reward-lasting")
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
    """Run the selected tests on every milestone; returns the written files by name."""
    tests = list(dict.fromkeys(tests))
    unknown = [t for t in tests if t not in TESTS]
    if unknown or not tests:
        raise ValueError(f"--test takes {', '.join(TESTS)}; got {tests}")
    if "follow-intrinsic-reward" in tests:
        raise NotImplementedError("follow-intrinsic-reward needs the restored scorer checkpoints "
                                  "(docs/analysis/restore_scorer.md); it is a separate follow-up")
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


def main(argv: Optional[Sequence[str]] = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0],
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--test", action="append", required=True, choices=TESTS,
                        help="analysis to run; repeat to combine")
    parser.add_argument("--milestone", action="append", required=True, type=Path,
                        help="milestone folder <run>/milestones/steps_N; repeat for several methods")
    parser.add_argument("--threshold", action="append", required=True, metavar="TASK=VALUE",
                        help="reward threshold of one task in stored units; repeat per task")
    parser.add_argument("--output", required=True, type=Path, help="folder for the CSV results and analysis.json")
    args = parser.parse_args(argv)
    written = run(args.test, args.milestone, parse_thresholds(args.threshold), args.output,
                  argv=sys.argv if argv is None else ["exploration_analysis.py", *argv])
    for name, path in written.items():
        print(f"{name}: {path.resolve()}")


if __name__ == "__main__":
    main()
