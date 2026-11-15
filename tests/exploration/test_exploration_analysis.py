"""exploration_analysis.py: task-reward coverage and lasting on real (tiny) milestones, read with the downstream
loader; bouts stay inside episodes, zero-bout episodes are reported, thresholds are strict, --test is repeatable."""
import csv
import json
from pathlib import Path

import numpy as np
import pytest
import torch

import exploration_analysis as ea
from data import Dataset, Episode
from data.milestone import publish_milestone

TASKS = ("stand", "hop")


def episode(stand, hop, finished=True):
    t = len(stand)
    end = torch.zeros(t, dtype=torch.uint8)
    trunc = torch.zeros(t, dtype=torch.uint8)
    if finished:
        trunc[-1] = 1
    return Episode(obs=torch.zeros(t, 3, 4, 4), act=torch.zeros(t, 1), rew=torch.zeros(t), end=end, trunc=trunc,
                   info={}, task_rew=torch.tensor([stand, hop], dtype=torch.float32).T.contiguous(), task_names=TASKS)


def make_milestone(root, episodes, method=None, domain="hopper", unfinished=()):
    """A published milestone (the exploration writer) of the given episodes; unfinished: ids cut at the milestone."""
    dataset = Dataset(root / "dataset", "train")
    for ep in episodes:
        dataset.add_episode(ep)
    steps = int(dataset.num_steps)
    manifest = dict(created_by="test", domain=domain, task_names=list(TASKS), collected_steps=steps,
                    num_episodes=int(dataset.num_episodes), epoch=1, counters=dict(denoiser_batches=0),
                    exploration=dict(collection_mode="policy" if method else "uniform",
                                     intrinsic_reward_enabled=method is not None,
                                     **(dict(method=method) if method else {})))
    successors = {i: torch.zeros(3, 4, 4) for i in unfinished}
    return publish_milestone(root / "milestones", f"steps_{steps:06d}", {"w": torch.zeros(1)}, dataset, successors,
                             manifest)


def rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


# stand: ep0 has bouts [1,3) and [4,5) (and ends above); ep1 starts above (a separate bout) and is otherwise 0;
# ep2 has none (the 1.0 equals the threshold: strict); ep3 is cut at the milestone while above.
STAND = [[0, 2, 2, 0, 2], [2, 0, 0, 0], [1.0, 0.5, 1.0], [0, 1.5, 1.5]]
HOP = [[0, 0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0], [0, 0, 0]]


@pytest.fixture()
def milestone(tmp_path):
    eps = [episode(s, h, finished=k != 3) for k, (s, h) in enumerate(zip(STAND, HOP))]
    return make_milestone(tmp_path / "lcg_run", eps, method="lcg", unfinished=[3])


def test_bouts_are_maximal_runs():
    assert ea.bouts(np.array([], bool)) == []
    assert ea.bouts(np.array([0, 0, 0], bool)) == []
    assert ea.bouts(np.array([1, 1, 1], bool)) == [(0, 3)]
    assert ea.bouts(np.array([1, 0, 1, 1, 0, 1], bool)) == [(0, 1), (2, 2), (5, 1)]


def test_thresholds_are_task_value_pairs_given_once():
    assert ea.parse_thresholds(["stand=1", "hop=0.5"]) == {"stand": 1.0, "hop": 0.5}
    for bad in (["stand"], ["=1"], ["stand=x"], ["stand=nan"], ["stand=1", "stand=2"], []):
        with pytest.raises(ValueError):
            ea.parse_thresholds(bad)


def test_coverage_counts_strictly_above_and_episodes_with_any(milestone, tmp_path):
    out = tmp_path / "out"
    ea.main(["--test", "task-reward-coverage", "--milestone", str(milestone), "--threshold", "stand=1.0",
             "--threshold", "hop=1.0", "--output", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["analysis.json", "coverage.csv"]
    stand, hop = rows(out / "coverage.csv")
    assert (stand["method"], stand["domain"], stand["task"], stand["milestone"], stand["collected_steps"],
            stand["threshold"]) == ("lcg", "hopper", "stand", "steps_000015", "15", "1.0")
    # Above: ep0 three, ep1 one, ep2 none (1.0 is not > 1.0), ep3 two -> 6 of 15; 3 of 4 episodes.
    assert (stand["num_episodes"], stand["num_transitions"], stand["transitions_above"],
            stand["episodes_with_above"]) == ("4", "15", "6", "3")
    assert float(stand["fraction_transitions_above"]) == pytest.approx(6 / 15)
    assert float(stand["fraction_episodes_with_above"]) == pytest.approx(3 / 4)
    assert (hop["transitions_above"], hop["episodes_with_above"]) == ("0", "0")


def test_lasting_keeps_bouts_inside_episodes_and_reports_zero_bout_episodes(milestone, tmp_path):
    out = tmp_path / "out"
    ea.main(["--test", "task-reward-lasting", "--milestone", str(milestone), "--threshold", "stand=1.0",
             "--threshold", "hop=1.0", "--output", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["analysis.json", "lasting_bouts.csv", "lasting_episodes.csv",
                                                     "lasting_summary.csv"]
    bouts = [(r["episode"], r["start"], r["length"], r["censored_at_cut"])
             for r in rows(out / "lasting_bouts.csv") if r["task"] == "stand"]
    # ep0 ends above and ep1 starts above: two separate bouts, never one across the boundary.
    assert bouts == [("0", "1", "2", "0"), ("0", "4", "1", "0"), ("1", "0", "1", "0"), ("3", "1", "2", "1")]
    episodes = {(r["task"], r["episode"]): r for r in rows(out / "lasting_episodes.csv")}
    assert len(episodes) == 8  # every episode for each task, zero-bout ones included
    assert (episodes["stand", "2"]["num_bouts"], episodes["stand", "2"]["longest_bout"]) == ("0", "0")
    assert all(episodes["hop", str(k)]["num_bouts"] == "0" for k in range(4))
    assert [episodes["stand", str(k)]["unfinished_at_cut"] for k in range(4)] == ["0", "0", "0", "1"]
    summary = {r["task"]: r for r in rows(out / "lasting_summary.csv")}
    s = summary["stand"]
    assert (s["num_episodes"], s["episodes_with_bouts"], s["num_bouts"], s["bout_length_max"],
            s["transitions_in_bouts"], s["bouts_censored_at_cut"]) == ("4", "3", "4", "2", "6", "1")
    assert float(s["bout_length_mean"]) == pytest.approx(1.5) and float(s["bouts_per_episode"]) == pytest.approx(1.0)
    assert (summary["hop"]["num_bouts"], summary["hop"]["bout_length_max"]) == ("0", "0")


def test_tests_combine_across_methods_with_one_threshold_per_task(milestone, tmp_path):
    uniform = make_milestone(tmp_path / "uniform_run", [episode([2, 2, 0], [0, 2, 2])])
    out = tmp_path / "both"
    ea.main(["--test", "task-reward-coverage", "--test", "task-reward-lasting", "--milestone", str(milestone),
             "--milestone", str(uniform), "--threshold", "stand=1.0", "--threshold", "hop=1.0", "--output", str(out)])
    assert sorted(p.name for p in out.iterdir()) == ["analysis.json", "coverage.csv", "lasting_bouts.csv",
                                                     "lasting_episodes.csv", "lasting_summary.csv"]
    coverage = {(r["method"], r["task"]): r for r in rows(out / "coverage.csv")}
    assert set(coverage) == {("lcg", "stand"), ("lcg", "hop"), ("uniform", "stand"), ("uniform", "hop")}
    assert {r["threshold"] for r in coverage.values()} == {"1.0"}
    assert coverage["uniform", "hop"]["transitions_above"] == "2"
    analysis = json.loads((out / "analysis.json").read_text())
    assert analysis["tests"] == ["task-reward-coverage", "task-reward-lasting"]
    assert analysis["thresholds"] == {"stand": 1.0, "hop": 1.0}
    assert [s["method"] for s in analysis["sources"]] == ["lcg", "uniform"]
    assert analysis["sources"][0]["unfinished_at_cut"] == [3]


def test_only_thresholded_tasks_are_reported(milestone, tmp_path):
    ea.main(["--test", "task-reward-coverage", "--milestone", str(milestone), "--threshold", "hop=0.5",
             "--output", str(tmp_path / "hop")])
    assert [r["task"] for r in rows(tmp_path / "hop" / "coverage.csv")] == ["hop"]


def test_refusals(milestone, tmp_path):
    base = ["--milestone", str(milestone), "--threshold", "stand=1.0"]
    with pytest.raises(ValueError, match="No recorded reward"):
        ea.main(["--test", "task-reward-coverage", "--milestone", str(milestone), "--threshold", "walk=1",
                 "--output", str(tmp_path / "a")])
    with pytest.raises(ValueError, match="runs on its own"):
        ea.run(["follow-intrinsic-reward"], [milestone], {"stand": 1.0}, tmp_path / "b")
    walker = make_milestone(tmp_path / "walker_run", [episode([2], [0])], domain="walker")
    with pytest.raises(ValueError, match="same domain"):
        ea.main(["--test", "task-reward-coverage", *base, "--milestone", str(walker), "--output", str(tmp_path / "c")])
    with pytest.raises(ValueError, match="more than once"):
        ea.main(["--test", "task-reward-coverage", *base, "--milestone", str(milestone), "--output", str(tmp_path / "d")])
    ea.main(["--test", "task-reward-coverage", *base, "--output", str(tmp_path / "e")])
    with pytest.raises(FileExistsError):
        ea.main(["--test", "task-reward-coverage", *base, "--output", str(tmp_path / "e")])
    with pytest.raises(SystemExit):  # argparse: unknown test name
        ea.main(["--test", "coverage", *base, "--output", str(tmp_path / "f")])


def test_follow_window_segments_in_collection_order():
    # Episodes of 500, 300 (cut at 800) and 200 rows: start_idx 0, 500, 800.
    start_idx, lengths = [0, 500, 800], [500, 300, 200]
    assert ea.window_segments(start_idx, lengths, 600, 700) == [(1, 100, 200)]  # inside one episode
    assert ea.window_segments(start_idx, lengths, 700, 900) == [(1, 200, 300), (2, 0, 100)]  # across a boundary
    assert ea.window_segments(start_idx, lengths, 0, 500) == [(0, 0, 500)]
    assert ea.window_segments(start_idx, lengths, 1000, 1100) == []


def test_follow_compares_successors_as_the_replay_stores_frames():
    frame = torch.rand(3, 4, 4) * 2 - 1
    stored = ea.as_stored(frame)
    assert torch.equal(ea.as_stored(stored), stored)  # idempotent on the replay's uint8 grid
    assert torch.equal(stored, frame.add(1).div(2).mul(255).round().byte().div(255).mul(2).sub(1))


def test_follow_command_line_refusals(milestone, tmp_path):
    follow = ["--test", "follow-intrinsic-reward", "--milestone", str(milestone), "--output", str(tmp_path / "f")]
    for extra in ([], ["--method", "lcg"], ["--uniform-run", str(tmp_path)],  # method and Uniform run are required
                  ["--method", "lcg", "--uniform-run", str(tmp_path), "--threshold", "stand=1"],
                  ["--method", "lcg", "--uniform-run", str(tmp_path), "--milestone", str(milestone)],
                  ["--method", "lcg", "--uniform-run", str(tmp_path), "--test", "task-reward-coverage"],
                  ["--method", "uniform", "--uniform-run", str(tmp_path)]):
        with pytest.raises(SystemExit):
            ea.main(follow + extra)
    with pytest.raises(SystemExit):  # --method belongs to the follow test
        ea.main(["--test", "task-reward-coverage", "--milestone", str(milestone), "--threshold", "stand=1",
                 "--method", "lcg", "--output", str(tmp_path / "g")])


def test_a_corrupted_episode_file_is_refused_by_the_loader(milestone, tmp_path):
    episode_file = next((tmp_path / "lcg_run" / "dataset").rglob("0_5.pt"))
    episode_file.write_bytes(episode_file.read_bytes() + b"x")
    with pytest.raises(ValueError, match="does not match its checksum"):
        ea.main(["--test", "task-reward-coverage", "--milestone", str(milestone), "--threshold", "stand=1",
                 "--output", str(tmp_path / "g")])


def test_the_module_imports_neither_trainer_nor_envs():
    """At import (the coverage and lasting tests); follow-intrinsic-reward imports the scorer modules only
    when it runs, and never a trainer."""
    import subprocess, sys
    code = ("import sys; sys.path.insert(0, 'src'); import exploration_analysis; "
            "bad = [m for m in ('exploration_trainer', 'downstream_trainer', 'envs', 'dm_control') if m in sys.modules]; "
            "print(bad); sys.exit(1 if bad else 0)")
    result = subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).resolve().parents[2], capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
