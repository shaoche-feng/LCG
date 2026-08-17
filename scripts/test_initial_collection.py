#! /usr/bin/env python
"""
Stage 7B verification: generalized initial dataset collection.

Part 1 (fast, isolated): exercises Trainer.collect_initial_dataset's logic directly
via a lightweight duck-typed stand-in (real Dataset, a fake collector coroutine
with fully controlled reward sequences, and a minimal agent.rew_end_model.continuous_reward
flag) -- mirroring how earlier stages tested model methods in isolation. No real
env, no real Agent, no Trainer.__init__ side effects.

Part 2 (real runtime path, one scenario): constructs a real Trainer against a real
DM Control task with a tiny initial-collection step budget, and calls
collect_initial_dataset() directly (not .run()), verifying the continuous-reward
branch engages correctly end-to-end.

Usage:
    python scripts/test_initial_collection.py
"""
import sys
import tempfile
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from omegaconf import OmegaConf

from coroutines import coroutine
from coroutines.collector import NumToCollect
from data import Dataset, Episode
from trainer import Trainer

REPO_ROOT = Path(__file__).resolve().parent.parent


# ---------------------------------------------------------------------------
# Fake collector: adds one synthetic Episode per `.send(NumToCollect(steps=n))`,
# with fully controlled reward content, directly into a real Dataset.
# ---------------------------------------------------------------------------

@coroutine
def make_fake_collector(dataset: Dataset, reward_fn):
    """reward_fn(start, n) -> Tensor of shape (n,): rewards for steps [start, start+n)."""
    num_to_collect = yield
    offset = 0
    while True:
        n = num_to_collect.steps
        rew = reward_fn(offset, n)
        obs = torch.zeros(n, 1, 1, 1)
        act = torch.zeros(n, 4)
        end = torch.zeros(n, dtype=torch.uint8)
        trunc = torch.zeros(n, dtype=torch.uint8)
        trunc[-1] = 1
        ep = Episode(obs, act, rew, end, trunc, info={})
        dataset.add_episode(ep)
        offset += n
        num_to_collect = yield []


def make_fake_trainer_self(dataset, collector, continuous_reward, min_steps, max_steps,
                            threshold_rew, steps_per_epoch, num_steps_total):
    cfg = OmegaConf.create({
        "collection": {
            "train": {
                "first_epoch": {"min": min_steps, "max": max_steps, "threshold_rew": threshold_rew},
                "steps_per_epoch": steps_per_epoch,
                "num_steps_total": num_steps_total,
            }
        }
    })
    return types.SimpleNamespace(
        _cfg=cfg,
        train_dataset=dataset,
        _train_collector=collector,
        agent=types.SimpleNamespace(rew_end_model=types.SimpleNamespace(continuous_reward=continuous_reward)),
    )


def run_collect(tmp_dir, reward_fn, continuous_reward, min_steps=20, max_steps=None,
                 threshold_rew=10, steps_per_epoch=20, num_steps_total=None):
    dataset = Dataset(Path(tmp_dir) / "train", "train_dataset")
    collector = make_fake_collector(dataset, reward_fn)
    if num_steps_total is None:
        num_steps_total = min_steps
    fake_self = make_fake_trainer_self(
        dataset, collector, continuous_reward, min_steps, max_steps, threshold_rew,
        steps_per_epoch, num_steps_total,
    )
    num_epochs_collect, to_log = Trainer.collect_initial_dataset(fake_self)
    return dataset, num_epochs_collect


# ---------------------------------------------------------------------------
# Part 1a: Atari regression -- reward-sign heuristic unchanged
# ---------------------------------------------------------------------------

def check_atari_early_stop():
    print("\n--- Atari: threshold reached within min_steps (early stop, unchanged formula) ---")
    # steps 0..19: rew[i] = -1 at i in {0,10}, +1 at i in {7,14}, else 0.
    # counts: {-1: 2, 0: 16, +1: 2} -> sorted [2,2,16] -> minority = 2+2 = 4 >= threshold_rew(3)
    def reward_fn(start, n):
        rew = torch.zeros(n)
        for j in range(n):
            i = start + j
            if i % 10 == 0:
                rew[j] = -1.0
            elif i % 7 == 0:
                rew[j] = 1.0
        return rew

    with tempfile.TemporaryDirectory() as tmp:
        dataset, num_epochs_collect = run_collect(
            tmp, reward_fn, continuous_reward=False,
            min_steps=20, max_steps=None, threshold_rew=3, steps_per_epoch=20, num_steps_total=20,
        )
    assert dataset.num_steps == 20, f"expected to stop exactly at min_steps=20, got {dataset.num_steps}"
    counts = dataset.counts_rew
    assert counts == [2, 16, 2], f"unexpected reward-sign counts: {counts}"
    assert num_epochs_collect == 0
    print(f"  stopped at num_steps={dataset.num_steps}, counts_rew={counts} (matches hand-computed [-1:2, 0:16, +1:2])  OK")


def check_atari_max_steps_fallback():
    print("\n--- Atari: threshold never reached, max_steps safety valve triggers (unchanged) ---")
    # all-positive rewards forever: minority = min(count(-1)=0, count(0)=0) = 0, never reaches
    # threshold_rew=5 -> loop must rely on max_steps to terminate, exactly as before.
    def reward_fn(start, n):
        return torch.ones(n)

    with tempfile.TemporaryDirectory() as tmp:
        dataset, num_epochs_collect = run_collect(
            tmp, reward_fn, continuous_reward=False,
            min_steps=20, max_steps=60, threshold_rew=5, steps_per_epoch=20, num_steps_total=60,
        )
    assert dataset.num_steps == 60, f"expected max_steps=60 to be the stopping point, got {dataset.num_steps}"
    assert dataset.counts_rew == [0, 0, 60]
    print(f"  stopped at num_steps={dataset.num_steps} via max_steps fallback (threshold never reached)  OK")


def check_atari_config_fields_still_used():
    print("\n--- Atari: config fields (min/max/threshold_rew) all actually consulted ---")
    # Two runs differing only in threshold_rew must produce different stopping points,
    # proving threshold_rew still drives the decision for continuous_reward=False.
    def reward_fn(start, n):
        rew = torch.zeros(n)
        rew[::5] = 1.0
        return rew

    with tempfile.TemporaryDirectory() as tmp:
        d_low, _ = run_collect(tmp, reward_fn, continuous_reward=False,
                                min_steps=20, max_steps=200, threshold_rew=1, steps_per_epoch=20, num_steps_total=200)
    with tempfile.TemporaryDirectory() as tmp:
        d_high, _ = run_collect(tmp, reward_fn, continuous_reward=False,
                                 min_steps=20, max_steps=200, threshold_rew=100, steps_per_epoch=20, num_steps_total=200)
    assert d_low.num_steps < d_high.num_steps, "threshold_rew no longer affects Atari stopping point"
    print(f"  threshold_rew=1 -> {d_low.num_steps} steps; threshold_rew=100 -> {d_high.num_steps} steps  OK")


# ---------------------------------------------------------------------------
# Part 1b: continuous-reward step-budget-only stopping
# ---------------------------------------------------------------------------

def check_continuous_step_budget_all_positive():
    print("\n--- Continuous: all-positive rewards, stop exactly at step budget ---")
    def reward_fn(start, n):
        return torch.full((n,), 0.73)

    with tempfile.TemporaryDirectory() as tmp:
        dataset, num_epochs_collect = run_collect(tmp, reward_fn, continuous_reward=True, min_steps=37, steps_per_epoch=37)
    assert dataset.num_steps == 37, f"expected exactly min_steps=37, got {dataset.num_steps}"
    assert num_epochs_collect == 0
    print(f"  all-positive (0.73) reward -> stopped at exactly {dataset.num_steps} steps (= min_steps)  OK")


def check_continuous_step_budget_all_zero():
    print("\n--- Continuous: all-zero rewards, stop exactly at step budget ---")
    def reward_fn(start, n):
        return torch.zeros(n)

    with tempfile.TemporaryDirectory() as tmp:
        dataset, num_epochs_collect = run_collect(tmp, reward_fn, continuous_reward=True, min_steps=37, steps_per_epoch=37)
    assert dataset.num_steps == 37, f"expected exactly min_steps=37, got {dataset.num_steps}"
    print(f"  all-zero reward -> stopped at exactly {dataset.num_steps} steps (= min_steps)  OK")


def check_continuous_step_count_independent_of_reward_content():
    print("\n--- Continuous: step count required is identical regardless of reward pattern ---")
    patterns = {
        "all_positive": lambda start, n: torch.full((n,), 5.0),
        "all_zero": lambda start, n: torch.zeros(n),
        "all_negative": lambda start, n: torch.full((n,), -3.0),
        "arbitrary": lambda start, n: torch.tensor([((start + j) * 0.6180339887) % 2.0 - 1.0 for j in range(n)]),
        "huge_values": lambda start, n: torch.full((n,), 1e6),
    }
    results = {}
    for name, fn in patterns.items():
        with tempfile.TemporaryDirectory() as tmp:
            dataset, num_epochs_collect = run_collect(tmp, fn, continuous_reward=True, min_steps=39, steps_per_epoch=13)
        results[name] = (dataset.num_steps, num_epochs_collect)
    steps_required = {v[0] for v in results.values()}
    print(f"  results per pattern: {results}")
    assert len(steps_required) == 1, f"step count required differs across reward patterns: {results}"
    print(f"  all {len(patterns)} reward patterns required exactly the same {steps_required.pop()} steps  OK")


def check_continuous_replay_raw_rewards():
    print("\n--- Continuous: replay episodes retain raw (non-sign-converted) rewards ---")
    raw_values = torch.tensor([-0.25, 0.13, 0.47, 0.92, -0.6] * 8)  # 40 values, matches min_steps below

    def reward_fn(start, n):
        return raw_values[start:start + n]

    with tempfile.TemporaryDirectory() as tmp:
        dataset, _ = run_collect(tmp, reward_fn, continuous_reward=True, min_steps=40, steps_per_epoch=40)
        ep = dataset.load_episode(0)
        assert torch.allclose(ep.rew, raw_values), f"stored rewards were altered: {ep.rew.tolist()}"
    print(f"  stored episode rewards exactly match raw input: {ep.rew[:5].tolist()} ...  OK")


def check_continuous_never_touches_counts_rew():
    print("\n--- Continuous: stopping decision provably independent of counts_rew ---")
    # A reward pattern engineered so counts_rew would look wildly different across two runs
    # (all steps positive vs. all steps exactly at the Atari class boundaries), yet the
    # continuous-mode step count required must be identical -- if the code path consulted
    # counts_rew at all, these would diverge.
    def all_plus_one(start, n):
        return torch.ones(n)  # counts_rew would be [0, 0, n] if it were consulted

    def all_zero(start, n):
        return torch.zeros(n)  # counts_rew would be [0, n, 0] if it were consulted

    with tempfile.TemporaryDirectory() as tmp:
        d1, _ = run_collect(tmp, all_plus_one, continuous_reward=True, min_steps=29, steps_per_epoch=29)
    with tempfile.TemporaryDirectory() as tmp:
        d2, _ = run_collect(tmp, all_zero, continuous_reward=True, min_steps=29, steps_per_epoch=29)
    assert d1.num_steps == d2.num_steps == 29
    print(f"  counts_rew for pattern A={d1.counts_rew}, pattern B={d2.counts_rew} (wildly different), "
          f"but both stopped at {d1.num_steps} steps  OK")


# ---------------------------------------------------------------------------
# Part 2: real runtime path, one real DM Control task, tiny step budget
# ---------------------------------------------------------------------------

def check_real_dm_control_trainer_run():
    print("\n--- Real runtime path: Trainer.collect_initial_dataset on real DM Control (cheetah-run) ---")
    import os
    from hydra import initialize, compose

    OmegaConf.register_new_resolver("eval", eval, replace=True)

    min_steps = 20
    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="trainer", overrides=[
            "env=dm_control",
            "env.train.domain_name=cheetah",
            "env.train.task_name=run",
            "wandb.mode=disabled",
            f"collection.train.first_epoch.min={min_steps}",
            f"collection.train.steps_per_epoch={min_steps}",
            f"collection.train.num_steps_total={min_steps}",
            "training.compile_wm=False",
        ])

    # Manual mkdtemp + single best-effort rmtree instead of TemporaryDirectory: the
    # AsyncVectorEnv subprocess spawned inside Trainer.__init__ can still hold file handles open
    # under this directory (e.g. in the copied ./src) by the time cleanup runs on Windows.
    # TemporaryDirectory(ignore_cleanup_errors=True)'s retry logic recurses on a handle that
    # never releases and hits Python's recursion limit; a single ignore_errors=True pass avoids
    # that. This is a subprocess-lifetime/OS cleanup quirk unrelated to
    # collect_initial_dataset's actual behavior, already fully exercised and asserted on above.
    tmp = tempfile.mkdtemp()
    try:
        old_cwd = os.getcwd()
        try:
            os.chdir(tmp)
            hydra_dir = Path(".hydra")
            hydra_dir.mkdir()
            OmegaConf.save(cfg, hydra_dir / "config.yaml")

            trainer = Trainer(cfg, REPO_ROOT)
            assert trainer.agent.rew_end_model.continuous_reward is True
            assert trainer.agent.actor_critic.continuous_action is True

            num_epochs_collect, to_log = trainer.collect_initial_dataset()

            assert trainer.train_dataset.num_steps >= min_steps, (
                f"expected >= {min_steps} steps, got {trainer.train_dataset.num_steps}"
            )
            # with num_envs=1, steps=min_steps collects *exactly* min_steps on the first send
            assert trainer.train_dataset.num_steps == min_steps
            assert num_epochs_collect == 0  # num_steps_total == min_steps here

            ep = trainer.train_dataset.load_episode(0)
            assert ep.rew.dtype == torch.float32
            print(f"  real Trainer + real cheetah-run env: collected exactly {trainer.train_dataset.num_steps} steps "
                  f"(budget={min_steps}), num_epochs_collect={num_epochs_collect}")
            print(f"  sample stored rewards: {ep.rew[:5].tolist()}")
            print("  continuous-reward branch engaged end-to-end through the real runtime path  OK")
        finally:
            os.chdir(old_cwd)
    finally:
        import shutil
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> None:
    check_atari_early_stop()
    check_atari_max_steps_fallback()
    check_atari_config_fields_still_used()

    check_continuous_step_budget_all_positive()
    check_continuous_step_budget_all_zero()
    check_continuous_step_count_independent_of_reward_content()
    check_continuous_replay_raw_rewards()
    check_continuous_never_touches_counts_rew()

    check_real_dm_control_trainer_run()

    print("\nAll Stage 7B checks passed.")


if __name__ == "__main__":
    main()
