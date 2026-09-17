"""Round-trip tests for the checkpoint-resume-fidelity work: RNG state, optimizer momentum/
variance, scheduler position, dataset order/counters/manifest, and (in isolation)
RunningRMS normalization state. CPU-only, fast, no GPU/dataset-on-disk-at-scale required.
"""
import random
import shutil
import tempfile
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW

from data import Dataset, Episode
from lcg.reward_normalization import RunningRMS, RunningRMSConfig
from utils import get_lr_sched, RNGState


# ---------------------------------------------------------------------------------------------
# RNGState
# ---------------------------------------------------------------------------------------------

def test_rng_state_round_trip_python_random():
    random.seed(0)
    for _ in range(5):
        random.random()
    snapshot = RNGState().state_dict()
    ref = [random.random() for _ in range(10)]

    random.seed(999)  # scramble
    for _ in range(37):
        random.random()

    RNGState().load_state_dict(snapshot)
    got = [random.random() for _ in range(10)]
    assert got == ref


def test_rng_state_round_trip_numpy():
    np.random.seed(0)
    np.random.rand(5)
    snapshot = RNGState().state_dict()
    ref = np.random.rand(10)

    np.random.seed(999)
    np.random.rand(37)

    RNGState().load_state_dict(snapshot)
    got = np.random.rand(10)
    assert np.array_equal(got, ref)


def test_rng_state_round_trip_torch_cpu():
    torch.manual_seed(0)
    torch.rand(5)
    snapshot = RNGState().state_dict()
    ref = torch.rand(10)

    torch.manual_seed(999)
    torch.rand(37)

    RNGState().load_state_dict(snapshot)
    got = torch.rand(10)
    assert torch.equal(got, ref)


def test_rng_state_round_trip_survives_map_location_relocation(tmp_path):
    """Regression test for a real bug found during the resume-fidelity integration test:
    Trainer.load_state_checkpoint() calls torch.load(..., map_location=self._device), which
    relocates EVERY tensor found during unpickling onto that device -- RNG state included, even
    though torch.set_rng_state()/cuda.set_rng_state_all() both require CPU-resident
    ByteTensors. Without RNGState.load_state_dict() defensively moving back to CPU, resuming
    on any CUDA device raised TypeError: RNG state must be a torch.ByteTensor."""
    if not torch.cuda.is_available():
        import pytest
        pytest.skip("requires CUDA to exercise the map_location relocation this bug depends on")

    torch.manual_seed(7)
    torch.rand(3)
    snapshot = RNGState().state_dict()
    ref = torch.rand(10)

    path = tmp_path / "rng.pt"
    torch.save(snapshot, path)

    torch.manual_seed(0)
    torch.rand(100)

    # The exact call Trainer.load_state_checkpoint() makes: map_location relocates tensors.
    loaded = torch.load(path, map_location=torch.device("cuda:0"), weights_only=False)
    assert loaded["torch_cpu"].is_cuda, "test setup sanity check: map_location should have relocated this"

    RNGState().load_state_dict(loaded)  # must not raise
    got = torch.rand(10)
    assert torch.equal(got, ref)


def test_rng_state_save_load_via_torch_save(tmp_path):
    """The actual persistence path: torch.save/load (not just in-memory), matching how
    Trainer actually checkpoints this."""
    torch.manual_seed(42)
    torch.rand(3)
    snapshot = RNGState().state_dict()
    ref = torch.rand(10)

    path = tmp_path / "rng.pt"
    torch.save(snapshot, path)

    torch.manual_seed(0)
    torch.rand(100)

    loaded = torch.load(path, weights_only=False)
    RNGState().load_state_dict(loaded)
    got = torch.rand(10)
    assert torch.equal(got, ref)


# ---------------------------------------------------------------------------------------------
# Optimizer momentum/variance, exactly
# ---------------------------------------------------------------------------------------------

def test_optimizer_state_round_trip_exact():
    torch.manual_seed(0)
    model_a = nn.Linear(8, 4)
    opt_a = AdamW(model_a.parameters(), lr=1e-3)

    for _ in range(5):
        opt_a.zero_grad()
        loss = model_a(torch.randn(3, 8)).sum()
        loss.backward()
        opt_a.step()

    sd = opt_a.state_dict()
    # exp_avg/exp_avg_sq must be genuinely populated (not all-zero) for this test to be meaningful
    any_state = list(sd["state"].values())
    assert len(any_state) > 0
    assert any(s["exp_avg"].abs().sum().item() > 0 for s in any_state)

    model_b = nn.Linear(8, 4)  # different random init
    opt_b = AdamW(model_b.parameters(), lr=1e-3)
    opt_b.load_state_dict(sd)

    for (pid_a, sa), (pid_b, sb) in zip(opt_a.state.items(), opt_b.state.items()):
        assert torch.equal(sa["exp_avg"], sb["exp_avg"])
        assert torch.equal(sa["exp_avg_sq"], sb["exp_avg_sq"])
        assert sa["step"] == sb["step"]


# ---------------------------------------------------------------------------------------------
# Scheduler position
# ---------------------------------------------------------------------------------------------

def test_scheduler_position_round_trip():
    model = nn.Linear(4, 2)
    opt_a = AdamW(model.parameters(), lr=1e-3)
    sched_a = get_lr_sched(opt_a, num_warmup_steps=10)
    for _ in range(4):
        opt_a.step()
        sched_a.step()

    sd = sched_a.state_dict()

    opt_b = AdamW(model.parameters(), lr=1e-3)
    sched_b = get_lr_sched(opt_b, num_warmup_steps=10)
    sched_b.load_state_dict(sd)

    assert sched_a.get_last_lr() == sched_b.get_last_lr()
    assert sched_a._step_count == sched_b._step_count

    # continuing from the restored position must match continuing the original
    for _ in range(6):
        opt_a.step()
        sched_a.step()
        opt_b.step()
        sched_b.step()
    assert sched_a.get_last_lr() == sched_b.get_last_lr()


# ---------------------------------------------------------------------------------------------
# Dataset order/counters/manifest
# ---------------------------------------------------------------------------------------------

def _make_episode(t: int, obs_val: float) -> Episode:
    return Episode(
        obs=torch.full((t, 3, 8, 8), obs_val),
        act=torch.zeros(t, 2),
        rew=torch.zeros(t),
        end=torch.zeros(t, dtype=torch.uint8),
        trunc=torch.zeros(t, dtype=torch.uint8),
        info={},
    )


def test_dataset_state_round_trip(tmp_path):
    ds_a = Dataset(tmp_path / "ds", "ds", cache_in_ram=False)
    for i in range(5):
        ds_a.add_episode(_make_episode(10 + i, float(i)))
    sd = ds_a.state_dict()

    ds_b = Dataset(tmp_path / "ds", "ds", cache_in_ram=False)
    ds_b.load_state_dict(sd)

    assert ds_b.num_episodes == ds_a.num_episodes
    assert ds_b.num_steps == ds_a.num_steps
    assert np.array_equal(ds_b.lengths, ds_a.lengths)
    assert np.array_equal(ds_b.start_idx, ds_a.start_idx)
    assert ds_b.counter_rew == ds_a.counter_rew
    assert ds_b.counter_end == ds_a.counter_end

    for eid in range(5):
        ep_a = ds_a.load_episode(eid)
        ep_b = ds_b.load_episode(eid)
        assert torch.equal(ep_a.obs, ep_b.obs)


def test_dataset_manifest_detects_length_mismatch(tmp_path):
    ds = Dataset(tmp_path / "ds", "ds", cache_in_ram=False)
    ds.add_episode(_make_episode(10, 0.0))
    manifest_before = ds.compute_manifest()
    assert manifest_before["length_mismatches"] == []
    assert manifest_before["episodes"][0]["bookkeeping_length"] == 10
    assert manifest_before["episodes"][0]["actual_length"] == 10

    # simulate the exact staleness bug found during the detach_value_trunk experiment:
    # extend episode 0's on-disk file without updating self.lengths[0] to match.
    longer = _make_episode(15, 0.0)
    path = ds._get_episode_path(0)
    longer.save(path)

    manifest_after = ds.compute_manifest()
    assert manifest_after["length_mismatches"] == [0]
    assert manifest_after["episodes"][0]["bookkeeping_length"] == 10
    assert manifest_after["episodes"][0]["actual_length"] == 15


def test_dataset_manifest_sha256_changes_when_content_changes(tmp_path):
    ds = Dataset(tmp_path / "ds", "ds", cache_in_ram=False)
    ds.add_episode(_make_episode(10, 0.0))
    h1 = ds.compute_manifest()["episodes"][0]["sha256"]

    ds.add_episode(_make_episode(10, 1.0), episode_id=0)
    h2 = ds.compute_manifest()["episodes"][0]["sha256"]
    assert h1 != h2


# ---------------------------------------------------------------------------------------------
# RunningRMS (isolation only -- not yet wired into Trainer checkpointing; LCG resume is
# explicitly out of scope for this pass)
# ---------------------------------------------------------------------------------------------

def test_running_rms_state_round_trip():
    rms_a = RunningRMS(RunningRMSConfig(enabled=True, ema_decay=0.9))
    for _ in range(5):
        rms_a(torch.randn(4) * 3 + 1)
    sd = rms_a.state_dict()
    assert sd["s2"] is not None

    rms_b = RunningRMS(RunningRMSConfig(enabled=True, ema_decay=0.9))
    rms_b.load_state_dict(sd)
    assert torch.equal(rms_a.s2, rms_b.s2)

    # continuing from the restored state must match continuing the original
    r = torch.randn(4)
    out_a = rms_a(r.clone())
    out_b = rms_b(r.clone())
    assert torch.allclose(out_a, out_b)


def test_running_rms_state_round_trip_before_any_call():
    """s2 is None before the first call -- must round-trip that too, not crash."""
    rms_a = RunningRMS(RunningRMSConfig(enabled=True))
    sd = rms_a.state_dict()
    assert sd["s2"] is None
    rms_b = RunningRMS(RunningRMSConfig(enabled=True))
    rms_b.load_state_dict(sd)
    assert rms_b.s2 is None
