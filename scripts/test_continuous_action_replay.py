#! /usr/bin/env python
"""
Stage 4 verification: continuous-action replay/data support.

Exercises Episode / Segment / Batch construction, save/load, segment slicing,
batching, .to(device), and pinned-memory behavior for:
  - continuous actions, float32, shape (T, action_dim) / (B, T, action_dim),
    for two different action dimensions (4 and 6), to confirm nothing is
    hard-coded to a specific action dimension,
  - the existing Atari-style discrete action path, int64, shape (T,) / (B, T),
    as a regression check.

Usage:
    python scripts/test_continuous_action_replay.py
"""
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch

from data import Batch, collate_segments_to_batch, Episode, make_segment, Segment, SegmentId


def make_episode(length: int, action_shape: tuple, action_dtype: torch.dtype, terminate: bool = True) -> Episode:
    obs = torch.rand(length, 3, 8, 8) * 2 - 1  # fake CHW obs in [-1, 1]
    if action_dtype.is_floating_point:
        act = torch.rand(length, *action_shape, dtype=action_dtype) * 2 - 1  # in [-1, 1], like DM Control bounds
    else:
        act = torch.randint(0, 5, (length, *action_shape), dtype=action_dtype)
    rew = torch.randn(length)
    end = torch.zeros(length, dtype=torch.uint8)
    trunc = torch.zeros(length, dtype=torch.uint8)
    if terminate:
        trunc[-1] = 1
    return Episode(obs, act, rew, end, trunc, info={})


def check_continuous(action_dim: int) -> None:
    print(f"\n{'=' * 60}\nContinuous action_dim = {action_dim}\n{'=' * 60}")
    length = 20
    ep = make_episode(length, (action_dim,), torch.float32)

    # 1. Episode stores continuous actions correctly
    assert ep.act.shape == (length, action_dim), ep.act.shape
    assert ep.act.dtype == torch.float32, ep.act.dtype
    print(f"Episode.act shape={tuple(ep.act.shape)} dtype={ep.act.dtype}  OK")

    # Episode.__add__ (used by the collector to append live steps to a not-yet-dead stored episode)
    ep_alive = make_episode(length, (action_dim,), torch.float32, terminate=False)
    ep2 = make_episode(5, (action_dim,), torch.float32, terminate=True)
    combined = ep_alive + ep2
    assert combined.act.shape == (length + 5, action_dim)
    assert combined.act.dtype == torch.float32
    print(f"Episode.__add__ preserves shape/dtype: {tuple(combined.act.shape)} {combined.act.dtype}  OK")

    # 2. save/load preserves dtype and shape (and exact values, since only obs is quantized)
    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ep.pt"
        ep.save(path)
        loaded = Episode.load(path)
    assert loaded.act.shape == ep.act.shape, (loaded.act.shape, ep.act.shape)
    assert loaded.act.dtype == ep.act.dtype, (loaded.act.dtype, ep.act.dtype)
    assert torch.equal(loaded.act, ep.act), "action values changed across save/load"
    print(f"save/load preserves act shape/dtype/values: {tuple(loaded.act.shape)} {loaded.act.dtype}  OK")

    # 3. segment slicing preserves action dimension (both padded and unpadded cases)
    seg_inrange = make_segment(ep, SegmentId(0, 2, 10), should_pad=True)
    assert seg_inrange.act.shape == (8, action_dim), seg_inrange.act.shape
    assert seg_inrange.act.dtype == torch.float32

    seg_padded = make_segment(ep, SegmentId(0, -3, 5), should_pad=True)  # pads before start
    assert seg_padded.act.shape == (8, action_dim), seg_padded.act.shape
    assert torch.equal(seg_padded.act[3:], ep.act[0:5])
    assert torch.all(seg_padded.act[:3] == 0), "left padding on act should be zeros"

    seg_padded_right = make_segment(ep, SegmentId(0, length - 2, length + 5), should_pad=True)  # pads after end
    assert seg_padded_right.act.shape == (7, action_dim), seg_padded_right.act.shape
    assert torch.equal(seg_padded_right.act[:2], ep.act[length - 2 :])
    assert torch.all(seg_padded_right.act[2:] == 0), "right padding on act should be zeros"
    print("Segment slicing (in-range, left-padded, right-padded) preserves action_dim  OK")

    # 4. batching produces (B, T, action_dim)
    segments = [make_segment(ep, SegmentId(0, i, i + 6), should_pad=True) for i in range(3)]
    batch = collate_segments_to_batch(segments)
    assert batch.act.shape == (3, 6, action_dim), batch.act.shape
    assert batch.act.dtype == torch.float32
    print(f"Batch.act shape={tuple(batch.act.shape)} dtype={batch.act.dtype}  OK")

    # 5. .to(device) and pinned-memory behavior
    pinned = batch.pin_memory()
    assert pinned.act.shape == batch.act.shape and pinned.act.dtype == batch.act.dtype
    assert pinned.act.is_pinned(), "Batch.pin_memory() did not pin the act tensor"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = pinned.to(device)
    assert moved.act.shape == batch.act.shape and moved.act.dtype == batch.act.dtype
    assert moved.act.device.type == device.type
    print(f"pin_memory()+to({device}) preserves shape/dtype, device={moved.act.device}  OK")

    ep_on_device = ep.to(device)
    assert ep_on_device.act.shape == ep.act.shape and ep_on_device.act.dtype == ep.act.dtype
    assert ep_on_device.act.device.type == device.type
    print(f"Episode.to({device}) preserves shape/dtype, device={ep_on_device.act.device}  OK")


def check_atari_discrete_regression() -> None:
    print(f"\n{'=' * 60}\nAtari-style discrete action regression\n{'=' * 60}")
    length = 20
    ep = make_episode(length, (), torch.int64)  # scalar action per timestep

    assert ep.act.shape == (length,), ep.act.shape
    assert ep.act.dtype == torch.int64
    print(f"Episode.act shape={tuple(ep.act.shape)} dtype={ep.act.dtype}  OK")

    with tempfile.TemporaryDirectory() as d:
        path = Path(d) / "ep.pt"
        ep.save(path)
        loaded = Episode.load(path)
    assert loaded.act.shape == ep.act.shape
    assert loaded.act.dtype == ep.act.dtype
    assert torch.equal(loaded.act, ep.act)
    print(f"save/load preserves act shape/dtype/values: {tuple(loaded.act.shape)} {loaded.act.dtype}  OK")

    seg = make_segment(ep, SegmentId(0, 2, 10), should_pad=True)
    assert seg.act.shape == (8,), seg.act.shape
    assert seg.act.dtype == torch.int64

    segments = [make_segment(ep, SegmentId(0, i, i + 6), should_pad=True) for i in range(3)]
    batch = collate_segments_to_batch(segments)
    assert batch.act.shape == (3, 6), batch.act.shape
    assert batch.act.dtype == torch.int64
    print(f"Batch.act shape={tuple(batch.act.shape)} dtype={batch.act.dtype}  OK")

    pinned = batch.pin_memory()
    assert pinned.act.is_pinned()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    moved = pinned.to(device)
    assert moved.act.shape == batch.act.shape and moved.act.dtype == batch.act.dtype
    print(f"pin_memory()+to({device}) preserves shape/dtype, device={moved.act.device}  OK")
    print("Atari discrete-action storage unaffected by continuous-action support  OK")


if __name__ == "__main__":
    for action_dim in (4, 6):
        check_continuous(action_dim)
    check_atari_discrete_regression()
    print("\nAll Stage 4 checks passed.")
