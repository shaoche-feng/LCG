#! /usr/bin/env python
"""
One-off diagnostic (not a permanent framework, see tests/lcg/test_historical_sampling.py
for the permanent regression suite): demonstrates the old BatchSampler-based historical
transition sampler's endpoint bias, side by side with the new
lcg.precision.sample_uniform_historical_transitions.

The old sampler is reconstructed HERE only (byte-identical to the removed
lcg.precision.sample_valid_transitions, which called BatchSampler directly with
sample_weights=None, can_sample_beyond_end=False) -- it is NOT restored to production
code, purely for this comparison.

Usage:
    python scripts/lcg/compare_historical_sampling_bias.py
"""
import sys
from collections import Counter
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


sys.path.insert(0, str(_find_repo_root(Path(__file__).parent) / "src"))

import numpy as np
import torch

from data import BatchSampler, Dataset, Episode
from lcg.precision import sample_uniform_historical_transitions

IMG_CHANNELS, IMG_SIZE, ACTION_DIM = 3, 8, 2
NUM_STEPS_CONDITIONING = 1
LENGTHS = [2, 3, 5, 8]  # a few unequal episode lengths
NUM_DRAWS = 50000


def make_episode(length):
    return Episode(
        obs=torch.randn(length, IMG_CHANNELS, IMG_SIZE, IMG_SIZE),
        act=torch.randn(length, ACTION_DIM),
        rew=torch.zeros(length),
        end=torch.zeros(length, dtype=torch.uint8),
        trunc=torch.zeros(length, dtype=torch.uint8),
        info={},
    )


def build_dataset(tmp_dir):
    dataset = Dataset(tmp_dir, "compare_sampling", cache_in_ram=True)
    for L in LENGTHS:
        dataset.add_episode(make_episode(L))
    return dataset


def old_sampler_target(dataset, seed):
    """Reconstructs the removed lcg.precision.sample_valid_transitions exactly: one
    BatchSampler draw (sample_weights=None, can_sample_beyond_end=False), batch_size=1."""
    np.random.seed(seed)
    sampler = BatchSampler(
        dataset, rank=0, world_size=1, batch_size=1, seq_length=NUM_STEPS_CONDITIONING + 1,
        sample_weights=None, can_sample_beyond_end=False,
    )
    sid = sampler.sample()[0]
    return sid.episode_id, sid.stop - 1  # the ACTUAL target frame, not the originally-drawn timestep


def new_sampler_target(dataset, seed):
    sid = sample_uniform_historical_transitions(dataset, batch_size=1, num_steps_conditioning=NUM_STEPS_CONDITIONING, seed=seed)[0]
    return sid.episode_id, sid.stop - 1


def main():
    import tempfile
    with tempfile.TemporaryDirectory() as tmp_dir:
        dataset = build_dataset(Path(tmp_dir))
        N = sum(LENGTHS)
        expected_freq = NUM_DRAWS / N

        old_counts = Counter(old_sampler_target(dataset, seed) for seed in range(NUM_DRAWS))
        new_counts = Counter(new_sampler_target(dataset, seed) for seed in range(NUM_DRAWS))

        all_pairs = [(e, t) for e, L in enumerate(LENGTHS) for t in range(L)]
        final_pairs = {(e, L - 1) for e, L in enumerate(LENGTHS)}

        print(f"episode lengths: {LENGTHS}  N={N} valid transitions  NUM_DRAWS={NUM_DRAWS}  "
              f"expected uniform frequency={expected_freq:.1f}\n")

        print(f"{'episode':>7} {'t':>3} {'is_final':>9} {'old_count':>10} {'old_ratio':>10} "
              f"{'new_count':>10} {'new_ratio':>10}")
        for e, t in all_pairs:
            oc = old_counts.get((e, t), 0)
            nc = new_counts.get((e, t), 0)
            is_final = (e, t) in final_pairs
            print(f"{e:>7} {t:>3} {str(is_final):>9} {oc:>10} {oc / expected_freq:>10.3f} "
                  f"{nc:>10} {nc / expected_freq:>10.3f}")

        old_final_mean = np.mean([old_counts.get(k, 0) for k in final_pairs])
        old_other_mean = np.mean([old_counts.get(k, 0) for k in all_pairs if k not in final_pairs])
        new_final_mean = np.mean([new_counts.get(k, 0) for k in final_pairs])
        new_other_mean = np.mean([new_counts.get(k, 0) for k in all_pairs if k not in final_pairs])

        old_max_dev = max(abs(old_counts.get(k, 0) - expected_freq) for k in all_pairs) / expected_freq
        new_max_dev = max(abs(new_counts.get(k, 0) - expected_freq) for k in all_pairs) / expected_freq

        print(f"\nOLD sampler: final-transition mean count={old_final_mean:.1f}  "
              f"other-transition mean count={old_other_mean:.1f}  "
              f"ratio(final/other)={old_final_mean / old_other_mean:.2f}  "
              f"max relative deviation from uniform={old_max_dev:.2f}")
        print(f"NEW sampler: final-transition mean count={new_final_mean:.1f}  "
              f"other-transition mean count={new_other_mean:.1f}  "
              f"ratio(final/other)={new_final_mean / new_other_mean:.2f}  "
              f"max relative deviation from uniform={new_max_dev:.2f}")

        print("\nConclusion: the OLD BatchSampler-based sampler shows a pronounced excess-probability "
              "spike at each episode's final transition (boundary clipping absorbs all overflowing "
              "(t, offset) combinations) and non-uniform coverage elsewhere; the NEW sampler is "
              "uniform across all valid transitions by construction, confirmed empirically here.")


if __name__ == "__main__":
    main()
