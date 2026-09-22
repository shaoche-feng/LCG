"""Phase 3: freeze held-out Walk/Run evaluation episodes and construct three
controlled 5,000-transition training mixtures per domain (run_scarce,
balanced, walk_scarce), from the Phase 2 source pools.

Does NOT train anything -- dataset construction and validation only. Uses
the existing DIAMOND Dataset/Episode format unmodified; provenance not
representable as a first-class Episode field is recorded in per-episode
info dict entries (source_type/orig_source_pool/orig_episode_id/orig_env_seed)
plus a companion manifest.json per output directory.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/construct_phase3_datasets.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import List

import torch

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))

from data import Dataset, DatasetTraverser  # noqa: E402

PROBE_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic"
SOURCE_ROOT = PROBE_ROOT / "source_pools"
EVAL_ROOT = PROBE_ROOT / "eval"
MIXTURE_ROOT = PROBE_ROOT / "mixtures"

DOMAINS = ["walker", "quadruped"]
ACTION_DIM = {"walker": 6, "quadruped": 12, "hopper": 4}

# Held-out selection: deterministic, final 2 episodes of each 12-episode task pool.
HELD_OUT_IDS = [10, 11]
# Train-available ids after excluding held-out (0..9), and random uses its first 2 of 6.
WALK_TRAIN_IDS = list(range(10))
RUN_TRAIN_IDS = list(range(10))
RANDOM_TRAIN_IDS = [0, 1]

MIXTURES = {
    "run_scarce":  {"random": [0, 1], "walk": [0, 1, 2, 3, 4, 5, 6], "run": [0]},
    "balanced":    {"random": [0, 1], "walk": [0, 1, 2, 3],          "run": [0, 1, 2, 3]},
    "walk_scarce": {"random": [0, 1], "walk": [0],                   "run": [0, 1, 2, 3, 4, 5, 6]},
}
EXPECTED_COUNTS = {
    "run_scarce":  {"random": 1000, "walk": 3500, "run": 500},
    "balanced":    {"random": 1000, "walk": 2000, "run": 2000},
    "walk_scarce": {"random": 1000, "walk": 500,  "run": 3500},
}
TASK_FRACTIONS = {
    "run_scarce":  {"walk_fraction_of_task_data": 0.875, "run_fraction_of_task_data": 0.125},
    "balanced":    {"walk_fraction_of_task_data": 0.5,   "run_fraction_of_task_data": 0.5},
    "walk_scarce": {"walk_fraction_of_task_data": 0.125, "run_fraction_of_task_data": 0.875},
}


def _load_source_dataset(domain: str, source: str) -> Dataset:
    ds = Dataset(SOURCE_ROOT / domain / slot_dir(domain, source) / "dataset", name=f"src_{domain}_{source}", cache_in_ram=True)
    ds.load_from_default_path()
    return ds


def _copy_episode_with_provenance(src_ds: Dataset, src_episode_id: int, source_type: str,
                                   domain: str, dest_ds: Dataset) -> dict:
    episode = src_ds.load_episode(src_episode_id)
    env_seed = episode.info.get("env_seed")
    episode.info = dict(episode.info)  # avoid mutating the cached source episode's info in place
    episode.info.update({
        "source_type": source_type,
        "orig_source_pool": f"{domain}/{source_type}",
        "orig_episode_id": src_episode_id,
        "orig_env_seed": env_seed,
    })
    new_id = dest_ds.add_episode(episode)
    return {"new_episode_id": new_id, "orig_episode_id": src_episode_id, "env_seed": env_seed}


def build_eval_sets(domain: str) -> dict:
    print(f"\n=== {domain}: freezing held-out eval episodes {HELD_OUT_IDS} ===")
    manifests = {}
    for source in ("walk", "run"):
        src_ds = _load_source_dataset(domain, source)
        out_dir = EVAL_ROOT / domain / slot_dir(domain, source)
        if out_dir.exists():
            shutil.rmtree(out_dir)
        dataset_dir = out_dir / "dataset"
        dest_ds = Dataset(dataset_dir, name=f"eval_{domain}_{source}", cache_in_ram=True)

        copied = [_copy_episode_with_provenance(src_ds, eid, source, domain, dest_ds) for eid in HELD_OUT_IDS]
        dest_ds.save_to_default_path()

        manifest = {
            "domain": domain, "source": source,
            "source_episode_ids": [c["orig_episode_id"] for c in copied],
            "environment_seeds": [c["env_seed"] for c in copied],
            "num_episodes": dest_ds.num_episodes,
            "num_transitions": dest_ds.num_steps,
            "observation_shape_chw": [3, 64, 64],
            "action_shape": [ACTION_DIM[domain]],
            "episode_length": 500,
            "source_pool_path": str((SOURCE_ROOT / domain / slot_dir(domain, source)).relative_to(_PROJECT_ROOT)),
            "dataset_path": str(dataset_dir.relative_to(_PROJECT_ROOT)),
            "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        }
        (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
        print(f"  {source}: episodes={dest_ds.num_episodes} transitions={dest_ds.num_steps} "
              f"orig_ids={manifest['source_episode_ids']} seeds={manifest['environment_seeds']}")
        manifests[source] = manifest
    return manifests


def build_mixture(domain: str, condition: str) -> dict:
    spec = MIXTURES[condition]
    print(f"\n=== {domain}/{condition} ===")
    src = {s: _load_source_dataset(domain, s) for s in ("random", "walk", "run")}

    out_dir = MIXTURE_ROOT / domain / cond_dir(domain, condition)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    dataset_dir = out_dir / "dataset"
    dest_ds = Dataset(dataset_dir, name=f"mix_{domain}_{condition}", cache_in_ram=True)

    per_source_copies = {}
    for source_type, local_ids in spec.items():
        train_pool_ids = {"random": RANDOM_TRAIN_IDS, "walk": WALK_TRAIN_IDS, "run": RUN_TRAIN_IDS}[source_type]
        orig_ids = [train_pool_ids[i] for i in local_ids]
        copied = [_copy_episode_with_provenance(src[source_type], eid, source_type, domain, dest_ds)
                  for eid in orig_ids]
        per_source_copies[source_type] = copied

    dest_ds.save_to_default_path()

    counts = {}
    for source_type in ("random", "walk", "run"):
        eids = [c["new_episode_id"] for c in per_source_copies[source_type]]
        n_steps = sum(len(dest_ds.load_episode(eid)) for eid in eids)
        counts[source_type] = n_steps

    manifest = {
        "domain": domain,
        "condition": condition,
        "total_episodes": dest_ds.num_episodes,
        "total_transitions": dest_ds.num_steps,
        "random": {
            "episode_ids": [c["orig_episode_id"] for c in per_source_copies["random"]],
            "env_seeds": [c["env_seed"] for c in per_source_copies["random"]],
            "num_episodes": len(per_source_copies["random"]),
            "num_transitions": counts["random"],
        },
        "walk": {
            "episode_ids": [c["orig_episode_id"] for c in per_source_copies["walk"]],
            "env_seeds": [c["env_seed"] for c in per_source_copies["walk"]],
            "num_episodes": len(per_source_copies["walk"]),
            "num_transitions": counts["walk"],
        },
        "run": {
            "episode_ids": [c["orig_episode_id"] for c in per_source_copies["run"]],
            "env_seeds": [c["env_seed"] for c in per_source_copies["run"]],
            "num_episodes": len(per_source_copies["run"]),
            "num_transitions": counts["run"],
        },
        "held_out_walk_episode_ids": HELD_OUT_IDS,
        "held_out_run_episode_ids": HELD_OUT_IDS,
        "observation_shape": [3, 64, 64],
        "action_shape": [ACTION_DIM[domain]],
        "action_repeat": 2,
        "source_pool_paths": {s: str((SOURCE_ROOT / domain / slot_dir(domain, s)).relative_to(_PROJECT_ROOT)) for s in ("random", "walk", "run")},
        **TASK_FRACTIONS[condition],
        "dataset_path": str(dataset_dir.relative_to(_PROJECT_ROOT)),
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

    print(f"  episodes={dest_ds.num_episodes} transitions={dest_ds.num_steps} "
          f"random={counts['random']} walk={counts['walk']} run={counts['run']}")
    return manifest


def validate_dataset(dataset_dir: Path, expected_episodes: int, expected_transitions: int,
                      action_dim: int) -> dict:
    fresh = Dataset(dataset_dir, name="validate", cache_in_ram=True)
    fresh.load_from_default_path()
    all_finite, all_500, obs_ok = True, True, True
    for eid in range(fresh.num_episodes):
        ep = fresh.load_episode(eid)
        all_finite = all_finite and bool(
            torch.isfinite(ep.obs).all() and torch.isfinite(ep.act).all() and torch.isfinite(ep.rew).all()
        )
        all_500 = all_500 and len(ep) == 500
        obs_ok = obs_ok and tuple(ep.obs.shape[1:]) == (3, 64, 64) and ep.act.shape[1] == action_dim

    traverser_ok = True
    try:
        traverser = DatasetTraverser(fresh, batch_num_samples=4, chunk_size=16)
        n_batches = sum(1 for _ in traverser)
    except Exception as e:  # noqa: BLE001
        traverser_ok = False
        n_batches = 0
        print(f"    DatasetTraverser FAILED: {e}")

    ok = (
        fresh.num_episodes == expected_episodes and fresh.num_steps == expected_transitions
        and all_finite and all_500 and obs_ok and traverser_ok
    )
    return {
        "num_episodes": fresh.num_episodes, "num_transitions": fresh.num_steps,
        "expected_episodes": expected_episodes, "expected_transitions": expected_transitions,
        "all_finite": all_finite, "all_episodes_500": all_500, "obs_action_shapes_ok": obs_ok,
        "traverser_batches": n_batches, "roundtrip_passed": ok,
    }


def leakage_check(domain: str, mixture_manifests: dict, eval_manifests: dict) -> dict:
    train_episode_keys = set()
    train_seeds = set()
    for condition, m in mixture_manifests.items():
        for source_type in ("random", "walk", "run"):
            for eid, seed in zip(m[source_type]["episode_ids"], m[source_type]["env_seeds"]):
                train_episode_keys.add((source_type, eid))
                train_seeds.add(seed)

    eval_episode_keys = set()
    eval_seeds = set()
    for source_type, m in eval_manifests.items():
        for eid, seed in zip(m["source_episode_ids"], m["environment_seeds"]):
            eval_episode_keys.add((source_type, eid))
            eval_seeds.add(seed)

    episode_intersection = train_episode_keys & eval_episode_keys
    seed_intersection = train_seeds & eval_seeds
    result = {
        "domain": domain,
        "train_episode_keys_count": len(train_episode_keys),
        "eval_episode_keys_count": len(eval_episode_keys),
        "episode_id_intersection": sorted(str(x) for x in episode_intersection),
        "environment_seed_intersection": sorted(seed_intersection),
        "leakage_free": len(episode_intersection) == 0 and len(seed_intersection) == 0,
    }
    print(f"\n=== {domain} leakage check ===")
    print(f"  episode-ID intersection: {result['episode_id_intersection']}")
    print(f"  env-seed intersection: {result['environment_seed_intersection']}")
    print(f"  LEAKAGE FREE: {result['leakage_free']}")
    return result


def main() -> None:
    EVAL_ROOT.mkdir(parents=True, exist_ok=True)
    MIXTURE_ROOT.mkdir(parents=True, exist_ok=True)

    all_eval_manifests, all_mixture_manifests, all_leakage = {}, {}, {}
    for domain in DOMAINS:
        all_eval_manifests[domain] = build_eval_sets(domain)
        all_mixture_manifests[domain] = {c: build_mixture(domain, c) for c in MIXTURES}
        all_leakage[domain] = leakage_check(domain, all_mixture_manifests[domain], all_eval_manifests[domain])

    print("\n\n=== VALIDATION: 4 eval datasets + 6 mixtures ===")
    validation_results = {}
    for domain in DOMAINS:
        for source in ("walk", "run"):
            key = f"eval/{domain}/{source}"
            r = validate_dataset(EVAL_ROOT / domain / slot_dir(domain, source) / "dataset", 2, 1000, ACTION_DIM[domain])
            validation_results[key] = r
            print(f"  [{('PASS' if r['roundtrip_passed'] else 'FAIL')}] {key}: {r}")
        for condition in MIXTURES:
            key = f"mixture/{domain}/{condition}"
            r = validate_dataset(MIXTURE_ROOT / domain / cond_dir(domain, condition) / "dataset", 10, 5000, ACTION_DIM[domain])
            validation_results[key] = r
            expected = EXPECTED_COUNTS[condition]
            m = all_mixture_manifests[domain][condition]
            counts_ok = (m["random"]["num_transitions"] == expected["random"]
                         and m["walk"]["num_transitions"] == expected["walk"]
                         and m["run"]["num_transitions"] == expected["run"])
            r["per_source_counts_ok"] = counts_ok
            print(f"  [{('PASS' if r['roundtrip_passed'] and counts_ok else 'FAIL')}] {key}: {r}")

    all_ok = all(r["roundtrip_passed"] for r in validation_results.values())
    all_ok = all_ok and all(r.get("per_source_counts_ok", True) for r in validation_results.values())
    all_ok = all_ok and all(l["leakage_free"] for l in all_leakage.values())

    summary = {
        "eval_manifests": all_eval_manifests,
        "mixture_manifests": all_mixture_manifests,
        "leakage_checks": all_leakage,
        "validation_results": validation_results,
        "all_ok": all_ok,
    }
    (PROBE_ROOT / "phase3_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(f"\n\nPHASE 3 OVERALL: {'ALL CHECKS PASSED' if all_ok else 'SOME CHECKS FAILED'}")
    assert all_ok, "Phase 3 validation failed -- see phase3_summary.json"


if __name__ == "__main__":
    main()
