"""Phase 3 step 1: verify the Phase 2 source pools before building anything.

Loads each pool's Dataset from disk (fresh instances, not reusing any
in-memory state from collection) and checks episode/transition counts,
ordered episode IDs, environment seeds, observation/action shapes, episode
lengths, and round-trip integrity. Does not modify the source pools.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/verify_source_pools.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import torch

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))

from data import Dataset  # noqa: E402

POOL_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "source_pools"
EXPECTED = {"random": (6, 3000), "walk": (12, 6000), "run": (12, 6000)}
DOMAINS = ["walker", "quadruped"]


def verify_pool(domain: str, source: str) -> dict:
    pool_dir = POOL_ROOT / domain / source
    dataset_dir = pool_dir / "dataset"
    manifest = json.loads((pool_dir / "manifest.json").read_text())

    ds = Dataset(dataset_dir, name=f"verify_{domain}_{source}", cache_in_ram=True)
    ds.load_from_default_path()

    exp_episodes, exp_transitions = EXPECTED[source]
    episode_ids, env_seeds, lengths, obs_shapes, action_shapes = [], [], [], [], []
    all_finite = True
    for eid in range(ds.num_episodes):
        ep = ds.load_episode(eid)
        episode_ids.append(eid)
        env_seeds.append(ep.info.get("env_seed"))
        lengths.append(len(ep))
        obs_shapes.append(tuple(ep.obs.shape[1:]))
        action_shapes.append(tuple(ep.act.shape[1:]))
        all_finite = all_finite and bool(
            torch.isfinite(ep.obs).all() and torch.isfinite(ep.act).all() and torch.isfinite(ep.rew).all()
        )

    ok = (
        ds.num_episodes == exp_episodes
        and ds.num_steps == exp_transitions
        and all(l == 500 for l in lengths)
        and all_finite
        and len(set(obs_shapes)) == 1
        and len(set(action_shapes)) == 1
    )
    result = {
        "domain": domain, "source": source,
        "num_episodes": ds.num_episodes, "num_transitions": ds.num_steps,
        "expected_episodes": exp_episodes, "expected_transitions": exp_transitions,
        "episode_ids_ordered": episode_ids,
        "env_seeds_ordered": env_seeds,
        "manifest_env_seeds": manifest["environment_seeds"],
        "seeds_match_manifest": env_seeds == manifest["environment_seeds"],
        "episode_lengths": lengths,
        "all_episodes_500": all(l == 500 for l in lengths),
        "obs_shape": obs_shapes[0] if obs_shapes else None,
        "action_shape": action_shapes[0] if action_shapes else None,
        "all_finite": all_finite,
        "roundtrip_ok": ok,
    }
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {domain}/{source}: episodes={ds.num_episodes} (exp {exp_episodes}), "
          f"transitions={ds.num_steps} (exp {exp_transitions}), obs={result['obs_shape']}, "
          f"act={result['action_shape']}, seeds_match_manifest={result['seeds_match_manifest']}, "
          f"all_500={result['all_episodes_500']}, finite={all_finite}")
    return result


def main() -> None:
    all_ok = True
    results = {}
    for domain in DOMAINS:
        for source in EXPECTED:
            r = verify_pool(domain, source)
            results[f"{domain}/{source}"] = r
            all_ok = all_ok and r["roundtrip_ok"]
    print("\nALL SOURCE POOLS VALID" if all_ok else "\nSOME SOURCE POOLS FAILED VERIFICATION")
    out_path = _THIS_DIR / "source_pool_verification.json"
    out_path.write_text(json.dumps(results, indent=2, default=str))
    assert all_ok, "source pool verification failed"


if __name__ == "__main__":
    main()
