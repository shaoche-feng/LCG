"""Phase 2 of the LCG controlled-undersampling diagnostic: collect the fixed
source pools P_R, P_walk, P_run independently for walker and quadruped, using
the Phase-1-validated TD-MPC2 collectors (deterministic) and uniform-random
native actions, through our unmodified DMControlEnv/DIAMOND Dataset format.

Does NOT construct undersampling mixtures, train world models, or compute
LCG -- source pools only. See docs/lcg_undersample_diagnostic/ for
the collected output.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/collect_source_pools.py
"""
from __future__ import annotations

from hopper_naming import cond_dir, slot_dir

import json
import shutil
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_THIS_DIR))
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))

from tdmpc2_adapter import load_pretrained_collector  # noqa: E402
from dm_control_env import DMControlEnv  # noqa: E402 -- unmodified production wrapper
from data import Dataset, DatasetTraverser, Episode  # noqa: E402 -- unmodified production data pipeline

OUT_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "source_pools"

ACTION_REPEAT = 2
IMG_SIZE = 64
CAMERA_ID = 0
COLLECTOR_SEED = 1  # which of TD-MPC2's 3 released training seeds (checkpoint identity, not env seed)

# Documented, non-overlapping environment-seed ranges (same ranges reused per domain --
# walker and quadruped are independent environments, so no cross-domain collision).
SOURCES = {
    "random": {"seed_base": 10_000, "target_transitions": 3_000, "env_task_for_instantiation": "walk"},
    "walk":   {"seed_base": 20_000, "target_transitions": 6_000, "env_task_for_instantiation": "walk"},
    "run":    {"seed_base": 30_000, "target_transitions": 6_000, "env_task_for_instantiation": "run"},
}
DOMAINS = ["walker", "quadruped"]

# Maps our internal pool/behavior slot ("walk"=calmer/abundant-by-default, "run"=more
# dynamic) to the ACTUAL dm_control task name for that domain. walker/quadruped really do
# have "walk"/"run" tasks, so this is the identity. hopper has no walk/run tasks at all
# (dm_control.suite.ALL_TASKS only exposes "stand"/"hop" for hopper) -- "stand" is aliased
# into our "walk" slot and "hop" into our "run" slot purely so every downstream script
# (Phase 5 onward: CSV "behavior" column, CONDITIONS naming, plotting scripts) keeps working
# completely unmodified. This alias is ONLY used here, for picking the right dm_control task
# / TD-MPC2 checkpoint file -- pool directories, manifests, and every later diagnostic still
# say "walk"/"run" throughout.
DOMAIN_BEHAVIOR_TASK = {
    "walker": {"walk": "walk", "run": "run"},
    "quadruped": {"walk": "walk", "run": "run"},
    "hopper": {"walk": "stand", "run": "hop"},
}


class RandomActionCollector:
    """Uniform-random-native-action policy. Not a checkpoint -- documented
    separately in the manifest as source='random_uniform', no checkpoint file."""

    def __init__(self, native_min: np.ndarray, native_max: np.ndarray, dtype) -> None:
        self._native_min = native_min
        self._native_max = native_max
        self._dtype = dtype
        self._rng = None

    def reset_episode(self, seed: int) -> None:
        self._rng = np.random.default_rng(seed)

    def act(self, state_observation, deterministic: bool = True) -> np.ndarray:
        del state_observation, deterministic
        return self._rng.uniform(self._native_min, self._native_max).astype(self._dtype)


def _forward_velocity(domain: str, dm_env: DMControlEnv) -> float:
    physics = dm_env._dm_env.physics
    if domain == "walker":
        return float(physics.horizontal_velocity())
    if domain == "hopper":
        # hopper's Physics exposes neither horizontal_velocity() nor torso_velocity() --
        # generic named-data lookup, matching the same "root body's x-velocity" idea.
        return float(physics.named.data.subtree_linvel["torso"][0])
    return float(physics.torso_velocity()[0])


def collect_episode(domain: str, task: str, collector, env_seed: int):
    dm_env = DMControlEnv(
        domain_name=domain, task_name=task, size=IMG_SIZE, camera_id=CAMERA_ID,
        action_repeat=ACTION_REPEAT, time_limit=None, seed=env_seed,
    )
    obs, _ = dm_env.reset(seed=env_seed)
    if hasattr(collector, "reset_episode"):
        try:
            collector.reset_episode(seed=env_seed)
        except TypeError:
            collector.reset_episode()  # TDMPC2Collector.reset_episode() takes no seed

    frames, actions, rewards, ends, truncs, fwd_vels = [], [], [], [], [], []
    final_obs = None
    terminated = truncated = False
    while not (terminated or truncated):
        state_obs = dm_env._dm_env.task.get_observation(dm_env._dm_env.physics)
        action = collector.act(state_obs, deterministic=True)

        frames.append(obs)
        actions.append(action)

        next_obs, reward, terminated, truncated, info = dm_env.step(action)
        rewards.append(reward)
        ends.append(1 if terminated else 0)
        truncs.append(1 if truncated else 0)
        fwd_vels.append(_forward_velocity(domain, dm_env))
        obs = next_obs

    final_obs = obs  # observation after the terminal/truncated step

    ep_obs = torch.from_numpy(np.stack(frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    ep_act = torch.from_numpy(np.stack(actions)).float()
    ep_rew = torch.tensor(rewards, dtype=torch.float32)
    ep_end = torch.tensor(ends, dtype=torch.uint8)
    ep_trunc = torch.tensor(truncs, dtype=torch.uint8)
    fo = torch.from_numpy(final_obs.copy()).float().div(255).mul(2).sub(1).permute(2, 0, 1).contiguous()
    info = {"final_observation": fo, "env_seed": env_seed}

    episode = Episode(obs=ep_obs, act=ep_act, rew=ep_rew, end=ep_end, trunc=ep_trunc, info=info)

    # per-transition finiteness/NaN checks (before any disk round trip)
    assert torch.isfinite(episode.obs).all(), f"non-finite obs, seed={env_seed}"
    assert torch.isfinite(episode.act).all(), f"non-finite act, seed={env_seed}"
    assert torch.isfinite(episode.rew).all(), f"non-finite rew, seed={env_seed}"

    return episode, np.array(fwd_vels)


def collect_pool(domain: str, source_name: str) -> dict:
    cfg = SOURCES[source_name]
    env_task = DOMAIN_BEHAVIOR_TASK[domain][cfg["env_task_for_instantiation"]]
    print(f"\n=== {domain}/{source_name} (target {cfg['target_transitions']} transitions) ===")

    probe_env = DMControlEnv(domain_name=domain, task_name=env_task, size=IMG_SIZE, camera_id=CAMERA_ID,
                              action_repeat=ACTION_REPEAT, time_limit=None)
    action_dim = probe_env.action_dim
    native_min, native_max = probe_env.action_low.copy(), probe_env.action_high.copy()

    checkpoint_name = None
    if source_name == "random":
        collector = RandomActionCollector(native_min, native_max, dtype=np.float32)
        collector_task_label = None
    else:
        checkpoint_task = DOMAIN_BEHAVIOR_TASK[domain][source_name]
        collector = load_pretrained_collector(source="tdmpc2", domain=domain, task=checkpoint_task, seed=COLLECTOR_SEED)
        checkpoint_name = f"{domain}-{checkpoint_task}-{COLLECTOR_SEED}.pt"
        collector_task_label = source_name

    pool_dir = OUT_ROOT / domain / slot_dir(domain, source_name)
    if pool_dir.exists():
        shutil.rmtree(pool_dir)
    dataset_dir = pool_dir / "dataset"
    dataset = Dataset(dataset_dir, name=f"{domain}_{source_name}", cache_in_ram=True)

    env_seeds_used, fwd_vel_chunks = [], []
    action_min_seen, action_max_seen = None, None
    all_finite = True
    ep_idx = 0
    while dataset.num_steps < cfg["target_transitions"]:
        env_seed = cfg["seed_base"] + ep_idx
        task_for_episode = env_task  # 'walk' env for random+walk sources, 'run' env for run source
        episode, fwd_vels = collect_episode(domain, task_for_episode, collector, env_seed)

        act_np = episode.act.numpy()
        action_min_seen = act_np.min(axis=0) if action_min_seen is None else np.minimum(action_min_seen, act_np.min(axis=0))
        action_max_seen = act_np.max(axis=0) if action_max_seen is None else np.maximum(action_max_seen, act_np.max(axis=0))
        all_finite = all_finite and bool(torch.isfinite(episode.obs).all() and torch.isfinite(episode.act).all())

        dataset.add_episode(episode)
        env_seeds_used.append(env_seed)
        if source_name != "random":
            fwd_vel_chunks.append(fwd_vels)
        ep_idx += 1
        print(f"  episode {ep_idx - 1} (env_seed={env_seed}): length={len(episode)}, "
              f"return={episode.rew.sum().item():.2f}, running total steps={dataset.num_steps}")

    dataset.save_to_default_path()
    print(f"  collected {dataset.num_episodes} episodes, {dataset.num_steps} transitions -> {dataset_dir}")

    # --- validation: round trip + shapes ---
    fresh = Dataset(dataset_dir, name=f"{domain}_{source_name}_reloaded", cache_in_ram=True)
    fresh.load_from_default_path()
    assert fresh.num_episodes == dataset.num_episodes
    assert fresh.num_steps == dataset.num_steps
    roundtrip_ok = True
    obs_shape = None
    try:
        for eid in range(fresh.num_episodes):
            ep = fresh.load_episode(eid)
            if obs_shape is None:
                obs_shape = tuple(ep.obs.shape[1:])
            assert ep.obs.shape[1:] == (3, IMG_SIZE, IMG_SIZE)
            assert ep.act.shape[1] == action_dim
            assert torch.isfinite(ep.obs).all() and torch.isfinite(ep.act).all() and torch.isfinite(ep.rew).all()
        traverser = DatasetTraverser(fresh, batch_num_samples=4, chunk_size=16)
        n_batches = sum(1 for _ in traverser)
        print(f"  round trip OK: {n_batches} DatasetTraverser batches")
    except Exception as e:  # noqa: BLE001
        roundtrip_ok = False
        print(f"  ROUND TRIP FAILED: {e}")

    velocity_stats = None
    if fwd_vel_chunks:
        v = np.concatenate(fwd_vel_chunks)
        velocity_stats = {
            "mean": float(v.mean()), "std": float(v.std()), "median": float(np.median(v)),
            "p10": float(np.percentile(v, 10)), "p90": float(np.percentile(v, 90)),
        }
        print(f"  forward velocity: mean={velocity_stats['mean']:.4f} std={velocity_stats['std']:.4f} "
              f"median={velocity_stats['median']:.4f} p10={velocity_stats['p10']:.4f} p90={velocity_stats['p90']:.4f}")

    manifest = {
        "source": "random_uniform" if source_name == "random" else "tdmpc2",
        "domain": domain,
        "task": collector_task_label,
        "collector_checkpoint": checkpoint_name,
        "collector_seed": COLLECTOR_SEED if checkpoint_name else None,
        "environment_task_used_for_instantiation": env_task,
        "environment_seeds": env_seeds_used,
        "seed_range_documented": f"[{cfg['seed_base']}, {cfg['seed_base'] + ep_idx - 1}]",
        "num_episodes": dataset.num_episodes,
        "num_transitions": dataset.num_steps,
        "target_transitions": cfg["target_transitions"],
        "action_repeat": ACTION_REPEAT,
        "observation_shape_chw": list(obs_shape) if obs_shape else None,
        "action_dim": action_dim,
        "action_min_observed": action_min_seen.tolist(),
        "action_max_observed": action_max_seen.tolist(),
        "action_native_bounds_min": native_min.tolist(),
        "action_native_bounds_max": native_max.tolist(),
        "all_values_finite": all_finite,
        "dataset_roundtrip_passed": roundtrip_ok,
        "forward_velocity_stats": velocity_stats,
        "collection_timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "dataset_path": str(dataset_dir.relative_to(_PROJECT_ROOT)),
    }
    with open(pool_dir / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)
    return manifest


def main() -> None:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    all_manifests = {}
    for domain in DOMAINS:
        for source_name in SOURCES:
            m = collect_pool(domain, source_name)
            all_manifests[f"{domain}/{source_name}"] = m

    with open(OUT_ROOT / "phase2_summary.json", "w") as f:
        json.dump(all_manifests, f, indent=2)

    print("\n\n=== PHASE 2 SUMMARY ===")
    for key, m in all_manifests.items():
        v = m["forward_velocity_stats"]
        v_str = f"fwd_vel_mean={v['mean']:.4f}" if v else "fwd_vel=n/a"
        print(f"{key:20s} episodes={m['num_episodes']:3d} transitions={m['num_transitions']:5d} "
              f"roundtrip={m['dataset_roundtrip_passed']} finite={m['all_values_finite']} {v_str}")


if __name__ == "__main__":
    main()
