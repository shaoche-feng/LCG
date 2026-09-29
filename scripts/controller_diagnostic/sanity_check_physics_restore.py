"""Standalone sanity check for DM Control physics save/restore/replay determinism,
before building the full advantage-magnitude diagnostic. No training/model code
involved -- pure environment-level check."""
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")
import numpy as np
from envs.dm_control_env import DMControlEnv

env = DMControlEnv(domain_name="walker", task_name="walk", size=64, camera_id=0, action_repeat=2, time_limit=None, seed=3001)
obs0, _ = env.reset(seed=3001)
physics = env._dm_env.physics  # captured AFTER reset, per the known bug from earlier work

rng = np.random.RandomState(0)
# advance a handful of real steps first, to get away from the exact reset state
for _ in range(10):
    a = rng.uniform(env.action_low, env.action_high).astype(np.float32)
    env.step(a)

print("get_state() shape:", physics.get_state().shape)
print("physics.data.time before save:", physics.data.time)

saved_state = physics.get_state().copy()
saved_time = physics.data.time

action_a = rng.uniform(env.action_low, env.action_high).astype(np.float32)
obs_a1, rew_a1, end_a1, trunc_a1, info_a1 = env.step(action_a)
obs_a2, rew_a2, end_a2, trunc_a2, info_a2 = env.step(action_a)
print(f"\nFirst pass: step1 reward={rew_a1:.6f} step2 reward={rew_a2:.6f}")
print("obs_a2 checksum:", obs_a2.astype(np.int64).sum())

# restore and replay the SAME two actions
physics.set_state(saved_state)
physics.data.time = saved_time
physics.forward()
print("\nphysics.data.time after restore:", physics.data.time)
print("state matches saved:", np.allclose(physics.get_state(), saved_state))

obs_b1, rew_b1, end_b1, trunc_b1, info_b1 = env.step(action_a)
obs_b2, rew_b2, end_b2, trunc_b2, info_b2 = env.step(action_a)
print(f"\nReplay: step1 reward={rew_b1:.6f} step2 reward={rew_b2:.6f}")
print("obs_b2 checksum:", obs_b2.astype(np.int64).sum())

print("\n=== SANITY CHECK RESULT ===")
print("reward match (step1):", abs(rew_a1 - rew_b1) < 1e-8, f"(diff={abs(rew_a1-rew_b1):.2e})")
print("reward match (step2):", abs(rew_a2 - rew_b2) < 1e-8, f"(diff={abs(rew_a2-rew_b2):.2e})")
print("obs exact match (step2):", np.array_equal(obs_a2, obs_b2))
print("obs max abs diff (step2):", np.abs(obs_a2.astype(np.int64) - obs_b2.astype(np.int64)).max())
print("end/trunc match:", (end_a1, trunc_a1, end_a2, trunc_a2) == (end_b1, trunc_b1, end_b2, trunc_b2))

# also test restoring to an EARLIER point after having advanced further, to make sure
# set_state correctly overrides subsequent drift (not just a no-op due to determinism alone)
for _ in range(5):
    a = rng.uniform(env.action_low, env.action_high).astype(np.float32)
    env.step(a)
physics.set_state(saved_state)
physics.data.time = saved_time
physics.forward()
obs_c1, rew_c1, *_ = env.step(action_a)
print("\nAfter drifting away and restoring again, reward matches original:", abs(rew_c1 - rew_a1) < 1e-8)
env.close()
