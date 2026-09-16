"""Fixed-seed deterministic vs. stochastic evaluation for the continuous-action actor-critic.

The routine training-loop evaluation (`evaluation.every`) collects only a handful of episodes
with the *stochastic* policy and reset seeds that are not fixed across checkpoints, which is too
small/noisy a sample to judge whether the policy is actually improving. This script instead:
  - Rolls out the SAME fixed set of environment seeds at every checkpoint it's run against, so
    results are directly comparable across epochs.
  - Reports BOTH deterministic (`action = tanh(mean)`, i.e. `sample_action(..., deterministic=True)`)
    and stochastic (sampled from the current policy) returns, since a policy can look fine in one
    mode and be dominated by noise in the other.
  - Uses the real DMControlEnv (test env config), not the world model -- this measures real-
    environment performance, not imagined-rollout performance.

This is read-only: it loads a checkpoint, runs no training step, and does not touch any existing
run's dataset/checkpoints/wandb state.

Usage (from the repo root, with the `lcg` conda env active):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<n> python scripts/diagnostics/fixed_seed_eval.py \\
        <path_to_checkpoint.pt> [--n-seeds 10] [--base-seed 100000] [--max-steps 2000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import numpy as np
import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from agent import Agent, get_action_space_kwargs  # noqa: E402
from envs import make_dm_control_env  # noqa: E402


def rollout_returns(env, model, deterministic: bool, seeds: list, max_steps: int) -> np.ndarray:
    """Runs every one of `env`'s sub-envs (one per fixed seed) to completion (real termination
    or truncation at the env's own time_limit), accumulating each env's total (unmodified) reward.
    `alive` freezes a sub-env's accumulated return the instant it reports end/trunc, so a vector
    env's automatic per-env reset afterward can never leak into the reported episode's return."""
    num_envs = env.num_envs
    assert num_envs == len(seeds)
    device = model.device
    hx = torch.zeros(num_envs, model.lstm_dim, device=device)
    cx = torch.zeros(num_envs, model.lstm_dim, device=device)
    obs, _ = env.reset(seed=list(seeds))
    returns = torch.zeros(num_envs, device=device)
    lengths = torch.zeros(num_envs, device=device)
    alive = torch.ones(num_envs, dtype=torch.bool, device=device)

    with torch.no_grad():
        for step in range(max_steps):
            logits_act, val, (hx, cx) = model.predict_act_value(obs, (hx, cx))
            act, z = model.sample_action(logits_act, deterministic=deterministic)
            obs, rew, end, trunc, info = env.step(act)
            returns += rew * alive.float()
            lengths += alive.float()
            dead_now = torch.logical_or(end.bool(), trunc.bool())
            alive = alive & (~dead_now)
            if not alive.any():
                break
        else:
            raise RuntimeError(
                f"{int(alive.sum().item())}/{num_envs} episode(s) did not terminate within "
                f"max_steps={max_steps}; pass a larger --max-steps"
            )

    return returns.cpu().numpy(), lengths.cpu().numpy()


def report(mode_name: str, returns: np.ndarray, lengths: np.ndarray, seeds: list) -> None:
    print(f"\n=== {mode_name} (n={len(returns)} fixed seeds) ===")
    for s, r, l in zip(seeds, returns, lengths):
        print(f"  seed={s}: return={r:.4f} length={int(l)}")
    print(
        f"mean={returns.mean():.4f} median={np.median(returns):.4f} std={returns.std():.4f} "
        f"min={returns.min():.4f} max={returns.max():.4f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("ckpt_path", type=str)
    parser.add_argument("--n-seeds", type=int, default=10)
    parser.add_argument("--base-seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=2000)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with initialize(version_base="1.3", config_path="../../config"):
        cfg = compose(config_name="trainer", overrides=[
            "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
        ])

    seeds = [args.base_seed + i for i in range(args.n_seeds)]
    env_kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = make_dm_control_env(num_envs=args.n_seeds, device=device, **env_kwargs)
    action_kwargs = get_action_space_kwargs(env)

    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.load(args.ckpt_path)
    agent.eval()
    print(f"loaded {args.ckpt_path}", flush=True)
    print(f"fixed seeds (n={args.n_seeds}): {seeds}", flush=True)

    model = agent.actor_critic

    det_returns, det_lengths = rollout_returns(env, model, True, seeds, args.max_steps)
    report("Deterministic (action = tanh(mean))", det_returns, det_lengths, seeds)

    sto_returns, sto_lengths = rollout_returns(env, model, False, seeds, args.max_steps)
    report("Stochastic (sampled from policy)", sto_returns, sto_lengths, seeds)


if __name__ == "__main__":
    main()
