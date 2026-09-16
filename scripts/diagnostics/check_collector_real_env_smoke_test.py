"""Real-environment collector smoke test for the continuous-action `z`-carrying rollout change.

Exercises the ACTUAL real DMControlEnv -> env_loop -> collector -> Dataset path (not just static
inspection of the tuple-unpacking) to confirm:
  - No tuple-unpacking or coroutine errors after env_loop's yielded tuple gained a `z` field.
  - The environment receives an action tensor of the expected shape and within its bounds.
  - `z` never enters the saved Episode (Episode's dataclass fields are exactly
    obs/act/rew/end/trunc/info; collector.py never extracts z from the yielded tuple at all).
  - An episode can be saved to disk and reloaded, byte-identical.
  - obs/act/rew/end/trunc shapes match expectations.

Usage (from the repo root, with the `lcg` conda env active):
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=<n> python scripts/diagnostics/check_collector_real_env_smoke_test.py [--steps 50]

Writes and removes a scratch dataset under outputs/_collector_smoke_test_dataset/; does not
touch any real training run's checkpoints or datasets.
"""
import argparse
import dataclasses
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import torch
from hydra import compose, initialize
from hydra.utils import instantiate
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from agent import Agent, get_action_space_kwargs  # noqa: E402
from coroutines.collector import make_collector, NumToCollect  # noqa: E402
from data import Dataset, Episode  # noqa: E402
from envs import make_dm_control_env  # noqa: E402

DATASET_TMP = "outputs/_collector_smoke_test_dataset"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--steps", type=int, default=50)
    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    with initialize(version_base="1.3", config_path="../../config"):
        cfg = compose(config_name="trainer", overrides=[
            "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
        ])

    env_kwargs = {k: v for k, v in cfg.env.train.items() if k != "type"}
    env = make_dm_control_env(num_envs=1, device=device, **env_kwargs)
    print(f"env: is_discrete={env.is_discrete} action_dim={env.action_dim} "
          f"action_low={env.action_low.tolist()} action_high={env.action_high.tolist()}", flush=True)

    action_kwargs = get_action_space_kwargs(env)
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.eval()

    captured_actions = []
    _orig_step = env.step

    def wrapped_step(actions):
        captured_actions.append(actions.detach().clone())
        return _orig_step(actions)

    env.step = wrapped_step

    shutil.rmtree(DATASET_TMP, ignore_errors=True)
    dataset = Dataset(DATASET_TMP, "collector_smoke_test")

    collector = make_collector(env, agent.actor_critic, dataset, epsilon=cfg.collection.train.epsilon)
    logs = collector.send(NumToCollect(steps=args.steps))
    print(f"\ncollector.send(NumToCollect(steps={args.steps})) returned without error. logs: {logs}", flush=True)

    print(f"\ncaptured {len(captured_actions)} calls to env.step()")
    all_actions = torch.cat(captured_actions, dim=0)
    print(f"action tensor shape per call: {captured_actions[0].shape} (expect (1, {env.action_dim}))")
    assert captured_actions[0].shape == (1, env.action_dim)
    print(f"action bounds actually sent: min={all_actions.min().item():.4f} max={all_actions.max().item():.4f} "
          f"(expect within [{env.action_low.min().item():.4f}, {env.action_high.max().item():.4f}])")
    assert (all_actions >= env.action_low - 1e-5).all() and (all_actions <= env.action_high + 1e-5).all()

    episode_fields = {f.name for f in dataclasses.fields(Episode)}
    print(f"\nEpisode dataclass fields: {sorted(episode_fields)}")
    assert episode_fields == {"obs", "act", "rew", "end", "trunc", "info"}, (
        f"unexpected Episode fields: {episode_fields}"
    )
    assert dataset.num_steps >= args.steps, f"expected >={args.steps} collected steps, got {dataset.num_steps}"
    ep = dataset.load_episode(0)
    assert not hasattr(ep, "z"), "Episode must not carry a z attribute"
    print(f"loaded episode 0: obs={ep.obs.shape} act={ep.act.shape} rew={ep.rew.shape} "
          f"end={ep.end.shape} trunc={ep.trunc.shape} dtype(act)={ep.act.dtype}")

    T = ep.obs.shape[0]
    assert ep.act.shape == (T, env.action_dim), f"act shape {ep.act.shape} != ({T}, {env.action_dim})"
    assert ep.act.dtype == torch.float32
    assert ep.rew.shape == (T,)
    assert ep.end.shape == (T,)
    assert ep.trunc.shape == (T,)
    assert ep.obs.ndim == 4 and ep.obs.shape[0] == T, f"obs shape {ep.obs.shape}"
    print(f"shape checks passed: obs={ep.obs.shape} act={ep.act.shape} rew={ep.rew.shape} "
          f"end={ep.end.shape} trunc={ep.trunc.shape}")

    dataset.save_to_default_path()
    dataset2 = Dataset(DATASET_TMP, "collector_smoke_test_reloaded")
    dataset2.load_from_default_path()
    assert dataset2.num_steps == dataset.num_steps, f"{dataset2.num_steps} != {dataset.num_steps}"
    ep2 = dataset2.load_episode(0)
    assert torch.equal(ep2.obs, ep.obs)
    assert torch.equal(ep2.act, ep.act)
    assert torch.equal(ep2.rew, ep.rew)
    assert torch.equal(ep2.end, ep.end)
    assert torch.equal(ep2.trunc, ep.trunc)
    print(f"\nsave/reload round-trip verified byte-identical: num_steps={dataset2.num_steps}, "
          f"num_episodes={dataset2.num_episodes}")

    shutil.rmtree(DATASET_TMP, ignore_errors=True)
    print("\nALL COLLECTOR SMOKE TEST CHECKS PASSED.")


if __name__ == "__main__":
    main()
