#! /usr/bin/env python
"""
Stage 7A verification: runtime integration of the already-implemented
continuous-control components (DM Control wrapper, continuous replay,
InnerModel/RewEndModel/ActorCritic continuous paths).

Everything here goes through the *real* runtime path: Hydra config composition
-> {make_atari_env, make_dm_control_env} dispatch -> get_action_space_kwargs ->
Agent(instantiate(cfg.agent, **kwargs)) -> coroutines.env_loop.make_env_loop.
Nothing here imports the DMControlEnv class or the Stage 1-3 smoke-test
wrapper directly.

Does NOT run Trainer.run() / initial dataset collection / training -- this is
integration-of-components verification only, per Stage 7A's scope.

Usage:
    python scripts/test_stage7a_integration.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import torch
from omegaconf import OmegaConf

OmegaConf.register_new_resolver("eval", eval, replace=True)

from hydra import initialize, compose
from hydra.utils import instantiate

from envs import make_atari_env, make_dm_control_env
from agent import Agent, get_action_space_kwargs
from coroutines.env_loop import make_env_loop


def build_cfg(overrides):
    with initialize(config_path="../config", version_base="1.3"):
        cfg = compose(config_name="trainer", overrides=overrides)
        OmegaConf.resolve(cfg)
    return cfg


def make_env_from_cfg(cfg, num_envs, device):
    make_env = {"atari": make_atari_env, "dm_control": make_dm_control_env}[cfg.env.train.type]
    kwargs = {k: v for k, v in cfg.env.train.items() if k != "type"}
    return make_env(num_envs=num_envs, device=device, **kwargs)


def act_emb_layer(module_act_emb):
    """InnerModel.act_emb is nn.Sequential(proj, Flatten); RewEndModel.act_emb is the bare proj."""
    return module_act_emb[0] if isinstance(module_act_emb, torch.nn.Sequential) else module_act_emb


# ---------------------------------------------------------------------------
# Atari regression
# ---------------------------------------------------------------------------

def check_atari_regression(device: torch.device) -> None:
    print(f"\n{'=' * 70}\nAtari regression [{device}]\n{'=' * 70}")
    cfg = build_cfg([])  # defaults: env=atari
    assert cfg.env.train.type == "atari"

    env = make_env_from_cfg(cfg, num_envs=1, device=device)
    try:
        assert env.is_discrete is True
        assert env.num_actions is not None and env.action_dim is None
        print(f"Environment construction OK: is_discrete={env.is_discrete}, num_actions={env.num_actions}")

        action_kwargs = get_action_space_kwargs(env)
        assert action_kwargs["num_actions"] == env.num_actions
        assert action_kwargs["continuous_action_dim"] is None
        assert action_kwargs["continuous_reward"] is False
        print(f"get_action_space_kwargs: {action_kwargs}")

        agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
        assert type(act_emb_layer(agent.denoiser.inner_model.act_emb)).__name__ == "Embedding"
        assert type(agent.rew_end_model.act_emb).__name__ == "Embedding"
        assert agent.rew_end_model.continuous_reward is False
        assert agent.actor_critic.continuous_action is False
        assert agent.actor_critic.continuous_reward is False
        assert agent.actor_critic.actor_linear.out_features == env.num_actions
        print("Agent received num_actions; all three model components on discrete path  OK")

        el = make_env_loop(env, agent.actor_critic, epsilon=0.0)
        _, act, rew, end, trunc, *_ = el.send(20)
        assert act.dtype == torch.int64, f"expected int64, got {act.dtype}"
        assert act.shape == (1, 20), f"expected (1,20), got {act.shape}"
        print(f"env_loop (epsilon=0.0): act shape={tuple(act.shape)} dtype={act.dtype}  OK")

        el_eps = make_env_loop(env, agent.actor_critic, epsilon=1.0)
        _, act_eps, *_ = el_eps.send(20)
        assert act_eps.dtype == torch.int64
        assert (act_eps >= 0).all() and (act_eps < env.num_actions).all()
        print(f"env_loop (epsilon=1.0): discrete random actions in [0,{env.num_actions})  OK")
    finally:
        env.close()


# ---------------------------------------------------------------------------
# DM Control
# ---------------------------------------------------------------------------

def check_dm_control(device: torch.device, domain_name: str, task_name: str) -> None:
    print(f"\n{'=' * 70}\nDM Control [{device}] domain_name={domain_name!r} task_name={task_name!r}\n{'=' * 70}")
    cfg = build_cfg([
        "env=dm_control",
        f"env.train.domain_name={domain_name}",
        f"env.train.task_name={task_name}",
    ])
    assert cfg.env.train.type == "dm_control"
    assert cfg.env.train.domain_name == domain_name and cfg.env.train.task_name == task_name

    env = make_env_from_cfg(cfg, num_envs=1, device=device)
    try:
        assert env.is_discrete is False
        assert env.num_actions is None
        action_dim = env.action_dim
        low, high = env.action_low, env.action_high
        assert low.shape == (action_dim,) and high.shape == (action_dim,)
        assert (low < high).all()
        print(f"Environment construction OK (real Hydra/runtime path): action_dim={action_dim}, "
              f"low={low.tolist()}, high={high.tolist()}")

        action_kwargs = get_action_space_kwargs(env)
        assert action_kwargs["num_actions"] is None
        assert action_kwargs["continuous_action_dim"] == action_dim
        assert action_kwargs["action_low"] == low.tolist()
        assert action_kwargs["action_high"] == high.tolist()
        assert action_kwargs["continuous_reward"] is True
        print(f"get_action_space_kwargs: continuous_action_dim={action_kwargs['continuous_action_dim']}, "
              f"continuous_reward={action_kwargs['continuous_reward']}")

        agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)

        inner_proj = act_emb_layer(agent.denoiser.inner_model.act_emb)
        assert type(inner_proj).__name__ == "Linear", "InnerModel must use Linear action projection"
        assert inner_proj.in_features == action_dim, "InnerModel action projection dim must not be hard-coded"
        print(f"InnerModel: act_emb projection = Linear(in_features={inner_proj.in_features})  OK")

        assert type(agent.rew_end_model.act_emb).__name__ == "Linear", "RewEndModel must use Linear action projection"
        assert agent.rew_end_model.act_emb.in_features == action_dim
        assert agent.rew_end_model.continuous_reward is True
        print(f"RewEndModel: act_emb = Linear(in_features={agent.rew_end_model.act_emb.in_features}), "
              f"continuous_reward=True  OK")

        assert agent.actor_critic.continuous_action is True
        assert agent.actor_critic.continuous_reward is True
        assert agent.actor_critic.actor_linear.out_features == 2 * action_dim, (
            "ActorCritic actor_linear width must be task-dependent (2*action_dim), not hard-coded"
        )
        print(f"ActorCritic: continuous_action=True, continuous_reward=True, "
              f"actor_linear.out_features={agent.actor_critic.actor_linear.out_features} (=2*{action_dim})  OK")

        # >= 100 real environment steps, epsilon=0 (policy-sampled actions)
        el = make_env_loop(env, agent.actor_critic, epsilon=0.0)
        _, act, rew, end, trunc, *_ = el.send(120)
        assert act.shape == (1, 120, action_dim), f"expected (1,120,{action_dim}), got {act.shape}"
        assert act.dtype == torch.float32, f"expected float32, got {act.dtype}"
        low_b = low.to(act.device)
        high_b = high.to(act.device)
        assert (act >= low_b - 1e-4).all() and (act <= high_b + 1e-4).all(), "policy action escaped bounds"
        assert rew.dtype.is_floating_point, f"expected floating-point reward, got {rew.dtype}"
        assert torch.isfinite(rew).all(), "non-finite reward during real rollout"
        # NOTE: this is env_loop stepping a real TorchEnv directly -- rew is TorchEnv.step()'s raw,
        # unmodified environment reward in every case (discrete or continuous); no sign() is ever
        # applied in this path regardless of mode, so a degenerate all-zero reward sequence here
        # (e.g. an untrained policy making a task collapse into its "no progress" reward-0 regime,
        # which does happen for hopper-hop) is legitimate continuous behavior, not evidence of
        # accidental clipping. The rew.sign() logic lives in RewEndModel's training targets and
        # ActorCritic.compute_lambda_returns, both already covered by Stages 5B/6 -- verified here
        # via the continuous_reward=True flags asserted above, not by re-deriving it from rollout
        # reward values.
        print(f"120 real env steps OK: act shape={tuple(act.shape)} dtype={act.dtype}, within bounds, "
              f"rewards finite float32 (sample={rew[0, :5].tolist()})")

        # epsilon=1 -> uniform random actions within bounds (continuous exploration replacement)
        el_eps = make_env_loop(env, agent.actor_critic, epsilon=1.0)
        _, act_eps, *_ = el_eps.send(30)
        assert act_eps.shape == (1, 30, action_dim)
        assert act_eps.dtype == torch.float32
        assert (act_eps >= low_b - 1e-4).all() and (act_eps <= high_b + 1e-4).all()
        # sanity: uniform random actions should spread noticeably across the bound range, unlike a
        # policy that hasn't learned anything yet but is still centered near a single mean
        spread = (act_eps.amax(dim=1) - act_eps.amin(dim=1))
        assert (spread > 0.1 * (high_b - low_b)).any(), "epsilon=1 actions suspiciously narrow for uniform sampling"
        print(f"epsilon=1.0: uniform random actions within bounds, shape={tuple(act_eps.shape)}  OK")
    finally:
        env.close()


def main() -> None:
    devices = [torch.device("cpu")]
    if torch.cuda.is_available():
        devices.append(torch.device("cuda"))
    print(f"Devices under test: {[str(d) for d in devices]}")

    for device in devices:
        check_atari_regression(device)
        check_dm_control(device, "cheetah", "run")
        check_dm_control(device, "hopper", "hop")

    print("\nAll Stage 7A integration checks passed.")


if __name__ == "__main__":
    main()
