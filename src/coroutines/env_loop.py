import random
from typing import Generator, Tuple, Union

import torch
import torch.nn as nn

from . import coroutine
from .frame_history import FrameHistory
from envs import TorchEnv, WorldModelEnv


@coroutine
def make_env_loop(
    env: Union[TorchEnv, WorldModelEnv], model: nn.Module, epsilon: float = 0.0,
    store_policy_observations: bool = False,
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    num_steps = yield

    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])
    history = FrameHistory(obs, model.frame_stack)

    while True:
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            # WorldModelEnv reset_dead() mutates its observation buffer in place.
            # Retain the actual pre-action state for likelihood replay.
            obs = obs.clone()
            policy_obs = history.state
            logits_act, val = model.predict_act_value(policy_obs)
            act = model.sample_action(logits_act)

            if random.random() < epsilon:
                act = torch.rand(obs.size(0), env.action_dim, device=obs.device)
                act = env.action_low + act * (env.action_high - env.action_low)

            next_obs, rew, end, trunc, info = env.step(act)

            if n > 0:
                val_bootstrap = val.detach().clone()
                if dead.any():
                    val_bootstrap[dead] = val_final_obs
                all_[-1][-1] = val_bootstrap

            dead = torch.logical_or(end, trunc)

            if dead.any():
                with torch.no_grad():
                    final_obs = info["final_observation"]
                    final_obs = history.successor(final_obs, dead)
                    _, val_final_obs = model.predict_act_value(final_obs)

            # Real collectors retain RGB for WM training. PMPO experience retains
            # exactly the temporal state on which its sampled action was based.
            stored_obs = policy_obs if store_policy_observations else obs
            all_.append([stored_obs, act, rew, end, trunc, logits_act, val, None])
            infos.append(info)

            history.advance(next_obs, dead)

            obs = next_obs
            n += 1

        with torch.no_grad():
            bootstrap_obs = history.state
            _, val_bootstrap = model.predict_act_value(bootstrap_obs)

        if dead.any():
            val_bootstrap[dead] = val_final_obs

        all_[-1][-1] = val_bootstrap

        all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap = (torch.stack(x, dim=1) for x in zip(*all_))

        num_steps = yield all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap, infos
