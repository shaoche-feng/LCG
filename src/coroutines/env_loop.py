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

    hx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)
    cx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)

    seed = random.randint(0, 2**31 - 1)
    obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])
    stack_length = getattr(model, "frame_stack", 1)
    history = FrameHistory(obs, stack_length) if stack_length > 1 else None

    while True:
        hx, cx = hx.detach(), cx.detach()
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            # WorldModelEnv reset_dead() mutates its observation buffer in place.
            # Retain the actual pre-action state for likelihood replay.
            obs = obs.clone()
            policy_obs = history.state if history is not None else obs
            logits_act, val, (hx, cx) = model.predict_act_value(policy_obs, (hx, cx))
            act = model.sample_action(logits_act)

            if random.random() < epsilon:
                if env.is_discrete:
                    act = torch.randint(low=0, high=env.num_actions, size=(obs.size(0),), device=obs.device)
                else:
                    # Uniform random action within the environment's configured bounds --
                    # the continuous analogue of the discrete random-replacement above.
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
                    if history is not None:
                        final_obs = history.successor(final_obs, dead)
                    _, val_final_obs, _ = model.predict_act_value(final_obs, (hx[dead], cx[dead]))
                reset_gate = 1 - dead.float().unsqueeze(1)
                hx = hx * reset_gate
                cx = cx * reset_gate
                if history is None and "burnin_obs" in info:
                    burnin_obs = info["burnin_obs"]
                    for i in range(burnin_obs.size(1)):
                        _, _, (hx[dead], cx[dead]) = model.predict_act_value(burnin_obs[:, i], (hx[dead], cx[dead]))

            # Real collectors retain RGB for WM training. PMPO experience retains
            # exactly the temporal state on which its sampled action was based.
            stored_obs = policy_obs if store_policy_observations else obs
            all_.append([stored_obs, act, rew, end, trunc, logits_act, val, None])
            infos.append(info)

            if history is not None:
                history.advance(next_obs, dead)

            obs = next_obs
            n += 1

        with torch.no_grad():
            bootstrap_obs = history.state if history is not None else next_obs
            _, val_bootstrap, _ = model.predict_act_value(bootstrap_obs, (hx, cx))  # do not update hx/cx

        if dead.any():
            val_bootstrap[dead] = val_final_obs

        all_[-1][-1] = val_bootstrap

        all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap = (torch.stack(x, dim=1) for x in zip(*all_))

        num_steps = yield all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap, infos
