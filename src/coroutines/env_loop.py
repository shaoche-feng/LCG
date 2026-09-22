import random
from typing import Generator, Optional, Tuple, Union

import torch
import torch.nn as nn

from . import coroutine
from envs import TorchEnv, WorldModelEnv


class RolloutHxCxState:
    """Mutable holder for ONE env_loop coroutine's model-LSTM rollout state (hx, cx), so it
    can be read for checkpointing and written for resume from OUTSIDE the coroutine --
    generators otherwise keep their locals opaque. One instance per env_loop; never share
    across multiple env_loop coroutines using the same model (e.g. a real-env collector and
    the imagined-rollout trainer each have their own env_loop over the SAME ActorCritic, and
    would corrupt each other's rollout state if they shared one holder).
    """

    def __init__(self) -> None:
        self.hx: Optional[torch.Tensor] = None
        self.cx: Optional[torch.Tensor] = None
        # False until env_loop's first real iteration in THIS process sets hx/cx -- guards
        # against "resuming" from a holder that was loaded from a checkpoint but never
        # actually populated (e.g. actor_critic training hadn't started yet when saved).
        self.initialized: bool = False

    def state_dict(self) -> dict:
        return {"hx": self.hx, "cx": self.cx, "initialized": self.initialized}

    def load_state_dict(self, state_dict: dict) -> None:
        self.hx = state_dict["hx"]
        self.cx = state_dict["cx"]
        self.initialized = state_dict["initialized"]


@coroutine
def make_env_loop(
    env: Union[TorchEnv, WorldModelEnv],
    model: nn.Module,
    epsilon: float = 0.0,
    hx_cx_state: Optional[RolloutHxCxState] = None,
) -> Generator[Tuple[torch.Tensor, ...], int, None]:
    """hx_cx_state=None (every existing call site: real-env train/test collectors) preserves
    the exact prior behavior -- hx/cx always start at zero, env.reset() always runs. Only
    ActorCritic.setup_training's imagined-rollout env_loop passes a real hx_cx_state, opting
    into resumable rollout continuity: if that state was already `initialized` (i.e. loaded
    from a checkpoint saved mid-rollout) AND `env` is a WorldModelEnv with its buffers already
    restored (see WorldModelEnv.load_rollout_state_dict, which Trainer calls before this
    coroutine's first real .send() on resume), hx/cx and the current observation are taken
    from that saved state instead of a fresh zero-init + env.reset() -- reset() would discard
    the just-restored WorldModelEnv buffers.
    """
    num_steps = yield

    resuming = hx_cx_state is not None and hx_cx_state.initialized
    if resuming:
        hx, cx = hx_cx_state.hx, hx_cx_state.cx
    elif hasattr(model, "initial_hx_cx"):
        # Optional model-provided override of the zero-init below -- e.g. DrQActorCritic uses
        # this to seed a NaN sentinel instead of zeros, so predict_act_value can reliably tell
        # "never touched, needs real seeding" apart from a mid-rollout reset_gate zero-out
        # (see DrQActorCritic.initial_hx_cx's docstring). Absent for every other model
        # (ActorCritic doesn't define this), so this branch changes nothing for them.
        hx, cx = model.initial_hx_cx(env.num_envs)
    else:
        hx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)
        cx = torch.zeros(env.num_envs, model.lstm_dim, device=model.device)

    if resuming and hasattr(env, "obs_buffer"):
        obs = env.obs_buffer[:, -1]
    else:
        seed = random.randint(0, 2**31 - 1)
        obs, _ = env.reset(seed=[seed + i for i in range(env.num_envs)])

    while True:
        hx, cx = hx.detach(), cx.detach()
        all_ = []
        infos = []
        n = 0

        while n < num_steps:
            logits_act, val, (hx, cx) = model.predict_act_value(obs, (hx, cx))
            # Captured HERE, immediately after the call that actually produced logits_act/act
            # for THIS step -- NOT the `hx` variable read later when building this step's row
            # (all_.append below), which by then may have been overwritten by the dead-env
            # reset_gate/burn-in handling for the NEXT step. Recording the wrong one would make
            # a caller reconstructing "the state that produced each action" (see
            # models.drq_actor_critic's collect_rollout) see the state AFTER a mid-rollout
            # reset instead of the state that was actually used.
            action_time_hx = hx
            act, z = model.sample_action(logits_act)

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
                    _, val_final_obs, _ = model.predict_act_value(info["final_observation"], (hx[dead], cx[dead]))
                reset_gate = 1 - dead.float().unsqueeze(1)
                hx = hx * reset_gate
                cx = cx * reset_gate
                if "burnin_obs" in info:
                    burnin_obs = info["burnin_obs"]
                    for i in range(burnin_obs.size(1)):
                        _, _, (hx[dead], cx[dead]) = model.predict_act_value(burnin_obs[:, i], (hx[dead], cx[dead]))

            all_.append([obs, act, rew, end, trunc, logits_act, val, z, action_time_hx, None])
            infos.append(info)

            obs = next_obs
            n += 1

        with torch.no_grad():
            _, val_bootstrap, _ = model.predict_act_value(next_obs, (hx, cx))  # do not update hx/cx

        if dead.any():
            val_bootstrap[dead] = val_final_obs

        all_[-1][-1] = val_bootstrap

        if hx_cx_state is not None:
            # Captured HERE (right before yielding, i.e. right before this coroutine suspends
            # and a checkpoint could be taken), using the hx/cx this rollout actually ENDED
            # on -- NOT at the top of the loop, which would capture the value this iteration
            # STARTED from (i.e. the previous call's ending state, one step stale). Found via
            # the resume-fidelity integration test: capturing at the top made a resumed run's
            # first post-resume step silently reuse the hx/cx from two calls before the save
            # point instead of the correct immediately-prior one.
            hx_cx_state.hx, hx_cx_state.cx, hx_cx_state.initialized = hx.detach(), cx.detach(), True

        def _maybe_stack(x):
            # `z` (the continuous policy's pre-tanh sample) is None for every step when the
            # action space is discrete -- sample_action() only returns it for continuous_action.
            return None if x[0] is None else torch.stack(x, dim=1)

        all_obs, act, rew, end, trunc, logits_act, val, z, all_hx, val_bootstrap = (
            _maybe_stack(x) for x in zip(*all_)
        )

        # all_hx: (num_envs, num_steps, *hx_shape) -- the exact per-step model state (hx, BEFORE
        # any dead-env reset_gate/burn-in touches it for the FOLLOWING step) that produced each
        # step's action. Added so a caller that needs to reproduce action-time state exactly at
        # loss-computation time (e.g. DrQActorCritic's frame stack across mid-rollout resets)
        # doesn't have to re-derive it from obs alone -- see models.drq_actor_critic's forward()
        # docstring. Positioned before `infos` (not appended after it) so existing callers using
        # `*_, [infos] = env_loop.send(...)` (coroutines.collector.make_collector) are
        # unaffected; callers unpacking every field individually (ActorCritic.forward) need one
        # extra placeholder added for it.
        num_steps = yield all_obs, act, rew, end, trunc, logits_act, val, val_bootstrap, z, all_hx, infos
