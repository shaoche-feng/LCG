"""Diagnostic adapter: use official TD-MPC2 pretrained checkpoints as pure
action generators for DM-Control locomotion tasks (walker-walk, walker-run,
quadruped-walk, quadruped-run).

Isolated from LCG/DIAMOND production code on purpose (per project instructions):
this module is only imported by scripts under scripts/lcg_diagnostic/, never by
src/. It vendors the minimal slice of nicklashansen/tdmpc2 (MIT license) needed
for inference -- tdmpc2.py + common/{world_model,layers,math,scale,init}.py --
under vendor/tdmpc2/, unmodified except for import paths.

Design (see docs discussion): the pretrained policy only ever sees dm_control's
native *state* observation (flattened observation_spec dict, exactly as
TD-MPC2's own envs/dmcontrol.py constructs it) and only ever produces a native
-bounded continuous action. It never sees our RGB frames. Reconstructing that
state observation from an existing DIAMOND DMControlEnv instance (or a raw
dm_control env) is done via env.task.get_observation(env.physics), which is
verified to exactly match a fresh timestep's .observation dict -- no changes to
src/envs/dm_control_env.py are needed or made.

Usage:
    collector = load_pretrained_collector(source="tdmpc2", domain="walker", task="walk")
    collector.reset_episode()  # call once per episode, before the first act()
    action = collector.act(state_observation, deterministic=True)  # native bounds
"""
from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch

_THIS_DIR = Path(__file__).resolve().parent
_VENDOR_DIR = _THIS_DIR / "vendor" / "tdmpc2"
_CHECKPOINT_DIR = _THIS_DIR / "checkpoints" / "tdmpc2"

if str(_VENDOR_DIR) not in sys.path:
    sys.path.insert(0, str(_VENDOR_DIR))

from tdmpc2 import TDMPC2  # noqa: E402 -- vendored module, path injected above


@dataclass
class _TDMPC2Config:
    """Plain attribute bag matching every `cfg.*` access in the vendored code
    (enumerated by grepping tdmpc2.py + common/*.py -- see adjacent notes.md).
    Values are the upstream defaults from tdmpc2/config.yaml, with model_size=5
    overrides applied (all released single-task DMControl checkpoints use
    model_size=5 per the official model zoo docs)."""

    # architecture (model_size=5)
    enc_dim: int = 256
    mlp_dim: int = 512
    latent_dim: int = 512
    num_enc_layers: int = 2
    num_channels: int = 32
    num_q: int = 5
    dropout: float = 0.01
    simnorm_dim: int = 8
    task_dim: int = 0

    # planning (latent-space MPC/CEM -- used at inference since mpc=True)
    mpc: bool = True
    iterations: int = 6
    num_samples: int = 512
    num_elites: int = 64
    num_pi_trajs: int = 24
    horizon: int = 3
    min_std: float = 0.05
    max_std: float = 2.0
    temperature: float = 0.5

    # actor / critic
    log_std_min: float = -10.0
    log_std_max: float = 2.0
    entropy_coef: float = 1e-4
    num_bins: int = 101
    vmin: float = -10.0
    vmax: float = 10.0

    # optimizer/loss hyperparameters -- unused at inference, but TDMPC2.__init__
    # constructs Adam optimizers unconditionally, so these must exist.
    lr: float = 3e-4
    enc_lr_scale: float = 0.3
    grad_clip_norm: float = 20.0
    tau: float = 0.01
    discount_denom: float = 5.0
    discount_min: float = 0.95
    discount_max: float = 0.995
    reward_coef: float = 0.1
    value_coef: float = 0.1
    termination_coef: float = 1.0
    consistency_coef: float = 20.0
    rho: float = 0.5

    # task/env -- set per collector instance in TDMPC2Collector.__init__
    obs: str = "state"
    episodic: bool = False
    multitask: bool = False
    compile: bool = False  # avoid requiring Triton (unavailable on Windows here)
    obs_shape: Dict[str, tuple] = field(default_factory=dict)
    action_dim: int = 0
    episode_length: int = 500
    tasks: List[str] = field(default_factory=list)
    bin_size: float = field(init=False, default=0.0)

    def __post_init__(self) -> None:
        self.bin_size = (self.vmax - self.vmin) / (self.num_bins - 1)


class TDMPC2Collector:
    """Wraps one loaded TD-MPC2 single-task checkpoint. Action-generator only:
    holds no reference to any DIAMOND/LCG object, and never touches RGB."""

    def __init__(self, domain: str, task: str, checkpoint_path: Path) -> None:
        if not torch.cuda.is_available():
            # TD-MPC2's own TDMPC2.__init__ hardcodes device=torch.device('cuda:0')
            # (see vendor/tdmpc2/tdmpc2.py) -- matches upstream evaluate.py's own
            # `assert torch.cuda.is_available()`. No CPU fallback exists upstream.
            raise RuntimeError("TD-MPC2 (vendored) requires a CUDA device; none is available.")

        from dm_control import suite  # local import: keep dm_control optional at module import time

        probe_env = suite.load(domain, task)
        action_spec = probe_env.action_spec()
        obs_spec = probe_env.observation_spec()

        self.domain = domain
        self.task = task
        self.action_dim = int(np.prod(action_spec.shape))
        self._obs_keys = list(obs_spec.keys())  # stable insertion order (OrderedDict)
        obs_dim = int(sum(int(np.prod(v.shape)) if v.shape else 1 for v in obs_spec.values()))

        # Native (unwrapped) action bounds -- what our DMControlEnv expects directly.
        self._native_min = np.asarray(action_spec.minimum, dtype=np.float64)
        self._native_max = np.asarray(action_spec.maximum, dtype=np.float64)
        self._native_dtype = action_spec.dtype

        cfg = _TDMPC2Config(
            obs_shape={"state": (obs_dim,)},
            action_dim=self.action_dim,
            episode_length=500,  # matches upstream envs/dmcontrol.py's Timeout(max_episode_steps=500)
            tasks=[f"{domain}-{task}"],
        )
        self.agent = TDMPC2(cfg)
        self.agent.load(str(checkpoint_path))
        self.agent.model.eval()
        self._t = 0

    def reset_episode(self) -> None:
        """Call once at the start of every episode, before the first act() call.
        Required for correct MPC warm-starting (TD-MPC2's `t0` flag)."""
        self._t = 0

    def _flatten_obs(self, state_observation: Dict[str, np.ndarray]) -> torch.Tensor:
        """Replicates upstream envs/dmcontrol.py's DMControlWrapper._obs_to_array:
        concatenate observation_spec values, in dict-insertion order, as float32."""
        parts = [np.atleast_1d(state_observation[k]).astype(np.float32).ravel() for k in self._obs_keys]
        return torch.from_numpy(np.concatenate(parts))

    def _rescale_action(self, action_normalized: np.ndarray) -> np.ndarray:
        """Replicates dm_control.suite.wrappers.action_scale.Wrapper(minimum=-1,
        maximum=1)'s exact transform (verified against its source): the policy
        was trained against an env whose action space was linearly rescaled to
        [-1, 1] per-dimension, so its raw output must be mapped back to this
        task's *native* per-dimension bounds before use -- this is NOT a no-op
        for quadruped, whose native bounds are asymmetric per-dimension
        (e.g. [-0.8, 0.8] / [-1, 1.1]), unlike walker's uniform [-1, 1]."""
        scale = (self._native_max - self._native_min) / 2.0
        native = self._native_min + scale * (action_normalized.astype(np.float64) + 1.0)
        native = np.clip(native, self._native_min, self._native_max)
        return native.astype(self._native_dtype)

    @torch.no_grad()
    def act(self, state_observation: Dict[str, np.ndarray], deterministic: bool = True) -> np.ndarray:
        """state_observation: the raw dm_control observation dict for this task
        (e.g. from env.task.get_observation(env.physics), or a fresh
        dm_env.reset()/step().observation). Returns a native-bounds action
        array, dtype matching this task's action_spec (ready to feed directly
        into DMControlEnv.step())."""
        obs_t = self._flatten_obs(state_observation)
        action_t = self.agent.act(obs_t, t0=(self._t == 0), eval_mode=deterministic)
        self._t += 1
        return self._rescale_action(action_t.numpy())


_SOURCES = {"tdmpc2": TDMPC2Collector}


def load_pretrained_collector(source: str, domain: str, task: str, seed: int = 1) -> TDMPC2Collector:
    """Load an official pretrained collector policy.

    source: currently only "tdmpc2" is implemented.
    domain, task: dm_control.suite naming, e.g. domain="walker", task="walk".
    seed: which of TD-MPC2's 3 released training seeds to use (1, 2, or 3).
    """
    if source not in _SOURCES:
        raise NotImplementedError(f"Unsupported pretrained-collector source: {source!r}")
    checkpoint_path = _CHECKPOINT_DIR / f"{domain}-{task}-{seed}.pt"
    if not checkpoint_path.exists():
        raise FileNotFoundError(
            f"Checkpoint not found: {checkpoint_path}. Expected one of the 4 official "
            f"TD-MPC2 dmcontrol checkpoints to already be downloaded into {_CHECKPOINT_DIR}."
        )
    return _SOURCES[source](domain=domain, task=task, checkpoint_path=checkpoint_path)
