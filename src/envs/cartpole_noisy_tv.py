"""Cartpole noisy TV: a random RGB panel in the rendered observation while the cart is inside a fixed
position interval (docs/implement/AlEATRORISTIC_EXPERIMENT_DESIGN.md).

The wrapper runs where DMControlEnv runs (inside each vector-env worker), after action repeat and rendering:
every observation it emits is final. Policy input, replay, frame history, final observations, denoiser
training, feature extraction and videos all read that one emitted array; nothing re-renders or resamples it.

- Trigger: I_t = 1 if x_L <= x_t <= x_R, x_t the cart (slider) position of the physics state o_t was rendered
  from. Reset observations follow the same rule.
- Active (enabled and I_t = 1): the panel's size x size x 3 values are drawn independently and uniformly from
  0..255 by the wrapper's own numpy Generator, one fresh texture per emitted observation. Inactive: the
  rendering is returned unchanged (Clean mode never draws).
- TV randomness: a private Generator, never the global numpy/torch/python RNGs or the physics' RandomState.
  A reset with an explicit seed s restarts it from SeedSequence([tv_seed, s]) (so evaluation episodes of the
  same reset seed get the same draws in every run of a condition, and TV16/TV24 share their stream); a reset
  without a seed (vector-env autoreset) continues it.
- Diagnostics (analysis only, never policy input): with record_physical_state, reset/step infos also carry
  tv_in_region, tv_active and tv_panel_size (0 when inactive) beside qpos/qvel (cart position/velocity and pole
  angle/angular velocity are qpos/qvel[0], [1]). Clean mode records region membership with tv_active False.
- Checkpoints: collection_state holds the wrapped env's state, the Generator state, the last emitted
  observation and its diagnostics, so a restored env continues exactly (the emitted observation is not
  regenerated).
"""
from __future__ import annotations

import copy
from typing import Any, Dict, Mapping, Optional, Tuple

import gymnasium
import numpy as np

PANEL_LOCATIONS = ("top_left", "top_right")
TV_STATE_KEYS = ("tv_in_region", "tv_active", "tv_panel_size")
CONFIG_KEYS = ("enabled", "size", "position_interval", "panel_location", "margin", "seed")


def check_noisy_tv_config(noisy_tv: Optional[Mapping[str, Any]], domain_name: str, image_size: int
                          ) -> Optional[Dict[str, Any]]:
    """The validated settings (plain dict), or None when the env is built without the wrapper. Any noisy-TV
    section, Clean (enabled false) included, needs Cartpole: the trigger reads the cart position."""
    if noisy_tv is None:
        return None
    cfg = dict(noisy_tv)
    missing, unknown = set(CONFIG_KEYS) - set(cfg), set(cfg) - set(CONFIG_KEYS)
    if missing or unknown:
        raise ValueError(f"noisy_tv needs exactly the keys {list(CONFIG_KEYS)}; missing {sorted(missing)}, "
                         f"unknown {sorted(unknown)}")
    if domain_name != "cartpole":
        raise ValueError(f"The noisy TV is implemented for cartpole only, not {domain_name!r}")
    if not isinstance(cfg["enabled"], bool):
        raise ValueError(f"noisy_tv.enabled must be true or false, got {cfg['enabled']!r}")
    size, margin, seed = cfg["size"], cfg["margin"], cfg["seed"]
    if not (isinstance(size, int) and size > 0) or not (isinstance(margin, int) and margin >= 0):
        raise ValueError(f"noisy_tv.size must be a positive and margin a non-negative integer, got {size!r}, {margin!r}")
    if size + margin > image_size:
        raise ValueError(f"noisy_tv panel {size} + margin {margin} does not fit a {image_size}-pixel image")
    if not (isinstance(seed, int) and seed >= 0):
        raise ValueError(f"noisy_tv.seed must be a non-negative integer, got {seed!r}")
    interval = list(cfg["position_interval"])
    if len(interval) != 2 or not all(isinstance(v, (int, float)) for v in interval) or not interval[0] < interval[1]:
        raise ValueError(f"noisy_tv.position_interval must be [x_L, x_R] with x_L < x_R, got {cfg['position_interval']!r}")
    if cfg["panel_location"] not in PANEL_LOCATIONS:
        raise ValueError(f"noisy_tv.panel_location must be one of {PANEL_LOCATIONS}, got {cfg['panel_location']!r}")
    cfg["position_interval"] = [float(interval[0]), float(interval[1])]
    return cfg


def panel_box(location: str, size: int, margin: int, image_size: int) -> Tuple[slice, slice]:
    """(rows, cols) of the panel in an image_size x image_size observation, anchored to the corner."""
    rows = slice(margin, margin + size)
    cols = slice(margin, margin + size) if location == "top_left" else slice(image_size - margin - size, image_size - margin)
    return rows, cols


def in_region(cart_position: float, interval) -> bool:
    return bool(interval[0] <= cart_position <= interval[1])


class CartpoleNoisyTV(gymnasium.Wrapper):
    """Noisy-TV observations around one DMControlEnv (cartpole). Physics, actions, rewards, task labels,
    termination, observation shape and dtype are the wrapped env's."""

    def __init__(self, env, noisy_tv: Mapping[str, Any]) -> None:
        super().__init__(env)
        self.tv = check_noisy_tv_config(noisy_tv, env.domain_name, env.size)
        self.rows, self.cols = panel_box(self.tv["panel_location"], self.tv["size"], self.tv["margin"], env.size)
        self.rng = np.random.default_rng(np.random.SeedSequence([self.tv["seed"]]))
        self.draws = 0  # textures drawn so far (diagnostics and tests)
        self.last_observation: Optional[np.ndarray] = None
        self.last_tv_state: Optional[Dict[str, Any]] = None

    # ------------------------------------------------------------------ emission
    def cart_position(self) -> float:
        return float(self.env._dm_env.physics.named.data.qpos["slider"][0])

    def emit(self, obs: np.ndarray) -> Tuple[np.ndarray, Dict[str, Any]]:
        """The observation of the current physics state, drawn once; cached for every later read."""
        region = in_region(self.cart_position(), self.tv["position_interval"])
        active = self.tv["enabled"] and region
        obs = np.array(obs, dtype=np.uint8, copy=True)  # rendered arrays may be views with negative strides
        if active:
            size = self.tv["size"]
            obs[self.rows, self.cols] = self.rng.integers(0, 256, size=(size, size, 3), dtype=np.uint8)
            self.draws += 1
        state = dict(tv_in_region=region, tv_active=active, tv_panel_size=self.tv["size"] if active else 0)
        self.last_observation, self.last_tv_state = obs.copy(), dict(state)
        return obs, state

    def reset(self, *, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None):
        obs, info = self.env.reset(seed=seed, options=options)
        if seed is not None:
            self.rng = np.random.default_rng(np.random.SeedSequence([self.tv["seed"], int(seed)]))
        obs, state = self.emit(obs)
        if self.env.record_physical_state:
            info.update(state)
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        obs, state = self.emit(obs)
        if self.env.record_physical_state:
            info.update(state)
        return obs, reward, terminated, truncated, info

    # ------------------------------------------------------------------ reads (no draws)
    @property
    def physical_state(self) -> Dict[str, Any]:
        """qpos/qvel and the TV diagnostics of the current (last emitted) observation."""
        if self.last_tv_state is None:
            raise RuntimeError("No observation emitted yet")
        return {**self.env.physical_state, **self.last_tv_state}

    @property
    def collection_state(self) -> Dict[str, Any]:
        return dict(env=self.env.collection_state, noisy_tv=dict(
            config=dict(self.tv), rng=copy.deepcopy(self.rng.bit_generator.state), draws=self.draws,
            last_observation=None if self.last_observation is None else self.last_observation.copy(),
            last_tv_state=None if self.last_tv_state is None else dict(self.last_tv_state)))

    @collection_state.setter
    def collection_state(self, state: Dict[str, Any]) -> None:
        saved = state.get("noisy_tv") if isinstance(state, dict) else None
        if saved is None:
            raise ValueError("Saved collection state has no noisy-TV state; it was saved without the wrapper")
        if saved["config"] != self.tv:
            raise ValueError(f"Saved noisy-TV settings {saved['config']} differ from this env's {self.tv}")
        self.env.collection_state = state["env"]
        self.rng.bit_generator.state = copy.deepcopy(saved["rng"])
        self.draws = int(saved["draws"])
        self.last_observation = None if saved["last_observation"] is None else saved["last_observation"].copy()
        self.last_tv_state = None if saved["last_tv_state"] is None else dict(saved["last_tv_state"])

    # Attributes the vector env and TorchEnv read from the worker's env.
    @property
    def record_physical_state(self) -> bool:
        return self.env.record_physical_state

    @property
    def task_names(self):
        return self.env.task_names
