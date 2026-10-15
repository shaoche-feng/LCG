from collections import defaultdict, namedtuple
from pathlib import Path
from typing import Any, Dict, List, Tuple

import torch
from torch import Tensor

from agent import Agent
from data import Dataset, Episode
from game.keymap import ActionNames, Keymap
from envs import WorldModelEnv
from coroutines.frame_history import FrameHistory


NamedEnv = namedtuple("NamedEnv", "name env")
OneStepData = namedtuple("OneStepData", "obs act rew end trunc")


class PlayEnv:
    def __init__(
        self,
        agent: Agent,
        envs: List[NamedEnv],
        action_names: ActionNames,
        keymap: Keymap,
        recording_mode: bool,
        store_denoising_trajectory: bool,
    ) -> None:
        self.agent = agent
        self.envs = envs
        self.action_names = action_names
        self.keymap = keymap
        self.recording_mode = recording_mode
        self.store_denoising_trajectory = store_denoising_trajectory
        self.env_id = 0
        self.env_name, self.env = self.envs[0]
        self.obs, self.t, self.return_, self.buffer, self.rec_dataset = (None,) * 5

    def print_controls(self) -> None:
        print("\nControls (policy viewer):\n")
        print("up/down: imagination horizon (+1/-1)")
        print("right/left: next/previous environment")

    def next_mode(self) -> bool:
        return False

    def next_axis_1(self) -> bool:
        self.update_wm_horizon(+1)
        return True

    def prev_axis_1(self) -> bool:
        self.update_wm_horizon(-1)
        return True

    def next_axis_2(self) -> bool:
        self.switch_env(self.env_id + 1)
        return True

    def prev_axis_2(self) -> bool:
        self.switch_env(self.env_id - 1)
        return True

    def is_wm_env(self) -> bool:
        return isinstance(self.env, WorldModelEnv)

    def print_env(self) -> None:
        print(f"> Environment: {self.env_name}")

    def switch_env(self, env_id: int) -> None:
        self.env_id = env_id % len(self.envs)
        self.env_name, self.env = self.envs[self.env_id]
        self.print_env()

    def update_wm_horizon(self, incr: int) -> None:
        if self.is_wm_env():
            self.env.horizon = max(1, self.env.horizon + incr)

    def reset_recording(self) -> None:
        self.buffer = defaultdict(list)
        self.buffer["info"] = defaultdict(list)
        directory = Path("dataset") / f"rec_{self.env_name}_π"
        self.rec_dataset = Dataset(directory)
        self.rec_dataset.load_from_default_path()

    def reset(self) -> Tuple[Tensor, None]:
        self.obs, env_info = self.env.reset()
        self.t, self.return_ = 0, 0
        # WorldModelEnv supplies its replay prompt; real resets repeat the first frame.
        self.history = FrameHistory(self.obs, self.agent.actor_critic.frame_stack, env_info.get("prompt"))
        if self.recording_mode:
            self.reset_recording()
        return self.obs, None

    @torch.no_grad()
    def step(self, act: int) -> Tuple[Tensor, Tensor, Tensor, Tensor, Dict[str, Any]]:
        model = self.agent.actor_critic
        logits_act, value = model.predict_act_value(self.history.state)
        act = model.sample_action(logits_act)
        _, entropy = model.log_prob_and_entropy(logits_act, act)
        entropy, value = f"{entropy.item():.2f}", f"{value.item():.2f}"
        current_obs = self.obs.clone()
        next_obs, rew, end, trunc, env_info = self.env.step(act)
        data = OneStepData(current_obs, act, rew, end, trunc)
        self.return_ += rew.item()
        control = "policy"
        header = [
            [
                f"Env     : {self.env_name}",
                f"Control : {control}",
                f"Timestep: {self.t + 1}",
                f"Horizon : {self.env.horizon}" if self.is_wm_env() else "",
            ],
            [
                f"Trunc : {bool(trunc)}",
                f"Done  : {bool(end)}",
                f"Reward: {rew.item():.2f}",
                f"Return: {self.return_:.2f}",
            ],
            [
                f"Action : {act[0].tolist()}",
                f"Entropy: {entropy}",
                f"Value  : {value}",
            ],
        ]
        info = {"header": header}

        if end or trunc:
            d = "Dead" if end else ("Horizon" if self.is_wm_env() else "Timed out")
            print(f"[{d}] return = {self.return_} - length = {self.t + 1}.")

        if self.recording_mode:
            for k, v in data._asdict().items():
                self.buffer[k].append(v)
            if self.store_denoising_trajectory and "denoising_trajectory" in env_info:
                self.buffer["info"]["denoising_trajectory"].append(env_info["denoising_trajectory"])
            if end or trunc:
                ep_dict = {k: torch.cat(v, dim=0) for k, v in self.buffer.items() if k != "info"}
                ep_info = {k: torch.cat(v, dim=0) for k, v in self.buffer["info"].items()}
                ep_info["final_observation"] = env_info["final_observation"][0].cpu()
                ep = Episode(**ep_dict, info=ep_info).to("cpu")
                self.rec_dataset.add_episode(ep)
                self.rec_dataset.save_to_default_path()

        self.history.advance(next_obs, torch.logical_or(end, trunc), env_info.get("prompt"))
        self.obs = next_obs
        self.t += 1

        return next_obs, rew, end, trunc, info
