"""World-model replay diagnostic: for each of the 4 real trajectories already recorded
(instrumented_results.json), pick several real starting points, build the exact real
K=4 conditioning window (frames + actions) at that point, then FORCE the same real
action sequence the policy actually took over the next `horizon` steps through the
checkpoint's own world model (denoiser + reward/end model) -- no new candidate actions,
no policy re-sampling. Compares predicted per-step reward against the real recorded
reward over that same window, and predicted-vs-real next-frame pixel MSE as a motion
fidelity proxy. Read-only, no training-state modification.
"""
import json
import os
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, f"{REPO_ROOT}/src")

import numpy as np
import torch
from torch.distributions.categorical import Categorical

CONFIGS = [
    dict(name="stage3", entropy_coef=0.01,
         checkpoint=f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt",
         seeds=[3013, 3030]),
    dict(name="stage3b", entropy_coef=0.02,
         checkpoint=f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3b/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt",
         seeds=[3013, 3011]),
]
HORIZON = 15
START_STEPS = [10, 60, 110, 160, 210, 260, 310, 360, 410, 460]  # 10 windows per trajectory, avoiding the very first few steps (need 3 prior real actions)
OUT_DIR = f"{REPO_ROOT}/outputs/diagnose_world_model_replay"


class ImaginationStepper:
    """Mirrors WorldModelEnv.step()'s exact logic, without the dataset-backed reset."""

    def __init__(self, sampler, rew_end_model, horizon):
        self.sampler = sampler
        self.rew_end_model = rew_end_model
        self.horizon = horizon

    def set_initial(self, obs_buffer, act_buffer, hx, cx):
        self.obs_buffer = obs_buffer.clone()
        self.act_buffer = act_buffer.clone()
        self.hx_rew_end = hx.clone()
        self.cx_rew_end = cx.clone()
        self.ep_len = torch.zeros(1, dtype=torch.long, device=obs_buffer.device)

    @torch.no_grad()
    def step(self, act):
        self.act_buffer[:, -1] = act
        next_obs, _ = self.sampler.sample(self.obs_buffer, self.act_buffer)
        rew_out, logits_end, (self.hx_rew_end, self.cx_rew_end) = self.rew_end_model.predict_rew_end(
            self.obs_buffer[:, -1:], self.act_buffer[:, -1:], next_obs.unsqueeze(1),
            (self.hx_rew_end, self.cx_rew_end))
        if self.rew_end_model.continuous_reward:
            rew = rew_out.squeeze(1)
        else:
            rew = Categorical(logits=rew_out).sample().squeeze(1) - 1.0
        end = Categorical(logits=logits_end).sample().squeeze(1)
        self.ep_len += 1
        trunc = (self.ep_len >= self.horizon).long()
        self.obs_buffer = self.obs_buffer.roll(-1, dims=1)
        self.act_buffer = self.act_buffer.roll(-1, dims=1)
        self.obs_buffer[:, -1] = next_obs
        return next_obs, rew, end.float(), trunc.float()


def build_agent(cfg, device):
    from agent import Agent, get_action_space_kwargs
    from envs import make_dm_control_env
    from hydra.utils import instantiate
    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.eval()
    return agent


def load_agent_checkpoint(agent, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    agent.load_state_dict(payload["trainer_state"]["agent"], strict=True)


def observation(img, device):
    return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)


@torch.no_grad()
def main():
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    with open(f"{REPO_ROOT}/outputs/diagnose_reward_decomposition/instrumented_results.json") as f:
        real_data = json.load(f)

    all_results = {}
    for cfg_spec in CONFIGS:
        with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
            cfg = compose(config_name="trainer", overrides=[
                "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
                "agent.actor_critic.prior_refresh_interval=500",
                f"agent.actor_critic.entropy_coef={cfg_spec['entropy_coef']}",
            ])
        agent = build_agent(cfg, device)
        load_agent_checkpoint(agent, cfg_spec["checkpoint"], device)
        from models.diffusion import DiffusionSampler
        from hydra.utils import instantiate as hydra_instantiate
        sampler = DiffusionSampler(agent.denoiser, hydra_instantiate(cfg.world_model_env.diffusion_sampler))
        stepper = ImaginationStepper(sampler, agent.rew_end_model, HORIZON)

        for seed in cfg_spec["seeds"]:
            key = f"{cfg_spec['name']}_seed{seed}"
            real_steps = real_data[key]["steps"]
            real_actions = [torch.tensor(s["action"], device=device, dtype=torch.float32).unsqueeze(0) for s in real_steps]
            real_rewards = [s["step_reward"] for s in real_steps]

            # regenerate the real frames by replaying the SAME env/seed/actions once more
            from envs.dm_control_env import DMControlEnv
            kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
            env = DMControlEnv(**kwargs, seed=int(seed))
            image, _ = env.reset(seed=int(seed))
            real_frames = [observation(image, device)]
            for s in real_steps:
                action_np = np.asarray(s["action"], dtype=np.float32)
                img, rew, end, trunc, _ = env.step(action_np)
                real_frames.append(observation(img, device))
                if end or trunc:
                    break
            env.close()

            window_results = []
            for start in START_STEPS:
                if start + HORIZON >= len(real_frames) - 1 or start < 3:
                    continue
                obs_buffer0 = torch.stack(real_frames[start - 3:start + 1], dim=1)  # (1,4,C,H,W)
                act_hist = real_actions[start - 3:start]  # 3 real actions leading up to 'start', each (1, act_dim)
                placeholder = torch.zeros_like(real_actions[0])
                act_buffer0 = torch.stack(act_hist + [placeholder], dim=1)  # (1, 4, act_dim)

                # burn in rew_end LSTM using the real 3-transition window
                hx, cx = agent.rew_end_model.predict_rew_end(obs_buffer0[:, :-1], act_buffer0[:, :-1], obs_buffer0[:, 1:])[2]

                stepper.set_initial(obs_buffer0, act_buffer0, hx, cx)
                predicted_rewards = []
                predicted_frames = []
                for h in range(HORIZON):
                    forced_action = real_actions[start + h]
                    next_obs, rew, end, trunc = stepper.step(forced_action)
                    predicted_rewards.append(float(rew.item()))
                    predicted_frames.append(next_obs)

                real_window_rewards = real_rewards[start:start + HORIZON]
                # pixel MSE between predicted and real next-frames (both normalized [-1,1])
                pixel_mse = []
                for h in range(HORIZON):
                    real_f = real_frames[start + 1 + h]
                    pred_f = predicted_frames[h]
                    mse = float(((real_f - pred_f) ** 2).mean().item())
                    pixel_mse.append(mse)

                window_results.append(dict(
                    start=start, predicted_rewards=predicted_rewards, real_rewards=real_window_rewards,
                    pixel_mse=pixel_mse,
                    predicted_reward_sum=sum(predicted_rewards), real_reward_sum=sum(real_window_rewards),
                ))
                print(f"{key} start={start}: predicted_sum={sum(predicted_rewards):.2f} real_sum={sum(real_window_rewards):.2f} "
                      f"mean_pixel_mse={np.mean(pixel_mse):.4f}", flush=True)

            all_results[key] = window_results

        del agent, sampler, stepper
        torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/replay_results.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
