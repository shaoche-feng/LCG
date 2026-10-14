"""Multi-checkpoint advantage-magnitude diagnostic (Stage 4 pre-check).

Read-only: loads Stage 3 checkpoints, evaluates on cuda:0, never touches the active
Stage 3 training directory/checkpoints/wandb run. Runs entirely standalone.

For each checkpoint: collects genuine real Walker states (with exact physics snapshot
+ K=4 real conditioning window), samples 12 candidate actions from that checkpoint's
own Beta policy at each state, computes the imagined PMPO advantage A_world = R_lambda -
V(s) for each candidate using that checkpoint's own denoiser/reward-end model/value net
(same DiffusionSampler/RewEndModel classes and step logic as WorldModelEnv.step(), just
without the dataset-backed reset()/data_loader dependency), and separately measures the
REAL outcome of executing that candidate action from the exact same restored physics
state, followed by the deterministic policy. Reports correlation between imagined
advantage and real relative return, globally and within-state, split by sign of A.
"""
import argparse
import json
import os
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
import time

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, f"{REPO_ROOT}/src")

import numpy as np
import torch
from torch.distributions.categorical import Categorical

RUN_DIR = f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic"
OUT_DIR = f"{REPO_ROOT}/outputs/stage4_advantage_magnitude_diagnostic"

CHECKPOINTS = [
    ("early_epoch34", f"{RUN_DIR}/agent_pmpo_update_014600_epoch.pt"),
    ("mid_early_epoch55", f"{RUN_DIR}/agent_pmpo_update_026600_epoch.pt"),
    ("strong_learning_epoch80", f"{RUN_DIR}/agent_pmpo_update_036600_epoch.pt"),
    ("peak_epoch108", f"{RUN_DIR}/agent_pmpo_update_047800_epoch.pt"),
    ("late_epoch118", f"{RUN_DIR}/agent_pmpo_update_051800_epoch.pt"),
]

TRAJ_SEEDS = [4001, 4002, 4003, 4004]  # fresh seeds, unused elsewhere in this session
GAMMA = 0.985
LAMBDA_ = 0.95
HORIZON = 15
REAL_HORIZON = 50


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--n_states", type=int, default=32)
    p.add_argument("--n_candidates", type=int, default=12)
    p.add_argument("--n_m", type=int, default=4)
    p.add_argument("--checkpoints", type=str, default="all")
    return p.parse_args()


def build_agent(cfg, action_kwargs, device):
    from agent import Agent
    from hydra.utils import instantiate
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    agent.eval()
    return agent


def load_agent_checkpoint(agent, path, device):
    payload = torch.load(path, map_location=device, weights_only=False)
    agent_sd = payload["trainer_state"]["agent"]
    agent.load_state_dict(agent_sd, strict=True)
    return payload["epoch"], payload["update"]


def observation(img, device):
    return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)


class ImaginationStepper:
    """Mirrors WorldModelEnv.step()'s exact logic (models/diffusion DiffusionSampler +
    RewEndModel.predict_rew_end), without the dataset-backed reset()/reset_dead() path --
    we always supply the initial obs_buffer/act_buffer/hx/cx ourselves from a real state."""

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


def collect_trajectory_states(agent, cfg, device, seed, n_states_per_seed):
    """Run the checkpoint's own deterministic policy for one real episode, recording,
    at evenly spaced steps (t >= 3 so a full K=4 real history exists), the physics
    snapshot, the last-4 real frames, and the last-3 real actions."""
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    img, _ = env.reset(seed=int(seed))
    physics = env._dm_env.physics  # captured AFTER reset

    frame_list = [observation(img, device)]
    act_list = []  # actions taken, aligned: act_list[i] led from frame_list[i] to frame_list[i+1]
    physics_states = [physics.get_state().copy()]

    max_steps = 480
    records = []
    total_return = 0.0
    model = agent.actor_critic

    for t in range(max_steps):
        if len(frame_list) < 4:
            policy_obs = torch.cat([frame_list[0]] * (4 - len(frame_list)) + frame_list, dim=1)
        else:
            policy_obs = torch.cat(frame_list[-4:], dim=1)
        raw = model.actor(policy_obs)
        act = model.sample_action(raw, deterministic=True)
        act_np = act[0].cpu().numpy()

        img, rew, end, trunc, info = env.step(act_np)
        total_return += float(rew)
        new_frame = observation(img, device)
        new_state = physics.get_state().copy()

        frame_list.append(new_frame)
        act_list.append(act[0].clone())
        physics_states.append(new_state)
        if len(frame_list) > 5:
            frame_list.pop(0); act_list.pop(0); physics_states.pop(0)

        # record a usable conditioning window once we have 4 real frames + 3 real actions
        if len(frame_list) >= 4 and (t + 1) % max(1, max_steps // n_states_per_seed) == 0:
            records.append(dict(
                seed=seed, step=t + 1,
                frames=[f.clone() for f in frame_list[-4:]],
                actions=[a.clone() for a in act_list[-3:]],
                physics_state=physics_states[-1].copy(),  # state matching frame_list[-1], the newest/current conditioning frame
            ))
        if end or trunc:
            break
    env.close()
    for r in records:
        r["episode_total_return"] = total_return
    return records, env


def build_conditioning(record, device):
    obs_buffer0 = torch.stack(record["frames"], dim=1)  # (1,4,C,H,W)
    act_dim = record["actions"][0].shape[-1]
    placeholder = torch.zeros(1, act_dim, device=device)
    act_buffer0 = torch.stack(record["actions"] + [placeholder[0]], dim=0).unsqueeze(0)  # (1,4,act_dim)
    return obs_buffer0, act_buffer0


@torch.no_grad()
def burn_in_rew_end(agent, obs_buffer0, act_buffer0):
    rew_end_model = agent.rew_end_model
    _, _, (hx, cx) = rew_end_model.predict_rew_end(obs_buffer0[:, :-1], act_buffer0[:, :-1], obs_buffer0[:, 1:])
    return hx, cx


@torch.no_grad()
def imagine_advantage(agent, sampler, stepper, obs_buffer0, act_buffer0, hx0, cx0, candidate_action, v_s, n_m, base_seed, device):
    model = agent.actor_critic
    targets_list = []
    for m in range(n_m):
        torch.manual_seed(base_seed * 1000 + m)
        stepper.set_initial(obs_buffer0, act_buffer0, hx0, cx0)
        frame_list = [obs_buffer0[:, i] for i in range(obs_buffer0.shape[1])]

        def policy_state():
            return torch.cat(frame_list[-4:], dim=1)

        rews, ends, truncs, vnexts = [], [], [], []
        act = candidate_action
        dead = False
        for t in range(HORIZON):
            if dead:
                rews.append(torch.zeros(1, device=device))
                ends.append(torch.ones(1, device=device))
                truncs.append(torch.zeros(1, device=device))
                vnexts.append(torch.zeros(1, device=device))
                continue
            next_obs, rew, end, trunc = stepper.step(act)
            rews.append(rew); ends.append(end); truncs.append(trunc)
            frame_list.append(next_obs)
            if len(frame_list) > 4:
                frame_list.pop(0)
            v_next = model.value(policy_state()).squeeze(-1)
            vnexts.append(v_next)
            if bool(end.item()):
                dead = True
            elif t < HORIZON - 1:
                raw = model.actor(policy_state())
                act = model.sample_action(raw, deterministic=False)
        rew_t = torch.stack(rews, dim=1)
        end_t = torch.stack(ends, dim=1)
        trunc_t = torch.stack(truncs, dim=1)
        vboot_t = torch.stack(vnexts, dim=1)
        from models.actor_critic import compute_lambda_returns
        targets = compute_lambda_returns(rew_t, end_t, trunc_t, vboot_t, GAMMA, LAMBDA_, continuous_reward=True)
        targets_list.append(targets[0, 0].item())
    mean_target = float(np.mean(targets_list))
    return mean_target - v_s, targets_list


def real_rollout_from_candidate(cfg, physics_state, candidate_action_np, model, device, horizon=REAL_HORIZON):
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=0)
    env.reset(seed=0)
    physics = env._dm_env.physics
    physics.set_state(physics_state)
    physics.forward()

    total = 0.0
    returns_at = {}
    action = candidate_action_np
    frame_list = None
    for t in range(horizon):
        img, rew, end, trunc, info = env.step(action)
        total += float(rew)
        if (t + 1) == 15:
            returns_at[15] = total
        if (t + 1) == 50:
            returns_at[50] = total
        if end or trunc:
            break
        new_frame = observation(img, device)
        if frame_list is None:
            frame_list = [new_frame] * 4
        else:
            frame_list = frame_list[1:] + [new_frame]
        policy_obs = torch.cat(frame_list, dim=1)
        raw = model.actor(policy_obs)
        action = model.sample_action(raw, deterministic=True)[0].cpu().numpy()
    env.close()
    if 15 not in returns_at:
        returns_at[15] = total
    if 50 not in returns_at:
        returns_at[50] = total
    return returns_at


def main():
    args = parse_args()
    from hydra import compose, initialize_config_dir
    from omegaconf import OmegaConf
    from agent import get_action_space_kwargs
    from envs import make_dm_control_env
    from models.diffusion import DiffusionSampler

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
            "agent.actor_critic.entropy_coef=0.01",
        ])

    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    sampler_cfg = cfg.world_model_env.diffusion_sampler

    checkpoints = CHECKPOINTS
    if args.checkpoints != "all":
        names = set(args.checkpoints.split(","))
        checkpoints = [c for c in CHECKPOINTS if c[0] in names]
    if args.smoke:
        checkpoints = checkpoints[:1]

    n_states_total = 2 if args.smoke else args.n_states
    n_states_per_seed = max(1, n_states_total // len(TRAJ_SEEDS))
    n_candidates = 3 if args.smoke else args.n_candidates
    n_m = 2 if args.smoke else args.n_m

    all_checkpoint_records = {}

    for ckpt_name, ckpt_path in checkpoints:
        t_ckpt_start = time.time()
        agent = build_agent(cfg, action_kwargs, device)
        epoch, update = load_agent_checkpoint(agent, ckpt_path, device)
        print(f"\n=== {ckpt_name}: epoch={epoch} update={update} path={ckpt_path} ===", flush=True)
        from hydra.utils import instantiate
        sampler = DiffusionSampler(agent.denoiser, instantiate(sampler_cfg))
        stepper = ImaginationStepper(sampler, agent.rew_end_model, HORIZON)
        model = agent.actor_critic

        # 1. collect real trajectory states
        state_records = []
        for seed in TRAJ_SEEDS:
            recs, _ = collect_trajectory_states(agent, cfg, device, seed, n_states_per_seed)
            state_records.extend(recs)
            if len(state_records) >= n_states_total:
                break
        state_records = state_records[:n_states_total]
        print(f"  collected {len(state_records)} states across {len(TRAJ_SEEDS)} seeds", flush=True)

        candidate_records = []
        for si, rec in enumerate(state_records):
            obs_buffer0, act_buffer0 = build_conditioning(rec, device)
            hx0, cx0 = burn_in_rew_end(agent, obs_buffer0, act_buffer0)
            policy_state = torch.cat(rec["frames"], dim=1)
            v_s = model.value(policy_state).squeeze(-1).item()

            candidates = []
            for c in range(n_candidates):
                raw = model.actor(policy_state)
                cand_act = model.sample_action(raw, deterministic=False)
                candidates.append(cand_act)

            for ci, cand_act in enumerate(candidates):
                base_seed = si * 10007 + ci * 131  # deterministic per (state, candidate), shared across checkpoints' M-index pattern
                adv_mean, adv_samples = imagine_advantage(
                    agent, sampler, stepper, obs_buffer0, act_buffer0, hx0, cx0, cand_act, v_s, n_m, base_seed, device)
                real_returns = real_rollout_from_candidate(cfg, rec["physics_state"], cand_act[0].cpu().numpy(), model, device)
                candidate_records.append(dict(
                    state_idx=si, seed=rec["seed"], step=rec["step"], episode_total_return=rec["episode_total_return"],
                    candidate_idx=ci, candidate_action=cand_act[0].cpu().numpy().tolist(),
                    v_s=v_s, imagined_advantage_samples=adv_samples, imagined_advantage_mean=adv_mean,
                    real_h15=real_returns[15], real_h50=real_returns[50],
                ))
            print(f"  state {si+1}/{len(state_records)} (seed={rec['seed']} step={rec['step']}) done, "
                  f"v_s={v_s:.2f} adv_range=[{min(r['imagined_advantage_mean'] for r in candidate_records[-n_candidates:]):.2f},"
                  f"{max(r['imagined_advantage_mean'] for r in candidate_records[-n_candidates:]):.2f}]", flush=True)

        all_checkpoint_records[ckpt_name] = dict(epoch=epoch, update=update, path=ckpt_path,
                                                  n_states=len(state_records), candidates=candidate_records)
        with open(f"{OUT_DIR}/raw_records.json", "w") as f:
            json.dump(all_checkpoint_records, f)
        print(f"  checkpoint done in {time.time()-t_ckpt_start:.1f}s, {len(candidate_records)} candidate records", flush=True)
        del agent, sampler, stepper
        torch.cuda.empty_cache()

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
