"""Characterization-only: no training. Deterministic Beta-mean evaluation of three
saved full-loop checkpoints (early/middle/final) on the same 30 fixed seeds, tracking
per-step reward so cumulative-at-N and full distributions can be reported, plus
optional per-step controller-health tracking and video saving for representative seeds.
"""
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import json
import os

import numpy as np
import torch

RUN_DIR = f"{REPO_ROOT}/outputs/k4-walker-fullloop/pmpo_diagnostic"
OUT_DIR = f"{REPO_ROOT}/outputs/characterization"
CKPTS = {
    "early_e6": f"{RUN_DIR}/agent_epoch_00006_pmpo_update_007000_epoch.pt",
    "middle_e10": f"{RUN_DIR}/agent_epoch_00010_pmpo_update_008600_epoch.pt",
    "final_e15": f"{RUN_DIR}/agent_epoch_00015_pmpo_update_010600_epoch.pt",
}
SEEDS = list(range(1001, 1031))


def build_agent_cfg(cfg):
    from agent import get_action_space_kwargs
    from envs import make_dm_control_env
    from hydra.utils import instantiate
    device = torch.device("cuda:0")
    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    return instantiate(cfg.agent, **action_kwargs), device


def load_checkpoint(model, path, device):
    from utils import extract_state_dict
    payload = torch.load(path, map_location=device, weights_only=False)
    ac_sd = extract_state_dict(payload["trainer_state"]["agent"], "actor_critic")
    model.load_state_dict(ac_sd)
    return payload["epoch"], payload["update"]


@torch.no_grad()
def evaluate_seed(model, cfg, seed, device, save_video_path=None, track_health=False):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    frames = [] if save_video_path else None
    try:
        image, _ = env.reset(seed=int(seed))

        def observation(img):
            return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)

        history = FrameHistory(observation(image), model.frame_stack)
        rewards, health = [], ([] if track_health else None)
        for step in range(500):
            state = history.state
            raw = model.actor(state)
            action = model.sample_action(raw, deterministic=True)
            if track_health:
                dist = model.distribution(raw)
                _, entropy = model.log_prob_and_entropy(raw, action)
                health.append(dict(step=step, alpha_mean=float(dist.concentration1.mean()),
                                    beta_mean=float(dist.concentration0.mean()),
                                    alpha_max=float(dist.concentration1.max()), beta_max=float(dist.concentration0.max()),
                                    entropy=float(entropy.mean()),
                                    action=action[0].cpu().tolist()))
            if frames is not None:
                frames.append(image.copy())
            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
            rewards.append(float(reward))
            if end or trunc:
                break
            history.advance(observation(image), torch.tensor([False], device=device))
        return dict(seed=seed, total_return=sum(rewards), length=len(rewards), rewards=rewards, health=health)
    finally:
        if save_video_path and frames:
            import cv2
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
        env.close()


def main():
    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
        ])

    agent_cfg, device = build_agent_cfg(cfg)

    all_results = {}
    for tag, path in CKPTS.items():
        model = PMPOBeta(agent_cfg.actor_critic).to(device)
        model.eval()
        epoch, update = load_checkpoint(model, path, device)
        print(f"=== {tag}: epoch={epoch} update={update} path={path} ===", flush=True)
        results = []
        for seed in SEEDS:
            r = evaluate_seed(model, cfg, seed, device)
            results.append(r)
            print(f"  seed={seed} return={r['total_return']:.2f} length={r['length']}", flush=True)
        all_results[tag] = dict(epoch=epoch, update=update, path=path, results=results)
        with open(f"{OUT_DIR}/{tag}.json", "w") as f:
            json.dump(all_results[tag], f)

    with open(f"{OUT_DIR}/all_checkpoints_summary.json", "w") as f:
        json.dump({tag: {"epoch": v["epoch"], "update": v["update"],
                         "returns": {r["seed"]: r["total_return"] for r in v["results"]}}
                   for tag, v in all_results.items()}, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
