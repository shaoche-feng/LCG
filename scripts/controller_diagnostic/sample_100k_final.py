"""Evaluation-only: sample a batch of seeds on the final 100k-run checkpoint, identify
a genuine high-return and low-return episode, and save videos of both. No training.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import json

import torch

RUN_DIR = f"{REPO_ROOT}/outputs/k4-walker-100k/pmpo_diagnostic"
CKPT = f"{RUN_DIR}/agent_pmpo_update_100600_epoch.pt"
OUT_DIR = f"{REPO_ROOT}/outputs/sample_100k_final"
SEEDS = list(range(2001, 2021))  # a fresh batch, distinct from the earlier 1001-1030 set


def build_agent_cfg(cfg, device):
    from agent import get_action_space_kwargs
    from envs import make_dm_control_env
    from hydra.utils import instantiate
    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    return instantiate(cfg.agent, **action_kwargs)


def load_checkpoint(model, path, device):
    from utils import extract_state_dict
    payload = torch.load(path, map_location=device, weights_only=False)
    ac_sd = extract_state_dict(payload["trainer_state"]["agent"], "actor_critic")
    model.load_state_dict(ac_sd, strict=True)
    return payload["epoch"], payload["update"]


@torch.no_grad()
def run_episode(model, cfg, seed, device, save_video_path=None):
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
        total = 0.0
        length = 0
        for step in range(500):
            state = history.state
            action = model.sample_action(model.actor(state), deterministic=True)
            if frames is not None:
                frames.append(image.copy())
            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
            total += float(reward)
            length += 1
            if end or trunc:
                break
            history.advance(observation(image), torch.tensor([False], device=device))
        return dict(seed=seed, total_return=total, length=length)
    finally:
        if save_video_path and frames:
            import cv2
            h, w = frames[0].shape[:2]
            writer = cv2.VideoWriter(save_video_path, cv2.VideoWriter_fourcc(*"mp4v"), 30, (w, h))
            for f in frames:
                writer.write(cv2.cvtColor(f, cv2.COLOR_RGB2BGR))
            writer.release()
            print(f"    saved video: {save_video_path}", flush=True)
        env.close()


def main():
    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    os.makedirs(f"{OUT_DIR}/videos", exist_ok=True)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
        ])

    agent_cfg = build_agent_cfg(cfg, device)
    model = PMPOBeta(agent_cfg.actor_critic).to(device)
    model.eval()
    epoch, update = load_checkpoint(model, CKPT, device)
    print(f"Loaded checkpoint: epoch={epoch} update={update}", flush=True)

    results = []
    for seed in SEEDS:
        r = run_episode(model, cfg, seed, device)
        results.append(r)
        print(f"seed={seed} return={r['total_return']:.2f} length={r['length']}", flush=True)

    results.sort(key=lambda r: r["total_return"])
    with open(f"{OUT_DIR}/sweep.json", "w") as f:
        json.dump(dict(checkpoint=CKPT, epoch=epoch, update=update, results=results), f, indent=2)

    low, high = results[0], results[-1]
    print(f"\nLOW: seed={low['seed']} return={low['total_return']:.2f}")
    print(f"HIGH: seed={high['seed']} return={high['total_return']:.2f}")

    run_episode(model, cfg, low["seed"], device, save_video_path=f"{OUT_DIR}/videos/seed_{low['seed']}_low_{low['total_return']:.0f}.mp4")
    run_episode(model, cfg, high["seed"], device, save_video_path=f"{OUT_DIR}/videos/seed_{high['seed']}_high_{high['total_return']:.0f}.mp4")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
