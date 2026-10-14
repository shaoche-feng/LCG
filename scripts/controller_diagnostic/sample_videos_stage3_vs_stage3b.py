"""Capture sample episode videos (low-return and high-return seed) for Stage 3's and
Stage 3B's final checkpoints. Read-only, deterministic policy, reuses already-computed
30-seed results to pick representative seeds -- no new sweep needed.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import torch

CONFIGS = [
    dict(name="stage3", entropy_coef=0.01,
         checkpoint=f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt",
         low_seed=3030, low_return=224.3, high_seed=3013, high_return=634.0),
    dict(name="stage3b", entropy_coef=0.02,
         checkpoint=f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3b/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt",
         low_seed=3011, low_return=177.9, high_seed=3013, high_return=641.9),
]
OUT_DIR = f"{REPO_ROOT}/outputs/sample_videos_stage3_vs_stage3b"


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
def run_episode(model, cfg, seed, device, save_video_path):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    frames = []
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
            frames.append(image.copy())
            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
            total += float(reward)
            length += 1
            if end or trunc:
                break
            history.advance(observation(image), torch.tensor([False], device=device))
        return dict(seed=seed, total_return=total, length=length)
    finally:
        if frames:
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
    device = torch.device("cuda:0")

    for cfg_spec in CONFIGS:
        with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
            cfg = compose(config_name="trainer", overrides=[
                "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
                "agent.actor_critic.prior_refresh_interval=500",
                f"agent.actor_critic.entropy_coef={cfg_spec['entropy_coef']}",
            ])

        agent_cfg = build_agent_cfg(cfg, device)
        model = PMPOBeta(agent_cfg.actor_critic).to(device)
        model.eval()
        epoch, update = load_checkpoint(model, cfg_spec["checkpoint"], device)
        print(f"\n=== {cfg_spec['name']}: epoch={epoch} update={update} ===", flush=True)

        low_path = f"{OUT_DIR}/{cfg_spec['name']}_seed{cfg_spec['low_seed']}_low.mp4"
        high_path = f"{OUT_DIR}/{cfg_spec['name']}_seed{cfg_spec['high_seed']}_high.mp4"

        r_low = run_episode(model, cfg, cfg_spec["low_seed"], device, low_path)
        print(f"  low: seed={r_low['seed']} return={r_low['total_return']:.2f} length={r_low['length']}", flush=True)

        r_high = run_episode(model, cfg, cfg_spec["high_seed"], device, high_path)
        print(f"  high: seed={r_high['seed']} return={r_high['total_return']:.2f} length={r_high['length']}", flush=True)

        del model
        torch.cuda.empty_cache()

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
