"""30-seed deterministic evaluation of Stage 3's (entropy_coef=0.01) actual final
checkpoint (epoch 240, update 100600), now that training has fully completed.
Runs on cuda:1 (free, since Stage 3's own training process has exited), read-only,
does not touch Stage 3B's GPU0 or its training process in any way.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "1")
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import json
import statistics as st
import torch

RUN_DIR = f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic"
CHECKPOINT = f"{RUN_DIR}/agent_pmpo_update_100600_epoch.pt"
OUT_DIR = f"{REPO_ROOT}/outputs/stage3_final_30seed_eval"
SEEDS = list(range(3001, 3031))


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
def run_episode(model, cfg, seed, device):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
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
            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
            total += float(reward)
            length += 1
            if end or trunc:
                break
            history.advance(observation(image), torch.tensor([False], device=device))
        return dict(seed=seed, total_return=total, length=length)
    finally:
        env.close()


def main():
    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")  # CUDA_VISIBLE_DEVICES=1 remaps physical GPU1 to logical cuda:0

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
            "agent.actor_critic.entropy_coef=0.01",
        ])

    agent_cfg = build_agent_cfg(cfg, device)
    model = PMPOBeta(agent_cfg.actor_critic).to(device)
    model.eval()
    epoch, update = load_checkpoint(model, CHECKPOINT, device)
    print(f"Loaded checkpoint: epoch={epoch} update={update} path={CHECKPOINT}", flush=True)

    results = []
    for seed in SEEDS:
        r = run_episode(model, cfg, seed, device)
        results.append(r)
        print(f"seed={seed} return={r['total_return']:.2f} length={r['length']}", flush=True)

    returns = [r["total_return"] for r in results]
    returns_sorted = sorted(returns)
    n = len(returns_sorted)
    mean_ = st.mean(returns_sorted)
    median_ = st.median(returns_sorted)
    std_ = st.pstdev(returns_sorted)
    min_ = returns_sorted[0]
    max_ = returns_sorted[-1]
    frac = lambda t: sum(1 for x in returns_sorted if x > t) / n

    summary = dict(
        checkpoint=CHECKPOINT, epoch=epoch, update=update,
        seeds=SEEDS, returns=returns,
        mean=mean_, median=median_, std=std_, min=min_, max=max_,
        frac_gt_200=frac(200), frac_gt_400=frac(400), frac_gt_500=frac(500), frac_gt_600=frac(600), frac_gt_800=frac(800),
    )
    print("\n=== SUMMARY ===")
    print(json.dumps(summary, indent=2))

    out_path = f"{OUT_DIR}/results.json"
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved {out_path}")
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
