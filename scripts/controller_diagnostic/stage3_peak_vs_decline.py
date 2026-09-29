"""Read-only 30-seed deterministic evaluation of 3 Stage 3 checkpoints (context/plateau,
peak, decline). Runs entirely on cuda:0, never touches outputs/k4-walker-100k-entropy-
stage3's own training process, checkpoints, jsonl logs, or wandb run -- only reads
checkpoint files and writes to a separate output directory.
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import json
import torch

RUN_DIR = f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic"
CHECKPOINTS = {
    "context_epoch103": f"{RUN_DIR}/agent_pmpo_update_045800_epoch.pt",
    "peak_epoch108": f"{RUN_DIR}/agent_pmpo_update_047800_epoch.pt",
    "decline_epoch118": f"{RUN_DIR}/agent_pmpo_update_051800_epoch.pt",
}
OUT_DIR = f"{REPO_ROOT}/outputs/stage3_peak_vs_decline_30seed"
SEEDS = list(range(3001, 3031))  # same seeds as Stage 1


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


@torch.no_grad()
def controller_health(model, cfg, device):
    """Lightweight concentration/entropy diagnostics on a batch of fixed real-env states,
    reusing the model's own distribution()/log_prob_and_entropy() -- no new instrumentation."""
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    states = []
    for seed in SEEDS[:8]:  # small fixed batch of real states, distinct from the 30 eval seeds' returns
        env = DMControlEnv(**kwargs, seed=int(seed))
        try:
            image, _ = env.reset(seed=int(seed))
            obs = torch.as_tensor(image.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)
            history = FrameHistory(obs, model.frame_stack)
            states.append(history.state)
        finally:
            env.close()
    batch = torch.cat(states, dim=0)
    raw = model.actor(batch)
    dist = model.distribution(raw)
    alpha, beta = dist.concentration1, dist.concentration0
    unit = model.to_environment(dist.mean)
    action_unit = (unit - model.action_low) / (model.action_high - model.action_low)
    means = model.sample_action(raw, deterministic=True)
    return dict(
        alpha_mean=alpha.mean().item(), alpha_median=alpha.median().item(), alpha_max=alpha.max().item(),
        beta_mean=beta.mean().item(), beta_median=beta.median().item(), beta_max=beta.max().item(),
        policy_entropy=(dist.entropy().sum(-1) + (model.action_high - model.action_low).log().sum()).mean().item(),
        near_lower_fraction=(action_unit < 0.01).float().mean().item(),
        near_upper_fraction=(action_unit > 0.99).float().mean().item(),
        fixed_state_policy_mean_std=means.std(0, unbiased=False).mean().item(),
    )


def main():
    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
            "agent.actor_critic.entropy_coef=0.01",
        ])

    agent_cfg = build_agent_cfg(cfg, device)
    model = PMPOBeta(agent_cfg.actor_critic).to(device)
    model.eval()

    all_results = {}
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        epoch, update = load_checkpoint(model, ckpt_path, device)
        print(f"\n=== {ckpt_name}: epoch={epoch} update={update} path={ckpt_path} ===", flush=True)

        results = []
        for seed in SEEDS:
            r = run_episode(model, cfg, seed, device)
            results.append(r)
            print(f"  seed={seed} return={r['total_return']:.2f}", flush=True)

        health = controller_health(model, cfg, device)
        print(f"  controller_health: {health}", flush=True)

        all_results[ckpt_name] = dict(epoch=epoch, update=update, path=ckpt_path,
                                      results=results, controller_health=health)
        with open(f"{OUT_DIR}/results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
