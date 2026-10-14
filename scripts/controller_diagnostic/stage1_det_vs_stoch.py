"""Stage 1: deterministic (Beta-mean) vs stochastic (Beta-sample) evaluation on
existing checkpoints from the completed 100k PMPO-Beta baseline. Read-only,
no training, does not modify outputs/k4-walker-100k in any way.
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
CHECKPOINTS = {
    "plateau_epoch150": f"{RUN_DIR}/agent_pmpo_update_064600_epoch.pt",
    "final_epoch240": f"{RUN_DIR}/agent_pmpo_update_100600_epoch.pt",
}
OUT_DIR = f"{REPO_ROOT}/outputs/stage1_det_vs_stoch"
SEEDS = list(range(3001, 3031))  # fresh 30-seed set, distinct from earlier 1001-1030 / 2001-2020
STOCH_REPEATS = 3  # independent policy-sampling repeats per env seed, to separate env-seed effect from sampling noise


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
def run_episode(model, cfg, seed, device, deterministic, policy_rng_seed=None):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    if policy_rng_seed is not None:
        torch.manual_seed(int(policy_rng_seed))
    try:
        image, _ = env.reset(seed=int(seed))

        def observation(img):
            return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)

        history = FrameHistory(observation(image), model.frame_stack)
        total = 0.0
        length = 0
        for step in range(500):
            state = history.state
            action = model.sample_action(model.actor(state), deterministic=deterministic)
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
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
        ])

    agent_cfg = build_agent_cfg(cfg, device)
    model = PMPOBeta(agent_cfg.actor_critic).to(device)
    model.eval()

    all_results = {}
    for ckpt_name, ckpt_path in CHECKPOINTS.items():
        epoch, update = load_checkpoint(model, ckpt_path, device)
        print(f"\n=== {ckpt_name}: epoch={epoch} update={update} ===", flush=True)

        det_results = []
        for seed in SEEDS:
            r = run_episode(model, cfg, seed, device, deterministic=True)
            det_results.append(r)
            print(f"  [det] seed={seed} return={r['total_return']:.2f}", flush=True)

        stoch_results = []  # list of dict(seed, repeat, total_return, length)
        for seed in SEEDS:
            for rep in range(STOCH_REPEATS):
                r = run_episode(model, cfg, seed, device, deterministic=False, policy_rng_seed=seed * 1000 + rep)
                r["repeat"] = rep
                stoch_results.append(r)
                print(f"  [stoch] seed={seed} rep={rep} return={r['total_return']:.2f}", flush=True)

        all_results[ckpt_name] = dict(
            epoch=epoch, update=update,
            deterministic=det_results,
            stochastic=stoch_results,
        )
        with open(f"{OUT_DIR}/results.json", "w") as f:
            json.dump(all_results, f, indent=2)

    print("\nDONE", flush=True)


if __name__ == "__main__":
    main()
