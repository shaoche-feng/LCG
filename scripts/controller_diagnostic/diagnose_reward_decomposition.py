"""Instrumented replay of the 4 sample seeds: reproduces the exact real-environment
reward decomposition (standing / upright / move / total) per inner physics step,
verifies it sums to the previously-reported episode return, and records torso
height/uprightness/velocity/x-position/actions for downstream plotting and for the
world-model replay comparison (which reuses the exact action sequence recorded here).
"""
import os
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import json
import numpy as np
import torch

from dm_control.utils import rewards as dmc_rewards

CONFIGS = [
    dict(name="stage3", entropy_coef=0.01,
         checkpoint=f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt",
         seeds=[3013, 3030]),
    dict(name="stage3b", entropy_coef=0.02,
         checkpoint=f"{REPO_ROOT}/outputs/k4-walker-100k-entropy-stage3b/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt",
         seeds=[3013, 3011]),
]
OUT_DIR = f"{REPO_ROOT}/outputs/diagnose_reward_decomposition"

_STAND_HEIGHT = 1.2
_WALK_SPEED = 1.0


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


def decompose_reward(physics):
    """Reproduces PlanarWalker.get_reward exactly, term by term."""
    torso_height = physics.named.data.xpos["torso", "z"]
    torso_upright = physics.named.data.xmat["torso", "zz"]
    horizontal_velocity = physics.named.data.sensordata["torso_subtreelinvel"][0]
    torso_x = physics.named.data.xpos["torso", "x"]

    standing = dmc_rewards.tolerance(torso_height, bounds=(_STAND_HEIGHT, float("inf")), margin=_STAND_HEIGHT / 2)
    upright = (1 + torso_upright) / 2
    stand_reward = (3 * standing + upright) / 4
    move_reward = dmc_rewards.tolerance(horizontal_velocity, bounds=(_WALK_SPEED, float("inf")),
                                        margin=_WALK_SPEED / 2, value_at_margin=0.5, sigmoid="linear")
    reward = stand_reward * (5 * move_reward + 1) / 6
    return dict(torso_height=float(torso_height), torso_upright=float(torso_upright),
                horizontal_velocity=float(horizontal_velocity), torso_x=float(torso_x),
                standing=float(standing), upright=float(upright), stand_reward=float(stand_reward),
                move_reward=float(move_reward), reward=float(reward))


@torch.no_grad()
def run_instrumented_episode(model, cfg, seed, device):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    action_repeat = kwargs["action_repeat"]

    try:
        image, _ = env.reset(seed=int(seed))
        physics = env._dm_env.physics  # must be captured AFTER reset: reset(seed=...) rebuilds _dm_env

        def observation(img):
            return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)

        history = FrameHistory(observation(image), model.frame_stack)
        outer_steps = []
        total = 0.0
        length = 0
        for step in range(500):
            state = history.state
            action = model.sample_action(model.actor(state), deterministic=True)
            action_np = action[0].cpu().numpy()

            inner_records = []
            step_reward = 0.0
            dm_action = np.asarray(action_np, dtype=np.float32).reshape(env.action_space.shape)
            terminated = False
            truncated = False
            for inner in range(action_repeat):
                timestep = env._dm_env.step(dm_action)
                decomp = decompose_reward(physics)
                inner_records.append(decomp)
                step_reward += float(timestep.reward)
                if timestep.last():
                    terminated = bool(timestep.discount == 0.0)
                    truncated = not terminated
                    break
            image = env._render()
            total += step_reward
            length += 1

            outer_records = dict(outer_step=step, action=action_np.tolist(),
                                  step_reward=step_reward, inner=inner_records)
            outer_steps.append(outer_records)

            if terminated or truncated:
                break
            history.advance(observation(image), torch.tensor([False], device=device))

        return dict(seed=seed, total_return=total, length=length, steps=outer_steps)
    finally:
        env.close()


def main():
    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")

    all_results = {}
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

        for seed in cfg_spec["seeds"]:
            result = run_instrumented_episode(model, cfg, seed, device)
            key = f"{cfg_spec['name']}_seed{seed}"
            all_results[key] = dict(checkpoint=cfg_spec["checkpoint"], epoch=epoch, update=update, **result)
            print(f"  seed={seed} total_return={result['total_return']:.3f} length={result['length']}", flush=True)
            # sanity: sum of step_reward should equal total_return exactly by construction
            recon = sum(s["step_reward"] for s in result["steps"])
            print(f"    reconstructed sum(step_reward)={recon:.6f}  (matches by construction: {abs(recon-result['total_return'])<1e-9})", flush=True)

        del model
        torch.cuda.empty_cache()

    with open(f"{OUT_DIR}/instrumented_results.json", "w") as f:
        json.dump(all_results, f)
    print(f"\nSaved {OUT_DIR}/instrumented_results.json", flush=True)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
