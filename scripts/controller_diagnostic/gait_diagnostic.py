"""Evaluation-only gait characterization diagnostic. No training. Runs on cuda:0
explicitly, isolated from the smoke test (which uses common.devices=1). Loads the
existing full-loop checkpoint read-only.
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

RUN_DIR = f"{REPO_ROOT}/outputs/k4-walker-fullloop/pmpo_diagnostic"
CKPT = f"{RUN_DIR}/agent_epoch_00015_pmpo_update_010600_epoch.pt"
OUT_DIR = f"{REPO_ROOT}/outputs/gait_diagnostic"

HIGH_SEEDS = [1016, 1009, 1006]
LOW_SEEDS = [1027, 1019, 1022]
PERTURB_STEP = 120
# qvel DOF order for dm_control walker (confirmed via physics.named.data.qvel.axes.row.names):
# ['rootz','rootx','rooty','right_hip','right_knee','right_ankle','left_hip','left_knee','left_ankle'].
# qvel[0] = rootz = vertical (height) velocity -- a small upward kick, visually salient
# (changes torso height/pose) regardless of camera tracking, unlike a horizontal-velocity
# perturbation which a tracking camera could render as invisible.
PERTURB_QVEL_DELTA = 0.8


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
    # This checkpoint predates the initial_actor diagnostic module (added after this
    # run finished); strict=False is safe here since this diagnostic only evaluates
    # model.actor, never initial_actor/kl_initial. Report exactly what's missing.
    missing, unexpected = model.load_state_dict(ac_sd, strict=False)
    missing = [k for k in missing if not k.startswith("initial_actor.")]
    if missing or unexpected:
        print(f"WARNING non-initial_actor missing={missing} unexpected={unexpected}", flush=True)
    assert not missing and not unexpected, "unexpected checkpoint/model mismatch beyond initial_actor"
    return payload["epoch"], payload["update"]


@torch.no_grad()
def run_episode(model, cfg, seed, device, perturb_step=None, capture_states_every=None):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    log = dict(seed=seed, perturbed=perturb_step is not None, actions=[], rewards=[],
              qpos=[], qvel=[], torso_z=[], torso_xvel=[], perturb_step=perturb_step)
    captured_states = []
    try:
        image, _ = env.reset(seed=int(seed))
        # DMControlEnv.reset(seed=...) REBUILDS self._dm_env internally (dm_control has
        # no in-place reseed hook) -- physics must be captured AFTER reset, never before,
        # or it silently points at an orphaned, never-stepped physics object.
        physics = env._dm_env.physics

        def observation(img):
            return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)

        history = FrameHistory(observation(image), model.frame_stack)
        for step in range(500):
            state = history.state
            if capture_states_every and step % capture_states_every == 0:
                captured_states.append(state.cpu().clone())
            raw = model.actor(state)
            action = model.sample_action(raw, deterministic=True)
            act_np = action[0].cpu().numpy()
            log["actions"].append(act_np.tolist())
            log["qpos"].append(physics.data.qpos.copy().tolist())
            log["qvel"].append(physics.data.qvel.copy().tolist())
            try:
                log["torso_z"].append(float(physics.named.data.xpos["torso", "z"]))
            except Exception:
                log["torso_z"].append(None)
            if perturb_step is not None and step == perturb_step:
                physics.data.qvel[0] += PERTURB_QVEL_DELTA
                physics.forward()
            image, reward, end, trunc, _ = env.step(act_np)
            log["rewards"].append(float(reward))
            log["torso_xvel"].append(float(physics.data.qvel[0]))
            if end or trunc:
                break
            history.advance(observation(image), torch.tensor([False], device=device))
        log["total_return"] = float(sum(log["rewards"]))
        log["length"] = len(log["rewards"])
        return log, captured_states
    finally:
        env.close()


@torch.no_grad()
def k4_diagnostic(model, states, device):
    frames = states.reshape(-1, model.frame_stack, model.cfg.img_channels, model.cfg.img_size, model.cfg.img_size)
    ordered = model.sample_action(model.actor(states), deterministic=True)
    repeated = model.sample_action(model.actor(frames[:, -1:].repeat(1, model.frame_stack, 1, 1, 1).flatten(1, 2)), deterministic=True)
    # history reversed but CURRENT frame (last) kept in the same final position:
    # [o(t-1), o(t-2), o(t-3), o(t)] rather than a full reverse of all four frames.
    hist_rev = torch.cat([frames[:, :-1].flip(1), frames[:, -1:]], dim=1).flatten(1, 2)
    reversed_preserve = model.sample_action(model.actor(hist_rev), deterministic=True)
    return dict(
        ordered_repeated_diff=float((ordered - repeated).abs().mean()),
        ordered_reversed_preserve_diff=float((ordered - reversed_preserve).abs().mean()),
    )


def main():
    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(OUT_DIR, exist_ok=True)
    device = torch.device("cuda:0")
    print(f"torch device={device}, CUDA_VISIBLE_DEVICES={os.environ.get('CUDA_VISIBLE_DEVICES')}", flush=True)

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
        ])

    agent_cfg = build_agent_cfg(cfg, device)
    model = PMPOBeta(agent_cfg.actor_critic).to(device)
    model.eval()
    epoch, update = load_checkpoint(model, CKPT, device)
    print(f"Loaded checkpoint: epoch={epoch} update={update} path={CKPT}", flush=True)

    results = {}
    all_captured = []
    for seed in HIGH_SEEDS + LOW_SEEDS:
        normal, captured = run_episode(model, cfg, seed, device, perturb_step=None, capture_states_every=20)
        print(f"seed={seed} normal return={normal['total_return']:.2f} length={normal['length']}", flush=True)
        results[f"{seed}_normal"] = normal
        all_captured.extend(captured)
        if seed in HIGH_SEEDS:
            perturbed, _ = run_episode(model, cfg, seed, device, perturb_step=PERTURB_STEP)
            print(f"seed={seed} perturbed return={perturbed['total_return']:.2f} length={perturbed['length']}", flush=True)
            results[f"{seed}_perturbed"] = perturbed

    results["_meta"] = dict(checkpoint=CKPT, epoch=epoch, update=update, perturb_step=PERTURB_STEP,
                            perturb_qvel_delta=PERTURB_QVEL_DELTA, device=str(device),
                            cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"))
    with open(f"{OUT_DIR}/episodes.json", "w") as f:
        json.dump(results, f)
    print("Saved episodes.json", flush=True)

    states_batch = torch.cat(all_captured).to(device)
    k4 = k4_diagnostic(model, states_batch, device)
    print("K=4 diagnostic:", k4, flush=True)
    with open(f"{OUT_DIR}/k4_diagnostic.json", "w") as f:
        json.dump(dict(**k4, n_states=states_batch.shape[0]), f)

    print("DONE", flush=True)


if __name__ == "__main__":
    main()
