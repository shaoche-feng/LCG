"""Diagnosis-only: verify the DrQ medium10 epoch-1 checkpoint's denoiser/rew_end_model
state dicts load cleanly (strict) into a freshly constructed PMPO-Beta Agent, before
launching the fixed-prior warm-started diagnostic. No training, no writes.
"""
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = "/home/bigking/scfeng/LCG/outputs/drq_actor_feature_l2_medium10/checkpoints/agent_versions/agent_epoch_00001.pt"


def main():
    from agent import Agent, get_action_space_kwargs
    from envs import make_dm_control_env
    from utils import extract_state_dict

    OmegaConf.register_new_resolver("eval", eval)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(
            config_name="trainer",
            overrides=["env=dm_control", "agent=pmpo_beta", "env.train.domain_name=walker", "env.train.task_name=walk"],
        )

    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    agent_cfg = instantiate(cfg.agent, **action_kwargs)
    agent = Agent(agent_cfg).to(device)

    sd = torch.load(CKPT, map_location=device, weights_only=False)
    denoiser_sd = extract_state_dict(sd, "denoiser")
    rew_end_sd = extract_state_dict(sd, "rew_end_model")

    missing_d, unexpected_d = agent.denoiser.load_state_dict(denoiser_sd, strict=True)
    print("denoiser strict load OK, missing=", missing_d, "unexpected=", unexpected_d)
    missing_r, unexpected_r = agent.rew_end_model.load_state_dict(rew_end_sd, strict=True)
    print("rew_end_model strict load OK, missing=", missing_r, "unexpected=", unexpected_r)
    print("COMPATIBILITY CONFIRMED")


if __name__ == "__main__":
    main()
