"""Diagnosis-only: deterministic Beta-mean evaluation of the fixed-prior diagnostic's
update-2000 checkpoint, on the exact fixed evaluation seeds. No training, no writes to
the stopped run's own directories.
"""
import sys
import pathlib
REPO_ROOT = pathlib.Path(__file__).resolve().parents[2]  # scripts/controller_diagnostic/<file>.py -> repo root
sys.path.insert(0, f"{REPO_ROOT}/src")

import json
import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf

CKPT = f"{REPO_ROOT}/outputs/k4-walker-fixed-prior-warmstart/pmpo_diagnostic/checkpoint_update_02000.pt"


def main():
    from agent import get_action_space_kwargs
    from envs import make_dm_control_env
    from models.pmpo_beta import PMPOBeta
    from pmpo_diagnostic import PMPORunDiagnostic
    from types import SimpleNamespace

    OmegaConf.register_new_resolver("eval", eval)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "+agent.actor_critic.fixed_prior=true",
        ])

    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    agent_cfg = instantiate(cfg.agent, **action_kwargs)
    model = PMPOBeta(agent_cfg.actor_critic).to(device)
    model.eval()

    sd = torch.load(CKPT, map_location=device, weights_only=False)
    model.actor.load_state_dict(sd["actor"])
    model.value.load_state_dict(sd["value"])
    model.prior_actor.load_state_dict(sd["prior_actor"])
    print("Loaded checkpoint at update", sd["update"])

    diag = PMPORunDiagnostic(SimpleNamespace(output_dir="/tmp/pmpo_fixed_prior_eval", evaluation_seeds=[1001, 1002, 1003, 1004]),
                             model, cfg.env.test)
    result = diag.evaluate()
    summary = {k: v for k, v in result.items() if not isinstance(v, list) or k in ("eval_returns", "eval_lengths")}
    print(json.dumps(summary, indent=2, default=str))


if __name__ == "__main__":
    main()
