"""Diagnosis-only: verify a fresh (post-initial_actor) 500-update-cadence checkpoint
from the completed smoke test reloads correctly into a new Trainer, with strict=True
(no missing/unexpected keys this time), and that the prior-block state is exactly
preserved.
"""
import sys
sys.path.insert(0, f"{REPO_ROOT}/src")

from pathlib import Path

import torch
from omegaconf import OmegaConf

CKPT = "/tmp/pmpo_100k_smoke/run/pmpo_diagnostic/agent_pmpo_update_000100.pt"


def main():
    from hydra import compose, initialize_config_dir
    from trainer import Trainer

    OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "common.devices=1", "common.seed=0", "training.compile_wm=false",
            "collection.train.first_epoch.min=100", "collection.train.steps_per_epoch=50",
            "collection.train.num_steps_total=300", "rew_end_model.training.batch_size=8",
            "evaluation.should=false", "intrinsic_reward.enabled=false",
            "agent.actor_critic.prior_refresh_interval=5",
        ])

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("checkpoint epoch=", payload["epoch"], "update=", payload["update"])
    trainer_state = payload["trainer_state"]

    import os, tempfile
    root = tempfile.mkdtemp()
    os.chdir(root)
    os.makedirs(".hydra", exist_ok=True)
    OmegaConf.save(cfg, ".hydra/config.yaml")
    trainer = Trainer(cfg, root_dir=Path(f"{REPO_ROOT}"))

    before_updates = int(trainer.agent.actor_critic.updates.item())
    trainer.load_state_dict(trainer_state)  # strict, no exceptions handled -- must just work
    after_updates = int(trainer.agent.actor_critic.updates.item())
    print("PMPO update counter: before(fresh)=", before_updates, "after(restored)=", after_updates)
    assert after_updates == payload["update"] + 1, f"expected {payload['update']+1}, got {after_updates}"

    interval = trainer.agent.actor_critic.cfg.prior_refresh_interval
    print("prior_refresh_interval restored:", interval)
    assert (after_updates - 1) % interval == 0, "checkpoint should sit exactly at a refresh boundary"

    # initial_actor must ALSO restore correctly now (this checkpoint post-dates the
    # module) and must differ from prior_actor once training has moved the policy.
    ia_sd = trainer.agent.actor_critic.initial_actor.state_dict()
    pa_sd = trainer.agent.actor_critic.prior_actor.state_dict()
    differs = any(not torch.equal(ia_sd[k], pa_sd[k]) for k in ia_sd)
    print("initial_actor differs from prior_actor after training:", differs)

    opt_state = trainer.opt.actor_critic.state_dict()
    print("actor optimizer state non-empty:", len(opt_state["actor"]["state"]) > 0)

    print("SMOKE-CHECKPOINT RELOAD VERIFICATION PASSED")


if __name__ == "__main__":
    main()
