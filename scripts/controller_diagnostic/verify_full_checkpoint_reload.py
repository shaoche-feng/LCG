"""Diagnosis-only: verify a 500-update full-loop checkpoint reloads correctly into a
fresh Trainer via Trainer.load_state_dict, and that PMPO update count / prior-refresh
timing / weights / optimizer state all restore correctly. No training, no writes back
to the running experiment's own directory.
"""
import sys
sys.path.insert(0, f"{REPO_ROOT}/src")

from pathlib import Path

import torch
from omegaconf import OmegaConf

CKPT = f"{REPO_ROOT}/outputs/k4-walker-fullloop/pmpo_diagnostic/agent_epoch_00001_pmpo_update_000500.pt"


def main():
    from hydra import compose, initialize_config_dir
    from trainer import Trainer

    OmegaConf.register_new_resolver("eval", eval)
    with initialize_config_dir(version_base="1.3", config_dir=f"{REPO_ROOT}/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "common.devices=1", "common.seed=0", "training.compile_wm=false",
            "collection.train.num_steps_total=10000", "collection.train.steps_per_epoch=500",
            "rew_end_model.training.batch_size=8", "evaluation.should=false", "intrinsic_reward.enabled=false",
            "agent.actor_critic.prior_refresh_interval=500",
        ])

    payload = torch.load(CKPT, map_location="cpu", weights_only=False)
    print("checkpoint epoch=", payload["epoch"], "update=", payload["update"])
    trainer_state = payload["trainer_state"]
    print("top-level trainer_state keys:", sorted(trainer_state.keys()))

    import tempfile, os
    root = tempfile.mkdtemp()
    os.chdir(root)
    os.makedirs(".hydra", exist_ok=True)
    OmegaConf.save(cfg, ".hydra/config.yaml")
    trainer = Trainer(cfg, root_dir=Path(f"{REPO_ROOT}"))
    before_actor = {k: v.clone() for k, v in trainer.agent.actor_critic.actor.state_dict().items()}
    before_updates = int(trainer.agent.actor_critic.updates.item())
    before_epoch = trainer.epoch

    trainer.load_state_dict(trainer_state)

    after_updates = int(trainer.agent.actor_critic.updates.item())
    print("PMPO update counter: before(fresh)=", before_updates, "after(restored)=", after_updates)
    # check_update's own "update" counter (used for the filename/cadence check) is the
    # num_batch_train.actor_critic value BEFORE it increments for this iteration, while
    # opt.step() (called just before check_update, same iteration) already incremented
    # model.updates -- so the restored buffer is consistently update+1, a benign,
    # pre-existing off-by-one in the label vs. the buffer, not a resume-fidelity bug.
    assert after_updates == payload["update"] + 1, f"expected {payload['update'] + 1}, got {after_updates}"
    print("epoch counter restored:", trainer.epoch)
    assert trainer.epoch == 1

    # prior-refresh timing: forward()'s refresh check reads self.updates BEFORE this
    # iteration's opt.step() increments it, so the restored buffer (already +1) should
    # satisfy (updates - 1) % interval == 0 -- i.e. this checkpoint sits exactly at a
    # refresh boundary, the state the 500-update cadence is meant to capture.
    interval = trainer.agent.actor_critic.cfg.prior_refresh_interval
    assert (trainer.agent.actor_critic.updates.item() - 1) % interval == 0
    print("prior-refresh block boundary check OK ((updates-1) % interval == 0)")

    after_actor = trainer.agent.actor_critic.actor.state_dict()
    changed = any(not torch.equal(before_actor[k], after_actor[k]) for k in before_actor)
    print("actor weights changed from fresh init after reload:", changed)
    assert changed

    opt_state = trainer.opt.actor_critic.state_dict()
    print("actor optimizer restored, has 'actor' Adam state groups:", "state" in opt_state["actor"])
    assert len(opt_state["actor"]["state"]) > 0, "actor optimizer state should be non-empty after real training"

    print("RELOAD VERIFICATION PASSED")


if __name__ == "__main__":
    main()
