"""Zero-shot PMPO adaptation-budget diagnostic.

Freezes the Stage-3 (entropy_coef=0.01) 100k-real-step DIAMOND world model completely
and trains a FRESH PMPO actor-critic entirely through world-model imagination, with
ZERO additional real-environment interaction for training. The real DM Control
environment is used ONLY for periodic 30-seed deterministic evaluation. This is NOT
another joint Stage-3 run: no denoiser/reward-model optimizer is ever constructed, no
real collection happens, and the existing Stage-3 replay dataset is used strictly as a
frozen, read-only source of K=4 imagination-starting histories via the exact same
BatchSampler machinery (episode-respecting, no boundary crossing) Stage 3 itself used
for its own actor-critic training.

Logging: per-PMPO-update metrics are accumulated in memory and averaged into ONE
epoch-level record every `epoch_size` updates (default 500) -- no per-update log file
is kept. 30-seed real-environment evaluations run at explicit update-count milestones
(including update 0, before any training) and are logged/saved separately.
"""
import argparse
import json
import os
import statistics as st
import sys
import time
from copy import deepcopy

os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
sys.path.insert(0, "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/src")

import torch

WM_CHECKPOINT = "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/outputs/k4-walker-100k-entropy-stage3/pmpo_diagnostic/agent_pmpo_update_100600_epoch.pt"
DATASET_DIR = "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/outputs/k4-walker-100k-entropy-stage3/dataset"
OUT_DIR = "/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/outputs/zero_shot_pmpo_adaptation"
WANDB_PROJECT = "LCG-zero-shot-adaptation"
WANDB_RUN_NAME = "walker_stage3wm_pmpo_budget"
WANDB_ENTITY = "models-national-tsing-hua-university7947"

EVAL_SEEDS = list(range(3001, 3031))
CHECKPOINT_MILESTONES = {10000, 25000, 50000, 75000, 100000}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--total_updates", type=int, default=100000)
    p.add_argument("--epoch_size", type=int, default=500)
    p.add_argument("--milestones", type=int, nargs="+", default=[0, 5000, 10000, 25000, 50000, 75000, 100000])
    p.add_argument("--eval_horizon", type=int, default=500)
    p.add_argument("--n_eval_seeds", type=int, default=30)
    p.add_argument("--out_dir", type=str, default=OUT_DIR)
    p.add_argument("--wandb_mode", type=str, default="online")
    return p.parse_args()


def build_agent(cfg, device):
    from agent import Agent, get_action_space_kwargs
    from envs import make_dm_control_env
    from hydra.utils import instantiate
    test_env = make_dm_control_env(
        num_envs=1, device=device,
        domain_name=cfg.env.test.domain_name, task_name=cfg.env.test.task_name,
        size=cfg.env.test.size, camera_id=cfg.env.test.camera_id,
        action_repeat=cfg.env.test.action_repeat, time_limit=cfg.env.test.time_limit,
    )
    action_kwargs = get_action_space_kwargs(test_env)
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    return agent


def load_frozen_wm(agent, path, device):
    """Loads ONLY denoiser + rew_end_model from the checkpoint. actor_critic is left
    at its fresh (just-constructed) initialization -- the checkpoint's own
    actor_critic weights are read but never applied."""
    from utils import extract_state_dict
    payload = torch.load(path, map_location=device, weights_only=False)
    agent_sd = payload["trainer_state"]["agent"]
    agent.denoiser.load_state_dict(extract_state_dict(agent_sd, "denoiser"))
    agent.rew_end_model.load_state_dict(extract_state_dict(agent_sd, "rew_end_model"))
    agent.denoiser.requires_grad_(False).eval()
    agent.rew_end_model.requires_grad_(False).eval()
    return payload["epoch"], payload["update"]


def snapshot_wm_state(agent):
    return deepcopy(agent.denoiser.state_dict()), deepcopy(agent.rew_end_model.state_dict())


def verify_wm_unchanged(agent, snapshot):
    denoiser_before, rew_end_before = snapshot
    ok = True
    diffs = []
    for name, before in denoiser_before.items():
        after = agent.denoiser.state_dict()[name]
        if not torch.equal(before, after):
            ok = False
            diffs.append(f"denoiser.{name}")
    for name, before in rew_end_before.items():
        after = agent.rew_end_model.state_dict()[name]
        if not torch.equal(before, after):
            ok = False
            diffs.append(f"rew_end_model.{name}")
    return ok, diffs


def build_frozen_dataset_world_model_env(agent, cfg, device, batch_size):
    from data import BatchSampler, Dataset, collate_segments_to_batch
    from envs import WorldModelEnv
    from hydra.utils import instantiate
    from pathlib import Path
    from torch.utils.data import DataLoader

    # save_on_disk=False: purely defensive -- this script never calls save_to_default_path()
    # anyway (no collection happens), but this makes "the frozen dataset is never written
    # back to" structurally explicit rather than merely a consequence of omission.
    train_dataset = Dataset(Path(DATASET_DIR) / "train", "train_dataset", cache_in_ram=False, save_on_disk=False)
    train_dataset.load_from_default_path()
    print(f"Loaded frozen dataset: {train_dataset.num_episodes} episodes, {train_dataset.num_steps} steps", flush=True)

    seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning  # K=4
    sample_weights = list(cfg.actor_critic.training.sample_weights)
    bs = BatchSampler(train_dataset, rank=0, world_size=1, batch_size=batch_size,
                       seq_length=seq_length, sample_weights=sample_weights, can_sample_beyond_end=False)
    use_cuda = device.type == "cuda"
    dl = DataLoader(dataset=train_dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch,
                    num_workers=0, pin_memory=use_cuda, pin_memory_device=str(device) if use_cuda else "")

    wm_env_cfg = instantiate(cfg.world_model_env)
    rl_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, wm_env_cfg)
    return rl_env, train_dataset


@torch.no_grad()
def run_eval_episode(model, cfg, seed, device, horizon):
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
        for step in range(horizon):
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


def run_milestone_eval(model, cfg, device, update, seeds, horizon, out_dir, wandb_run=None):
    results = [run_eval_episode(model, cfg, s, device, horizon) for s in seeds]
    returns = [r["total_return"] for r in results]
    n = len(returns)
    s = sorted(returns)
    mean_ = st.mean(s)
    median_ = st.median(s)
    std_ = st.pstdev(s)
    frac = lambda t: sum(1 for x in s if x > t) / n
    summary = dict(update=update, seeds=seeds, returns=returns,
                   mean=mean_, median=median_, std=std_, min=s[0], max=s[-1],
                   frac_gt_200=frac(200), frac_gt_400=frac(400), frac_gt_500=frac(500), frac_gt_600=frac(600))
    with open(f"{out_dir}/eval30_update_{update}.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[eval] update={update} mean={mean_:.1f} median={median_:.1f} std={std_:.1f} "
          f">400={frac(400)*100:.0f}% >500={frac(500)*100:.0f}% >600={frac(600)*100:.0f}%", flush=True)
    if wandb_run is not None:
        wandb_run.log({
            "eval30/update": update, "eval30/return_mean": mean_, "eval30/return_median": median_,
            "eval30/return_std": std_, "eval30/return_min": s[0], "eval30/return_max": s[-1],
            "eval30/frac_gt_200": frac(200), "eval30/frac_gt_400": frac(400),
            "eval30/frac_gt_500": frac(500), "eval30/frac_gt_600": frac(600),
        })
    return summary


def save_pmpo_checkpoint(pmpo_model, opt, update, epoch_count, out_dir, wm_checkpoint_ref):
    payload = dict(
        actor_critic=pmpo_model.state_dict(),
        opt_actor_critic=opt.state_dict(),
        update=update, epoch_count=epoch_count,
        frozen_wm_checkpoint_ref=wm_checkpoint_ref,
        dataset_ref=DATASET_DIR,
    )
    path = f"{out_dir}/pmpo_update_{update:06d}.pt"
    torch.save(payload, path)
    print(f"[checkpoint] saved {path} ({os.path.getsize(path)/1e6:.1f} MB)", flush=True)
    return path


def aggregate_epoch(buffer):
    keys = buffer[0].keys()
    row = {}
    for k in keys:
        vals = [b[k] for b in buffer if k in b]
        try:
            row[k] = float(sum(float(v) for v in vals) / len(vals))
        except (TypeError, ValueError):
            continue
    return row


def main():
    args = parse_args()
    if args.smoke:
        args.total_updates = 20
        args.epoch_size = 5
        args.milestones = [0, 10, 20]
        args.n_eval_seeds = 2
        args.eval_horizon = 20
        args.out_dir = OUT_DIR + "_smoke"
        args.wandb_mode = "offline"

    from hydra import compose, initialize_config_dir
    from models.pmpo_beta import PMPOBeta, PMPOOptimizers
    from omegaconf import OmegaConf

    OmegaConf.register_new_resolver("eval", eval)
    os.makedirs(args.out_dir, exist_ok=True)
    device = torch.device("cuda:0")

    with initialize_config_dir(version_base="1.3", config_dir="/home/bigking/scfeng/LCG-pmpo-beta-k4-20260923/config"):
        cfg = compose(config_name="trainer", overrides=[
            "agent=pmpo_beta", "env=dm_control", "env.train.domain_name=walker", "env.train.task_name=walk",
            "agent.actor_critic.prior_refresh_interval=500",
            "agent.actor_critic.entropy_coef=0.01",
        ])

    with open(f"{args.out_dir}/resolved_config.yaml", "w") as f:
        f.write(OmegaConf.to_yaml(cfg))

    # 1. Agent: fresh actor_critic, WM loaded + frozen from the Stage-3 checkpoint.
    agent = build_agent(cfg, device)
    agent.eval()  # denoiser/rew_end_model in eval mode; actor_critic mode doesn't matter (no BN/dropout)
    wm_epoch, wm_update = load_frozen_wm(agent, WM_CHECKPOINT, device)
    print(f"Loaded frozen WM from epoch={wm_epoch} update={wm_update} (actor_critic left fresh-initialized)", flush=True)
    wm_snapshot = snapshot_wm_state(agent)

    # 2. Dataset-backed WorldModelEnv (read-only source of K=4 imagination starts).
    batch_size = cfg.actor_critic.training.batch_size
    rl_env, train_dataset = build_frozen_dataset_world_model_env(agent, cfg, device, batch_size)
    agent.actor_critic.setup_training(rl_env, None)

    # 3. Fresh optimizers.
    opt = PMPOOptimizers(agent.actor_critic, warmup_steps=cfg.actor_critic.training.lr_warmup_steps)

    # 4. W&B.
    import wandb
    wandb_run = wandb.init(project=WANDB_PROJECT, entity=WANDB_ENTITY, name=WANDB_RUN_NAME,
                            mode=args.wandb_mode, dir=args.out_dir,
                            config=dict(wm_checkpoint=WM_CHECKPOINT, dataset_dir=DATASET_DIR,
                                       epoch_size=args.epoch_size, total_updates=args.total_updates,
                                       entropy_coef=0.01, alpha_pmpo=0.5, beta_kl=0.3,
                                       prior_refresh_interval=500, imagination_horizon=15))

    run_metadata = dict(
        experiment="zero_shot_pmpo_adaptation_budget", frozen_wm_checkpoint=WM_CHECKPOINT,
        frozen_wm_epoch=wm_epoch, frozen_wm_update=wm_update, dataset_dir=DATASET_DIR,
        epoch_size=args.epoch_size, total_updates=args.total_updates, milestones=args.milestones,
        checkpoint_milestones=sorted(CHECKPOINT_MILESTONES), eval_seeds=EVAL_SEEDS[:args.n_eval_seeds],
        smoke_test=args.smoke,
    )
    with open(f"{args.out_dir}/run_metadata.json", "w") as f:
        json.dump(run_metadata, f, indent=2)

    milestones_remaining = sorted(args.milestones)

    def maybe_eval_and_checkpoint(update, epoch_count):
        nonlocal milestones_remaining
        while milestones_remaining and milestones_remaining[0] <= update:
            m = milestones_remaining.pop(0)
            run_milestone_eval(agent.actor_critic, cfg, device, m, EVAL_SEEDS[:args.n_eval_seeds],
                               args.eval_horizon, args.out_dir, wandb_run)
            if m in CHECKPOINT_MILESTONES or (args.smoke and m == args.total_updates):
                save_pmpo_checkpoint(agent.actor_critic, opt, m, epoch_count, args.out_dir, WM_CHECKPOINT)

    # Update-0 eval BEFORE any training.
    maybe_eval_and_checkpoint(0, 0)

    epoch_buffer = []
    epoch_count = 0
    t_start = time.time()
    for update in range(1, args.total_updates + 1):
        opt.zero_grad()
        loss, metrics = agent.actor_critic()
        loss.backward()
        # fail-fast: frozen WM must never receive gradients
        for p in agent.denoiser.parameters():
            assert p.grad is None, "Frozen denoiser received a gradient -- WM is not frozen!"
        for p in agent.rew_end_model.parameters():
            assert p.grad is None, "Frozen rew_end_model received a gradient -- WM is not frozen!"
        opt_metrics = opt.step()
        metrics.update(opt_metrics)
        if not bool(metrics.get("all_finite", 1.0)):
            raise FloatingPointError(f"Non-finite metrics at update {update}")
        epoch_buffer.append({k: (v.item() if torch.is_tensor(v) else v) for k, v in metrics.items()})

        if update % args.epoch_size == 0:
            epoch_count += 1
            row = aggregate_epoch(epoch_buffer)
            row["epoch"] = epoch_count
            row["update"] = update
            row["elapsed_s"] = time.time() - t_start
            with open(f"{args.out_dir}/epochs.jsonl", "a") as f:
                f.write(json.dumps(row) + "\n")
            wandb_run.log({f"train/{k}": v for k, v in row.items()})
            print(f"[epoch {epoch_count}] update={update} loss_actor={row.get('loss_actor', float('nan')):.4f} "
                  f"alpha_mean={row.get('alpha_mean', float('nan')):.2f} entropy={row.get('policy_entropy', float('nan')):.3f} "
                  f"elapsed={row['elapsed_s']:.1f}s", flush=True)
            epoch_buffer = []

        maybe_eval_and_checkpoint(update, epoch_count)

    # Final verification: frozen WM must be bit-identical to its state before training.
    ok, diffs = verify_wm_unchanged(agent, wm_snapshot)
    print(f"\n=== FROZEN WM VERIFICATION: {'PASS' if ok else 'FAIL'} ===", flush=True)
    if not ok:
        print("Changed tensors:", diffs[:20], flush=True)
        raise RuntimeError("Frozen world model was modified during training!")

    with open(f"{args.out_dir}/run_metadata.json") as f:
        meta = json.load(f)
    meta["wm_verification_passed"] = ok
    meta["total_wall_time_s"] = time.time() - t_start
    with open(f"{args.out_dir}/run_metadata.json", "w") as f:
        json.dump(meta, f, indent=2)

    wandb_run.finish()
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
