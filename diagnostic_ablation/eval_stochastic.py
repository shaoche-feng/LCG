"""Stochastic-policy evaluation of diagnostic checkpoints (baseline follow-up D). Same episode loop as eval_arm.py
(run_eval30.py: real-history four-frame stack, 500 steps, environment seeds 3001-3030), but the action is SAMPLED from
the policy used in imagination training: pre-tanh x ~ Normal(mu, std), unit = (tanh(x) + 1) / 2, mapped to the action
bounds (the same transform as PMPOBeta.sample_action with deterministic=False). The Gaussian noise is drawn from a
private torch.Generator on the CPU, seeded per (environment seed, action seed) as env_seed * 1000 + action_seed, so
evaluation randomness is separate from learning and from the global RNG, and identical across checkpoints.
Action seeds: 0, 1, 2 (three episodes per environment seed).
usage: python eval_stochastic.py --source-run DIR --out-dir DIR --spec REL:CKPT [--spec ...] [--wandb-id ID --wandb-project P]"""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))
sys.path.insert(0, str(Path(__file__).parent))
import torch  # noqa: E402
from eval_arm import load_model, summarize, WANDB_ENTITY  # noqa: E402

ENV_SEEDS = list(range(3001, 3031))
ACTION_SEEDS = [0, 1, 2]


@torch.no_grad()
def run_episode(model, cfg, seed, action_seed, device):
    from coroutines.frame_history import FrameHistory
    from envs.dm_control_env import DMControlEnv
    kwargs = {k: v for k, v in cfg.env.test.items() if k != "type"}
    env = DMControlEnv(**kwargs, seed=int(seed))
    gen = torch.Generator().manual_seed(int(seed) * 1000 + int(action_seed))
    try:
        image, _ = env.reset(seed=int(seed))

        def observation(img):
            return torch.as_tensor(img.copy(), device=device).permute(2, 0, 1).float().div(255).mul(2).sub(1).unsqueeze(0)

        history = FrameHistory(observation(image), model.frame_stack)
        total, length = 0.0, 0
        for step in range(500):
            dist = model.distribution(model.actor(history.state))
            noise = torch.randn(dist.mean.shape, generator=gen).to(device)
            pre_tanh = dist.mean + dist.stddev * noise
            action = model.to_environment((torch.tanh(pre_tanh) + 1) / 2)
            image, reward, end, trunc, _ = env.step(action[0].cpu().numpy())
            total += float(reward)
            length += 1
            if end or trunc:
                break
            history.advance(observation(image), torch.tensor([False], device=device))
        return dict(seed=seed, action_seed=action_seed, total_return=total, length=length)
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--spec", action="append", required=True)
    p.add_argument("--env-seeds", default=None, help="smoke only: comma-separated subset")
    p.add_argument("--wandb-id")
    p.add_argument("--wandb-project")
    args = p.parse_args()
    if not ACTION_SEEDS or not ENV_SEEDS:
        raise SystemExit("no seeds")
    env_seeds = [int(s) for s in args.env_seeds.split(",")] if args.env_seeds else ENV_SEEDS
    device = torch.device("cuda:0")
    cfg, model = load_model(args.source_run, device)
    if not model.gaussian:
        raise SystemExit("stochastic evaluation is written for the tanh-Gaussian actor")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for spec in args.spec:
        rel, ckpt = spec.split(":", 1)
        out = args.out_dir / f"eval_stochastic30x3_update_{int(rel):05d}.json"
        if out.exists():
            results.append(json.loads(out.read_text()))
            continue
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["actor_critic"], strict=True)
        global_cpu = torch.get_rng_state()
        rows = []
        for seed in env_seeds:
            for a in ACTION_SEEDS:
                r = run_episode(model, cfg, seed, a, device)
                rows.append(r)
                print(f"update={rel} stochastic seed={seed} action_seed={a} return={r['total_return']:.2f}", flush=True)
        if not torch.equal(global_cpu, torch.get_rng_state()):
            raise RuntimeError("stochastic evaluation touched the global CPU RNG")
        per_env = {s: st.mean(r["total_return"] for r in rows if r["seed"] == s) for s in env_seeds}
        summary = dict(checkpoint=ckpt, diagnostic_update=int(rel), kind="stochastic30x3", env_seeds=env_seeds, action_seeds=ACTION_SEEDS,
                       noise_seed_rule="torch.Generator(cpu).manual_seed(env_seed * 1000 + action_seed)", episodes=rows,
                       all_episodes=summarize([r["total_return"] for r in rows]), per_env_seed_mean=per_env,
                       per_env_seed_mean_stats=summarize(list(per_env.values())),
                       action_seed_spread_mean_sd=st.mean(st.pstdev([r["total_return"] for r in rows if r["seed"] == s]) for s in env_seeds))
        out.write_text(json.dumps(summary, indent=1))
        results.append(summary)
        print(json.dumps({k: summary[k] for k in ("diagnostic_update", "all_episodes", "per_env_seed_mean_stats", "action_seed_spread_mean_sd")}), flush=True)
    if args.wandb_id:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=WANDB_ENTITY, id=args.wandb_id, resume="must")
        run.define_metric("diagnostic_update")
        run.define_metric("*", step_metric="diagnostic_update")
        for s in results:
            a = s["all_episodes"]
            run.log({"diagnostic_update": s["diagnostic_update"], **{f"eval_stochastic/return_{m}": a[m] for m in ("mean", "median", "std", "min", "max")},
                     **{f"eval_stochastic/{m}": a[m] for m in ("frac_gt_200", "frac_gt_400", "frac_gt_500", "frac_gt_600")}})
        run.finish()


if __name__ == "__main__":
    main()
