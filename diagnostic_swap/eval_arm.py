"""Deterministic evaluation of ablation diagnostic checkpoints. The episode loop is copied verbatim from the source
run's eval30/run_eval30.py (itself verbatim from run_stage3_30seed_eval.py): deterministic tanh(mu) actions,
real-history four-frame stack, 500 steps. Seeds: eval30 = 3001-3030, eval4 = 3001, 3004, 3016, 3021.

usage: python eval_arm.py --source-run DIR --out-dir DIR --spec REL:KIND:CKPT [--spec ...]
                          [--saved-reference JSON] [--wandb-ids ID,ID,...] [--wandb-project P]
Writes OUT_DIR/eval_<kind>_update_<rel>.json per spec. --saved-reference: compare the per-seed returns of the first
spec with a saved eval30 JSON (exact equality required). Results are logged as eval30/* or eval4/* against
diagnostic_update to every listed W&B run (resume='must'), one run at a time."""
import argparse
import json
import statistics as st
import sys
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))
import torch  # noqa: E402

SEEDS = dict(eval30=list(range(3001, 3031)), eval4=[3001, 3004, 3016, 3021])
WANDB_ENTITY = "models-national-tsing-hua-university7947"


@torch.no_grad()
def run_episode(model, cfg, seed, device):  # verbatim from run_stage3_30seed_eval.py
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


def load_model(source_run, device):
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import get_action_space_kwargs
    from envs import make_dm_control_env
    from models.pmpo_beta import PMPOBeta
    from utils import normalize_saved_config
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = normalize_saved_config(OmegaConf.load(source_run / "config/trainer.yaml"))
    OmegaConf.resolve(cfg)
    t = cfg.env.test
    env = make_dm_control_env(num_envs=1, device=device, domain_name=t.domain_name, task_name=t.task_name, size=t.size,
                              camera_id=t.camera_id, action_repeat=t.action_repeat, time_limit=t.time_limit)
    model = PMPOBeta(instantiate(cfg.agent, **get_action_space_kwargs(env)).actor_critic).to(device).eval()
    env.close()
    return cfg, model


def summarize(returns):
    frac = lambda thr: sum(x > thr for x in returns) / len(returns)
    return dict(mean=st.mean(returns), median=st.median(returns), std=st.pstdev(returns), min=min(returns), max=max(returns),
                frac_gt_200=frac(200), frac_gt_400=frac(400), frac_gt_500=frac(500), frac_gt_600=frac(600))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--out-dir", required=True, type=Path)
    p.add_argument("--spec", action="append", required=True)
    p.add_argument("--saved-reference", type=Path)
    p.add_argument("--wandb-ids", default="")
    p.add_argument("--wandb-project")
    args = p.parse_args()
    device = torch.device("cuda:0")
    cfg, model = load_model(args.source_run, device)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for i, spec in enumerate(args.spec):
        rel, kind, ckpt = spec.split(":", 2)
        out = args.out_dir / f"eval_{kind}_update_{int(rel):05d}.json"
        if out.exists():
            results.append(json.loads(out.read_text()))
            print(f"{out} exists, reusing", flush=True)
            continue
        ck = torch.load(ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["actor_critic"], strict=True)
        rows = []
        for seed in SEEDS[kind]:
            r = run_episode(model, cfg, seed, device)
            rows.append(r)
            print(f"update={rel} {kind} seed={seed} return={r['total_return']:.2f} length={r['length']}", flush=True)
        returns = [r["total_return"] for r in rows]
        summary = dict(checkpoint=ckpt, diagnostic_update=int(rel), kind=kind, pmpo_updates_counter=ck["pmpo_updates_counter"],
                       seeds=SEEDS[kind], returns=returns, lengths=[r["length"] for r in rows], **summarize(returns))
        if i == 0 and args.saved_reference is not None:
            ref = json.loads(args.saved_reference.read_text())
            ref_ret = dict(zip(ref["seeds"], ref["returns"]))
            summary["saved_reference"] = str(args.saved_reference)
            summary["reproduces_saved_reference"] = all(ref_ret[s] == r for s, r in zip(SEEDS[kind], returns))
            summary["max_abs_diff_vs_saved_reference"] = max(abs(ref_ret[s] - r) for s, r in zip(SEEDS[kind], returns))
            print(f"reproduces saved reference: {summary['reproduces_saved_reference']} "
                  f"(max abs diff {summary['max_abs_diff_vs_saved_reference']})", flush=True)
        out.write_text(json.dumps(summary, indent=1))
        results.append(summary)
        print(json.dumps({k: v for k, v in summary.items() if k not in ("returns", "lengths", "seeds")}), flush=True)
    if args.wandb_ids:
        import wandb
        for wid in args.wandb_ids.split(","):
            run = wandb.init(project=args.wandb_project, entity=WANDB_ENTITY, id=wid, resume="must")
            run.define_metric("diagnostic_update")
            run.define_metric("*", step_metric="diagnostic_update")
            for s in results:
                k = s["kind"]
                run.log({"diagnostic_update": s["diagnostic_update"], **{f"{k}/return_{m}": s[m] for m in ("mean", "median", "std", "min", "max")},
                         **{f"{k}/{m}": s[m] for m in ("frac_gt_200", "frac_gt_400", "frac_gt_500", "frac_gt_600")}})
            run.finish()
            print(f"logged {len(results)} evaluations to W&B run {wid}", flush=True)


if __name__ == "__main__":
    main()
