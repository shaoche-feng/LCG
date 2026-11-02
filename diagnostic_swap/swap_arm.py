"""Bounded controller-only continuation with a controller bundle and a model/data bundle from (possibly different)
Trainer snapshots (component-swap diagnostic). Adapted from the replay-value ablation's validated run_arm.py:
  controller bundle  <- --snapshot of --source-run: actor, online critic, EMA critic, prior and initial actors, PMPO
                        update counter and private imagination RNG (all of agent.actor_critic), PMPO optimizers and
                        schedulers; the resolved config also comes from this run (configs of the two source runs differ
                        only in seed and W&B name/id)
  model/data bundle  <- --model-snapshot of --model-source-run: denoiser and reward/end model parameters and the
                        snapshot's dataset state, with episodes read in place from that run's append-only dataset
                        directory (only episodes < the snapshot's num_episodes, at the snapshot's lengths, can be sampled)
Without --model-snapshot both bundles come from --snapshot (the original behaviour, used for reproduction checks).

Rebuilds exactly the objects Trainer uses for controller updates -- Agent, PMPOOptimizers, the actor-critic DataLoader
(BatchSampler, batch 32, 4-frame prompts, configured sample weights), WorldModelEnv and agent.setup_training -- from the
source run's own saved config, restores the complete snapshot state (agent incl. actor, online/EMA/prior/initial critic
and actor copies, PMPO update counter and private imagination RNG; PMPO optimizers and schedulers; dataset state), and
runs the Trainer.train_component('actor_critic') loop body (zero_grad, forward, backward, PMPOOptimizers.step: one actor
and one critic Adam step per update). World and reward models are frozen (requires_grad False; they receive no
optimizer steps in the controller phase of the original either) and hashed. The replay dataset is read in place
(no writes) at the snapshot's dataset state; no environment experience is collected.

Arms (agent.actor_critic overrides on top of the source config; everything else unchanged):
  A  replay_value_coef 0.0, replay_sampling False  -> original BatchSampler, 32 independent starts
  B  replay_value_coef 0.0, replay_sampling True   -> grouped 8 segments x 4 consecutive starts, replay loss off
                                                       (replay statistics computed without gradients, monitoring only)
  C  replay_value_coef 0.3, replay_sampling True   -> grouped sampling + replay-value loss
  asis  the source config unchanged (reproduction checks against the source run's own later snapshot)

Prompt stream (--prompt-stream):
  shared     the original coupling: prompt indices drawn from the controller's private numpy stream (with action,
             world-model and end sampling sharing that stream's torch/python parts). Used for reproduction checks.
  dedicated  prompt indices drawn by the unchanged sampler code from a separate numpy stream re-seeded at the start
             of every update to RandomState(prompt_seed + relative_update), so prompt selection is independent of any
             difference in action/world-model RNG consumption and B and C draw the same first (replay) batch every
             update. The private stream still drives actions, world-model sampling, end sampling and loader seeds.

Records per update: all controller metrics, grad-clip flags, every sampled segment batch (episode, start, stop), the
derived imagination start (episode, t) of each row, and (every --check-every updates) an element-wise check that each
imagination start state equals the dataset frame at its recorded index repeated frame_stack times.
Checkpoints at --checkpoints (relative updates; 0 = restored state before any update): controller state, PMPO
optimizers, private RNG, prompt-stream position, plus fixed-input online/EMA values and frozen-model hashes.
Resumable: restarts from the latest saved diagnostic checkpoint (the prompt stream is per-update seeded)."""
import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

WORKTREE = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKTREE / "src"))

import numpy as np  # noqa: E402
import torch  # noqa: E402

# Component swap: the scalar-EMA source has no replay-value options; every arm runs the source config unchanged.
ARMS = dict(asis={})
WANDB_ENTITY = "models-national-tsing-hua-university7947"


def canon_hash(obj, h=None):
    """SHA256 over a nested structure: dict keys sorted; tensors by dtype, shape and bytes."""
    top = h is None
    h = h or hashlib.sha256()
    if isinstance(obj, dict):
        h.update(b"{")
        for k in sorted(obj, key=str):
            h.update(repr(k).encode()); canon_hash(obj[k], h)
        h.update(b"}")
    elif isinstance(obj, (list, tuple)):
        h.update(b"[")
        for v in obj:
            canon_hash(v, h)
        h.update(b"]")
    elif isinstance(obj, torch.Tensor):
        t = obj.detach().cpu().contiguous().reshape(-1)
        shape = tuple(obj.shape)
        h.update(f"T{t.dtype}{shape}".encode()); h.update(t.view(torch.uint8).numpy().tobytes() if t.numel() else b"")
    elif isinstance(obj, np.ndarray):
        h.update(f"N{obj.dtype}{obj.shape}".encode()); h.update(np.ascontiguousarray(obj).tobytes())
    else:
        h.update(repr(obj).encode())
    return h.hexdigest() if top else h


def file_sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


class PromptStream:
    """Wraps the prompt sampler's sample() (unchanged code) to record every batch and, in dedicated mode, to draw it
    from a per-update seeded numpy stream instead of the controller's private numpy stream."""

    def __init__(self, sampler, mode, seed):
        self.sampler, self.mode, self.seed = sampler, mode, seed
        self.orig = sampler.sample
        sampler.sample = self.sample
        self.state, self.batches = None, []

    def begin(self, relative_update):
        self.batches = []
        if self.mode == "dedicated":
            self.state = np.random.RandomState(self.seed + relative_update).get_state()

    def sample(self):
        if self.mode == "dedicated":
            saved = np.random.get_state()
            np.random.set_state(self.state)
        try:
            ids = self.orig()
        finally:
            if self.mode == "dedicated":
                self.state = np.random.get_state()
                np.random.set_state(saved)
        self.batches.append([(int(i.episode_id), int(i.start), int(i.stop)) for i in ids])
        return ids


def start_indices(first_batch, grouped, starts):
    """Imagination start (episode, t) of every rollout row. Independent: the prompt's last frame (stop - 1). Grouped:
    row s*starts+k ends at segment frame window-1+k with window = seq_length of the prompt loader (4)."""
    if not grouped:
        return [(e, stop - 1) for e, _, stop in first_batch]
    return [(e, start + 3 + k) for e, start, _ in first_batch for k in range(starts)]


def episode_manifest_sha256(dataset):
    """SHA256 over the files of the sampleable episodes (ids < num_episodes) with their recorded lengths."""
    h = hashlib.sha256()
    for i in range(int(dataset.num_episodes)):
        h.update(f"{i}:{int(dataset.lengths[i])}:".encode()); h.update(file_sha256(dataset._get_episode_path(i)).encode())
    return h.hexdigest()


def build(source_run, arm, device, data_run=None):
    from functools import partial
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from torch.utils.data import DataLoader
    from agent import Agent, get_action_space_kwargs
    from data import BatchSampler, collate_segments_to_batch, Dataset
    from envs import make_dm_control_env, WorldModelEnv
    from models.pmpo_beta import PMPOOptimizers
    from utils import normalize_saved_config
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    cfg = normalize_saved_config(OmegaConf.load(source_run / "config/trainer.yaml"))
    for k, v in ARMS[arm].items():
        cfg.agent.actor_critic[k] = v
    OmegaConf.resolve(cfg)
    t = cfg.env.test
    env = make_dm_control_env(num_envs=1, device=device, domain_name=t.domain_name, task_name=t.task_name, size=t.size,
                              camera_id=t.camera_id, action_repeat=t.action_repeat, time_limit=t.time_limit)
    action_kwargs = get_action_space_kwargs(env)
    env.close()
    agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(device)
    opt = PMPOOptimizers(agent.actor_critic, cfg.actor_critic.training.lr_warmup_steps)
    dataset = Dataset((data_run or source_run) / "dataset/train", "train_dataset", cfg.training.cache_in_ram, False)
    c = cfg.actor_critic.training
    bs = BatchSampler(dataset, 0, 1, c.batch_size, cfg.agent.denoiser.inner_model.num_steps_conditioning, c.sample_weights)
    dl = DataLoader(dataset=dataset, collate_fn=collate_segments_to_batch, num_workers=cfg.training.num_workers_data_loaders,
                    persistent_workers=False, pin_memory=True, pin_memory_device=str(device), batch_sampler=bs)
    rl_env = WorldModelEnv(agent.denoiser, agent.rew_end_model, dl, instantiate(cfg.world_model_env), return_imagined_candidate=False)
    agent.setup_training(instantiate(cfg.denoiser.sigma_distribution), rl_env)
    return cfg, agent, opt, dataset, rl_env


def wm_state(agent):
    return {**{f"denoiser.{k}": v for k, v in agent.denoiser.state_dict().items()},
            **{f"rew_end_model.{k}": v for k, v in agent.rew_end_model.state_dict().items()}}


@torch.no_grad()
def fixed_values(model, fixed, device):
    """Online and EMA critic values and deterministic actions on the fixed real-state inputs, one state at a time
    (batch size 1 reproduces the behaviour traces' saved values exactly; batched evaluation differs by up to ~0.007)."""
    out = {}
    for kind in ("real_history", "repeated_frame"):
        x = fixed[kind]
        v, vs, a = [], [], []
        for i in range(len(x)):
            s = x[i:i + 1].to(device).float().div(255).mul(2).sub(1)
            v.append(model.value(s).squeeze(-1).cpu())
            vs.append(model.slow_value(s).squeeze(-1).cpu())
            a.append(model.sample_action(model.actor(s), deterministic=True).cpu())
        out[kind] = dict(v_online=torch.cat(v), v_ema=torch.cat(vs), action=torch.cat(a))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--source-run", required=True, type=Path)
    p.add_argument("--snapshot", required=True, type=Path)
    p.add_argument("--arm", default="asis", choices=list(ARMS))
    p.add_argument("--arm-name", default=None, help="label, e.g. S01")
    p.add_argument("--model-source-run", type=Path, default=None)
    p.add_argument("--model-snapshot", type=Path, default=None)
    p.add_argument("--out", required=True, type=Path)
    p.add_argument("--updates", type=int, default=None, help="default: up to --compare-to's update counter")
    p.add_argument("--checkpoints", default="0")
    p.add_argument("--prompt-stream", choices=("shared", "dedicated"), default="dedicated")
    p.add_argument("--prompt-seed", type=int, default=190_000_000)
    p.add_argument("--check-every", type=int, default=50)
    p.add_argument("--fixed-inputs", type=Path, default=None)
    p.add_argument("--compare-to", type=Path, default=None)
    p.add_argument("--wandb-id"), p.add_argument("--wandb-name"), p.add_argument("--wandb-project")
    p.add_argument("--stop-after", type=int, default=None, help="smoke only: exit after this relative update (resume test)")
    args = p.parse_args()

    torch.backends.cuda.matmul.allow_tf32 = True  # as Trainer
    torch.backends.cudnn.deterministic = True
    device = torch.device("cuda:0")
    torch.cuda.set_device(0)
    from utils import set_seed
    from data import SegmentId
    out = args.out
    (out / "checkpoints").mkdir(parents=True, exist_ok=True)

    model_run = args.model_source_run or args.source_run
    model_snap = args.model_snapshot or args.snapshot
    if (args.model_source_run is None) != (args.model_snapshot is None):
        raise SystemExit("--model-source-run and --model-snapshot go together")
    cfg, agent, opt, dataset, rl_env = build(args.source_run, args.arm, device, data_run=model_run)
    set_seed(cfg.common.seed)  # as Trainer (the controller's imagination uses its own checkpointed stream)
    payload = torch.load(args.snapshot, map_location=device, weights_only=False)
    ts = payload["trainer_state"]
    mpayload = payload if model_snap == args.snapshot else torch.load(model_snap, map_location=device, weights_only=False)
    mts = mpayload["trainer_state"]
    from utils import extract_state_dict
    agent.load_state_dict(ts["agent"])  # controller bundle (and, provisionally, its own models)
    agent.denoiser.load_state_dict(extract_state_dict(mts["agent"], "denoiser"), strict=True)  # model/data bundle
    agent.rew_end_model.load_state_dict(extract_state_dict(mts["agent"], "rew_end_model"), strict=True)
    opt.load_state_dict(ts["opt"]["actor_critic"])
    dataset.load_state_dict(mts["train_dataset"])  # model snapshot's dataset state; episodes read in place from its run
    model = agent.actor_critic
    for m in (agent.denoiser, agent.rew_end_model):
        m.requires_grad_(False)
    snapshot_updates = int(model.updates)
    n_updates = args.updates
    target = None
    if args.compare_to is not None:
        target = torch.load(args.compare_to, map_location="cpu", weights_only=False)
        n_updates = n_updates or int(target["trainer_state"]["agent"]["actor_critic.updates"]) - snapshot_updates
    checkpoints = sorted(int(x) for x in args.checkpoints.split(","))
    grouped = getattr(model, "replay_loader", None) is not None
    sampler = model.replay_loader.batch_sampler if grouped else rl_env.data_loader.batch_sampler
    stream = PromptStream(sampler, args.prompt_stream, args.prompt_seed)
    fixed = torch.load(args.fixed_inputs, weights_only=False) if args.fixed_inputs else None

    wm_hash0 = canon_hash(wm_state(agent))
    provenance = dict(
        source_run=str(args.source_run), snapshot=str(args.snapshot), snapshot_sha256=file_sha256(args.snapshot),
        snapshot_epoch=payload["epoch"], snapshot_update=payload["update"], snapshot_pmpo_updates_counter=snapshot_updates,
        arm=args.arm, arm_name=args.arm_name, arm_overrides=ARMS[args.arm], prompt_stream=args.prompt_stream, prompt_seed=args.prompt_seed,
        model_source_run=str(model_run), model_snapshot=str(model_snap), model_snapshot_sha256=file_sha256(model_snap),
        model_snapshot_epoch=mpayload["epoch"], model_snapshot_update=mpayload["update"],
        grouped_sampling=grouped, replay_value_coef=float(getattr(model.cfg, "replay_value_coef", 0.0)),
        actor_distribution=model.cfg.actor_distribution, action_low=list(map(float, model.action_low)), action_high=list(map(float, model.action_high)),
        slow_critic_coef=float(model.cfg.slow_critic_coef), slow_critic_rate=float(model.cfg.slow_critic_rate),
        horizon=int(model.cfg.imagination_horizon), num_envs=int(rl_env.num_envs), max_grad_norm=float(model.cfg.max_grad_norm),
        prior_refresh_interval=int(model.cfg.prior_refresh_interval), updates=n_updates, checkpoints=checkpoints,
        sampler=type(sampler).__name__, sampler_batch_size=int(sampler.batch_size), sampler_seq_length=int(sampler.seq_length),
        sample_weights=list(sampler.sample_weights) if sampler.sample_weights is not None else None,
        dataset_dir=str(dataset._directory), dataset_num_episodes=int(dataset.num_episodes), dataset_num_steps=int(dataset.num_steps),
        dataset_state_sha256=canon_hash(mts["train_dataset"]), dataset_episode_manifest_sha256=episode_manifest_sha256(dataset),
        restored_hashes=dict(actor_critic=canon_hash(model.state_dict()), pmpo_optimizers=canon_hash(opt.state_dict()),
                             world_and_reward_models=wm_hash0,
                             model_snapshot_world_and_reward_models=canon_hash({k: v for k, v in mts["agent"].items() if k.startswith(("denoiser.", "rew_end_model."))}),
                             controller_snapshot_world_and_reward_models=canon_hash({k: v for k, v in ts["agent"].items() if k.startswith(("denoiser.", "rew_end_model."))}),
                             model_snapshot_actor_critic=canon_hash({k[len("actor_critic."):]: v for k, v in mts["agent"].items() if k.startswith("actor_critic.")}),
                             snapshot_actor_critic=canon_hash({k[len("actor_critic."):]: v for k, v in ts["agent"].items() if k.startswith("actor_critic.")}),
                             snapshot_pmpo_optimizers=canon_hash(ts["opt"]["actor_critic"])),
        torch=torch.__version__, cuda_visible_devices=os.environ.get("CUDA_VISIBLE_DEVICES"),
        egl_device=os.environ.get("MUJOCO_EGL_DEVICE_ID"), gpu_name=torch.cuda.get_device_name(0),
        fixed_inputs=str(args.fixed_inputs) if args.fixed_inputs else None,
        fixed_inputs_sha256=file_sha256(args.fixed_inputs) if args.fixed_inputs else None)
    rh = provenance["restored_hashes"]
    if rh["actor_critic"] != rh["snapshot_actor_critic"] or rh["pmpo_optimizers"] != rh["snapshot_pmpo_optimizers"]:
        raise RuntimeError("Restored controller/optimizer state differs from the controller snapshot")
    if rh["world_and_reward_models"] != rh["model_snapshot_world_and_reward_models"]:
        raise RuntimeError("Restored world/reward models differ from the model snapshot")
    # wiring identities: the controller bundle is the controller snapshot's; the model/data bundle is the model snapshot's
    provenance["controller_bundle_sha256"] = canon_hash(dict(actor_critic=rh["actor_critic"], pmpo_optimizers=rh["pmpo_optimizers"]))
    provenance["model_data_bundle_sha256"] = canon_hash(dict(models=rh["world_and_reward_models"], dataset_state=provenance["dataset_state_sha256"],
                                                             episodes=provenance["dataset_episode_manifest_sha256"]))

    # Resume from the latest diagnostic checkpoint if any.
    start = 0
    done = sorted(int(f.stem.split("_")[1]) for f in (out / "checkpoints").glob("update_*.pt"))
    if done:
        start = done[-1]
        ck = torch.load(out / "checkpoints" / f"update_{start:05d}.pt", map_location=device, weights_only=False)
        model.load_state_dict(ck["actor_critic"])
        opt.load_state_dict(ck["pmpo_optimizers"])
        provenance["resumed_from"] = start
    (out / "provenance.json").write_text(json.dumps(provenance, indent=1))

    run = None
    if args.wandb_id:
        import wandb
        run = wandb.init(project=args.wandb_project, entity=WANDB_ENTITY, id=args.wandb_id, name=args.wandb_name,
                         resume="allow", config=dict(provenance=provenance, source_config=json.loads(json.dumps(
                             __import__("omegaconf").OmegaConf.to_container(cfg, resolve=True), default=str))))
        run.define_metric("diagnostic_update")
        run.define_metric("*", step_metric="diagnostic_update")

    # Per-update logs, truncated to the resume point.
    logs = {}
    for name in ("updates.jsonl", "indices.jsonl"):
        path = out / name
        rows = [l for l in path.read_text().splitlines() if json.loads(l)["diagnostic_update"] <= start] if path.exists() else []
        path.write_text("".join(r + "\n" for r in rows))
        logs[name] = open(path, "a")

    def save_checkpoint(rel):
        state = dict(actor_critic=model.state_dict(), pmpo_optimizers=opt.state_dict(), relative_update=rel,
                     pmpo_updates_counter=int(model.updates), prompt_stream=args.prompt_stream, prompt_seed=args.prompt_seed,
                     next_prompt_seed=args.prompt_seed + rel, provenance=provenance)
        path = out / "checkpoints" / f"update_{rel:05d}.pt"
        torch.save(state, path)
        rec = dict(diagnostic_update=rel, pmpo_updates_counter=int(model.updates), path=str(path), sha256=file_sha256(path),
                   actor_critic_hash=canon_hash(state["actor_critic"]), pmpo_optimizers_hash=canon_hash(state["pmpo_optimizers"]),
                   world_and_reward_models_hash=canon_hash(wm_state(agent)), world_and_reward_models_unchanged=canon_hash(wm_state(agent)) == wm_hash0)
        if fixed is not None:
            fv = fixed_values(model, fixed, device)
            torch.save(fv, out / "checkpoints" / f"fixed_values_{rel:05d}.pt")
            for kind, d in fv.items():
                rec[f"fixed_{kind}_v_online_mean"] = float(d["v_online"].mean())
                rec[f"fixed_{kind}_v_ema_mean"] = float(d["v_ema"].mean())
                rec[f"fixed_{kind}_online_minus_ema_abs"] = float((d["v_online"] - d["v_ema"]).abs().mean())
            for src in sorted(set(fixed["source"])):
                m = torch.tensor([s == src for s in fixed["source"]])
                for kind, d in fv.items():
                    rec[f"fixed_{src}_{kind}_v_online_mean"] = float(d["v_online"][m].mean())
                    rec[f"fixed_{src}_{kind}_v_ema_mean"] = float(d["v_ema"][m].mean())
        with open(out / "checkpoints.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        if not rec["world_and_reward_models_unchanged"]:
            raise RuntimeError("World/reward model parameters changed")
        if run is not None:
            run.log({"diagnostic_update": rel, **{f"fixed/{k}": v for k, v in rec.items() if k.startswith("fixed_")}})
        print(f"checkpoint {rel} saved", flush=True)

    if start == 0 and 0 in checkpoints:
        save_checkpoint(0)

    captured = {}
    orig_collect = model.collect_imagination

    def collect():
        res = orig_collect()
        captured["obs0"] = res[0][:, 0]
        return res
    model.collect_imagination = collect

    agent.train()  # as Trainer.train_agent
    agent.zero_grad()
    model.train()
    t0 = time.time()
    rel = start - 1
    for rel in range(start, n_updates):
        stream.begin(rel)
        opt.zero_grad()
        loss, metrics = model()
        loss.backward()
        metrics.update(opt.step())
        metrics["actor_grad_clipped"] = (metrics["actor_grad_norm"] > model.cfg.max_grad_norm).float()
        metrics["value_grad_clipped"] = (metrics["value_grad_norm"] > model.cfg.max_grad_norm).float()
        row = {k: float(v) for k, v in metrics.items()}
        row.update(diagnostic_update=rel + 1, update_index=rel, pmpo_updates_counter=int(model.updates),
                   prior_refresh_event=float((int(model.updates) - 1) % model.cfg.prior_refresh_interval == 0))
        logs["updates.jsonl"].write(json.dumps(row) + "\n")
        first = stream.batches[0]
        starts = start_indices(first, grouped, getattr(model.cfg, "replay_starts", 4))
        idx = dict(diagnostic_update=rel + 1, batches=stream.batches, n_batches=len(stream.batches), starts=starts)
        if rel % args.check_every == 0 or rel < 5:
            obs0 = captured["obs0"]
            ok = all(torch.equal(obs0[r], dataset.load_episode(e).obs[tt].to(device).repeat(model.frame_stack, 1, 1))
                     for r, (e, tt) in enumerate(starts))
            idx["start_frames_checked"] = True
            idx["start_frames_match"] = ok
            if not ok:
                raise RuntimeError(f"Imagination start states differ from the recorded dataset indices at update {rel}")
        logs["indices.jsonl"].write(json.dumps(idx) + "\n")
        if run is not None:
            run.log({"diagnostic_update": rel + 1, **{f"train/{k}": v for k, v in row.items() if k != "diagnostic_update"}})
        if (rel + 1) % 50 == 0:
            for f in logs.values():
                f.flush()
            print(f"update {rel + 1}/{n_updates} {time.time() - t0:.0f}s loss_value {row['loss_value']:.3f} "
                  f"return_mean {row['return_mean']:.2f}", flush=True)
        if rel + 1 in checkpoints:
            for f in logs.values():
                f.flush()
            save_checkpoint(rel + 1)
        if args.stop_after is not None and rel + 1 >= args.stop_after:
            print(f"stopping after {rel + 1} (smoke resume test)", flush=True)
            break
    for f in logs.values():
        f.close()
    final = dict(actor_critic_hash=canon_hash(model.state_dict()), pmpo_optimizers_hash=canon_hash(opt.state_dict()),
                 world_and_reward_models_unchanged=canon_hash(wm_state(agent)) == wm_hash0, pmpo_updates_counter=int(model.updates),
                 completed_relative_updates=rel + 1, seconds=time.time() - t0)
    if target is not None:
        tsd = {k[len("actor_critic."):]: v for k, v in target["trainer_state"]["agent"].items() if k.startswith("actor_critic.")}
        msd = model.state_dict()
        mismatched = [k for k in tsd if canon_hash(tsd[k]) != canon_hash(msd[k])]
        final.update(compare_to=str(args.compare_to), target_actor_critic_hash=canon_hash(tsd),
                     actor_critic_identical=canon_hash(tsd) == canon_hash(msd), mismatched_actor_critic_keys=mismatched,
                     pmpo_optimizers_identical=canon_hash(target["trainer_state"]["opt"]["actor_critic"]) == canon_hash(opt.state_dict()),
                     max_abs_param_diff=max((float((tsd[k].float() - msd[k].cpu().float()).abs().max()) for k in tsd
                                             if isinstance(tsd[k], torch.Tensor) and tsd[k].is_floating_point()), default=0.0))
        # per-update metrics against the source run's own updates.jsonl
        src_rows = {}
        src = args.source_run / "pmpo_diagnostic/updates.jsonl"
        for line in open(src):
            r = json.loads(line)
            src_rows[r["update"]] = r
        mine = [json.loads(l) for l in open(out / "updates.jsonl")]
        diffs = {}
        for r in mine:
            s = src_rows.get(snapshot_updates + r["update_index"])
            if s is None:
                continue
            for k in ("loss_value", "loss_actor", "return_mean", "value_mean", "actor_grad_norm", "value_grad_norm", "kl_prior", "loss_repval"):
                if k in s and k in r:
                    diffs[k] = max(diffs.get(k, 0.0), abs(s[k] - r[k]))
        final.update(metric_rows_compared=sum(1 for r in mine if snapshot_updates + r["update_index"] in src_rows), max_abs_metric_diff=diffs)
    (out / "final.json").write_text(json.dumps(final, indent=1))
    print(json.dumps(final, indent=1), flush=True)
    if run is not None:
        run.summary.update({f"final/{k}": v for k, v in final.items() if isinstance(v, (int, float, bool, str))})
        run.finish()


if __name__ == "__main__":
    main()
