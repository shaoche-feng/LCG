"""Downstream training on a frozen exploration milestone (Step 5).

Phase 1 trains a fresh reward/end model on the selected task's recorded rewards. Phase 2
freezes it and trains a fresh PMPO policy in imagination on its predicted task reward. The
milestone's denoiser stays frozen throughout. Training runs in `.seed_<seed>.tmp/` next to
the reserved output `<run>/downstream/steps_N/<task>/seed_<seed>/` and is renamed to it when
complete. Checkpoints at completed updates allow a faithful resume; the real task is then
evaluated with a deterministic policy.
"""
from copy import deepcopy
import hashlib
import json
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional, Tuple

from dm_control import suite
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
import wandb

from agent import Agent
from coroutines.frame_history import FrameHistory
from data import BatchSampler, collate_segments_without_rewards
from data.milestone import (SUMMARY_FILE, downstream_output_dir, downstream_summary, load_milestone, milestone_members,
                            read_json, state_sha256, write_json)
from data.utils import make_task_reward_collate
from device_report import physical_gpu_index
from envs import DMControlEnv, TIME_LIMIT_ONLY_DOMAINS, WorldModelEnv
from models.pmpo_beta import (PMPOBeta, PMPOOptimizers, check_controller_state, controller_settings, metric_path,
                              reward_statistics)
from utils import (Progress, check_same_packages, configure_opt, file_sha256, get_lr_sched, rng_state, save_atomically,
                   say, set_rng_state, set_seed, source_identity)

# Bumped when the checkpoint contents needed for faithful resume change. 2: milestone
# replay content checksum (dataset.pt, which lists every episode file's checksum).
DOWNSTREAM_RESUME_FORMAT = 2
# Settings a resume may change: how it is launched, where the (moved) milestone is, and logging
# (W&B mode/names and the diagnostics-only debug flags; they change no loss, update or RNG draw).
RESUME_MAY_CHANGE = ("downstream.mode", "initialization.milestone_path", "wandb", "rew_end_model.debug_metrics",
                     "agent.actor_critic.debug_metrics")
RETURN_CONVENTION = ("sum over environment steps of the requested task's reward; each environment step "
                     "sums the reward of its executed action-repeat substeps")

# W&B (wandb.mode: disabled | offline | online). One run per milestone x task x adaptation seed x denoiser
# member. Rows: one per update (reward model, then PMPO) and one per evaluation; the W&B step of an update
# is its index in the whole training (reward-model updates first), so a resume knows which rows the
# checkpoint committed. Every update row is derived from its metrics.jsonl row (wandb_row), which is
# therefore also the history a resume backfills from.
REWARD_AXIS = "reward_model/train/core/num_updates"
POLICY_AXIS = "actor_critic/train/core/num_batch_train_actor_critic"
EVAL_AXIS = "exploration/train_env_steps"  # the milestone's recorded collected_steps (fixed per run)
# Reward-model keys below reward_model/train/: W&B name -> metrics.jsonl name. Debug keys only with
# rew_end_model.debug_metrics (rows that recorded prediction_mae).
REWARD_CORE = dict(loss_reward="loss_rew", predicted_reward_mean="predicted_reward_mean",
                   target_reward_mean="target_reward_mean")
REWARD_DEBUG = dict(loss_total="loss_total", loss_end="loss_end", grad_norm="grad_norm_before_clip",
                    prediction_mae="prediction_mae")
# Controller metrics not published downstream: advantage-weight statistics, which sign-only PMPO does not use.
UNPUBLISHED_POLICY_PREFIXES = ("weight_clipped_fraction_", "weight_mean_", "weight_max_", "weight_std_")
NOT_RECORDED = "not recorded"


class DownstreamTrainer:
    def __init__(self, cfg: DictConfig, resume: bool = False) -> None:
        for name in ("rew_end_model", "actor_critic"):
            steps = cfg[name].training.steps
            if not isinstance(steps, int) or steps < 1:
                raise ValueError(f"{name}.training.steps must be a positive integer, got {steps!r}")
        every = cfg.checkpointing.every
        if every is not None and (not isinstance(every, int) or every < 1):
            raise ValueError(f"checkpointing.every must be null or a positive integer, got {every!r}")
        check_evaluation_config(cfg.evaluation)
        if cfg.agent.actor_critic.actor_objective != "sign":
            raise ValueError("Downstream PMPO is sign-only (agent.actor_critic.actor_objective=sign), got "
                             f"{cfg.agent.actor_critic.actor_objective!r}")
        task_name, seed = cfg.downstream.task_name, cfg.downstream.seed
        set_seed(seed)  # downstream randomness comes only from its own seed
        torch.backends.cudnn.deterministic = True
        use_cuda = cfg.common.devices != "cpu" and torch.cuda.is_available()
        self._device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")  # reported by downstream_main

        # The exact milestone: validated denoiser, replay versions and task labels.
        # Ensemble milestones: one configured member (default 0) is the frozen denoiser.
        member_index = OmegaConf.select(cfg, "initialization.member_index", default=0)
        self.milestone = load_milestone(cfg.initialization.milestone_path, task_name, map_location=self._device,
                                        cache_in_ram=True, member_index=member_index)
        manifest = self.milestone.manifest
        self.member_sha256 = state_sha256(self.milestone.denoiser_state)
        if manifest["domain"] not in TIME_LIMIT_ONLY_DOMAINS:
            raise ValueError(f"Domain {manifest['domain']!r} is not verified to end only at its time limit")
        if self.milestone.dataset.counts_end[1] > 0:
            raise ValueError("Milestone replay has true terminations; downstream supports time-limit-only tasks")

        # The milestone's own env and denoiser configuration define the models.
        cfg = deepcopy(cfg)
        with open_dict(cfg):
            if cfg.initialization.pop("milestones_dir", None) is not None:  # launcher-level (downstream_main)
                raise ValueError("A downstream trainer takes one milestone; initialization.milestones_dir is for downstream_main.py")
            cfg.env.train = manifest["model"]["env"]
            cfg.agent.denoiser = manifest["model"]["denoiser"]
        OmegaConf.resolve(cfg)
        self.cfg = cfg

        # Outputs: never overwrite; trained in a work folder renamed when complete.
        self.output_dir = downstream_output_dir(self.milestone.path, task_name, seed, self.milestone.member_index)
        self._work_dir = self.output_dir.with_name(f".{self.output_dir.name}.tmp")
        self._path_checkpoint = self._work_dir / "checkpoints" / "state.pt"
        if self.output_dir.exists():
            raise FileExistsError(f"Downstream output {self.output_dir} already exists; not overwriting it")
        if resume:
            if not self._path_checkpoint.is_file():
                raise FileNotFoundError(f"No resumable downstream checkpoint at {self._path_checkpoint}")
        else:
            if self._work_dir.exists():
                raise FileExistsError(f"Incomplete downstream output {self._work_dir} from an earlier attempt; "
                                      f"resume it with downstream.mode=resume or delete it to rerun")
            self._work_dir.mkdir(parents=True)
            OmegaConf.save(cfg, self._work_dir / "config.yaml")

        # Models: frozen milestone denoiser, fresh reward/end model and PMPO.
        spec = action_spec(manifest["domain"], task_name)
        self.agent = Agent(instantiate(cfg.agent, **spec))
        self.agent.to(self._device)
        self.agent.denoiser.load_state_dict(self.milestone.denoiser_state)
        self.agent.denoiser.requires_grad_(False).eval()
        self.agent.rew_end_model.debug_metrics = bool(OmegaConf.select(cfg, "rew_end_model.debug_metrics", default=False))
        self.rew_end_opt = configure_opt(self.agent.rew_end_model, **cfg.rew_end_model.optimizer)
        self.rew_end_lr_sched = get_lr_sched(self.rew_end_opt, cfg.rew_end_model.training.lr_warmup_steps)
        self.actor_critic_opt = PMPOOptimizers(self.agent.actor_critic, cfg.actor_critic.training.lr_warmup_steps)

        # Replay sampling over the milestone's fixed dataset (see BatchSampler): reward-model
        # windows may pad on both sides; imagination prompts pad only before an episode start.
        dataset = self.milestone.dataset
        c = cfg.rew_end_model.training
        sampler = BatchSampler(dataset, 0, 1, c.batch_size, c.seq_length, c.sample_weights, can_sample_beyond_end=True)
        collate = make_task_reward_collate(self.milestone.task_index, self.milestone.successor_observations)
        # Creating the iterator draws one global Torch number; a resume restores RNG afterwards.
        self._reward_batches = iter(DataLoader(dataset, batch_sampler=sampler, collate_fn=collate))
        c = cfg.actor_critic.training
        sampler = BatchSampler(dataset, 0, 1, c.batch_size, cfg.agent.denoiser.inner_model.num_steps_conditioning,
                               c.sample_weights)
        prompts = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_segments_without_rewards)
        # Predicted task reward; no end sampling: imagined episodes end by horizon truncation.
        # PMPO draws a fresh prompt iterator from its own RNG stream for every update.
        self.rl_env = WorldModelEnv(self.agent.denoiser, self.agent.rew_end_model, prompts,
                                    instantiate(cfg.world_model_env), sample_ends=False)
        self.agent.actor_critic.setup_training(self.rl_env)

        self.phase = "rew_end_model"  # then "actor_critic", then "done"
        self.num_updates = dict(rew_end_model=0, actor_critic=0)
        self.source = source_identity(Path(__file__).resolve().parents[1])
        self._wandb_next = 0  # the next W&B step to log (rows below it are on the W&B server)
        self._replayed = None  # resume: updates the server already holds from the interrupted segment
        self._resume_event = {}  # resume: what the W&B reconciliation found (summary "resume")
        rng = None
        if resume:
            rng = self.load_checkpoint()
            recover_metrics_log(self._work_dir / "metrics.jsonl")
            self.log_row(dict(phase="resume", resumed_phase=self.phase, num_updates=dict(self.num_updates)))
        state = rng_state()
        self.start_wandb(resume)  # W&B draws no training random numbers: the state is put back
        set_rng_state(rng if resume else state)  # last: everything above may consume random numbers

    # ------------------------------------------------------------------ training

    def run(self) -> Path:
        try:
            output_dir = self._run()
        except BaseException as error:  # the W&B run is marked failed (a killed process stays "running")
            finish_wandb("failed", error)
            raise
        finish_wandb("completed")
        return output_dir

    def _run(self) -> Path:
        # Phases completed before a resume are skipped (no bar).
        if self.phase == "rew_end_model":
            self.train_reward_model()
        if self.phase == "actor_critic":
            self.train_policy()
        output_dir = self.save()
        if wandb.run is not None:
            wandb.summary["output"] = str(output_dir.resolve())
        try:
            if self.cfg.evaluation.after_training:
                result = evaluate_policy(self.agent.actor_critic, self.cfg.env.train, self.milestone.task_name,
                                         self.cfg.evaluation.seeds, self.cfg.evaluation.max_steps)
                path = save_evaluation(output_dir, result, self.agent.actor_critic.state_dict())
                report_evaluation(result, path)
                log_evaluation(result, path, self.milestone.manifest["collected_steps"], "after_training",
                               self.total_updates(), self._wandb_next)
        finally:  # also when the evaluation fails: the published outputs stay
            say(f"Downstream outputs saved: > {output_dir.resolve()}")
        return output_dir

    def train_reward_model(self) -> None:
        steps = self.cfg.rew_end_model.training.steps
        self.agent.rew_end_model.train()
        pbar = Progress(total=steps, initial=self.num_updates["rew_end_model"], desc="Reward-predictor training",
                        unit="updates")
        while self.num_updates["rew_end_model"] < steps:
            self.reward_model_update()
            pbar.update(1)
            self.maybe_checkpoint("rew_end_model", steps)
        pbar.close()
        # Frozen from here: PMPO only reads its predictions.
        self.agent.rew_end_model.requires_grad_(False).eval()
        self.phase = "actor_critic"
        self.save_checkpoint()

    def train_policy(self) -> None:
        steps = self.cfg.actor_critic.training.steps
        self.agent.actor_critic.train()
        pbar = Progress(total=steps, initial=self.num_updates["actor_critic"], desc="Task-policy training", unit="updates")
        while self.num_updates["actor_critic"] < steps:
            self.policy_update()
            pbar.update(1)
            self.maybe_checkpoint("actor_critic", steps)
        pbar.close()
        self.phase = "done"
        self.save_checkpoint()

    def reward_model_update(self) -> None:
        model, c = self.agent.rew_end_model, self.cfg.rew_end_model.training
        batch = next(self._reward_batches).to(self._device)
        self.rew_end_opt.zero_grad()
        loss, metrics = model(batch)
        loss.backward()
        metrics.pop("confusion_matrix")
        metrics["grad_norm_before_clip"] = torch.nn.utils.clip_grad_norm_(model.parameters(), c.max_grad_norm)
        self.rew_end_opt.step()
        metrics["lr"] = self.rew_end_lr_sched.get_last_lr()[0]
        self.rew_end_lr_sched.step()
        self.num_updates["rew_end_model"] += 1
        self.log("rew_end_model", metrics)

    def policy_update(self) -> None:
        self.actor_critic_opt.zero_grad()
        loss, metrics = self.agent.actor_critic()
        loss.backward()
        metrics.update(self.actor_critic_opt.step())
        # The predicted task rewards this update gave the lambda returns (no second reward-model pass).
        metrics.update(reward_statistics(self.agent.actor_critic.last_reward, "task_reward",
                                         self.agent.actor_critic.debug_metrics))
        self.num_updates["actor_critic"] += 1
        self.log("actor_critic", metrics)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the current policy in the real task without changing training state."""
        return evaluate_policy(self.agent.actor_critic, self.cfg.env.train, self.milestone.task_name,
                               self.cfg.evaluation.seeds, self.cfg.evaluation.max_steps)

    # ------------------------------------------------------------------ logging and outputs

    def log(self, phase: str, metrics: Dict[str, Any]) -> None:
        # (phase, update) identifies a row; updates replayed after a resume repeat it.
        row = dict(phase=phase, update=self.num_updates[phase], **{k: float(v) for k, v in metrics.items()})
        self.log_row(row)  # first: the local record a later resume can backfill W&B from
        self.publish(row)

    def log_row(self, row: Dict[str, Any]) -> None:
        with open(self._work_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    # ------------------------------------------------------------------ W&B

    def total_updates(self) -> int:
        return self.cfg.rew_end_model.training.steps + self.cfg.actor_critic.training.steps

    def wandb_step(self, phase: str, update: int) -> int:
        """The W&B step of an update: its index in the whole training, reward-model updates first."""
        return update - 1 + (self.cfg.rew_end_model.training.steps if phase == "actor_critic" else 0)

    def publish(self, row: Dict[str, Any]) -> None:
        if wandb.run is None:
            return
        step = self.wandb_step(row["phase"], row["update"])
        if step < self._wandb_next:  # replayed after a resume; the server holds this update already
            self.check_replayed(step, row)
            return
        wandb.log(wandb_row(row, self.cfg.agent.actor_critic.actor_distribution), step=step)
        self._wandb_next = step + 1

    def check_replayed(self, step: int, row: Dict[str, Any]) -> None:
        """A replayed update is not logged again; its values are compared with the local record of
        the interrupted segment (what was sent to the server)."""
        earlier = self._replayed.pop(step, None)
        outcome = "unverified" if earlier is None else "matched" if earlier == row else "mismatched"
        key = f"replayed_{outcome}"
        self._resume_event[key] = self._resume_event.get(key, 0) + 1
        wandb.summary["resume"] = self._resume_event  # kept here: W&B returns nested summary values flattened
        if outcome == "mismatched":
            say(f"W&B warning: replayed {row['phase']} update {row['update']} (W&B step {step}) differs from the "
                f"value the interrupted segment sent; the server keeps the earlier value")

    def start_wandb(self, resume: bool) -> None:
        """One W&B run per output (milestone x task x adaptation seed x member); its identity is kept
        in the work folder's wandb_run.json (published with the output), so a resume continues the
        same run and evaluate-only can add to it. Nothing with wandb.mode=disabled."""
        mode = OmegaConf.select(self.cfg, "wandb.mode", default="disabled")
        if mode == "disabled":
            return
        path_identity = self._work_dir / "wandb_run.json"
        identity = json.loads(path_identity.read_text()) if resume and path_identity.is_file() else None
        name, group = run_names(self.cfg, self.milestone.path, self.milestone.member_index,
                                milestone_members(self.milestone.manifest), self.milestone.manifest)
        init_wandb(self.cfg, identity, name, group, "adaptation", self._device,
                   config=dict(OmegaConf.to_container(self.cfg), provenance=self.provenance()))
        path_identity.write_text(json.dumps(run_identity(), indent=2))
        define_axes()
        if resume:
            self.reconcile_wandb(identity)  # reads the interrupted segment's status first
        begin_status("adaptation")

    def reconcile_wandb(self, identity: Optional[Dict[str, Any]]) -> None:
        """On resume. The checkpoint committed one W&B row per completed update. Committed rows the
        server lacks (lost with a crash) are backfilled from metrics.jsonl; updates the server holds
        beyond the checkpoint (logged before the crash, now replayed) are not logged twice. An offline
        segment of an existing run cannot know the server's rows: nothing is backfilled."""
        committed = sum(self.num_updates.values())
        rows = metrics_rows(self._work_dir / "metrics.jsonl")
        event = dict(count=sum(1 for r in rows if r.get("phase") == "resume"), from_phase=self.phase,
                     from_updates=dict(self.num_updates), committed_rows=committed,
                     previous_status=(wandb.summary.get("status") or NOT_RECORDED) if identity else "no earlier W&B run",
                     mode=self.cfg.wandb.mode)
        offline = bool(getattr(wandb.run, "offline", False))
        server = int(getattr(wandb.run, "starting_step", 0) or 0)
        by_step = {}
        for r in rows:  # the last record of each update (a replayed update repeats an earlier one)
            if r.get("phase") in ("rew_end_model", "actor_critic"):
                by_step[self.wandb_step(r["phase"], r["update"])] = r
        if offline and identity is not None:
            event["note"] = "offline segment: the server's rows are unknown; nothing backfilled"
            self._wandb_next = committed
        else:
            event["server_rows"] = server
            if server < committed:
                missing = [s for s in range(server, committed) if s not in by_step]
                if missing:
                    raise RuntimeError(f"metrics.jsonl lacks committed updates (W&B steps {missing[:5]}...); cannot backfill")
                for step in range(server, committed):
                    wandb.log(wandb_row(by_step[step], self.cfg.agent.actor_critic.actor_distribution), step=step)
                event["backfilled_rows"] = committed - server
                say(f"W&B: backfilled {committed - server} committed updates the server did not receive")
            elif server > committed:
                event["server_rows_beyond_checkpoint"] = server - committed
                self._replayed = {s: by_step.get(s) for s in range(committed, server)}
                say(f"W&B: {server - committed} updates after the checkpoint are on the server; "
                    "they are replayed without being logged again")
            self._wandb_next = max(server, committed)
        self._resume_event = event
        wandb.summary["resume"] = event

    def provenance(self) -> Dict[str, Any]:
        """What this adaptation run is (W&B config). Missing historical metadata is labelled, never guessed."""
        m, cfg, ac = self.milestone.manifest, self.cfg, self.agent.actor_critic
        exploration = m.get("exploration") or {}
        return dict(
            domain=m["domain"], task=self.milestone.task_name, adaptation_seed=cfg.downstream.seed,
            exploration=dict(method=exploration.get("method", NOT_RECORDED),
                             collection_mode=exploration.get("collection_mode", NOT_RECORDED),
                             intrinsic_reward_enabled=exploration.get("intrinsic_reward_enabled", NOT_RECORDED),
                             seed=exploration_seed(self.milestone.path, m),
                             run_folder=self.milestone.path.parent.parent.name),
            milestone=dict(name=self.milestone.path.name, path=str(self.milestone.path.resolve()),
                           collected_steps=m["collected_steps"], epoch=m.get("epoch"), identity=m["identity"],
                           replay_sha256=m["files"]["dataset"]["sha256"]),
            denoiser_member=dict(index=self.milestone.member_index, members=milestone_members(m), sha256=self.member_sha256),
            controller=dict(controller_settings(ac.cfg), actor_objective=ac.objective_settings(),
                            debug_metrics=bool(ac.debug_metrics)),
            reward_model=dict(debug_metrics=bool(self.agent.rew_end_model.debug_metrics)),
            budgets=dict(rew_end_model=OmegaConf.to_container(cfg.rew_end_model.training),
                         actor_critic=OmegaConf.to_container(cfg.actor_critic.training)),
            evaluation=evaluation_protocol(cfg.evaluation), source=self.source)

    def save(self) -> Path:
        torch.save(self.agent.rew_end_model.state_dict(), self._work_dir / "rew_end_model.pt")
        torch.save(self.agent.actor_critic.state_dict(), self._work_dir / "actor_critic.pt")
        manifest = self.milestone.manifest
        summary = dict(
            milestone=dict(path=Path(os.path.relpath(self.milestone.path, self.output_dir)).as_posix(),
                           identity=manifest["identity"], collected_steps=manifest["collected_steps"],
                           member_index=self.milestone.member_index,
                           member_sha256=state_sha256(self.milestone.denoiser_state)),
            task_name=self.milestone.task_name,
            task_index=self.milestone.task_index,
            seed=self.cfg.downstream.seed,
            num_updates=self.num_updates,
            files=dict(config="config.yaml", metrics="metrics.jsonl", rew_end_model="rew_end_model.pt",
                       actor_critic="actor_critic.pt"),
            note="Completed training: final weights. Evaluations are in evaluations/.",
        )
        write_json(self._work_dir / SUMMARY_FILE, summary)
        os.replace(self._work_dir, self.output_dir)
        shutil.rmtree(self.output_dir / "checkpoints", ignore_errors=True)  # training is complete
        return self.output_dir

    # ------------------------------------------------------------------ checkpoints

    def maybe_checkpoint(self, phase: str, steps: int) -> None:
        every, done = self.cfg.checkpointing.every, self.num_updates[phase]
        if every is not None and done % every == 0 and done < steps:  # phase ends save anyway
            self.save_checkpoint()

    def save_checkpoint(self) -> None:
        """Atomic: an interrupted write leaves the previous checkpoint usable."""
        state = dict(
            format=DOWNSTREAM_RESUME_FORMAT,
            phase=self.phase,
            num_updates=dict(self.num_updates),
            rew_end_model=self.agent.rew_end_model.state_dict(),
            actor_critic=self.agent.actor_critic.state_dict(),  # incl. prior and PMPO's private RNG
            rew_end_opt=self.rew_end_opt.state_dict(),
            rew_end_lr_sched=self.rew_end_lr_sched.state_dict(),
            actor_critic_opt=self.actor_critic_opt.state_dict(),
            rng=rng_state(),
            config=OmegaConf.to_container(self.cfg),
            # The frozen denoiser and replay are referenced, not copied. dataset.pt's checksum
            # (verified by load_milestone) pins every episode file's contents.
            milestone=dict(path=Path(os.path.relpath(self.milestone.path, self._work_dir)).as_posix(),
                           identity=self.milestone.manifest["identity"],
                           dataset_sha256=self.milestone.manifest["files"]["dataset"]["sha256"]),
            source=self.source,
        )
        self._path_checkpoint.parent.mkdir(exist_ok=True)
        save_atomically(state, self._path_checkpoint)

    def load_checkpoint(self) -> Dict[str, Any]:
        # Trusted, locally written checkpoint (RNG states, numpy arrays): weights_only=False.
        state = torch.load(self._path_checkpoint, map_location=self._device, weights_only=False)
        if state.get("format") == 1:
            raise ValueError(f"{self._path_checkpoint} predates the replay content checksum (format 1); "
                             f"it cannot be resumed faithfully. Start a new run.")
        if state.get("format") != DOWNSTREAM_RESUME_FORMAT:
            raise ValueError(f"{self._path_checkpoint} is not a downstream training checkpoint")
        check_same_packages(state["source"])
        if state["milestone"]["identity"] != self.milestone.manifest["identity"]:
            raise ValueError("Checkpoint was trained on a different milestone")
        if state["milestone"]["dataset_sha256"] != self.milestone.manifest["files"]["dataset"]["sha256"]:
            raise ValueError("Checkpoint was trained on a different milestone replay (dataset.pt checksum differs)")
        if (self._work_dir / state["milestone"]["path"]).resolve() != self.milestone.path.resolve():
            raise ValueError("Checkpoint refers to a milestone at a different place in the run folder")
        changed = changed_settings(normalized_saved_config(state["config"]), OmegaConf.to_container(self.cfg))
        if changed:
            raise ValueError(f"Cannot resume with different settings: {changed}")
        check_controller_state(state["actor_critic"].get("_extra_state"), self.agent.actor_critic.cfg,
                               source=f"{self._path_checkpoint}'s controller")  # before any weights
        self.agent.rew_end_model.load_state_dict(state["rew_end_model"])
        self.agent.actor_critic.load_state_dict(state["actor_critic"])
        self.rew_end_opt.load_state_dict(state["rew_end_opt"])
        self.rew_end_lr_sched.load_state_dict(state["rew_end_lr_sched"])
        self.actor_critic_opt.load_state_dict(state["actor_critic_opt"])
        self.phase, self.num_updates, self.source = state["phase"], dict(state["num_updates"]), state["source"]
        if self.phase != "rew_end_model":  # trained and frozen before the checkpoint
            self.agent.rew_end_model.requires_grad_(False).eval()
        return state["rng"]


def wandb_row(row: Dict[str, Any], actor_distribution: str) -> Dict[str, float]:
    """The W&B row of one metrics.jsonl update row: reward_model/train/{core,debug}/ or the accepted
    actor_critic/train/<group>/ paths (metric_path), with the update axis."""
    if row["phase"] == "rew_end_model":
        out = {f"reward_model/train/core/{k}": row[v] for k, v in REWARD_CORE.items() if v in row}
        if "prediction_mae" in row:  # recorded with rew_end_model.debug_metrics
            out.update({f"reward_model/train/debug/{k}": row[v] for k, v in REWARD_DEBUG.items() if v in row})
        out[REWARD_AXIS] = row["update"]
        return out
    out = {f"actor_critic/train/{metric_path(k, actor_distribution)}": v for k, v in row.items()
           if k not in ("phase", "update") and not k.startswith(UNPUBLISHED_POLICY_PREFIXES)}
    out[POLICY_AXIS] = row["update"]
    return out


def evaluation_row(result: Dict[str, Any], collected_steps: Optional[int]) -> Dict[str, float]:
    """eval/seed_<s>/return per configured seed and the saved aggregates, at the milestone's data budget."""
    row = {f"eval/seed_{e['seed']}/return": e["episode_return"] for e in result["episodes"]}
    row.update({f"eval/return_{k}": result[f"return_{k}"] for k in ("max", "median", "mean", "min", "std")})
    if collected_steps is not None:
        row[EVAL_AXIS] = collected_steps
    return row


def metrics_rows(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
        if path.is_file() else []


def run_names(cfg: DictConfig, milestone_path: Path, member: int, members: Optional[int],
              manifest: Optional[Dict[str, Any]] = None, domain: Optional[str] = None) -> Tuple[str, str]:
    """W&B name <prefix>-<milestone>-<task>-adaptation-seed<s>[-member<k>] and group
    <prefix>-<task>-adaptation-seed<s>[-member<k>]: the milestones of one exploration run, task, adaptation seed and
    member share the group. prefix: wandb.name, else <method>-<domain>-exploration-seed<N> (exploration_label).
    wandb.group replaces the group."""
    prefix = OmegaConf.select(cfg, "wandb.name") or exploration_label(Path(milestone_path).parent.parent, manifest, domain)
    suffix = f"-{cfg.downstream.task_name}-adaptation-seed{cfg.downstream.seed}" + (
        f"-member{member}" if member or (members or 1) > 1 else "")
    return f"{prefix}-{Path(milestone_path).name}{suffix}", OmegaConf.select(cfg, "wandb.group") or f"{prefix}{suffix}"


def exploration_label(run_dir: Path, manifest: Optional[Dict[str, Any]] = None, domain: Optional[str] = None) -> str:
    """<method>-<domain>-exploration-seed<N> of the exploration run: from its saved configuration
    (config/exploration.yaml); without it, from what the milestone manifest records (method, domain) or the
    given domain. What nothing records is "unrecorded"; folder names are never used."""
    try:
        identity = exploration_identity(run_dir)
        method, domain, seed = identity["method"], identity["domain"], identity["exploration_seed"]
    except FileNotFoundError:
        from exploration_trainer import method_label
        recorded = (manifest or {}).get("exploration") or {}
        method = method_label(True, recorded["method"]) if "method" in recorded else \
            "uniform" if recorded.get("intrinsic_reward_enabled") is False else None
        domain, seed = (manifest or {}).get("domain") or domain, None
    return (f"{method or 'unrecorded'}-{domain or 'unrecorded'}-exploration-"
            + (f"seed{seed}" if seed is not None else "seed-unrecorded"))


def init_wandb(cfg: DictConfig, identity: Optional[Dict[str, Any]], name: str, group: str, job_type: str,
               device: torch.device, config: Optional[Dict[str, Any]] = None) -> None:
    """wandb.init: a new run, or the run of `identity` (resume="allow"); system charts of the used GPU only."""
    w = cfg.wandb
    kwargs = dict(mode=w.mode, project=w.project, entity=w.entity, name=name, group=group, tags=w.tags, notes=w.notes,
                  job_type=job_type, config=config)
    if identity is not None:  # the existing run keeps its own name and group
        kwargs.update(id=identity["id"], project=identity.get("project") or w.project,
                      entity=identity.get("entity") or w.entity, resume="allow",
                      name=identity.get("name") or name, group=identity.get("group") or group)
    gpu = physical_gpu_index(device)
    if gpu is not None:  # W&B numbers GPUs like nvidia-smi
        kwargs["settings"] = wandb.Settings(x_stats_gpu_device_ids=[gpu])
    wandb.init(**kwargs)


def run_identity() -> Dict[str, Any]:
    r = wandb.run
    return dict(id=r.id, project=r.project, entity=r.entity, name=r.name, group=r.group, url=r.url)


def define_axes() -> None:
    wandb.define_metric(REWARD_AXIS, hidden=True)
    wandb.define_metric("reward_model/train/*", step_metric=REWARD_AXIS)
    wandb.define_metric(POLICY_AXIS)
    wandb.define_metric("actor_critic/train/*", step_metric=POLICY_AXIS)
    wandb.define_metric(EVAL_AXIS, hidden=True)
    wandb.define_metric("eval/*", step_metric=EVAL_AXIS)


def begin_status(activity: str) -> None:
    """The run's current status becomes running for this activity (adaptation or evaluate_only). A
    previous failure is kept as history (status_history/<UTC time>), and its error is no longer
    presented as the current one."""
    previous, error = wandb.summary.get("status"), wandb.summary.get("error")
    if previous == "failed":
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S.%fZ")
        wandb.summary[f"status_history/{stamp}"] = dict(
            status="failed", activity=wandb.summary.get("activity") or NOT_RECORDED,
            error=error if error is not None else NOT_RECORDED)
    if error is not None:
        del wandb.summary["error"]
    wandb.summary["status"] = "running"
    wandb.summary["activity"] = activity


def finish_wandb(status: str, error: Optional[BaseException] = None) -> None:
    """completed, or failed with the error as the current one (a killed process stays running)."""
    if wandb.run is None:
        return
    wandb.summary["status"] = status
    if error is not None:
        wandb.summary["error"] = repr(error)[:500]
    wandb.finish(exit_code=0 if error is None else 1)


def log_evaluation(result: Dict[str, Any], path: Path, collected_steps: Optional[int], kind: str,
                   total_updates: int, next_step: int) -> None:
    """One W&B row per evaluation file: after training at step total_updates, later evaluate-only
    ones after it (one step per earlier evaluation file of the output). The summary records what each
    evaluation was (after_training or evaluate_only) and its file."""
    if wandb.run is None:
        return
    earlier = sum(1 for p in Path(path).parent.glob("eval_*.json") if p != Path(path))
    step = max(next_step, int(getattr(wandb.run, "starting_step", 0) or 0), total_updates + earlier)
    wandb.log(evaluation_row(result, collected_steps), step=step)
    wandb.summary[f"evaluations/{Path(path).stem}"] = dict(
        kind=kind, step=step, file=str(Path(path).resolve()), seeds=result["seeds"], max_steps=result["max_steps"],
        train_env_steps=collected_steps if collected_steps is not None else NOT_RECORDED)


def exploration_seed(milestone_path: Path, manifest: Dict[str, Any]) -> Any:
    """The exploration seed from the run configuration the manifest refers to (run_config), if it is
    there and fixed; otherwise labelled. Milestone manifests do not record the seed themselves."""
    path = Path(milestone_path) / manifest.get("run_config", "")
    if not manifest.get("run_config") or not path.is_file():
        return NOT_RECORDED + " (the exploration run configuration is not available)"
    seed = OmegaConf.select(OmegaConf.load(path), "common.seed")
    return seed if isinstance(seed, int) else NOT_RECORDED + " (common.seed was null: drawn at random)"


def evaluation_protocol(c: DictConfig) -> Dict[str, Any]:
    return dict(seeds=list(c.seeds), max_steps=int(c.max_steps), after_training=bool(c.get("after_training", True)),
                policy="deterministic (mean action)", return_convention=RETURN_CONVENTION)


def recover_metrics_log(path: Path) -> Optional[Path]:
    """Repair a log whose last row a crash cut short; returns the recovery file, if any.
    Complete rows (including updates newer than the checkpoint) stay unchanged; the cut
    bytes are kept verbatim in metrics.jsonl.interrupted-<n>. Running it again is a no-op."""
    if not path.is_file():
        return None
    data = path.read_bytes()
    *rows, tail = data.split(b"\n")
    for number, row in enumerate(rows, start=1):
        try:
            json.loads(row)
        except ValueError:
            raise ValueError(f"{path}: row {number} is corrupt; not resuming over it") from None
    if not tail.strip():
        return None
    try:
        json.loads(tail)
    except ValueError:
        pass
    else:
        with open(path, "a", encoding="utf-8") as f:  # a complete record that lost only its newline
            f.write("\n")
        return None
    number = 1
    while path.with_name(f"{path.name}.interrupted-{number}").exists():
        number += 1
    recovery = path.with_name(f"{path.name}.interrupted-{number}")
    recovery.write_bytes(tail)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(data[:len(data) - len(tail)])
    tmp.replace(path)
    say(f"Recovered an interrupted metrics row into {recovery}")
    return recovery


def normalized_saved_config(saved: Dict[str, Any]) -> Dict[str, Any]:
    """A saved downstream config as the current code reads it: configs from before ensemble
    milestones have no initialization.member_index, which meant the only denoiser (member 0).
    Returns a copy; the saved file is not changed."""
    saved = deepcopy(saved)
    saved.setdefault("initialization", {}).setdefault("member_index", 0)
    return saved


def changed_settings(saved: Dict[str, Any], current: Dict[str, Any], prefix: str = "") -> List[str]:
    keys = []
    for key in sorted(set(saved) | set(current)):
        name = f"{prefix}{key}"
        if name in RESUME_MAY_CHANGE:
            continue
        a, b = saved.get(key), current.get(key)
        if isinstance(a, dict) and isinstance(b, dict):
            keys += changed_settings(a, b, f"{name}.")
        elif a != b:
            keys.append(name)
    return keys


def action_spec(domain_name: str, task_name: str) -> Dict[str, Any]:
    """Action dimension and bounds. The dm_control environment is built only to read its
    action spec, then closed; it is never reset or stepped."""
    spec_env = suite.load(domain_name, task_name)
    try:
        spec = spec_env.action_spec()
    finally:
        spec_env.close()
    return dict(continuous_action_dim=int(np.prod(spec.shape)), action_low=spec.minimum.tolist(),
                action_high=spec.maximum.tolist())


# ---------------------------------------------------------------------- evaluation

def check_evaluation_config(c: DictConfig) -> None:
    seeds = list(c.seeds)
    if not seeds or not all(isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"evaluation.seeds must be a non-empty list of distinct integers, got {seeds}")
    if not isinstance(c.max_steps, int) or c.max_steps < 1:
        raise ValueError(f"evaluation.max_steps must be a positive integer, got {c.max_steps!r}")


@torch.no_grad()
def evaluate_policy(model: PMPOBeta, env_cfg: DictConfig, task_name: str, seeds: List[int],
                    max_steps: int) -> Dict[str, Any]:
    """One episode per seed in the real `task_name`, with the milestone's image size,
    camera, action repeat and time limit, and deterministic (mean) PMPO actions. Nothing is
    stored or trained; global RNG states and the model's mode are restored afterwards."""
    rng = rng_state()
    modes = {name: module.training for name, module in model.named_modules()}  # mixed (e.g. frozen prior)
    model.eval()
    episodes = []
    pbar = Progress(total=len(seeds), desc=f"Evaluation episodes (seeds {seed_span(seeds, '-')})", unit="ep")
    try:
        for seed in seeds:
            env = DMControlEnv(env_cfg.domain_name, [task_name], size=env_cfg.size, camera_id=env_cfg.camera_id,
                               action_repeat=env_cfg.action_repeat, time_limit=env_cfg.time_limit, seed=int(seed))
            try:
                image, _ = env.reset(seed=int(seed))
                history = FrameHistory(frame_tensor(image, model.device), model.frame_stack)  # new episode
                total, terminated, truncated = 0.0, False, False
                for length in range(1, max_steps + 1):
                    action = model.sample_action(model.actor(history.state), deterministic=True)
                    image, reward, terminated, truncated, _ = env.step(action[0].cpu().numpy())
                    total += float(reward)
                    if terminated or truncated:
                        break
                    history.advance(frame_tensor(image, model.device), torch.tensor([False], device=model.device))
                episodes.append(dict(seed=int(seed), episode_return=total, length=length, terminated=bool(terminated),
                                     truncated=bool(truncated), step_limit=not (terminated or truncated)))
            finally:
                env.close()
            pbar.update(1)
    finally:
        pbar.close()
        for name, module in model.named_modules():
            module.training = modes[name]
        set_rng_state(rng)
    returns = np.array([e["episode_return"] for e in episodes])
    return dict(task_name=task_name, domain_name=env_cfg.domain_name, seeds=[int(s) for s in seeds],
                max_steps=int(max_steps), action_repeat=int(env_cfg.action_repeat), policy="deterministic (Beta mean)",
                return_convention=RETURN_CONVENTION, episodes=episodes,
                return_mean=float(returns.mean()), return_median=float(np.median(returns)),
                return_std=float(returns.std()), return_min=float(returns.min()), return_max=float(returns.max()))


def frame_tensor(image: np.ndarray, device: torch.device) -> torch.Tensor:
    # Same conversion as TorchEnv: uint8 HWC -> float in [-1, 1], NCHW.
    # Rendered frames can have negative strides; the vector env's np.stack copies them too.
    image = np.ascontiguousarray(image)[None]
    return torch.tensor(image, device=device).div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()


def evaluation_path(output_dir: Path, task_name: str, seeds: List[int], max_steps: int,
                    policy_state: Dict[str, Any]) -> Path:
    """evaluations/eval_<task>_policy-<weights hash>_protocol-<seeds and step limit hash>.json"""
    policy = state_sha256({k: v for k, v in policy_state.items() if torch.is_tensor(v)})[:12]
    protocol = hashlib.sha256(repr(([int(s) for s in seeds], int(max_steps))).encode()).hexdigest()[:12]
    return Path(output_dir) / "evaluations" / f"eval_{task_name}_policy-{policy}_protocol-{protocol}.json"


def save_evaluation(output_dir: Path, result: Dict[str, Any], policy_state: Dict[str, Any]) -> Path:
    """Write to evaluations/, named by policy weights and protocol; never overwritten."""
    path = evaluation_path(output_dir, result["task_name"], result["seeds"], result["max_steps"], policy_state)
    policy = path.stem.split("_policy-")[1].split("_protocol-")[0]
    if path.exists():
        raise FileExistsError(f"Evaluation {path} already exists; not overwriting it")
    path.parent.mkdir(exist_ok=True)
    write_json(path, dict(result, policy_sha256_prefix=policy))
    return path


def seed_span(seeds: List[int], dash: str) -> str:
    """Evaluation seeds as shown in the terminal: "3001-3030" for a consecutive run, else listed."""
    seeds = [int(s) for s in seeds]
    if len(seeds) > 1 and seeds == list(range(seeds[0], seeds[0] + len(seeds))):
        return f"{seeds[0]}{dash}{seeds[-1]}"
    return ", ".join(map(str, seeds))


def report_evaluation(result: Dict[str, Any], path: Path) -> None:
    """The terminal summary of an evaluation (per-seed returns stay in its file)."""
    values = " | ".join(f"{name} {result[f'return_{name}']:.2f}" for name in ("max", "median", "mean", "min", "std"))
    say(f"Evaluation summary: {values}")
    say(f"Evaluation saved: > {Path(path).resolve()}")


def evaluate_saved_output(cfg: DictConfig) -> Path:
    """Evaluation-only: evaluate the final policy of a completed downstream output (also
    weights-only outputs from before checkpoints existed)."""
    check_evaluation_config(cfg.evaluation)
    task_name, seed = cfg.downstream.task_name, cfg.downstream.seed
    output_dir = downstream_output_dir(Path(cfg.initialization.milestone_path), task_name, seed,
                                       OmegaConf.select(cfg, "initialization.member_index", default=0))
    if downstream_summary(output_dir) is None:
        raise FileNotFoundError(f"No completed downstream output at {output_dir} (no {SUMMARY_FILE})")
    saved = OmegaConf.load(output_dir / "config.yaml")
    use_cuda = cfg.common.devices != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    model = PMPOBeta(instantiate(saved.agent.actor_critic, **action_spec(saved.env.train.domain_name, task_name)))
    # Trusted, locally written weights (PMPO's RNG state is numpy objects): weights_only=False.
    model.load_state_dict(torch.load(output_dir / "actor_critic.pt", map_location=device, weights_only=False))
    model.to(device)
    # Refused before anything runs (and before W&B): an existing evaluation is never a failed run.
    existing = evaluation_path(output_dir, task_name, cfg.evaluation.seeds, cfg.evaluation.max_steps, model.state_dict())
    if existing.exists():
        raise FileExistsError(f"Evaluation {existing} already exists; not overwriting it")
    logged = None
    if OmegaConf.select(cfg, "wandb.mode", default="disabled") != "disabled":
        logged = start_saved_evaluation(cfg, output_dir, saved, device)  # status running
    try:
        result = evaluate_policy(model, saved.env.train, task_name, cfg.evaluation.seeds, cfg.evaluation.max_steps)
        path = save_evaluation(output_dir, result, model.state_dict())
        report_evaluation(result, path)
        if logged is not None:
            log_evaluation(result, path, logged["steps"], "evaluate_only", logged["total"], 0)
            wandb.summary["last_evaluate_only"] = dict(file=str(Path(path).resolve()), seeds=result["seeds"],
                                                       max_steps=result["max_steps"])
    except BaseException as error:
        finish_wandb("failed", error)
        raise
    finish_wandb("completed")
    return path


def start_saved_evaluation(cfg: DictConfig, output_dir: Path, saved: DictConfig, device: torch.device) -> Dict[str, Any]:
    """Evaluate-only to W&B: the output's adaptation run when its identity (wandb_run.json) exists,
    else a new run of job type evaluate-only; its status becomes running (begin_status). Everything
    comes from the completed output. Returns the evaluation row's axis value and update count."""
    summary = downstream_summary(output_dir) or {}
    recorded = summary.get("milestone") or {}
    steps = recorded.get("collected_steps")
    path_identity = output_dir / "wandb_run.json"
    identity = json.loads(path_identity.read_text()) if path_identity.is_file() else None
    milestone_path = Path(cfg.initialization.milestone_path)
    member = recorded.get("member_index", 0)
    name, group = run_names(cfg, milestone_path, member, None, domain=saved.env.train.domain_name)
    config = None
    if identity is None:
        config = dict(config=OmegaConf.to_container(saved), provenance=dict(
            domain=saved.env.train.domain_name, task=summary.get("task_name"), adaptation_seed=summary.get("seed"),
            exploration=dict(method=NOT_RECORDED + " in the saved output", seed=NOT_RECORDED + " in the saved output"),
            milestone=dict(name=milestone_path.name, collected_steps=steps if steps is not None else NOT_RECORDED,
                           identity=recorded.get("identity", NOT_RECORDED)),
            denoiser_member=dict(index=member, sha256=recorded.get("member_sha256", NOT_RECORDED)),
            output=str(output_dir.resolve()), evaluation=evaluation_protocol(cfg.evaluation),
            note="evaluate-only run: the output has no W&B adaptation run (wandb_run.json)"))
    init_wandb(cfg, identity, name + ("" if identity else "-evaluate-only"), group, "evaluate-only", device, config)
    define_axes()
    begin_status("evaluate_only")
    return dict(steps=steps, total=sum((summary.get("num_updates") or {}).values()))


# ---------------------------------------------------------------------- results (CPU only)
# downstream_main.py downstream.mode=collect: one exploration run's completed downstream outputs for one task,
# adaptation seed and denoiser member -> <run>/result/<task>/seed_<s>[_member_<k>]/result.csv (one row per
# milestone). downstream.mode=compare: the result.csv of every method's run of one domain and exploration seed
# -> comparison.csv, analysis.json and plots. Nothing is trained or evaluated; source files are only read.

THRESHOLDS = (200, 400, 500, 600)  # fractions of evaluation episodes with return strictly above each
METHOD_COLORS = {"uniform": "#7f7f7f", "lcg": "#d62728", "full-ensemble": "#1f77b4", "laplace-ensemble": "#9467bd",
                 "plan2explore": "#2ca02c"}
# Marker and line style per method too, so curves that coincide (e.g. all at 0) stay distinguishable.
METHOD_STYLES = {"uniform": ("o", "-"), "lcg": ("s", "--"), "full-ensemble": ("^", "-."),
                 "laplace-ensemble": ("D", ":"), "plan2explore": ("v", "--")}
RESULT_COLUMNS = ["method", "domain", "exploration_seed", "task", "adaptation_seed", "member_index", "milestone",
                  "train_env_steps", "num_eval_episodes", "return_mean", "return_median", "return_std", "return_min",
                  "return_max", *(f"frac_over_{t}" for t in THRESHOLDS), "protocol", "evaluation_max_steps",
                  "compatibility_sha256"]  # then return_seed_<s> per evaluation seed, then source_evaluation_file
# What must match for results to be compared (compatibility_settings), as effective (resolved) values: the
# environment the policy was trained and evaluated in (domain, tasks, action repeat, time limit, camera, image
# size), the controller and its imagination, both training phases, and the evaluation's environment, policy and
# return convention. Not compared: paths, run identity (seeds and task are the selection itself), checkpoint
# cadence, W&B and the diagnostics-only switches below. The column was downstream_settings_sha256 before the
# environment and evaluation were included; tables with that column must be collected again.
COMPARED_SETTINGS = ("env.train", "agent", "world_model_env", "rew_end_model.training", "rew_end_model.optimizer",
                     "actor_critic.training")
NOT_COMPARED = ("agent.actor_critic.debug_metrics", "agent.actor_critic.log_diagnostics")
COMPARED_REPORT_FIELDS = ("domain_name", "task_name", "action_repeat", "policy", "return_convention")


def exploration_identity(run_dir: Path) -> Dict[str, Any]:
    """Method label, domain and exploration seed of an exploration run, from its saved configuration
    (config/exploration.yaml), never from folder names."""
    from exploration_trainer import method_label
    path = Path(run_dir) / "config" / "exploration.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: not an exploration run folder")
    cfg = OmegaConf.load(path)
    seed = OmegaConf.select(cfg, "common.seed")
    return dict(method=method_label(bool(OmegaConf.select(cfg, "intrinsic_reward.enabled", default=False)),
                                    OmegaConf.select(cfg, "intrinsic_reward.method", default=None)),
                domain=OmegaConf.select(cfg, "env.train.domain_name"), exploration_seed=seed)


def result_dir(run_dir: Path, task_name: str, seed: int, member_index: int = 0) -> Path:
    """<run>/result/<task>/seed_<s>[_member_<k>] (member suffix as for downstream outputs)."""
    return Path(run_dir) / "result" / task_name / (f"seed_{seed}" + (f"_member_{member_index}" if member_index else ""))


def protocol_id(path: Path) -> str:
    return Path(path).stem.rsplit("_protocol-", 1)[1]


def compatibility_settings(output_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    """The settings that make two results comparable (COMPARED_SETTINGS), resolved, from the output's saved
    config.yaml, and the evaluation report's environment, policy and return convention."""
    saved = OmegaConf.load(Path(output_dir) / "config.yaml")
    settings = {}
    for key in COMPARED_SETTINGS:
        value = OmegaConf.select(saved, key)
        settings[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    for dotted in NOT_COMPARED:
        parent, _, leaf = dotted.rpartition(".")
        node = settings.get(parent.split(".", 1)[0], {})
        for part in parent.split(".")[1:]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(leaf, None)
    settings["evaluation"] = {k: report.get(k) for k in COMPARED_REPORT_FIELDS}
    return settings


def compatibility_sha256(settings: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and value:
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def settings_difference(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    fa, fb = flatten(a), flatten(b)
    keys = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    return "; ".join(f"{k}: {fa.get(k)!r} vs {fb.get(k)!r}" for k in keys[:8]) + (" ..." if len(keys) > 8 else "")


def check_compatible(sources: List[Tuple[str, Dict[str, Any]]], what: str) -> None:
    """(label, compatibility settings) of every result: an error naming the differing settings unless all match."""
    first_label, first = sources[0]
    for label, settings in sources[1:]:
        if compatibility_sha256(settings) != compatibility_sha256(first):
            raise ValueError(f"{what}: not comparable: {first_label} and {label} differ in "
                             f"{settings_difference(first, settings)}")


def result_row(report_path: Path, report: Dict[str, Any], identity: Dict[str, Any], summary: Dict[str, Any],
               output_dir: Path, run_dir: Path) -> Dict[str, Any]:
    """One milestone's row: statistics recomputed from the episode returns (the saved aggregates are checked
    against them), strict threshold fractions and every evaluation seed's return."""
    seeds = [int(s) for s in report["seeds"]]
    episodes = report["episodes"]
    if [int(e["seed"]) for e in episodes] != seeds:
        raise ValueError(f"{report_path}: episode seeds {[e['seed'] for e in episodes]} differ from the protocol {seeds}")
    expected = hashlib.sha256(repr((seeds, int(report["max_steps"]))).encode()).hexdigest()[:12]
    if expected != protocol_id(report_path):
        raise ValueError(f"{report_path}: its seeds and step limit do not match the protocol in its name")
    returns = np.array([float(e["episode_return"]) for e in episodes])
    stats = dict(return_mean=float(returns.mean()), return_median=float(np.median(returns)),
                 return_std=float(returns.std()), return_min=float(returns.min()), return_max=float(returns.max()))
    for key, value in stats.items():
        if not np.isclose(value, float(report[key]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{report_path}: saved {key} {report[key]} differs from the episodes' {value}")
    row = dict(identity, task=summary["task_name"], adaptation_seed=summary["seed"],
               member_index=summary["milestone"].get("member_index", 0), milestone=output_dir.parent.parent.name,
               train_env_steps=summary["milestone"]["collected_steps"], num_eval_episodes=len(returns), **stats,
               **{f"frac_over_{t}": float((returns > t).mean()) for t in THRESHOLDS},
               protocol=protocol_id(report_path), evaluation_max_steps=int(report["max_steps"]),
               compatibility_sha256=compatibility_sha256(compatibility_settings(output_dir, report)))
    row.update({f"return_seed_{s}": float(r) for s, r in zip(seeds, returns)})
    row["source_evaluation_file"] = Path(os.path.relpath(report_path, run_dir)).as_posix()
    return row


def collect_results(run_dir: Path, task_name: str, seed: int, member_index: int = 0,
                    protocol: Optional[str] = None) -> Tuple[Optional[Path], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every milestone of the run, in collected-step order: its completed downstream output's evaluation of
    the one (selected) protocol -> result.csv. Returns (csv path or None, rows, missing). Different
    protocols or downstream settings are never mixed; ambiguity is an error asking for a selection."""
    from data.milestone import discover_milestones
    run_dir = Path(run_dir)
    identity = exploration_identity(run_dir)
    found, missing = [], []
    for path, manifest in discover_milestones(run_dir / "milestones"):
        output = downstream_output_dir(path, task_name, seed, member_index)
        summary = downstream_summary(output)
        entry = dict(milestone=path.name, train_env_steps=manifest["collected_steps"])
        if summary is None:
            work = output.with_name(f".{output.name}.tmp")
            missing.append(dict(entry, reason="incomplete (work folder " + work.name + " only)" if work.exists()
                                else "no downstream output"))
            continue
        if summary["milestone"]["collected_steps"] != manifest["collected_steps"] or \
                summary["milestone"].get("identity") != manifest["identity"]:
            raise ValueError(f"{output}: its summary does not match milestone {path.name}")
        reports = {}  # protocol -> every report of it (a completed output holds one policy)
        for p in sorted((output / "evaluations").glob("eval_*.json")):
            reports.setdefault(protocol_id(p), []).append(p)
        found.append((entry, output, summary, reports))
    protocols = sorted({p for *_, reports in found for p in reports})
    if protocol is None and len(protocols) > 1:
        raise ValueError(f"{run_dir}: several evaluation protocols ({', '.join(protocols)}); select one with "
                         "analysis.protocol=<id> (the eval_*_protocol-<id>.json suffix)")
    chosen = protocol if protocol is not None else (protocols[0] if protocols else None)
    selected = []
    for entry, output, summary, reports in found:
        if chosen not in reports:
            missing.append(dict(entry, reason=f"completed, but no evaluation with protocol {chosen}"))
            continue
        if len(reports[chosen]) > 1:  # never chosen by name, time or return
            raise ValueError(f"{output}: {len(reports[chosen])} evaluation reports of different policies with protocol "
                             f"{chosen}: {', '.join(p.name for p in reports[chosen])}; a completed output has one final "
                             "policy, so this is ambiguous. Nothing was collected.")
        selected.append((entry, output, summary, reports[chosen][0]))
    rows, sources = [], []
    for entry, output, summary, path in selected:
        report = read_json(path, "evaluation report")
        rows.append(result_row(path, report, identity, summary, output, run_dir))
        sources.append((Path(os.path.relpath(path, run_dir)).as_posix(), compatibility_settings(output, report)))
    if rows:
        check_compatible(sources, f"{run_dir}")
    if not rows:
        return None, rows, missing
    rows.sort(key=lambda r: r["train_env_steps"])
    seed_columns = [k for k in rows[0] if k.startswith("return_seed_")]
    path = result_dir(run_dir, task_name, seed, member_index) / "result.csv"
    write_csv(path, rows, RESULT_COLUMNS + seed_columns + ["source_evaluation_file"])
    return path, rows, missing


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    import csv
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    import csv
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key.startswith(("return_", "frac_over_")):
                row[key] = float(value)
            elif key in ("exploration_seed", "adaptation_seed", "member_index", "train_env_steps", "num_eval_episodes",
                         "evaluation_max_steps"):
                row[key] = int(value)
    return rows


def compare_methods(output_root: Path, domain: str, exploration_seed: int, task_name: str, seed: int,
                    member_index: int = 0, protocol: Optional[str] = None) -> Path:
    """Every method's run outputs/<method>/<domain>/seed_<exploration seed> with a result.csv for the task,
    adaptation seed and member: comparison.csv, analysis.json, task_return.{png,pdf} and
    fraction_over_thresholds.{png,pdf}. Methods come from each run's saved configuration. Rows of different
    evaluation protocols or downstream settings are not compared; missing milestones are recorded, never
    interpolated."""
    output_root = Path(output_root)
    comparison_dir = output_root / "comparisons" / domain / f"exploration_seed_{exploration_seed}" / task_name / \
        f"adaptation_seed_{seed}"
    runs, missing, inputs, rows = {}, [], [], []
    for run_dir in sorted(output_root.glob(f"*/{domain}/seed_{exploration_seed}")):
        if not (run_dir / "config" / "exploration.yaml").is_file():
            continue
        identity = exploration_identity(run_dir)
        if (identity["domain"], identity["exploration_seed"]) != (domain, exploration_seed):
            raise ValueError(f"{run_dir}: its configuration records {identity['domain']}, seed "
                             f"{identity['exploration_seed']}; not {domain}, seed {exploration_seed}")
        if identity["method"] in runs:
            raise ValueError(f"Two runs record method {identity['method']}: {runs[identity['method']]} and {run_dir}")
        runs[identity["method"]] = run_dir
        csv_path = result_dir(run_dir, task_name, seed, member_index) / "result.csv"
        if not csv_path.is_file():
            missing.append(dict(method=identity["method"], run=str(run_dir), reason=f"no {csv_path.relative_to(run_dir)} "
                                "(run downstream_main.py downstream.mode=collect)"))
            continue
        method_rows = read_csv(csv_path)
        if method_rows and "compatibility_sha256" not in method_rows[0]:
            raise ValueError(f"{csv_path} was collected before the compatibility fingerprint covered the environment and "
                             "evaluation settings; collect it again (downstream_main.py downstream.mode=collect)")
        for row in method_rows:
            if (row["method"], row["domain"], row["exploration_seed"], row["task"], row["adaptation_seed"],
                    row["member_index"]) != (identity["method"], domain, exploration_seed, task_name, seed, member_index):
                raise ValueError(f"{csv_path}: a row of {row['method']}/{row['task']}/seed {row['adaptation_seed']}/"
                                 f"member {row['member_index']} does not belong to this selection")
        steps = [r["train_env_steps"] for r in method_rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{csv_path}: duplicate milestones {sorted(s for s in steps if steps.count(s) > 1)}")
        evaluations = [run_dir / r["source_evaluation_file"] for r in method_rows]
        inputs.append(dict(method=identity["method"], run=str(run_dir.resolve()), result_csv=str(csv_path.resolve()),
                           result_csv_sha256=file_sha256(csv_path), evaluations={str(p.resolve()): file_sha256(p)
                                                                                for p in evaluations}))
        rows += method_rows
    if not rows:
        raise ValueError(f"No result.csv under {output_root}/*/{domain}/seed_{exploration_seed} for {task_name}, "
                         f"adaptation seed {seed}, member {member_index}; missing: {missing}")
    protocols = sorted({r["protocol"] for r in rows})
    if protocol is not None:
        rows = [r for r in rows if r["protocol"] == protocol]
        missing += [dict(method=m, reason=f"no rows with protocol {protocol}") for m in runs
                    if not any(r["method"] == m for r in rows)]
    elif len(protocols) > 1:
        raise ValueError(f"The methods' results use different evaluation protocols ({', '.join(protocols)}); "
                         "select one with analysis.protocol=<id>")
    if len({r["compatibility_sha256"] for r in rows}) > 1:  # name the differing settings
        sources = []
        for r in rows:
            report_path = runs[r["method"]] / r["source_evaluation_file"]
            sources.append((f"{r['method']} {r['milestone']}", compatibility_settings(
                report_path.parent.parent, read_json(report_path, "evaluation report"))))
        check_compatible(sources, "The methods' results")
        raise ValueError("The methods' results have different compatibility fingerprints; collect them again")
    seed_columns = sorted({k for r in rows for k in r if k.startswith("return_seed_")})
    if any(set(k for k in r if k.startswith("return_seed_")) != set(seed_columns) for r in rows):
        raise ValueError("Rows with different evaluation seeds; not comparing them")
    methods = [m for m in METHOD_COLORS if any(r["method"] == m for r in rows)]
    # Every known milestone of the selected methods and every observed row: a milestone that no method has a
    # result for still breaks the curves instead of being skipped.
    known = {m["collected_steps"] for method in methods for _, m in discover_milestones_safe(runs[method] / "milestones")}
    grid = sorted(known | {r["train_env_steps"] for r in rows})
    for method in methods:
        have = {r["train_env_steps"] for r in rows if r["method"] == method}
        for steps in sorted(set(grid) - have):
            missing.append(dict(method=method, train_env_steps=steps, reason="no result row"))
    rows.sort(key=lambda r: (methods.index(r["method"]), r["train_env_steps"]))
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_dir / "comparison.csv", rows, RESULT_COLUMNS + seed_columns + ["source_evaluation_file"])
    first = rows[0]
    eval_seeds = [int(k.rsplit("_", 1)[1]) for k in seed_columns]
    analysis = dict(
        selection=dict(output_root=str(output_root.resolve()), domain=domain, exploration_seed=exploration_seed,
                       task=task_name, adaptation_seed=seed, member_index=member_index, protocol=protocol),
        evaluation_protocol=dict(protocol=first["protocol"], seeds=eval_seeds, max_steps=first["evaluation_max_steps"],
                                 num_eval_episodes=first["num_eval_episodes"]),
        compatibility_sha256=first["compatibility_sha256"], plot_grid=grid, methods=methods, inputs=inputs,
        missing=missing,
        note="One exploration seed and one adaptation seed per method. return_* and frac_over_* summarize the "
             "evaluation episodes (seeds) of one adapted policy per milestone; their spread is not uncertainty over "
             "training seeds. Missing milestones are not interpolated.",
        outputs=["comparison.csv", "task_return.png", "task_return.pdf", "fraction_over_thresholds.png",
                 "fraction_over_thresholds.pdf"])
    write_json(comparison_dir / "analysis.json", analysis)
    plot_comparison(comparison_plot_spec(rows, methods, grid, analysis, comparison_dir))
    return comparison_dir


def discover_milestones_safe(milestones_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    from data.milestone import discover_milestones
    return discover_milestones(milestones_dir) if Path(milestones_dir).is_dir() else []


# The figures, drawn in a separate Python process that loads matplotlib but not torch (as device_report's
# EGL probe runs in its own process): in this project's Windows conda environment torch's bundled OpenMP
# runtime and the one matplotlib's numpy work loads abort a process holding both ("OMP: Error #15").
PLOT_PROGRAM = r"""
import json, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
spec = json.loads(sys.stdin.read())
grid, out = spec["grid"], spec["out"]
nan = lambda values: [math.nan if v is None else v for v in values]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for method in spec["methods"]:
    marker, line = spec["styles"][method]
    ax.plot(grid, nan(spec["series"][method]["return_mean"]), marker=marker, linestyle=line, color=spec["colors"][method],
            label=method, alpha=0.85)
ax.set_xlabel("Exploration training environment steps")
ax.set_ylabel("Mean downstream task return")
ax.set_title(spec["title"] + "\n" + spec["subtitle"], fontsize=9)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(f"{out}/task_return.{suffix}", dpi=150)
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
for ax, t in zip(axes.flat, spec["thresholds"]):
    for method in spec["methods"]:
        marker, line = spec["styles"][method]
        ax.plot(grid, nan(spec["series"][method][f"frac_over_{t}"]), marker=marker, linestyle=line,
                color=spec["colors"][method], label=method, alpha=0.85)
    ax.set_title(f"Fraction of evaluation episodes with return > {t}", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("Exploration training environment steps")
for ax in axes[:, 0]:
    ax.set_ylabel("Fraction")
axes[0, 0].legend()
fig.suptitle(spec["subtitle"], fontsize=9)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(f"{out}/fraction_over_thresholds.{suffix}", dpi=150)
"""


def comparison_plot_spec(rows: List[Dict[str, Any]], methods: List[str], grid: List[int], analysis: Dict[str, Any],
                         out: Path) -> Dict[str, Any]:
    """The data both figures plot: per method and value, one entry per grid step, None where the method has no
    result (a gap, never interpolated)."""
    s, p = analysis["selection"], analysis["evaluation_protocol"]
    seeds = p["seeds"]
    seed_text = f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else ", ".join(map(str, seeds))
    series = {}
    for method in methods:
        by_steps = {r["train_env_steps"]: r for r in rows if r["method"] == method}
        series[method] = {key: [by_steps[x][key] if x in by_steps else None for x in grid]
                          for key in ["return_mean", *(f"frac_over_{t}" for t in THRESHOLDS)]}
    spec = dict(grid=grid, out=str(Path(out).resolve()), methods=methods, colors=METHOD_COLORS, styles=METHOD_STYLES,
                series=series,
                thresholds=list(THRESHOLDS), title=f"Downstream {s['task']} return",
                subtitle=(f"{s['domain']} {s['task']} | exploration seed {s['exploration_seed']} | adaptation seed "
                          f"{s['adaptation_seed']}\n{p['num_eval_episodes']} evaluation episodes per point (seeds "
                          f"{seed_text}, <= {p['max_steps']} steps); no interpolation across missing milestones"))
    return spec


def plot_comparison(spec: Dict[str, Any]) -> None:
    """task_return: mean return per milestone, one curve per method; fraction_over_thresholds: four panels.
    Drawn by PLOT_PROGRAM from comparison_plot_spec."""
    import subprocess
    import sys
    result = subprocess.run([sys.executable, "-c", PLOT_PROGRAM], input=json.dumps(spec), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Plotting failed:\n{result.stderr[-3000:]}")
