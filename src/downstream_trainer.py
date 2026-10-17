"""Downstream training on a frozen exploration milestone (Step 5, first increment).

Phase 1 trains a fresh reward/end model on the selected task's recorded rewards. Phase 2
freezes it and trains a fresh PMPO policy in imagination on its predicted task reward. The
milestone's denoiser stays frozen throughout. Outputs go to the reserved
`<run>/downstream/steps_N/<task>/seed_<seed>/`; they are model weights, not resumable
checkpoints.
"""
from copy import deepcopy
import json
import os
from pathlib import Path
from typing import Any, Dict

from dm_control import suite
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import trange
import yaml

from agent import Agent
from data import BatchSampler, collate_segments_without_rewards
from data.milestone import downstream_output_dir, load_milestone
from data.utils import make_task_reward_collate
from envs import TIME_LIMIT_ONLY_DOMAINS, WorldModelEnv
from models.pmpo_beta import PMPOOptimizers
from utils import configure_opt, get_lr_sched, set_seed


class DownstreamTrainer:
    def __init__(self, cfg: DictConfig) -> None:
        for name in ("rew_end_model", "actor_critic"):
            steps = cfg[name].training.steps
            if not isinstance(steps, int) or steps < 1:
                raise ValueError(f"{name}.training.steps must be a positive integer, got {steps!r}")
        task_name, seed = cfg.downstream.task_name, cfg.downstream.seed
        set_seed(seed)  # downstream randomness comes only from its own seed
        use_cuda = cfg.common.devices != "cpu" and torch.cuda.is_available()
        self._device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
        print(f"Starting on {self._device}")

        # The exact milestone: validated denoiser, replay versions and task labels.
        self.milestone = load_milestone(cfg.initialization.milestone_path, task_name, map_location=self._device,
                                        cache_in_ram=True)
        manifest = self.milestone.manifest
        if manifest["domain"] not in TIME_LIMIT_ONLY_DOMAINS:
            raise ValueError(f"Domain {manifest['domain']!r} is not verified to end only at its time limit")
        if self.milestone.dataset.counts_end[1] > 0:
            raise ValueError("Milestone replay has true terminations; downstream supports time-limit-only tasks")

        # The milestone's own env and denoiser configuration define the models.
        cfg = deepcopy(cfg)
        with open_dict(cfg):
            cfg.env.train = manifest["model"]["env"]
            cfg.agent.denoiser = manifest["model"]["denoiser"]
        OmegaConf.resolve(cfg)
        self.cfg = cfg

        # Outputs: never overwrite; written aside and renamed when complete.
        self.output_dir = downstream_output_dir(self.milestone.path, task_name, seed)
        self._work_dir = self.output_dir.with_name(f".{self.output_dir.name}.tmp")
        if self.output_dir.exists():
            raise FileExistsError(f"Downstream output {self.output_dir} already exists; not overwriting it")
        if self._work_dir.exists():
            raise FileExistsError(f"Incomplete downstream output {self._work_dir} from an earlier attempt; delete it to rerun")
        self._work_dir.mkdir(parents=True)
        OmegaConf.save(cfg, self._work_dir / "config.yaml")

        # Models: frozen milestone denoiser, fresh reward/end model and PMPO.
        # One dm_control environment is built only to read the action spec; it is never reset
        # or stepped. All training data comes from the milestone.
        spec_env = suite.load(manifest["domain"], task_name)
        try:
            spec = spec_env.action_spec()
        finally:
            spec_env.close()
        self.agent = Agent(instantiate(cfg.agent, continuous_action_dim=int(np.prod(spec.shape)),
                                       action_low=spec.minimum.tolist(), action_high=spec.maximum.tolist()))
        self.agent.to(self._device)
        self.agent.denoiser.load_state_dict(self.milestone.denoiser_state)
        self.agent.denoiser.requires_grad_(False).eval()
        self.rew_end_opt = configure_opt(self.agent.rew_end_model, **cfg.rew_end_model.optimizer)
        self.rew_end_lr_sched = get_lr_sched(self.rew_end_opt, cfg.rew_end_model.training.lr_warmup_steps)
        self.actor_critic_opt = PMPOOptimizers(self.agent.actor_critic, cfg.actor_critic.training.lr_warmup_steps)

        # Replay sampling over the milestone's fixed dataset (see BatchSampler): reward-model
        # windows may pad on both sides; imagination prompts pad only before an episode start.
        dataset = self.milestone.dataset
        c = cfg.rew_end_model.training
        sampler = BatchSampler(dataset, 0, 1, c.batch_size, c.seq_length, c.sample_weights, can_sample_beyond_end=True)
        collate = make_task_reward_collate(self.milestone.task_index, self.milestone.successor_observations)
        self._reward_batches = iter(DataLoader(dataset, batch_sampler=sampler, collate_fn=collate))
        c = cfg.actor_critic.training
        sampler = BatchSampler(dataset, 0, 1, c.batch_size, cfg.agent.denoiser.inner_model.num_steps_conditioning,
                               c.sample_weights)
        prompts = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_segments_without_rewards)
        # Predicted task reward; no end sampling: imagined episodes end by horizon truncation.
        self.rl_env = WorldModelEnv(self.agent.denoiser, self.agent.rew_end_model, prompts,
                                    instantiate(cfg.world_model_env), sample_ends=False)
        self.agent.actor_critic.setup_training(self.rl_env)

        self.num_updates = dict(rew_end_model=0, actor_critic=0)

    def run(self) -> None:
        self.train_reward_model()
        self.train_policy()
        self.save()

    def train_reward_model(self) -> None:
        model, c = self.agent.rew_end_model, self.cfg.rew_end_model.training
        model.train()
        for _ in trange(c.steps, desc="Training rew_end_model (task reward)"):
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
        # Frozen from here: PMPO only reads its predictions.
        model.requires_grad_(False).eval()

    def train_policy(self) -> None:
        model = self.agent.actor_critic
        model.train()
        for _ in trange(self.cfg.actor_critic.training.steps, desc="Training actor_critic (PMPO, predicted task reward)"):
            self.actor_critic_opt.zero_grad()
            loss, metrics = model()
            loss.backward()
            metrics.update(self.actor_critic_opt.step())
            self.num_updates["actor_critic"] += 1
            self.log("actor_critic", metrics)

    def log(self, phase: str, metrics: Dict[str, Any]) -> None:
        row = dict(phase=phase, update=self.num_updates[phase], **{k: float(v) for k, v in metrics.items()})
        with open(self._work_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

    def save(self) -> Path:
        torch.save(self.agent.rew_end_model.state_dict(), self._work_dir / "rew_end_model.pt")
        torch.save(self.agent.actor_critic.state_dict(), self._work_dir / "actor_critic.pt")
        manifest = self.milestone.manifest
        summary = dict(
            milestone=dict(path=Path(os.path.relpath(self.milestone.path, self.output_dir)).as_posix(),
                           identity=manifest["identity"], collected_steps=manifest["collected_steps"]),
            task_name=self.milestone.task_name,
            task_index=self.milestone.task_index,
            seed=self.cfg.downstream.seed,
            num_updates=self.num_updates,
            files=dict(config="config.yaml", metrics="metrics.jsonl", rew_end_model="rew_end_model.pt",
                       actor_critic="actor_critic.pt"),
            note="Model weights only; not a resumable checkpoint. Real-environment evaluation is not run yet.",
        )
        with open(self._work_dir / "summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        os.replace(self._work_dir, self.output_dir)
        print(f"Downstream outputs: {self.output_dir}")
        return self.output_dir
