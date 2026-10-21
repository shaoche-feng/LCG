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
import os
from pathlib import Path
import shutil
from typing import Any, Dict, List, Optional

from dm_control import suite
from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf, open_dict
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import yaml

from agent import Agent
from coroutines.frame_history import FrameHistory
from data import BatchSampler, collate_segments_without_rewards
from data.milestone import downstream_output_dir, load_milestone, state_sha256
from data.utils import make_task_reward_collate
from envs import DMControlEnv, TIME_LIMIT_ONLY_DOMAINS, WorldModelEnv
from models.pmpo_beta import PMPOBeta, PMPOOptimizers
from utils import (check_same_packages, configure_opt, get_lr_sched, rng_state, save_atomically, set_rng_state,
                   set_seed, source_identity)

# Bumped when the checkpoint contents needed for faithful resume change. 2: milestone
# replay content checksum (dataset.pt, which lists every episode file's checksum).
DOWNSTREAM_RESUME_FORMAT = 2
# Settings a resume may change: how it is launched, and where the (moved) milestone is.
RESUME_MAY_CHANGE = ("downstream.mode", "initialization.milestone_path")
RETURN_CONVENTION = ("sum over environment steps of the requested task's reward; each environment step "
                     "sums the reward of its executed action-repeat substeps")


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
        task_name, seed = cfg.downstream.task_name, cfg.downstream.seed
        set_seed(seed)  # downstream randomness comes only from its own seed
        torch.backends.cudnn.deterministic = True
        use_cuda = cfg.common.devices != "cpu" and torch.cuda.is_available()
        self._device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
        print(f"Starting on {self._device}")

        # The exact milestone: validated denoiser, replay versions and task labels.
        # Ensemble milestones: one configured member (default 0) is the frozen denoiser.
        member_index = OmegaConf.select(cfg, "initialization.member_index", default=0)
        self.milestone = load_milestone(cfg.initialization.milestone_path, task_name, map_location=self._device,
                                        cache_in_ram=True, member_index=member_index)
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
        if resume:
            rng = self.load_checkpoint()
            recover_metrics_log(self._work_dir / "metrics.jsonl")
            self.log_row(dict(phase="resume", resumed_phase=self.phase, num_updates=dict(self.num_updates)))
            set_rng_state(rng)  # last: everything above may consume random numbers

    # ------------------------------------------------------------------ training

    def run(self) -> Path:
        if self.phase == "rew_end_model":
            self.train_reward_model()
        if self.phase == "actor_critic":
            self.train_policy()
        output_dir = self.save()
        if self.cfg.evaluation.after_training:
            result = evaluate_policy(self.agent.actor_critic, self.cfg.env.train, self.milestone.task_name,
                                     self.cfg.evaluation.seeds, self.cfg.evaluation.max_steps)
            save_evaluation(output_dir, result, self.agent.actor_critic.state_dict())
        return output_dir

    def train_reward_model(self) -> None:
        steps = self.cfg.rew_end_model.training.steps
        self.agent.rew_end_model.train()
        with tqdm(total=steps, initial=self.num_updates["rew_end_model"], desc="Training rew_end_model (task reward)") as pbar:
            while self.num_updates["rew_end_model"] < steps:
                self.reward_model_update()
                pbar.update(1)
                self.maybe_checkpoint("rew_end_model", steps)
        # Frozen from here: PMPO only reads its predictions.
        self.agent.rew_end_model.requires_grad_(False).eval()
        self.phase = "actor_critic"
        self.save_checkpoint()

    def train_policy(self) -> None:
        steps = self.cfg.actor_critic.training.steps
        self.agent.actor_critic.train()
        with tqdm(total=steps, initial=self.num_updates["actor_critic"], desc="Training actor_critic (PMPO, predicted task reward)") as pbar:
            while self.num_updates["actor_critic"] < steps:
                self.policy_update()
                pbar.update(1)
                self.maybe_checkpoint("actor_critic", steps)
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
        self.num_updates["actor_critic"] += 1
        self.log("actor_critic", metrics)

    def evaluate(self) -> Dict[str, Any]:
        """Evaluate the current policy in the real task without changing training state."""
        return evaluate_policy(self.agent.actor_critic, self.cfg.env.train, self.milestone.task_name,
                               self.cfg.evaluation.seeds, self.cfg.evaluation.max_steps)

    # ------------------------------------------------------------------ logging and outputs

    def log(self, phase: str, metrics: Dict[str, Any]) -> None:
        # (phase, update) identifies a row; updates replayed after a resume repeat it.
        self.log_row(dict(phase=phase, update=self.num_updates[phase], **{k: float(v) for k, v in metrics.items()}))

    def log_row(self, row: Dict[str, Any]) -> None:
        with open(self._work_dir / "metrics.jsonl", "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")

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
        with open(self._work_dir / "summary.yaml", "w", encoding="utf-8") as f:
            yaml.safe_dump(summary, f, sort_keys=False)
        os.replace(self._work_dir, self.output_dir)
        shutil.rmtree(self.output_dir / "checkpoints", ignore_errors=True)  # training is complete
        print(f"Downstream outputs: {self.output_dir}")
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
        self.agent.rew_end_model.load_state_dict(state["rew_end_model"])
        self.agent.actor_critic.load_state_dict(state["actor_critic"])
        self.rew_end_opt.load_state_dict(state["rew_end_opt"])
        self.rew_end_lr_sched.load_state_dict(state["rew_end_lr_sched"])
        self.actor_critic_opt.load_state_dict(state["actor_critic_opt"])
        self.phase, self.num_updates, self.source = state["phase"], dict(state["num_updates"]), state["source"]
        if self.phase != "rew_end_model":  # trained and frozen before the checkpoint
            self.agent.rew_end_model.requires_grad_(False).eval()
        return state["rng"]


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
    print(f"Recovered an interrupted metrics row into {recovery}")
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
    finally:
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


def save_evaluation(output_dir: Path, result: Dict[str, Any], policy_state: Dict[str, Any]) -> Path:
    """Write to evaluations/, named by policy weights and protocol; never overwritten."""
    policy = state_sha256({k: v for k, v in policy_state.items() if torch.is_tensor(v)})[:12]
    protocol = hashlib.sha256(repr((result["seeds"], result["max_steps"])).encode()).hexdigest()[:12]
    path = Path(output_dir) / "evaluations" / f"eval_{result['task_name']}_policy-{policy}_protocol-{protocol}.yaml"
    if path.exists():
        raise FileExistsError(f"Evaluation {path} already exists; not overwriting it")
    path.parent.mkdir(exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        yaml.safe_dump(dict(result, policy_sha256_prefix=policy), f, sort_keys=False)
    tmp.replace(path)
    print(f"Evaluation: mean return {result['return_mean']:.2f} over {len(result['seeds'])} seeds -> {path}")
    return path


def evaluate_saved_output(cfg: DictConfig) -> Path:
    """Evaluation-only: evaluate the final policy of a completed downstream output (also
    weights-only outputs from before checkpoints existed)."""
    check_evaluation_config(cfg.evaluation)
    task_name, seed = cfg.downstream.task_name, cfg.downstream.seed
    output_dir = downstream_output_dir(Path(cfg.initialization.milestone_path), task_name, seed,
                                       OmegaConf.select(cfg, "initialization.member_index", default=0))
    if not (output_dir / "summary.yaml").is_file():
        raise FileNotFoundError(f"No completed downstream output at {output_dir}")
    saved = OmegaConf.load(output_dir / "config.yaml")
    use_cuda = cfg.common.devices != "cpu" and torch.cuda.is_available()
    device = torch.device("cuda", 0) if use_cuda else torch.device("cpu")
    model = PMPOBeta(instantiate(saved.agent.actor_critic, **action_spec(saved.env.train.domain_name, task_name)))
    # Trusted, locally written weights (PMPO's RNG state is numpy objects): weights_only=False.
    model.load_state_dict(torch.load(output_dir / "actor_critic.pt", map_location=device, weights_only=False))
    model.to(device)
    result = evaluate_policy(model, saved.env.train, task_name, cfg.evaluation.seeds, cfg.evaluation.max_steps)
    return save_evaluation(output_dir, result, model.state_dict())
