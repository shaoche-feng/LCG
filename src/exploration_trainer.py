from functools import partial
from pathlib import Path
import shutil
import time
import json
from typing import Any, Dict, List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent, get_action_space_kwargs
from utils import normalize_saved_config
from coroutines.collector import COLLECTION_MODES, CollectorState, make_collector, NumToCollect
from data import BatchSampler, collate_segments_without_rewards, Dataset, DatasetTraverser
from data.milestone import milestone_name, publish_milestone, referenced_files
from envs import make_dm_control_env, TIME_LIMIT_ONLY_DOMAINS, WorldModelEnv
from ensemble import check_ensemble_settings, EnsembleDisagreementConfig, EnsembleScorer, EnsembleTraining
from lcg import LCGLifecycle, selected_named_parameters
from models.pmpo_beta import PMPOOptimizers, summary_statistics
from pmpo_diagnostic import PMPORunDiagnostic
from utils import (
    broadcast_if_needed,
    build_ddp_wrapper,
    check_same_packages,
    CommonTools,
    configure_opt,
    count_parameters,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    rng_state,
    save_info_for_import_script,
    save_atomically,
    set_rng_state,
    set_seed,
    source_identity,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)


class ExplorationTrainer(StateDictMixin):
    """Reward-free exploration: collect real data, train the denoiser, refresh the intrinsic
    scorer, and train the exploration policy in imagination on intrinsic reward only.

    collection.train.mode selects the method: "policy" is learned (LCG) exploration and
    needs intrinsic_reward.enabled; "uniform" is the uniform-random baseline, which trains
    the denoiser only. Recorded task rewards stay in the dataset for downstream training;
    every exploration learner input has them removed (collate_segments_without_rewards).
    Test diagnostics (evaluation.*, see test_diagnostics) only read them for return summaries.

    The intrinsic reward is intrinsic_reward.method: "lcg" (default; one denoiser) or
    "ensemble_disagreement" (ensemble.members denoisers, see ensemble.py).
    """

    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = normalize_saved_config(cfg)
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        if self._world_size != 1 or cfg.actor_critic.training.grad_acc_steps != 1:
            raise ValueError("Minimal PMPO supports one device and grad_acc_steps=1")
        seed = cfg.common.seed if cfg.common.seed is not None else torch.seed() % 10 ** 9
        set_seed(seed)
        # Test collection and diagnostics draw from their own RNG stream (saved in checkpoints),
        # so they never shift the random numbers of training sampling and collection.
        set_seed(int(np.random.SeedSequence([seed, 1]).generate_state(1)[0]))
        self.test_rng = rng_state()
        set_seed(seed)
        torch.backends.cudnn.deterministic = True

        # Device
        self._device = torch.device("cuda", self._rank) if torch.cuda.is_available() else torch.device("cpu")
        print(f"Starting on {self._device}")
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)  # fix compilation error on multi-gpu nodes

        # Init wandb
        if self._rank == 0:
            try_until_no_except(
                partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True, resume=True, **cfg.wandb)
            )

        # Flags
        self._is_static_dataset = cfg.static_dataset.path is not None
        for key in ("collection.train.first_epoch.mode", "collection.train.mode"):
            mode = OmegaConf.select(cfg, key)
            if mode not in COLLECTION_MODES:
                raise ValueError(f"{key} must be one of {COLLECTION_MODES}, got {mode!r}")
        self._learned_exploration = cfg.collection.train.mode == "policy"
        lcg_enabled = bool(cfg.intrinsic_reward.enabled)
        if self._learned_exploration and not lcg_enabled:
            raise ValueError("Learned exploration (collection.train.mode=policy) needs intrinsic_reward.enabled=true")
        if not self._learned_exploration and lcg_enabled:
            raise ValueError("The uniform baseline trains no exploration policy; set intrinsic_reward.enabled=false")
        if not self._learned_exploration and cfg.get("pmpo_diagnostic") is not None:
            raise ValueError("pmpo_diagnostic needs PMPO updates; the uniform baseline has none")
        # Intrinsic-reward method; configs without `method` are LCG. Only the ensemble method
        # trains more than one denoiser (ensemble.* is ignored otherwise).
        self._method = OmegaConf.select(cfg, "intrinsic_reward.method", default="lcg") if self._learned_exploration else None
        if self._method not in (None, "lcg", "ensemble_disagreement"):
            raise ValueError(f"intrinsic_reward.method must be lcg or ensemble_disagreement, got {self._method!r}")
        self._ensemble_members, self._ensemble_cfg = 1, None
        if self._method == "ensemble_disagreement":
            check_ensemble_settings(OmegaConf.select(cfg, "ensemble.members"), OmegaConf.select(cfg, "ensemble.bootstrap"))
            self._ensemble_members = cfg.ensemble.members
            fields = {k: v for k, v in OmegaConf.to_container(cfg.intrinsic_reward).items() if k != "_target_"}
            self._ensemble_cfg = EnsembleDisagreementConfig(**fields)  # plain ValueError on bad settings
            if cfg.training.compile_wm:
                raise ValueError("Ensemble imagination is not compiled; set training.compile_wm=false")
            if cfg.world_model_env.diffusion_sampler.s_churn != 0:
                raise ValueError("Ensemble disagreement shares the initial noise only; it needs s_churn=0")
        keep_last = cfg.milestones.keep_last
        if keep_last is not None and keep_last < 1:
            raise ValueError(f"milestones.keep_last must be null or at least 1, got {keep_last}")
        check_collection_budget(cfg.collection.train)
        self._should_test, collect_test = check_evaluation_config(cfg.evaluation)
        self._collect_test = collect_test and not self._is_static_dataset  # static datasets never collect
        if self._should_test and not collect_test and not self._is_static_dataset and not cfg.common.resume:
            raise ValueError("evaluation.collect_test=false reuses the run's test dataset, but a new run has none; "
                             "set evaluation.collect_test=true or evaluation.should=false")
        # Imagined episodes never truly end, so real ones must not either.
        if cfg.env.train.domain_name not in TIME_LIMIT_ONLY_DOMAINS:
            raise ValueError(f"Domain {cfg.env.train.domain_name!r} is not verified to end only at its time limit")

        # Checkpointing
        self._path_ckpt_dir = Path("checkpoints")
        self._path_state_ckpt = self._path_ckpt_dir / "state.pt"
        self._keep_agent_copies = partial(
            keep_agent_copies_every,
            every=cfg.checkpointing.save_agent_every,
            path_ckpt_dir=self._path_ckpt_dir,
            num_to_keep=cfg.checkpointing.num_to_keep,
        )
        self._save_info_for_import_script = partial(
            save_info_for_import_script, run_name=cfg.wandb.name, path_ckpt_dir=self._path_ckpt_dir
        )

        # First time, init files hierarchy
        if not cfg.common.resume and self._rank == 0:
            self._path_ckpt_dir.mkdir(exist_ok=False, parents=False)
            path_config = Path("config") / "trainer.yaml"
            path_config.parent.mkdir(exist_ok=False, parents=False)
            shutil.move(".hydra/config.yaml", path_config)
            try:
                wandb.save(str(path_config))
            except OSError:
                # wandb.save() symlinks the file into the run's local wandb dir; this
                # raises WinError 1314 on Windows without admin/Developer Mode. The
                # config is already saved locally above, so this is safe to skip.
                pass
            shutil.copytree(src=root_dir / "src", dst="./src")
            shutil.copytree(src=root_dir / "scripts", dst="./scripts")

        # Datasets
        num_workers = cfg.training.num_workers_data_loaders
        use_manager = cfg.training.cache_in_ram and (num_workers > 0)
        p = Path(cfg.static_dataset.path) if self._is_static_dataset else Path("dataset")
        self.train_dataset = Dataset(p / "train", "train_dataset", cfg.training.cache_in_ram, use_manager)
        self.test_dataset = Dataset(p / "test", "test_dataset", cache_in_ram=True)
        self.train_dataset.load_from_default_path()
        self.test_dataset.load_from_default_path()

        # Envs
        if self._rank == 0:
            train_env = make_dm_control_env(num_envs=cfg.collection.train.num_envs, device=self._device, **cfg.env.train)
            test_env = make_dm_control_env(num_envs=cfg.collection.test.num_envs, device=self._device, **cfg.env.test)
            action_kwargs = get_action_space_kwargs(test_env)
        else:
            action_kwargs = None
        action_kwargs, = broadcast_if_needed(action_kwargs)

        # Create models (no reward/end model: exploration never uses task reward)
        self.agent = Agent(instantiate(cfg.agent, rew_end_model=None, ensemble_members=self._ensemble_members,
                                       **action_kwargs)).to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if dist.is_initialized() else self.agent

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Collectors. The train collector's state (unfinished episodes, env loop position)
        # is kept here so checkpoints can save and restore it.
        self._train_collector_state = CollectorState()
        if not self._is_static_dataset and self._rank == 0:
            self._train_env = train_env
            self._train_collector = make_collector(
                train_env, self.agent.actor_critic, self.train_dataset, cfg.collection.train.epsilon,
                state=self._train_collector_state,
            )
            self._test_collector = make_collector(
                test_env, self.agent.actor_critic, self.test_dataset, cfg.collection.test.epsilon, reset_every_collect=True
            )

        ######################################################

        # Optimizers and LR schedulers (CommonTools' rew_end_model slot stays None)

        self._model_names = ["denoiser", "actor_critic"] if self._learned_exploration else ["denoiser"]
        denoiser_opt = configure_opt(self.agent.denoiser, **cfg.denoiser.optimizer)
        actor_critic_opt = None
        if self._learned_exploration:
            # Both PMPO schedulers are owned and checkpointed by PMPOOptimizers.
            actor_critic_opt = PMPOOptimizers(self.agent.actor_critic, cfg.actor_critic.training.lr_warmup_steps)
        self.opt = CommonTools(denoiser_opt, None, actor_critic_opt)
        self.lr_sched = CommonTools(get_lr_sched(denoiser_opt, cfg.denoiser.training.lr_warmup_steps), None, None)
        # Ensemble members 1..K-1: their own optimizers and schedulers (member 0 uses the above).
        self.ensemble = None
        if self._ensemble_members > 1:
            opts = [configure_opt(m, **cfg.denoiser.optimizer) for m in self.agent.ensemble_members]
            scheds = [get_lr_sched(o, cfg.denoiser.training.lr_warmup_steps) for o in opts]
            self.ensemble = EnsembleTraining(self._ensemble_members, bool(cfg.ensemble.bootstrap), seed, opts, scheds)

        # Data loaders: task rewards are removed from every exploration learner input.

        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_without_rewards,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        make_batch_sampler = partial(BatchSampler, self.train_dataset, self._rank, self._world_size)

        def get_sample_weights(sample_weights: List[float]) -> Optional[List[float]]:
            return None if (self._is_static_dataset and cfg.static_dataset.ignore_sample_weights) else sample_weights

        c = cfg.denoiser.training
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
        bs = make_batch_sampler(c.batch_size, seq_length, get_sample_weights(c.sample_weights))
        dl_denoiser_train = make_data_loader(batch_sampler=bs)
        dl_denoiser_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length, collate_segments_without_rewards)

        self._data_loader_train = CommonTools(dl_denoiser_train, None, None)
        # Ensemble: one sampler per member (bootstrap multiplicities when enabled), shared replay.
        self._member_samplers, self._member_loaders = [], []
        if self.ensemble is not None:
            for k in range(self._ensemble_members):
                mult = partial(self.ensemble.multiplicity, k) if self.ensemble.bootstrap else None
                bs_k = make_batch_sampler(c.batch_size, seq_length, get_sample_weights(c.sample_weights),
                                          episode_multiplicity=mult)
                self._member_samplers.append(bs_k)
                self._member_loaders.append(make_data_loader(batch_sampler=bs_k))
        self._data_loader_test = CommonTools(dl_denoiser_test, None, None)

        # RL env: imagination without a reward/end model (zero reward placeholder, no
        # true endings, truncation at the horizon). Only learned exploration uses it.

        rl_env = None
        if self._learned_exploration:
            c = cfg.actor_critic.training
            sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
            bs = make_batch_sampler(c.batch_size, sl, get_sample_weights(c.sample_weights))
            dl_actor_critic = make_data_loader(batch_sampler=bs)
            wm_env_cfg = instantiate(cfg.world_model_env)
            ensemble = self.agent.denoisers if self.ensemble is not None else None
            rl_env = WorldModelEnv(self.agent.denoiser, None, dl_actor_critic, wm_env_cfg,
                                   return_imagined_candidate=self._method == "lcg", ensemble=ensemble,
                                   ensemble_noise_samples=self._ensemble_cfg.num_noise_samples if ensemble else 1)
            if cfg.training.compile_wm:
                rl_env.predict_next_obs = torch.compile(rl_env.predict_next_obs, mode="reduce-overhead")

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        self.agent.setup_training(sigma_distribution_cfg, rl_env)
        diagnostic_cfg = cfg.get("pmpo_diagnostic")
        self._pmpo_diagnostic = None
        if diagnostic_cfg is not None and diagnostic_cfg.enabled:
            # Upper bound is a sanity check against typos, not a protocol limit -- the
            # 100k-real-step baseline (steps_per_epoch=500) resolves to 241 epochs, so
            # 20 (fine for short bounded diagnostics) was too low for that run.
            if not 1 <= diagnostic_cfg.max_epochs <= 2000:
                raise ValueError("PMPO diagnostic epoch bound must be in [1, 2000]")
            self._pmpo_diagnostic = PMPORunDiagnostic(diagnostic_cfg, self.agent.actor_critic, cfg.env.test,
                                                      optimizers=self.opt.actor_critic, trainer_state_fn=self.state_dict)

        # LCG intrinsic-reward lifecycle, the only exploration scorer (learned exploration).
        self._lcg_lifecycle = None
        self._ensemble_scorer = None
        self._lcg_reward_scaling = None  # intrinsic reward settings a resume must not change (any method)
        if self._method == "ensemble_disagreement":
            self._lcg_reward_scaling = self._ensemble_cfg.reward_scaling_settings()
            self._ensemble_scorer = EnsembleScorer(self._ensemble_cfg)
        elif self._learned_exploration:
            lcg_cfg = instantiate(cfg.intrinsic_reward)
            selected_named_parameters(self.agent.denoiser, lcg_cfg.theta_s)  # fail now on a wrong theta_S, not at epoch 1
            self._lcg_reward_scaling = lcg_cfg.reward_scaling_settings()
            self._lcg_lifecycle = LCGLifecycle(
                lcg_cfg, sigma_distribution_cfg,
                img_channels=cfg.agent.denoiser.inner_model.img_channels,
                img_size=cfg.env.train.size,
                device=self._device,
            )

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_epochs_collect = None
        self.num_episodes_test = 0
        self.num_test_steps = 0  # environment steps of all fresh test collection, final included
        self.final_test_done = False
        self.num_batch_train = CommonTools(0, None, 0)
        self.num_batch_test = CommonTools(0, None, 0)
        self.source = source_identity(root_dir)  # replaced by the checkpoint's on resume
        self.milestone_targets_done = []  # real-step targets already given a milestone
        self.milestones = []  # dict(name, epoch, retired); retired ones are deleted by retention

        self._path_milestones = Path("milestones")
        self._start_logs = []
        self._rng_to_restore = None
        if cfg.common.resume:
            self.load_state_checkpoint()
            self._start_logs.append({"resume/from_epoch": self.epoch})
        if self._should_test and not self._collect_test:
            self.check_reusable_test_dataset()
        if not cfg.common.resume:
            self.save_checkpoint()

        if self._rank == 0:
            for name in self._model_names:
                print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
            print(self.train_dataset)
            print(self.test_dataset)

        if cfg.common.resume:
            # A crash between committing a checkpoint and publishing its milestone.
            self.publish_pending_milestones()
            # Last: nothing after this may consume random numbers before training resumes.
            set_rng_state(self._rng_to_restore)

    def run(self) -> None:
        try:
            self._run()
        except Exception as error:
            if self._pmpo_diagnostic is not None:
                # Do not replace the last healthy checkpoint with failed state.
                self._path_ckpt_dir.mkdir(exist_ok=True)
                torch.save(self.agent.state_dict(), self._path_ckpt_dir / "agent_failed.pt")
                (self._pmpo_diagnostic.path / "failure.txt").write_text(repr(error), encoding="utf-8")
            raise

    def _run(self) -> None:
        to_log = list(self._start_logs)  # resumed runs log where they resumed from

        if self.epoch == 0:
            if self._is_static_dataset:
                self.num_epochs_collect = 0
            else:
                if self._rank == 0:
                    self.num_epochs_collect, to_log_ = self.collect_initial_dataset()
                    to_log += to_log_
                self.num_epochs_collect, sd_train_dataset = broadcast_if_needed(self.num_epochs_collect, self.train_dataset.state_dict())
                self.train_dataset.load_state_dict(sd_train_dataset)
                self.check_no_true_termination(self.train_dataset)

        num_epochs = self.num_epochs_collect + self._cfg.training.num_final_epochs
        if self._pmpo_diagnostic is not None:
            num_epochs = min(num_epochs, self._pmpo_diagnostic.cfg.max_epochs)

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            # Training
            should_collect_train = (self._rank == 0 and not self._is_static_dataset and self.epoch <= self.num_epochs_collect)

            if should_collect_train:
                c = self._cfg.collection.train
                to_log += self._train_collector.send(NumToCollect(steps=c.steps_per_epoch, mode=c.mode))
                self.check_no_true_termination(self.train_dataset)
            sd_train_dataset, = broadcast_if_needed(self.train_dataset.state_dict())  # update dataset for ranks > 0
            self.train_dataset.load_state_dict(sd_train_dataset)
            
            if self._cfg.training.should:
                to_log += self.train_agent()

            if self._pmpo_diagnostic is not None:
                self._pmpo_diagnostic.finish_epoch(self.epoch, to_log, self.train_dataset.num_steps)

            # Evaluation (test diagnostics)
            if self._rank == 0 and self._should_test and self.epoch % self._cfg.evaluation.every == 0:
                to_log += self.test_diagnostics()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            if self._rank == 0:
                wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()
            
            if dist.is_initialized():
                dist.barrier()

        # Final diagnostics, committed once: a resumed finished run does not repeat them.
        if self._rank == 0 and self._should_test and self._pmpo_diagnostic is None and not self.final_test_done:
            wandb_log(self.test_diagnostics(final=True), self.epoch)
            self.final_test_done = True
            self.save_checkpoint()

    def collect_initial_dataset(self) -> Tuple[int, Logs]:
        print("\nInitial collect\n")
        to_log = []
        c = self._cfg.collection.train
        min_steps = c.first_epoch.min
        steps_per_epoch = c.steps_per_epoch
        assert min_steps % steps_per_epoch == 0

        steps = min_steps
        while True:
            to_log += self._train_collector.send(NumToCollect(steps=steps, mode=c.first_epoch.mode))
            num_steps = self.train_dataset.num_steps
            if num_steps >= min_steps:
                break
            steps = steps_per_epoch

        print("\nSummary of initial collect:")
        print(f"Num steps: {num_steps} / {c.num_steps_total}")

        remaining_steps = c.num_steps_total - num_steps
        assert remaining_steps % c.steps_per_epoch == 0
        num_epochs_collect = remaining_steps // c.steps_per_epoch

        return num_epochs_collect, to_log

    def check_no_true_termination(self, dataset: Dataset) -> None:
        if dataset.counts_end[1] > 0:
            raise RuntimeError(f"{dataset.name} has a true termination; exploration supports time-limit-only tasks")

    def collect_test(self, final: bool = False) -> Logs:
        c = self._cfg.collection.test
        episodes = c.num_final_episodes if final else c.num_episodes
        td = self.test_dataset
        td.clear(keep_files=True)  # the last checkpoint's test files are pruned after the next commit
        # Same action source as training collection: the baseline has no trained policy.
        mode = self._cfg.collection.train.mode
        to_log = self._test_collector.send(NumToCollect(episodes=episodes, mode=mode))
        key_ep_id = f"{td.name}/episode_id"
        to_log = [{k: v + self.num_episodes_test if k == key_ep_id else v for k, v in x.items()} for x in to_log]

        self.check_no_true_termination(td)
        print(f"\nSummary of {'final' if final else 'test'} collect: {td.num_episodes} episodes ({td.num_steps} steps)")
        keys = [key_ep_id, "return", "length"]
        to_log_episodes = [x for x in to_log if set(x.keys()) == set(keys)]
        episode_ids, returns, lengths = [[d[k] for d in to_log_episodes] for k in keys]
        for i, (ep_id, ret, length) in enumerate(zip(episode_ids, returns, lengths)):
            print(f"  Episode {ep_id}: return = {ret} length = {length}\n", end="\n" if i == episodes - 1 else "")

        self.num_episodes_test += episodes

        if final:
            to_log.append({"final_return_mean": np.mean(returns), "final_return_std": np.std(returns)})
            print(to_log[-1])

        return to_log

    def train_agent(self) -> Logs:
        self.agent.train()
        self.agent.zero_grad()
        to_log = []
        model_names = self._model_names
        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                if name == "actor_critic":
                    if self._use_cuda:
                        torch.cuda.reset_peak_memory_stats(self._device)
                    t_refresh = time.time()
                    if self._lcg_lifecycle is not None:
                        # Refresh the intrinsic scorer on this epoch's denoiser, then rewire the
                        # PMPO hook. The round seed is the checkpointed epoch, so a resumed run
                        # continues the seed sequence (the lifecycle itself is not checkpointed).
                        self._lcg_lifecycle.refresh(self.agent.denoiser, self.train_dataset, round_identifier=self.epoch)
                        self.agent.actor_critic.set_intrinsic_reward_fn(self._lcg_lifecycle.intrinsic_reward_fn)
                    else:
                        # Ensemble: frozen members (eval mode, no optimizer step) score imagination.
                        for denoiser in self.agent.denoisers:
                            denoiser.eval()
                        self._ensemble_scorer.reset()
                        self.agent.actor_critic.set_intrinsic_reward_fn(self._ensemble_scorer)
                    t_refresh = time.time() - t_refresh
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                t0 = time.time()
                if name == "denoiser" and self.ensemble is not None:
                    to_log += self.train_ensemble(steps)
                else:
                    to_log += self.train_component(name, steps)
                if self._use_cuda:
                    torch.cuda.synchronize(self._device)
                print(f"[TIMING] epoch={self.epoch} component={name} time={time.time() - t0:.2f}s", flush=True)
                if name == "actor_critic":
                    # Wall-clock monitoring (timing/*), not compared between runs.
                    timing = {"timing/pmpo_update_seconds": time.time() - t0, "timing/pmpo_updates": steps}
                    lcg = self._lcg_lifecycle
                    if lcg is not None:
                        timing.update({"timing/lcg_refresh_seconds": t_refresh,
                                       "timing/lcg_precision_seconds": lcg.precision_seconds,
                                       "timing/lcg_scoring_seconds": lcg.scoring_seconds})
                    if self._use_cuda:
                        timing["timing/gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(self._device) / 2 ** 30
                        timing["timing/gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(self._device) / 2 ** 30
                    print(f"[TIMING] epoch={self.epoch} {timing}", flush=True)
                    to_log.append(timing)
        return to_log

    def train_ensemble(self, steps: int) -> Logs:
        """Each member trains `steps` updates (the per-member schedule; K members cost K times the
        updates) on its own RNG stream and sampler."""
        to_log, timing = [], {}
        ens = self.ensemble
        for k in range(ens.members):
            t0 = time.time()
            with ens.member_rng(k):
                to_log += self.train_component("denoiser", steps, member=k)
            if self._use_cuda:
                torch.cuda.synchronize(self._device)
            timing[f"timing/denoiser_member_{k}_seconds"] = time.time() - t0
            sampler = self._member_samplers[k]
            ens.empty_bootstrap_batches[k] += sampler.empty_bootstrap_batches
            sampler.empty_bootstrap_batches = 0
        summary = {"ensemble/total_denoiser_updates": sum(ens.updates)}
        for k in range(ens.members):
            summary[f"ensemble/member_{k}_updates"] = ens.updates[k]
            summary[f"ensemble/member_{k}_empty_bootstrap_batches"] = ens.empty_bootstrap_batches[k]
            if ens.bootstrap:
                m = ens.multiplicity(k, self.train_dataset.num_episodes)
                summary[f"ensemble/member_{k}_bootstrap_zero_episode_fraction"] = float((m == 0).mean())
        to_log.append({**timing, **summary})
        print(f"[TIMING] epoch={self.epoch} {timing}", flush=True)
        return to_log

    def lcg_reward_metrics(self) -> Dict[str, Any]:
        """Raw intrinsic scores and the policy reward (after scaling) of the last PMPO update
        (lcg_* for LCG, ens_* for ensemble disagreement)."""
        scorer = self._lcg_lifecycle or self._ensemble_scorer
        if scorer is None or scorer.last_raw is None:
            return {}
        name = "lcg" if scorer is self._lcg_lifecycle else "ens"
        metrics = {}
        for prefix, values in ((f"{name}_raw_reward", scorer.last_raw), (f"{name}_policy_reward", scorer.last_scaled)):
            values = values.float()
            metrics.update(summary_statistics(values, prefix))
            metrics[f"{prefix}_p99"] = torch.quantile(values.flatten(), 0.99)
        return metrics

    def test_diagnostics(self, final: bool = False) -> Logs:
        """A scheduled (or the final) test diagnostic: collect fresh test episodes if
        evaluation.collect_test, else reuse the test dataset; then the denoiser test loss on
        reward-free batches, and per-task return summaries from the test episodes' recorded
        task rewards. Uses the test RNG stream and restores module modes; no weights,
        optimizers, schedules or training data change."""
        td = self.test_dataset
        training_rng = rng_state()
        modes = {name: module.training for name, module in self.agent.named_modules()}
        set_rng_state(self.test_rng)
        try:
            to_log, new_steps = [], 0
            if self._collect_test:
                to_log = self.collect_test(final)
                new_steps = next(x[f"{td.name}/new_env_steps"] for x in to_log if f"{td.name}/new_env_steps" in x)
            self.num_test_steps += new_steps
            to_log += self.test_agent()
            summary = dict(fresh=int(self._collect_test), num_episodes=td.num_episodes, num_steps=td.num_steps,
                           new_env_steps=new_steps, total_new_env_steps=self.num_test_steps,
                           **task_return_summary(td))
            prefix = "final_test" if final else "test"
            to_log.append({f"{prefix}/{k}": v for k, v in summary.items()})
            print(f"{prefix}: {'fresh' if self._collect_test else 'reused'} test data, {summary}")
        finally:
            self.test_rng = rng_state()
            set_rng_state(training_rng)
            for name, module in self.agent.named_modules():
                module.training = modes[name]
        return to_log

    def check_reusable_test_dataset(self) -> None:
        """Reuse-only diagnostics need test episodes on disk; training replay is never used."""
        td = self.test_dataset
        if td.num_episodes == 0:
            raise ValueError(f"evaluation.collect_test=false needs an existing test dataset, but {td.directory} is empty")
        for path in td.episode_files():
            if not path.is_file():
                raise FileNotFoundError(f"Test dataset episode file {path} is missing")

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        if self.epoch > self._cfg.denoiser.training.start_after_epochs:
            to_log += self.test_component("denoiser")
        return to_log

    def train_component(self, name: str, steps: int, member: Optional[int] = None) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self._agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)
        if member is not None:  # ensemble denoiser member
            model = self.agent.denoisers[member]
            if member > 0:
                opt, lr_sched = self.ensemble.optimizers[member - 1], self.ensemble.schedulers[member - 1]
            data_loader = self._member_loaders[member]

        if name == "actor_critic":
            if cfg.grad_acc_steps != 1 or self._world_size != 1:
                raise ValueError("Minimal PMPO supports one device and grad_acc_steps=1")
            model.train()
            logs = []
            for _ in trange(steps, desc="Training actor_critic (PMPO)"):
                opt.zero_grad()
                loss, metrics = model()
                loss.backward()
                metrics.update(opt.step())
                metrics.update(self.lcg_reward_metrics())
                if self._pmpo_diagnostic is not None:
                    self._pmpo_diagnostic.check_update(metrics, self.epoch, self.num_batch_train.actor_critic,
                                                       real_steps=self.train_dataset.num_steps)
                if self.agent.actor_critic.cfg.log_diagnostics:
                    print("PMPO_DIAGNOSTICS " + json.dumps({k: float(v) for k, v in metrics.items()}), flush=True)
                self.num_batch_train.actor_critic += 1
                metrics["num_batch_train_actor_critic"] = self.num_batch_train.actor_critic
                logs.append({f"actor_critic/train/{k}": v for k, v in metrics.items()})
            return logs

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader)
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        for i in trange(num_steps, desc=f"Training {name}", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device)
            loss, metrics = model(batch)
            if self._pmpo_diagnostic is not None and not torch.isfinite(loss).all():
                raise FloatingPointError(f"Non-finite {name} loss")
            loss.backward()

            if member is None or member == 0:
                num_batch = self.num_batch_train.get(name)
                metrics[f"num_batch_train_{name}"] = num_batch
                self.num_batch_train.set(name, num_batch + 1)
            if member is not None:
                metrics[f"num_batch_train_member"] = self.ensemble.updates[member]
                self.ensemble.updates[member] += 1

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm,
                                                              error_if_nonfinite=self._pmpo_diagnostic is not None)
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()

            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        prefix = f"{name}/train" if member is None else f"{name}/member_{member}/train"
        to_log = [{f"{prefix}/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    @torch.no_grad()
    def test_component(self, name: str) -> Logs:
        model = getattr(self.agent, name)
        data_loader = self._data_loader_test.get(name)
        model.eval()
        to_log = []
        for batch in tqdm(data_loader, desc=f"Evaluating {name}"):
            batch = batch.to(self._device)
            _, metrics = model(batch)
            num_batch = self.num_batch_test.get(name)
            metrics[f"num_batch_test_{name}"] = num_batch
            self.num_batch_test.set(name, num_batch + 1)
            to_log.append(metrics)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/test/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    def state_dict(self) -> dict:
        state = super().state_dict()
        if self._pmpo_diagnostic is not None:
            state["pmpo_diagnostic"] = self._pmpo_diagnostic.state_dict()
        state["resume_format"] = RESUME_FORMAT
        state["lcg_reward_scaling"] = self._lcg_reward_scaling
        state["collection"] = self.collection_state()
        state["rng"] = rng_state()
        return state

    def load_state_dict(self, state_dict: dict) -> None:
        # Checkpoints saved without a diagnostic carry no "pmpo_diagnostic" entry.
        state_dict = dict(state_dict)
        if state_dict.pop("resume_format", None) != RESUME_FORMAT:
            raise ValueError(f"Checkpoint has no exploration resume state of format {RESUME_FORMAT} (saved by "
                             "older code); it cannot be resumed faithfully. Start a new run.")
        collection = state_dict.pop("collection")
        scaling = state_dict.pop("lcg_reward_scaling")
        if scaling != self._lcg_reward_scaling:
            raise ValueError(f"Checkpoint was trained with LCG reward scaling {scaling}, the config now gives "
                             f"{self._lcg_reward_scaling}; a resume must not switch it")
        self._rng_to_restore = state_dict.pop("rng")
        diagnostic_state = state_dict.pop("pmpo_diagnostic", None)
        state_dict.setdefault("ensemble", None)  # checkpoints from before ensembles: single denoiser
        saved = (state_dict["ensemble"] or {}).get("settings")
        expected = self.ensemble.settings() if self.ensemble is not None else None
        if saved != expected:  # checked before any weights are loaded
            raise ValueError(f"Checkpoint was trained with ensemble settings {saved}, the config gives {expected} "
                             "(None: single denoiser); a resume must not change them")
        check_same_packages(state_dict["source"])
        super().load_state_dict(state_dict)
        self.restore_collection_state(collection)
        if self._pmpo_diagnostic is not None:
            # After the agent: the history is validated against the restored PMPOBeta.updates.
            self._pmpo_diagnostic.load_state_dict(diagnostic_state)

    def collection_state(self) -> Optional[Dict[str, Any]]:
        """The train collector at this epoch boundary: unfinished episode ids, the current
        observation and frame history, and each real env's full state."""
        if self._is_static_dataset or self._rank != 0:
            return None
        state = self._train_collector_state
        started = state.loop.obs is not None
        return dict(
            env_config=OmegaConf.to_container(self._cfg.env.train),
            episode_ids=list(state.episode_ids),
            obs=state.loop.obs.cpu() if started else None,
            history=state.loop.history.cpu() if started else None,
            envs=self._train_env.get_env_states() if started else None,
        )

    def restore_collection_state(self, saved: Optional[Dict[str, Any]]) -> None:
        if self._is_static_dataset or self._rank != 0:
            return
        if saved is None:
            raise ValueError("Checkpoint has no collection state; it cannot resume collection")
        if saved["env_config"] != OmegaConf.to_container(self._cfg.env.train):
            raise ValueError("Checkpoint was collected with a different env.train config")
        for episode_id in saved["episode_ids"]:
            if episode_id is not None and (episode_id >= self.train_dataset.num_episodes
                                           or self.train_dataset.load_episode(episode_id).dead[-1]):
                raise ValueError(f"Checkpoint's unfinished episode {episode_id} is not an unfinished episode of its dataset")
        state = self._train_collector_state
        state.episode_ids[:] = saved["episode_ids"]  # in place: the collector holds this list
        if saved["obs"] is not None:
            self._train_env.set_env_states(saved["envs"])
            state.loop.obs = saved["obs"].to(self._device)
            state.loop.history = saved["history"].to(self._device)

    def load_state_checkpoint(self) -> None:
        # Trusted, locally generated DIAMOND state (includes Dataset state: numpy arrays, Counters,
        # etc.), not just tensor weights -- weights_only=False is required for PyTorch >=2.6, whose
        # default changed to weights_only=True.
        self.load_state_dict(torch.load(self._path_state_ckpt, map_location=self._device, weights_only=False))

    def save_checkpoint(self) -> None:
        """Commit point: state.pt, which includes the replay index and collection state, is
        replaced atomically. Episode versions it refers to were written before; milestones
        are published and unreferenced versions pruned only after the commit."""
        if self._rank != 0:
            return
        self.mark_due_milestone()
        save_atomically(self.state_dict(), self._path_state_ckpt)
        if not self._is_static_dataset:  # never write into a static dataset
            self.train_dataset.save_to_default_path()
            self.test_dataset.save_to_default_path()
        self._keep_agent_copies(self.agent.state_dict(), self.epoch)
        self._save_info_for_import_script(self.epoch)
        self.publish_pending_milestones()
        if not self._is_static_dataset:
            self.prune_episode_versions()

    def mark_due_milestone(self) -> None:
        """Before the commit: record a milestone for this boundary if a real-step target was
        reached, and mark the ones retention will delete."""
        c = self._cfg.milestones
        num_steps = self.train_dataset.num_steps
        targets = set(c.at_steps)
        if c.every_steps:
            targets.update(range(c.every_steps, num_steps + 1, c.every_steps))
        due = sorted(t for t in targets if t <= num_steps and t not in self.milestone_targets_done)
        if due:
            self.milestone_targets_done += due
            self.milestones.append(dict(name=milestone_name(num_steps), epoch=self.epoch, retired=False))
        if c.keep_last is not None:
            active = [m for m in self.milestones if not m["retired"]]
            for milestone in active[:-c.keep_last]:
                milestone["retired"] = True

    def publish_pending_milestones(self) -> None:
        """After the commit (or on resume): publish this boundary's milestone, then delete
        retired ones. Both steps can be repeated after an interruption."""
        for milestone in self.milestones:
            path = self._path_milestones / milestone["name"]
            if milestone["retired"] or path.exists():
                continue
            if milestone["epoch"] != self.epoch:
                raise RuntimeError(f"Milestone {milestone['name']} is missing but was not retired")
            self.publish_milestone(milestone["name"])
        for milestone in self.milestones:
            path = self._path_milestones / milestone["name"]
            if milestone["retired"] and path.exists():
                shutil.rmtree(path)

    def publish_milestone(self, name: str) -> Path:
        cfg = self._cfg
        state = self._train_collector_state
        successors = {}
        if state.loop.obs is not None:
            for i, episode_id in enumerate(state.episode_ids):
                if episode_id is not None:
                    successors[episode_id] = state.loop.obs[i]
        manifest = dict(
            created_by="ExplorationTrainer",
            domain=cfg.env.train.domain_name,
            task_names=list(cfg.env.train.task_names),
            collected_steps=int(self.train_dataset.num_steps),
            num_episodes=int(self.train_dataset.num_episodes),
            epoch=int(self.epoch),
            counters=dict(denoiser_batches=int(self.num_batch_train.denoiser),
                          pmpo_updates=int(self.agent.actor_critic.updates.item()),
                          **(dict(member_denoiser_batches=list(self.ensemble.updates)) if self.ensemble else {})),
            exploration=dict(collection_mode=cfg.collection.train.mode,
                             intrinsic_reward_enabled=bool(cfg.intrinsic_reward.enabled),
                             **(dict(method=self._method) if self._method else {})),
            model=dict(denoiser=OmegaConf.to_container(cfg.agent.denoiser),
                       env=OmegaConf.to_container(cfg.env.train)),
            run_config="../../config/trainer.yaml",
            source=self.source,
        )
        member_states = None
        if self.ensemble is not None:
            manifest["ensemble"] = dict(members=self.ensemble.members, bootstrap=self.ensemble.bootstrap,
                                        downstream_default_member=0, note="denoiser.pt is member 0")
            member_states = [m.state_dict() for m in self.agent.ensemble_members]
        return publish_milestone(self._path_milestones, name, self.agent.denoiser.state_dict(),
                                 self.train_dataset, successors, manifest, member_states)

    def prune_episode_versions(self) -> None:
        """Delete replay versions no longer referenced by this checkpoint or any milestone."""
        keep = set(self.train_dataset.episode_files())
        for path in self._path_milestones.glob("steps_*"):
            keep.update(referenced_files(path))
        self.train_dataset.prune_versions(keep)
        # Earlier test collections' files: deleted only once this checkpoint no longer needs them.
        self.test_dataset.prune_versions(set(self.test_dataset.episode_files()))


# Bumped when the checkpoint contents needed for faithful resume change (2: test RNG and
# counts; 3: LCG reward scaling settings).
RESUME_FORMAT = 3


def check_evaluation_config(e: DictConfig) -> Tuple[bool, bool]:
    """Returns (diagnostics enabled, collect fresh test data). A missing collect_test means
    True, the behaviour before the flag existed."""
    should, collect = e.should, OmegaConf.select(e, "collect_test", default=True)
    for name, value in (("should", should), ("collect_test", collect)):
        if not isinstance(value, bool):
            raise ValueError(f"evaluation.{name} must be true or false, got {value!r}")
    if should and (isinstance(e.every, bool) or not isinstance(e.every, int) or e.every < 1):
        raise ValueError(f"evaluation.every must be a positive integer, got {e.every!r}")
    return should, collect


def task_return_summary(dataset: Dataset) -> Dict[str, float]:
    """Per-task episode returns (sum of each recorded task-reward column) of a test dataset:
    mean, std, min, max. Read only for logging; never a learner input."""
    if dataset.task_names is None or dataset.num_episodes == 0:
        return {}
    returns = np.stack([dataset.load_episode(i).task_rew.double().sum(0).numpy() for i in range(dataset.num_episodes)])
    summary = {}
    for task, r in zip(dataset.task_names, returns.T):
        summary.update({f"return_{task}/mean": r.mean(), f"return_{task}/std": r.std(),
                        f"return_{task}/min": r.min(), f"return_{task}/max": r.max()})
    return {k: float(v) for k, v in summary.items()}


def check_collection_budget(c: DictConfig) -> None:
    """The training-data budget (environment steps) must end exactly at num_steps_total:
    first_epoch.min initial steps plus whole epochs of steps_per_epoch, each a multiple of
    num_envs (a collect stops at the first multiple of num_envs reaching its target)."""
    if not 0 < c.first_epoch.min <= c.num_steps_total:
        raise ValueError(f"collection.train.first_epoch.min must be in (0, num_steps_total], got {c.first_epoch.min}")
    for name, steps in (("first_epoch.min", c.first_epoch.min), ("steps_per_epoch", c.steps_per_epoch)):
        if steps % c.steps_per_epoch or steps % c.num_envs:
            raise ValueError(f"collection.train.{name}={steps} must be a multiple of steps_per_epoch "
                             f"({c.steps_per_epoch}) and num_envs ({c.num_envs})")
    if (c.num_steps_total - c.first_epoch.min) % c.steps_per_epoch:
        raise ValueError(f"collection.train.num_steps_total={c.num_steps_total} minus first_epoch.min "
                         f"must be a multiple of steps_per_epoch ({c.steps_per_epoch})")
