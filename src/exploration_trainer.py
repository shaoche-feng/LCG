from functools import partial
from pathlib import Path
import shutil
import time
import json
import os
from typing import Any, Dict, List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader
import wandb

from agent import Agent, get_action_space_kwargs
from utils import normalize_saved_config
from coroutines.collector import COLLECTION_MODES, CollectorState, make_collector, NumToCollect
from data import BatchSampler, collate_segments_without_rewards, Dataset, DatasetTraverser
from data.milestone import milestone_name, publish_milestone, read_manifest, referenced_files
from device_report import device_lines
from envs import make_dm_control_env, TIME_LIMIT_ONLY_DOMAINS, WorldModelEnv
from ensemble import EnsembleDisagreementConfig, EnsembleScorer, EnsembleTraining, migrate_ensemble_layout
from lcg import LCGLifecycle, selected_named_parameters
from models.pmpo_beta import PMPOOptimizers, check_controller_state, controller_settings, summary_statistics
from plan2explore import check_feature_layers, Plan2Explore, Plan2ExploreConfig
from laplace import LaplaceConfig, LaplaceEnsemble, LaplaceScorer
from intrinsic_eval import aggregate, episode_record, IntrinsicEvaluator
from pmpo_diagnostic import PMPORunDiagnostic
from utils import (
    broadcast_if_needed,
    build_ddp_wrapper,
    check_same_packages,
    CommonTools,
    configure_opt,
    configure_progress,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    history_records,
    repair_history,
    phase,
    Progress,
    rng_state,
    save_info_for_import_script,
    save_atomically,
    say,
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

    The intrinsic reward is intrinsic_reward.method: "lcg" (default; one denoiser),
    "ensemble_disagreement" (intrinsic_reward.num_members denoisers, see ensemble.py) or
    "plan2explore_disagreement" (one denoiser plus intrinsic_reward.num_members lightweight
    feature predictors, see plan2explore.py) or "laplace_disagreement" (one denoiser and
    intrinsic_reward.num_members Gaussian samples of its LCG-selected parameters, see laplace.py).
    """

    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        cfg = migrate_ensemble_layout(normalize_saved_config(cfg))  # older saved ensemble layout, in memory
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
        self._use_cuda = self._device.type == "cuda"
        if self._use_cuda:
            torch.cuda.set_device(self._rank)  # fix compilation error on multi-gpu nodes

        # Terminal output (configs without a logging section use the defaults)
        configure_progress(OmegaConf.select(cfg, "logging.progress", default="bar"))
        self._epoch_summary = OmegaConf.select(cfg, "logging.epoch_summary", default=True)
        # Local mirror of every W&B row (run folder), used to backfill rows a crash kept from W&B.
        self._local_history = OmegaConf.select(cfg, "logging.local_history", default=True)
        self._path_history = Path("wandb_history.jsonl")
        self._seed = seed

        # Init wandb. A run folder keeps its W&B run id (wandb_run.json), so a resume continues
        # the same W&B run; folders without the file resume as before (wandb's own resume file).
        if self._rank == 0:
            wandb_kwargs = {k: v for k, v in OmegaConf.to_container(cfg.wandb).items()}
            path_run_id = Path("wandb_run.json")
            if wandb_kwargs.get("id") is None and path_run_id.is_file():
                wandb_kwargs["id"] = json.loads(path_run_id.read_text())["id"]
            try_until_no_except(
                partial(wandb.init, config=OmegaConf.to_container(cfg, resolve=True), reinit=True,
                        resume="allow" if wandb_kwargs.get("id") else True, **wandb_kwargs)
            )
            if wandb.run is not None and wandb_kwargs.get("mode") != "disabled":
                path_run_id.write_text(json.dumps(dict(id=wandb.run.id, project=wandb.run.project, entity=wandb.run.entity,
                                                       name=wandb.run.name, group=wandb.run.group, url=wandb.run.url),
                                                  indent=2))

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
        # trains more than one denoiser (intrinsic_reward.num_members).
        self._method = OmegaConf.select(cfg, "intrinsic_reward.method", default="lcg") if self._learned_exploration else None
        if self._method not in (None, "lcg", "ensemble_disagreement", "plan2explore_disagreement", "laplace_disagreement"):
            raise ValueError(f"intrinsic_reward.method must be lcg, ensemble_disagreement, plan2explore_disagreement or "
                             f"laplace_disagreement, got {self._method!r}")
        self._ensemble_members, self._ensemble_cfg, self._p2e_cfg, self._laplace_cfg = 1, None, None, None
        if self._method == "laplace_disagreement":
            fields = {k: v for k, v in OmegaConf.to_container(cfg.intrinsic_reward).items() if k != "_target_"}
            self._laplace_cfg = LaplaceConfig(**fields)  # plain ValueError on bad settings
            if cfg.training.compile_wm:
                raise ValueError("Laplace ensemble imagination is not compiled; set training.compile_wm=false")
            if cfg.world_model_env.diffusion_sampler.s_churn != 0:
                raise ValueError("Laplace members share the initial noise only; it needs s_churn=0")
        if self._method == "plan2explore_disagreement":
            fields = {k: v for k, v in OmegaConf.to_container(cfg.intrinsic_reward).items() if k != "_target_"}
            self._p2e_cfg = Plan2ExploreConfig(**fields)  # plain ValueError on bad settings
            if cfg.training.compile_wm:
                raise ValueError("Plan2Explore imagination is validated uncompiled; set training.compile_wm=false")
            check_feature_layers(self._p2e_cfg, instantiate(cfg.agent.denoiser), cfg.env.train.size)
        if self._method == "ensemble_disagreement":
            fields = {k: v for k, v in OmegaConf.to_container(cfg.intrinsic_reward).items() if k != "_target_"}
            self._ensemble_cfg = EnsembleDisagreementConfig(**fields)  # plain ValueError on bad settings
            self._ensemble_members = self._ensemble_cfg.num_members
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
        # Fixed test reset seeds (one episode per seed) for periodic and final collections, and
        # whether test episodes are scored with the frozen intrinsic scorer.
        self._reset_seeds = {final: check_reset_seeds(cfg, final) for final in (False, True)}
        self._intrinsic_eval = OmegaConf.select(cfg, "evaluation.intrinsic", default=True)
        if not isinstance(self._intrinsic_eval, bool):
            raise ValueError(f"evaluation.intrinsic must be true or false, got {self._intrinsic_eval!r}")
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

        # Plan2Explore: feature extractor on the (single) denoiser and the predictor heads. The
        # selected blocks and pooling are shape-checked on the model here, before any data.
        self.plan2explore = None
        if self._p2e_cfg is not None:
            self.plan2explore = Plan2Explore(self._p2e_cfg, self.agent.denoiser, cfg.env.train.size,
                                             action_kwargs["action_low"], action_kwargs["action_high"], seed, self._device)

        # Laplace: LCG's selection (and its guards) on the single denoiser; samples are drawn at
        # each refresh. The agent holds one denoiser; members exist only as a sample bank.
        self.laplace = None
        if self._laplace_cfg is not None:
            self.laplace = LaplaceEnsemble(self._laplace_cfg, self.agent.denoiser,
                                           instantiate(cfg.denoiser.sigma_distribution), seed, self._device)

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
            self.ensemble = EnsembleTraining(self._ensemble_members, self._ensemble_cfg.bootstrap, seed, opts, scheds)

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
                                   ensemble_noise_samples=(self._ensemble_cfg.num_noise_samples if ensemble else
                                                           self._laplace_cfg.num_noise_samples if self.laplace else 1),
                                   feature_encoder=self.plan2explore.extractor if self.plan2explore else None,
                                   member_predictor=self.laplace.predict_members if self.laplace else None)
            if cfg.training.compile_wm:
                rl_env.predict_next_obs = torch.compile(rl_env.predict_next_obs, mode="reduce-overhead")

        self._rl_env = rl_env

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
        elif self._method == "plan2explore_disagreement":
            self._lcg_reward_scaling = self._p2e_cfg.reward_scaling_settings()
        elif self._method == "laplace_disagreement":
            self._lcg_reward_scaling = self._laplace_cfg.reward_scaling_settings()
            self._ensemble_scorer = LaplaceScorer(self._laplace_cfg)
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
            self._lcg_lifecycle.verbose = False  # refresh details go to W&B (lcg/*)

        # Real-environment intrinsic evaluation with the method's frozen scorer (learned methods).
        self._intrinsic = None
        self._scorer_restore_note = None  # set when a resumed checkpoint lacks frozen scorer state
        if self._method is not None:
            noise = (self._ensemble_cfg or self._laplace_cfg).num_noise_samples if self._method in (
                "ensemble_disagreement", "laplace_disagreement") else 1
            self._intrinsic = IntrinsicEvaluator(
                self._method, cfg.agent.denoiser.inner_model.num_steps_conditioning, self._device,
                lcg=self._lcg_lifecycle, ensemble_samplers=rl_env.ensemble_samplers, ensemble_scorer=self._ensemble_scorer,
                laplace=self.laplace, sampler=rl_env.sampler, plan2explore=self.plan2explore, noise_samples=noise)

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
        self.test_episode_seeds = []  # reset seed of each current test episode (None: drawn at random)
        self.resume_count = 0  # resumes of this run folder (logging only)
        self.wandb_steps = 0  # W&B rows logged (and mirrored) up to this checkpoint
        self.wandb_history_bytes = 0  # wandb_history.jsonl length at this checkpoint: its committed records
        self._committed_history_bytes = None  # on resume: the loaded checkpoint's value (backfill reads only these)

        self._path_milestones = Path("milestones")
        self._start_logs = []
        self._rng_to_restore = None
        if cfg.common.resume:
            self.load_state_checkpoint()
            self.resume_count += 1
            if self._rank == 0 and self._local_history:
                self.repair_wandb_history()
            self._start_logs.append({"resume/from_epoch": self.epoch, "resume/count": self.resume_count})
        if self._should_test and not self._collect_test:
            self.check_reusable_test_dataset()
        if not cfg.common.resume:
            self.save_checkpoint()

        if self._rank == 0:
            self.setup_wandb()
            self.announce_start()

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
            self._phase_seconds, self._gpu_peak = {}, 0.0
            steps_before = self.train_dataset.num_steps
            if self._use_cuda:
                torch.cuda.reset_peak_memory_stats(self._device)  # epoch peak (monitoring only)

            # Training
            should_collect_train = (self._rank == 0 and not self._is_static_dataset and self.epoch <= self.num_epochs_collect)

            if self._rank == 0:
                budget = self._cfg.collection.train.num_steps_total
                kind = "exploration phase" if should_collect_train else "training-only phase (no new collection)"
                phase()
                say(f"=== Epoch {self.epoch}/{num_epochs} | {kind} | training env steps "
                    f"{self.train_dataset.num_steps}/{budget} ===")

            if should_collect_train:
                c = self._cfg.collection.train
                t0 = time.time()
                to_log += without_task_returns(self._train_collector.send(
                    NumToCollect(steps=c.steps_per_epoch, mode=c.mode, desc=f"Collection ({c.mode} actions)")))
                self._phase_seconds["collection"] = time.time() - t0
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
                if self._epoch_summary:
                    to_log.append(self.epoch_summary(to_log, steps_before, time.time() - start_time))
                self.log_rows(to_log)
            to_log = []

            # Checkpointing
            self.save_checkpoint()
            
            if dist.is_initialized():
                dist.barrier()

        # Final diagnostics, committed once: a resumed finished run does not repeat them.
        if self._rank == 0 and self._should_test and self._pmpo_diagnostic is None and not self.final_test_done:
            self.log_rows(self.test_diagnostics(final=True))
            self.final_test_done = True
            self.save_checkpoint()

    def collect_initial_dataset(self) -> Tuple[int, Logs]:
        to_log = []
        c = self._cfg.collection.train
        min_steps = c.first_epoch.min
        phase()
        steps_per_epoch = c.steps_per_epoch
        assert min_steps % steps_per_epoch == 0

        steps = min_steps
        while True:
            to_log += without_task_returns(self._train_collector.send(
                NumToCollect(steps=steps, mode=c.first_epoch.mode, desc=f"Initial collection ({c.first_epoch.mode} actions)")))
            num_steps = self.train_dataset.num_steps
            if num_steps >= min_steps:
                break
            steps = steps_per_epoch

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
        seeds = self._reset_seeds[final]
        desc = f"{'Final evaluation' if final else 'Evaluation'} episodes"
        if seeds is None:
            to_log = self._test_collector.send(NumToCollect(episodes=episodes, mode=mode, desc=desc))
            self.test_episode_seeds = [None] * episodes
        else:
            # One episode per fixed reset seed (the collector resets for every collect); one bar.
            to_log = []
            pbar = Progress(total=len(seeds), desc=f"{desc} (seeds {seeds[0]}-{seeds[-1]})", unit="ep")
            for seed in seeds:
                to_log += self._test_collector.send(NumToCollect(episodes=1, mode=mode, reset_seed=seed, progress=False))
                pbar.update(1)
            pbar.close()
            if td.num_episodes != len(seeds):
                raise RuntimeError(f"Seeded test collection gave {td.num_episodes} episodes for {len(seeds)} seeds")
            self.test_episode_seeds = list(seeds)
        key_ep_id = f"{td.name}/episode_id"
        to_log = [{k: v + self.num_episodes_test if k == key_ep_id else v for k, v in x.items()} for x in to_log]

        self.check_no_true_termination(td)
        self.num_episodes_test += episodes
        return without_task_returns(to_log)

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
                        self.note_gpu_peak()  # before the refresh + PMPO peak window restarts
                        torch.cuda.reset_peak_memory_stats(self._device)
                    say(f"Scorer refresh ({METHOD_NAMES[self._method]}): {REFRESH_WORK[self._method]}")
                    refresh_logs = []
                    t_refresh = time.time()
                    if self._lcg_lifecycle is not None:
                        # Refresh the intrinsic scorer on this epoch's denoiser, then rewire the
                        # PMPO hook. The round seed is the checkpointed epoch, so a resumed run
                        # continues the seed sequence (the lifecycle itself is not checkpointed).
                        self._lcg_lifecycle.refresh(self.agent.denoiser, self.train_dataset, round_identifier=self.epoch)
                        self.agent.actor_critic.set_intrinsic_reward_fn(self._lcg_lifecycle.intrinsic_reward_fn)
                        refresh_logs.append(lcg_refresh_metrics(self._lcg_lifecycle, self.train_dataset))
                    elif self.plan2explore is not None:
                        # Encode the replay with this epoch's (frozen) denoiser, fit fresh heads, then
                        # freeze them; the denoiser stays in eval mode for PMPO (only read).
                        to_log += self.plan2explore.refresh(self.agent.denoiser, self.train_dataset,
                                                            self.successor_observations(), epoch=self.epoch)
                        if self._use_cuda:  # the refresh alone (gpu_peak_* below covers refresh + PMPO)
                            to_log.append({"timing/p2e_refresh_gpu_peak_allocated_gb":
                                           torch.cuda.max_memory_allocated(self._device) / 2 ** 30})
                        self.agent.denoiser.eval()
                        self.agent.actor_critic.set_intrinsic_reward_fn(self.plan2explore)
                    elif self.laplace is not None:
                        # LCG's precision on this epoch's denoiser, then a fresh sample bank, frozen
                        # for the PMPO phase; the base denoiser is only read (eval mode).
                        to_log.append(self.laplace.refresh(self.train_dataset, epoch=self.epoch))
                        if self._use_cuda:  # the refresh alone (gpu_peak_* below covers refresh + PMPO)
                            to_log.append({"timing/laplace_refresh_gpu_peak_allocated_gb":
                                           torch.cuda.max_memory_allocated(self._device) / 2 ** 30})
                        self.agent.denoiser.eval()
                        self._ensemble_scorer.reset()
                        self.agent.actor_critic.set_intrinsic_reward_fn(self._ensemble_scorer)
                    else:
                        # Ensemble: frozen members (eval mode, no optimizer step) score imagination.
                        for denoiser in self.agent.denoisers:
                            denoiser.eval()
                        self._ensemble_scorer.reset()
                        self.agent.actor_critic.set_intrinsic_reward_fn(self._ensemble_scorer)
                    if self._use_cuda:
                        torch.cuda.synchronize(self._device)
                    t_refresh = time.time() - t_refresh
                    self._phase_seconds["scorer_refresh"] = t_refresh
                    to_log += refresh_logs
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                t0 = time.time()
                if name == "denoiser" and self.ensemble is not None:
                    to_log += self.train_ensemble(steps)
                else:
                    rows = self.train_component(name, steps)
                    to_log += rows
                    if name == "actor_critic":
                        say(training_intrinsic_line(rows))
                if self._use_cuda:
                    torch.cuda.synchronize(self._device)
                elapsed = time.time() - t0
                self._phase_seconds["denoiser" if name == "denoiser" else "pmpo"] = elapsed
                if name == "actor_critic":
                    # Wall-clock monitoring (timing/*), not compared between runs. pmpo_update_seconds
                    # is the whole PMPO phase (all updates, imagination and scoring included).
                    timing = {"timing/pmpo_update_seconds": elapsed, "timing/pmpo_updates": steps,
                              "timing/pmpo_seconds_per_update": elapsed / max(steps, 1),
                              "timing/scorer_refresh_seconds": t_refresh}
                    lcg = self._lcg_lifecycle
                    if lcg is not None:
                        timing.update({"timing/lcg_refresh_seconds": t_refresh,
                                       "timing/lcg_precision_seconds": lcg.precision_seconds,
                                       "timing/lcg_scoring_seconds": lcg.scoring_seconds})
                    if self.plan2explore is not None:
                        # Imagination feature extraction (one extractor pass per imagined step) and
                        # head scoring, both inside pmpo_update_seconds; the refresh is not.
                        timing.update({"timing/p2e_refresh_seconds": t_refresh,
                                       "timing/p2e_imagination_feature_seconds": self.plan2explore.extractor.seconds,
                                       "timing/p2e_scoring_seconds": self.plan2explore.scoring_seconds})
                    if self.laplace is not None:
                        # Member predictions (all K complete chains per imagined step) are inside
                        # pmpo_update_seconds; the refresh (precision + sampling) is not.
                        timing.update({"timing/laplace_refresh_seconds": t_refresh,
                                       "timing/laplace_prediction_seconds": self.laplace.prediction_seconds,
                                       "timing/laplace_prediction_calls": self.laplace.prediction_calls,
                                       "timing/laplace_shared_prefix_evaluations": self.laplace.prefix_evaluations})
                    if self._use_cuda:
                        timing["timing/gpu_peak_allocated_gb"] = torch.cuda.max_memory_allocated(self._device) / 2 ** 30
                        timing["timing/gpu_peak_reserved_gb"] = torch.cuda.max_memory_reserved(self._device) / 2 ** 30
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
        return to_log

    def lcg_reward_metrics(self) -> Dict[str, Any]:
        """Raw intrinsic scores and the policy reward (after scaling) of the last PMPO update
        (lcg_* for LCG, ens_* for ensemble disagreement)."""
        scorer = self._lcg_lifecycle or self._ensemble_scorer or self.plan2explore
        if scorer is None or scorer.last_raw is None:
            return {}
        name = "lcg" if scorer is self._lcg_lifecycle else scorer.metric_prefix
        metrics = {}
        for prefix, values in ((f"{name}_raw_reward", scorer.last_raw), (f"{name}_policy_reward", scorer.last_scaled)):
            values = values.float()
            metrics.update(summary_statistics(values, prefix))
            metrics[f"{prefix}_p99"] = torch.quantile(values.flatten(), 0.99)
        return metrics

    # ---------------------------------------------------------------- run logging (terminal, W&B)

    def note_gpu_peak(self) -> None:
        if self._use_cuda:
            self._gpu_peak = max(getattr(self, "_gpu_peak", 0.0), torch.cuda.max_memory_allocated(self._device) / 2 ** 30)

    def progress_counters(self) -> Dict[str, Any]:
        counters = {"progress/train_env_steps": int(self.train_dataset.num_steps),
                    "progress/test_env_steps": int(self.num_test_steps),
                    "progress/denoiser_updates": int(self.num_batch_train.denoiser),
                    "progress/pmpo_updates": int(self.agent.actor_critic.updates.item())}
        if self.ensemble is not None:
            counters["progress/member_denoiser_updates_total"] = int(sum(self.ensemble.updates))
        return counters

    def with_env_steps(self, rows: Logs) -> Logs:
        """Every W&B row also carries the training env steps at the time it is logged (collection
        precedes training within an epoch), so any chart can use an environment-step axis."""
        steps = int(self.train_dataset.num_steps)
        return [{**row, "progress/train_env_steps": steps} for row in rows]

    def log_rows(self, rows: Logs) -> None:
        """Log rows to W&B (each with the env-step axis) and mirror each one, keyed by its W&B
        step, to wandb_history.jsonl before the checkpoint that commits it."""
        rows = self.with_env_steps(rows)
        if self._local_history:
            with open(self._path_history, "a", encoding="utf-8") as f:
                for i, row in enumerate(rows):
                    values = {k: float(v) for k, v in row.items() if np.ndim(v) == 0 and not isinstance(v, str)}
                    f.write(json.dumps(dict(step=self.wandb_steps + i, epoch=self.epoch, segment=self.resume_count,
                                            values=values)) + "\n")
                f.flush()
                os.fsync(f.fileno())  # on disk before the checkpoint that commits these rows
                self.wandb_history_bytes = f.tell()
        wandb_log(rows, self.epoch)
        self.wandb_steps += len(rows)

    def repair_wandb_history(self) -> None:
        """On resume, before anything is appended or backfilled: keep an interrupted last record
        in a recovery file (or complete one that only lost its newline) and check that every
        record the checkpoint committed is intact (utils.repair_history)."""
        self._committed_history_bytes = self.wandb_history_bytes
        action = repair_history(self._path_history, self._committed_history_bytes)
        if action is not None:
            say(f"W&B mirror: {action}")
        # Rows after the committed offset (an interrupted segment) stay; the next commit covers them.
        self.wandb_history_bytes = self._path_history.stat().st_size if self._path_history.is_file() else 0

    def reconcile_wandb_history(self) -> Dict[str, int]:
        """On resume: W&B's starting step is the number of rows its server holds. Committed rows
        it lacks (lost when a hard crash also killed the upload) are re-logged from the mirror,
        each with resume/backfilled=1; rows beyond the checkpoint (an uncommitted epoch that is
        now replayed) are counted, never silently duplicated. Returns the counts."""
        server = int(getattr(wandb.run, "starting_step", 0) or 0)
        committed = self.wandb_steps
        if getattr(wandb.run, "offline", False) and committed is not None:
            # Offline W&B knows no server rows (starting_step 0) and writes a new offline file per
            # segment: backfilling would re-log every committed row, duplicated when both are synced.
            say("W&B offline: the server's row count is unknown, so no rows are backfilled "
                f"({self._path_history} keeps every committed row)")
            return {"resume/wandb_committed_rows": committed}
        stats = {"resume/wandb_server_rows": server}
        if committed is None:  # checkpoint from before the mirror
            self.wandb_steps = server
            return stats
        stats["resume/wandb_committed_rows"] = committed
        if server < committed and self._local_history and self._path_history.is_file():
            # Only records the checkpoint committed (before its byte offset; checkpoints from before
            # the offset: all complete records). Within them the last write of a step is the
            # committed one: a replayed epoch rewrites its steps after the interrupted attempt.
            latest = {}
            for record in history_records(self._path_history, end=self._committed_history_bytes):
                if server <= record["step"] < committed:
                    latest[record["step"]] = record
            missing = [s for s in range(server, committed) if s not in latest]
            if missing:
                raise RuntimeError(f"W&B rows {missing[:5]}... are missing from {self._path_history}; cannot backfill")
            for step in range(server, committed):
                record = latest[step]
                wandb.log({"epoch": record["epoch"], **record["values"], "resume/backfilled": 1})
            stats["resume/backfilled_rows"] = committed - server
            say(f"W&B: backfilled {committed - server} committed rows the server did not receive before the crash")
        elif server > committed:
            stats["resume/server_rows_beyond_checkpoint"] = server - committed
            say(f"W&B: {server - committed} rows logged after the checkpoint are on the server; their epochs are replayed")
            self.wandb_steps = server
        else:
            self.wandb_steps = max(committed, server)
        return stats

    def epoch_summary(self, rows: Logs, steps_before: int, seconds: float) -> Dict[str, Any]:
        """One compact W&B row (epoch_summary/*, progress/*, timing/epoch_*) of this epoch's
        existing metrics (not shown on the terminal). Monitoring only."""
        def values(match):
            return [float(v) for row in rows for k, v in row.items() if match(k) and np.ndim(v) == 0]

        def mean(match):
            v = values(match)
            return float(np.mean(v)) if v else None

        summary = {}
        summary["epoch_summary/denoiser_loss_mean"] = mean(
            lambda k: k == "denoiser/train/loss_denoising" or (k.startswith("denoiser/member_") and k.endswith("/train/loss_denoising")))
        if self.ensemble is not None:
            for k in range(self.ensemble.members):
                summary[f"epoch_summary/denoiser_member_{k}_loss_mean"] = mean(lambda key: key == f"denoiser/member_{k}/train/loss_denoising")
        summary["epoch_summary/denoiser_test_loss_mean"] = mean(lambda k: k == "denoiser/test/loss_denoising")
        for name in PMPO_SUMMARY:
            summary[f"epoch_summary/pmpo_{name}_mean"] = mean(lambda k: k == f"actor_critic/train/{name}")
        summary["epoch_summary/intrinsic_raw_reward_mean"] = mean(lambda k: k.startswith("actor_critic/train/") and k.endswith("_raw_reward_mean"))
        raw_max = values(lambda k: k.startswith("actor_critic/train/") and k.endswith("_raw_reward_max"))
        summary["epoch_summary/intrinsic_raw_reward_max"] = max(raw_max) if raw_max else None
        summary["epoch_summary/intrinsic_scaled_reward_mean"] = mean(lambda k: k.startswith("actor_critic/train/") and k.endswith("_policy_reward_mean"))
        summary = {k: v for k, v in summary.items() if v is not None}
        summary.update(self.progress_counters())
        summary["progress/new_train_env_steps"] = int(self.train_dataset.num_steps - steps_before)
        self.note_gpu_peak()
        p = self._phase_seconds
        timing = {f"timing/epoch_{k}_seconds": v for k, v in p.items()}
        timing["timing/epoch_seconds"] = seconds
        if self._use_cuda:
            timing["timing/epoch_gpu_peak_allocated_gb"] = self._gpu_peak

        return {**summary, **timing}

    def announce_start(self) -> None:
        """Startup summary (stdout) once the run's state is final."""
        cfg, c = self._cfg, self._cfg.collection.train
        method = METHOD_NAMES[self._method]
        budget = (f"{c.num_steps_total} env steps = {c.first_epoch.min} initial ({c.first_epoch.mode}) + "
                  f"{(c.num_steps_total - c.first_epoch.min) // c.steps_per_epoch} epochs x {c.steps_per_epoch} "
                  f"({c.mode}); then {cfg.training.num_final_epochs} training-only epochs")
        lines = ["", "=" * 78, "Reward-free exploration",
                 f"  Domain: {cfg.env.train.domain_name}; downstream tasks: {', '.join(cfg.env.train.task_names)}",
                 f"  Exploration method: {method}",
                 f"  Seed: {self._seed}",
                 *(f"  {line}" for line in device_lines(self._device)),
                 f"  Training-data budget: {budget}",
                 f"  Output directory: {Path.cwd()}"]
        if cfg.common.resume:
            lines.append(f"  Resumed at epoch {self.epoch} (resume #{self.resume_count})")
        if self._method is None:
            lines.append("  Uniform random: no scorer refresh, no exploration-policy training, no intrinsic scores")
        lines.append("=" * 78)
        say("\n".join(lines))

    def setup_wandb(self) -> None:
        """W&B axes and provenance (no-op without an active run)."""
        if self._rank != 0 or wandb.run is None:
            return
        wandb.define_metric("epoch")
        wandb.define_metric("actor_critic/train/num_batch_train_actor_critic")
        wandb.define_metric("actor_critic/train/*", step_metric="actor_critic/train/num_batch_train_actor_critic")
        wandb.define_metric("denoiser/train/num_batch_train_denoiser")
        wandb.define_metric("denoiser/train/*", step_metric="denoiser/train/num_batch_train_denoiser")
        for k in range(self.ensemble.members if self.ensemble is not None else 0):
            wandb.define_metric(f"denoiser/member_{k}/train/num_batch_train_member")
            wandb.define_metric(f"denoiser/member_{k}/train/*", step_metric=f"denoiser/member_{k}/train/num_batch_train_member")
        # Everything else is per epoch. The globs do not overlap the per-update ones above, so
        # the result does not depend on how a W&B version orders glob matches.
        for name in EPOCH_AXIS_METRICS:
            wandb.define_metric(name, step_metric="epoch")
        wandb.config.update({"provenance": self.provenance()}, allow_val_change=True)
        wandb.summary["resume_count"] = self.resume_count
        if self._cfg.common.resume:
            self._start_logs[-1].update(self.reconcile_wandb_history())
            for milestone in self.milestones:
                if not milestone["retired"]:
                    self.milestone_summary(milestone["name"])

    def provenance(self) -> Dict[str, Any]:
        settings = None
        if self._method is not None:
            settings = dict(reward_scaling=self._lcg_reward_scaling)
            if self.laplace is not None:
                settings["laplace"] = self.laplace.settings()
            if self.plan2explore is not None:
                settings["plan2explore"] = self.plan2explore.settings()
            if self.ensemble is not None:
                settings["ensemble"] = self.ensemble.settings()
        controller = None
        if self._method is not None:  # uniform random trains no exploration policy
            ac = self.agent.actor_critic
            controller = dict(controller_settings(ac.cfg), actor_objective=ac.objective_settings(),
                              entropy_coef=float(ac.cfg.entropy_coef), beta_kl=float(ac.cfg.beta_kl),
                              prior_refresh_interval=int(ac.cfg.prior_refresh_interval))
        return dict(seed=self._seed, method=self._method or "uniform", method_name=METHOD_NAMES[self._method],
                    method_settings=settings, controller=controller, source=self.source, evaluation=self.evaluation_protocol(),
                    milestones=[m["name"] for m in self.milestones if not m["retired"]])

    def evaluation_protocol(self) -> Dict[str, Any]:
        cfg = self._cfg
        return dict(should=self._should_test, every=cfg.evaluation.every, collect_test=self._collect_test,
                    reset_seeds=self._reset_seeds[False], final_reset_seeds=self._reset_seeds[True],
                    num_episodes=cfg.collection.test.num_episodes, num_final_episodes=cfg.collection.test.num_final_episodes,
                    epsilon=cfg.collection.test.epsilon, action_source=cfg.collection.train.mode,
                    env=OmegaConf.to_container(cfg.env.test), intrinsic=self._intrinsic_eval and self._method is not None,
                    intrinsic_note=INTRINSIC_UNITS_NOTE)

    def intrinsic_evaluation(self, final: bool) -> Logs:
        """Per test episode: length, task returns and (learned methods) raw and scaled intrinsic
        returns from the frozen scorer; logged under <block>/ and saved to evaluation/*.json.
        Fresh episodes are a current-policy evaluation; reused ones a dataset diagnostic."""
        td, fresh = self.test_dataset, self._collect_test
        block = ("final_" if final else "") + ("test_eval" if fresh else "test_dataset_diagnostic")
        labels = self.test_episode_seeds
        if labels is None or len(labels) != td.num_episodes:
            labels = [None] * td.num_episodes  # unknown: never relabeled with the configured seeds
        reason = None
        if self._method is None:
            reason = "not applicable: uniform random has no intrinsic scorer"
        elif not self._intrinsic_eval:
            reason = "disabled (evaluation.intrinsic=false)"
        else:
            reason = self._intrinsic.unavailable_reason()
            if reason is not None and self._scorer_restore_note is not None:
                reason = self._scorer_restore_note
        records = []
        for i in range(td.num_episodes):
            episode = td.load_episode(i)
            record = dict(episode=i, seed=labels[i], length=len(episode))
            if reason is None:
                record.update(episode_record(*self._intrinsic.score_episode(episode)))
            records.append(record)
        row = {f"{block}/intrinsic_applicable": int(self._method is not None), f"{block}/intrinsic_scored": int(reason is None),
               f"{block}/num_episodes": td.num_episodes}
        for record in records:
            label = f"seed_{record['seed']}" if record["seed"] is not None else f"episode_{record['episode']}"
            row.update({f"{block}/{label}/{k}": v for k, v in record.items() if k not in ("episode", "seed")})
        row.update({f"{block}/{k}": v for k, v in aggregate(records).items()})

        if not fresh:
            kind = "dataset_diagnostic (reused test episodes, not the current policy)"
        else:
            kind = "current_policy_evaluation" if self._method is not None else "fresh_episodes (uniform-random actions, no policy)"
        say(evaluation_heading(final, reason if self._method is not None else None))
        for r in records:
            say(evaluation_line(r, scored=reason is None))
        if reason is None and any("intrinsic_scaled_return" not in r for r in records):
            say(f"Warning: no frozen reward scale{'; ' + self._scorer_restore_note if self._scorer_restore_note else ''}"
                "; scaled returns are N/A")
        result = dict(kind=kind, block=block, epoch=self.epoch, final=final, fresh=fresh,
                      train_env_steps=int(self.train_dataset.num_steps), total_test_env_steps=int(self.num_test_steps),
                      pmpo_updates=int(self.agent.actor_critic.updates.item()), protocol=self.evaluation_protocol(),
                      intrinsic=dict(applicable=self._method is not None, scored=reason is None, reason=reason,
                                     scorer=self._intrinsic.settings() if reason is None else None),
                      episodes=records, aggregate=aggregate(records))
        path = Path("evaluation") / (f"final_epoch_{self.epoch:05d}.json" if final else f"epoch_{self.epoch:05d}.json")
        path.parent.mkdir(exist_ok=True)
        path.write_text(json.dumps(result, indent=2, default=str))
        return [row]

    def test_diagnostics(self, final: bool = False) -> Logs:
        """A scheduled (or the final) test diagnostic: collect fresh test episodes if
        evaluation.collect_test, else reuse the test dataset; then the denoiser test loss on
        reward-free batches and the intrinsic evaluation. Task returns are neither shown nor
        logged (the test episodes keep their task-reward columns). Uses the test RNG stream and restores module modes; no weights,
        optimizers, schedules or training data change."""
        td = self.test_dataset
        t0 = time.time()
        training_rng = rng_state()
        modes = {name: module.training for name, module in self.agent.named_modules()}
        set_rng_state(self.test_rng)
        if final:
            phase()  # the final evaluation is not part of an epoch
        try:
            to_log, new_steps = [], 0
            if self._collect_test:
                to_log = self.collect_test(final)
                new_steps = sum(x[f"{td.name}/new_env_steps"] for x in to_log if f"{td.name}/new_env_steps" in x)
            self.num_test_steps += new_steps
            to_log += self.test_agent()
            # The frozen scorer on the real test episodes (test RNG stream; nothing refreshed).
            to_log += self.intrinsic_evaluation(final)
            summary = dict(fresh=int(self._collect_test), num_episodes=td.num_episodes, num_steps=td.num_steps,
                           new_env_steps=new_steps, total_new_env_steps=self.num_test_steps)
            prefix = "final_test" if final else "test"
            to_log.append({f"{prefix}/{k}": v for k, v in summary.items()})
        finally:
            self.test_rng = rng_state()
            set_rng_state(training_rng)
            for name, module in self.agent.named_modules():
                module.training = modes[name]
        seconds = time.time() - t0
        if not final:
            self._phase_seconds["test"] = seconds
        to_log.append({f"timing/{'final_' if final else ''}test_diagnostic_seconds": seconds})
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
            for _ in Progress(range(steps), desc="Exploration-policy training", unit="updates"):
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

        desc = "Denoiser training" + ("" if member is None else f" (member {member + 1}/{self.ensemble.members})")
        for i in Progress(range(num_steps), desc=desc, unit="updates", disable=self._rank > 0):
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
        for batch in data_loader:  # no terminal output: the test loss is logged to W&B only
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
        # Frozen intrinsic scorer of the latest refresh (what a resume needs to evaluate without
        # refreshing). Members, Laplace banks and Plan2Explore heads are checkpointed with their
        # own fields; LCG's round and the full ensemble's RunningRMS scale only here.
        state["lcg_scorer"] = None if self._lcg_lifecycle is None else self._lcg_lifecycle.state_dict()
        state["ensemble_scorer"] = None if self._ensemble_scorer is None else self._ensemble_scorer.state_dict()
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
        # Absent in checkpoints from before frozen scorer state was saved (MISSING, not None).
        lcg_state, scale_state = state_dict.pop("lcg_scorer", MISSING), state_dict.pop("ensemble_scorer", MISSING)
        diagnostic_state = state_dict.pop("pmpo_diagnostic", None)
        state_dict.setdefault("test_episode_seeds", None)  # before seed labels: unknown (never relabeled)
        state_dict.setdefault("resume_count", 0)
        state_dict.setdefault("wandb_steps", None)  # before the mirror: no backfill
        state_dict.setdefault("wandb_history_bytes", None)  # before committed offsets: all complete records count
        state_dict.setdefault("ensemble", None)  # checkpoints from before ensembles: single denoiser
        saved = (state_dict["ensemble"] or {}).get("settings")
        expected = self.ensemble.settings() if self.ensemble is not None else None
        if saved != expected:  # checked before any weights are loaded
            raise ValueError(f"Checkpoint was trained with ensemble settings {saved}, the config gives {expected} "
                             "(None: single denoiser); a resume must not change them")
        state_dict.setdefault("laplace", None)  # checkpoints from before the Laplace ensemble
        if self.laplace is not None:
            self.laplace.check_settings(state_dict["laplace"])  # before any weights are loaded
        elif state_dict["laplace"] is not None:
            raise ValueError("Checkpoint was trained with a Laplace sample bank, the config selects "
                             f"intrinsic_reward method {self._method!r}; a resume must not change it")
        state_dict.setdefault("plan2explore", None)  # checkpoints from before Plan2Explore
        if self.plan2explore is not None:
            self.plan2explore.check_settings(state_dict["plan2explore"])  # before any weights are loaded
        elif state_dict["plan2explore"] is not None:
            raise ValueError("Checkpoint was trained with Plan2Explore feature predictors, the config selects "
                             f"intrinsic_reward method {self._method!r}; a resume must not change it")
        # The controller (Beta/Gaussian, slow critic, frame-history convention, horizon) before any
        # weights are loaded: a mismatch is refused, never reinterpreted.
        check_controller_state(state_dict["agent"].get("actor_critic._extra_state"), self.agent.actor_critic.cfg,
                               source="Checkpoint's exploration controller")
        check_same_packages(state_dict["source"])
        super().load_state_dict(state_dict)
        self.restore_collection_state(collection)
        self.restore_scorers(lcg_state, scale_state)
        if self._pmpo_diagnostic is not None:
            # After the agent: the history is validated against the restored PMPOBeta.updates.
            self._pmpo_diagnostic.load_state_dict(diagnostic_state)

    def restore_scorers(self, lcg_state: Any, scale_state: Any) -> None:
        """After the weights: the latest refresh's frozen scorer, wired as the refresh left it.
        Nothing is refreshed, redrawn, refitted or trained; the next epoch's refresh replaces it
        as usual. A checkpoint without this state (older code) keeps the scorer unavailable and
        says why at evaluation; missing state is never re-created."""
        actor_critic = self.agent.actor_critic
        if self._lcg_lifecycle is None and lcg_state not in (MISSING, None):
            raise ValueError("Checkpoint has an LCG scorer, the config selects another method")
        if self._ensemble_scorer is None and scale_state not in (MISSING, None):
            raise ValueError("Checkpoint has an ensemble-disagreement scorer, the config selects another method")
        if self._lcg_lifecycle is not None:
            if lcg_state is MISSING:
                self._scorer_restore_note = ("the checkpoint predates saved LCG scorer state (precision, candidate bank, "
                                             "RunningRMS); the frozen scorer cannot be restored and is not re-created")
            elif lcg_state is not None:
                self._lcg_lifecycle.load_state_dict(lcg_state, self.agent.denoiser)
                actor_critic.set_intrinsic_reward_fn(self._lcg_lifecycle.intrinsic_reward_fn)
        if self._ensemble_scorer is not None:
            if scale_state is MISSING:
                if self._ensemble_scorer.cfg.reward_scaling == "rms":
                    self._scorer_restore_note = ("the checkpoint predates the saved RunningRMS scale of the "
                                                 f"{METHOD_NAMES[self._method]} scorer; scaled returns cannot be reproduced")
            elif scale_state is not None:
                self._ensemble_scorer.load_state_dict(scale_state, self._device)
            if self.epoch > 0 and (self.laplace is None or self.laplace.perturbations is not None):
                actor_critic.set_intrinsic_reward_fn(self._ensemble_scorer)
        if self.plan2explore is not None and self.plan2explore.heads is not None:
            actor_critic.set_intrinsic_reward_fn(self.plan2explore)

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

    def successor_observations(self) -> Dict[int, torch.Tensor]:
        """Observation after the last stored step of each unfinished training episode (the
        collector's current observation), by episode id."""
        state = self._train_collector_state
        successors = {}
        if state.loop.obs is not None:
            for i, episode_id in enumerate(state.episode_ids):
                if episode_id is not None:
                    successors[episode_id] = state.loop.obs[i]
        return successors

    def publish_milestone(self, name: str) -> Path:
        cfg = self._cfg
        successors = self.successor_observations()
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
        if self.plan2explore is not None:
            # Provenance only: the heads are exploration state (checkpoints/state.pt); downstream
            # training needs just denoiser.pt and the replay.
            p2e = self.plan2explore
            manifest["plan2explore"] = dict(
                num_members=p2e.cfg.num_members, feature_layers=list(p2e.cfg.feature_layers),
                feature_dim=int(p2e.feature_dim), feature_sigma=float(p2e.cfg.feature_sigma),
                pool_size=int(p2e.cfg.pool_size), probe_sha256=p2e.extractor.probe_sha256,
                last_refresh_epoch=p2e.last_refresh_epoch, refreshes=int(p2e.refreshes),
                note="single denoiser; feature predictors are not part of the milestone")
        if self.laplace is not None:
            # Provenance only: denoiser.pt is the unperturbed trained model; the sample bank is
            # exploration state (checkpoints/state.pt), not used downstream.
            lap = self.laplace
            manifest["laplace"] = dict(
                num_members=lap.cfg.num_members, posterior_temperature=float(lap.cfg.posterior_temperature),
                selected_include=list(lap.cfg.theta_s.include), selected_dim=int(lap.dim),
                refresh_epoch=lap.refresh_epoch, refreshes=int(lap.refreshes),
                note="denoiser.pt is the unperturbed base model; sampled members are not part of the milestone")
        path = publish_milestone(self._path_milestones, name, self.agent.denoiser.state_dict(),
                                 self.train_dataset, successors, manifest, member_states)
        say(f"Milestone {name} published: > {Path(path).resolve()}")
        self.milestone_summary(name)
        return path

    def milestone_summary(self, name: str) -> None:
        """A published milestone's identity in the W&B summary (milestones/<name>)."""
        path = self._path_milestones / name
        if self._rank == 0 and wandb.run is not None and path.exists():
            wandb.summary[f"milestones/{name}"] = dict(path=str(path.resolve()), **read_manifest(path).get("identity", {}))

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

METHOD_NAMES = {None: "uniform random", "lcg": "LCG", "ensemble_disagreement": "full diffusion ensemble",
                "plan2explore_disagreement": "Plan2Explore-style ensemble", "laplace_disagreement": "Laplace ensemble"}
REFRESH_WORK = {"lcg": "historical precision and candidate bank on this epoch's denoiser",
                "ensemble_disagreement": "freezing the trained members (no refresh computation)",
                "plan2explore_disagreement": "replay feature encoding and fresh predictor heads",
                "laplace_disagreement": "historical precision and a new posterior sample bank"}
# Existing PMPO metrics averaged over the epoch's updates in epoch_summary/*.
PMPO_SUMMARY = ("loss_actor", "loss_value", "policy_entropy", "kl_prior", "positive_fraction", "negative_fraction",
                "actor_grad_norm", "value_grad_norm", "return_mean")
# W&B metrics plotted against epoch (setup_wandb): every logged key outside the per-update
# denoiser/actor_critic train rows.
EPOCH_AXIS_METRICS = ("epoch_summary/*", "progress/*", "timing/*", "duration", "resume/*", "milestone/*",
                      "train_dataset/*", "test_dataset/*", "denoiser/test/*", "test/*", "final_test/*",
                      "test_eval/*", "final_test_eval/*", "test_dataset_diagnostic/*", "final_test_dataset_diagnostic/*",
                      "ensemble/*", "plan2explore/*", "laplace/*", "lcg/*", "classification_metrics/*", "length")
INTRINSIC_UNITS_NOTE = ("Intrinsic returns are in each method's own units (LCG score, pixel variance, feature "
                        "variance, ...); they compare checkpoints within one method, never methods with each other.")


MISSING = object()  # a checkpoint field that older code did not save


def without_task_returns(rows: Logs) -> Logs:
    """Collection rows without the per-episode return (the env reward is a downstream task's
    reward, so it is a task return); episode ids, lengths and counts stay. Replay is unchanged."""
    return [{k: v for k, v in row.items() if k != "return"} for row in rows]


def lcg_refresh_metrics(lcg, dataset: Dataset) -> Dict[str, Any]:
    """LCG refresh identity for W&B (the lifecycle's stdout lines are off in exploration)."""
    if not lcg.ready:
        return {}  # no round was set up; the missing hook is reported by PMPO
    return {"lcg/round": lcg.round_id, "lcg/replay_steps": int(dataset.num_steps),
            "lcg/theta_s_dim": int(lcg._theta_s_dim), "lcg/h_D_fingerprint": float(lcg.h_D.sum().item()),
            "lcg/banks_fingerprint": float(sum(s.item() for s in lcg.bank.sigmas))}


def training_intrinsic_line(rows: Logs) -> str:
    """Raw and scaled intrinsic reward per imagined step over this epoch's PMPO updates: mean of
    the per-update means, max of the per-update maxima (the epoch_summary/* definitions)."""
    def stat(suffix, reduce):
        values = [float(v) for row in rows for k, v in row.items()
                  if k.startswith("actor_critic/train/") and k.endswith(suffix)]
        return reduce(values) if values else None
    parts = []
    for label, key in (("raw", "_raw_reward"), ("scaled", "_policy_reward")):
        mean, high = stat(f"{key}_mean", lambda v: float(np.mean(v))), stat(f"{key}_max", max)
        if mean is not None:
            parts.append(f"{label} mean {mean:.4g}, max {high:.4g}")
    return "Imagined intrinsic reward/step: " + (" | ".join(parts) if parts else "N/A")


def evaluation_heading(final: bool, reason: Optional[str]) -> str:
    """Per-seed episode totals on real test episodes from the frozen scorer (definitions in
    REWARD_FREE_EXPERIMENT_NOTES.md and evaluation/*.json). A reason is shown only when a
    learned method is not scored (uniform random: its lines say intrinsic N/A)."""
    heading = "Final intrinsic evaluation:" if final else "Intrinsic evaluation:"
    return heading if reason is None else f"{heading} N/A ({reason})"


def evaluation_line(record: Dict[str, Any], scored: bool) -> str:
    name = f"seed {record['seed']}" if record["seed"] is not None else f"episode {record['episode']}"
    text = f"  {name} | {record['length']} steps"
    if scored:
        text += f" | raw return {record['intrinsic_raw_return']:.4g}"
        if "intrinsic_scaled_return" in record:
            text += f" | scaled return {record['intrinsic_scaled_return']:.4g}"
    else:
        text += " | intrinsic N/A"
    return text


def check_reset_seeds(cfg: DictConfig, final: bool) -> Optional[List[int]]:
    """evaluation.reset_seeds / final_reset_seeds: null (a random reset seed per collection, the
    earlier behaviour) or distinct non-negative integers, one test episode each; the matching
    episode count must equal their number and test collection must use one env."""
    key = "final_reset_seeds" if final else "reset_seeds"
    seeds = OmegaConf.select(cfg, f"evaluation.{key}", default=None)
    if seeds is None:
        return None
    seeds = list(seeds) if not isinstance(seeds, (int, str)) else seeds
    if not isinstance(seeds, list) or not seeds or len(set(seeds)) != len(seeds) or any(
            isinstance(x, bool) or not isinstance(x, int) or x < 0 for x in seeds):
        raise ValueError(f"evaluation.{key} must be null or a list of distinct non-negative integers, got {seeds!r}")
    count_key = "num_final_episodes" if final else "num_episodes"
    if cfg.collection.test[count_key] != len(seeds):
        raise ValueError(f"evaluation.{key} has {len(seeds)} seeds but collection.test.{count_key}="
                         f"{cfg.collection.test[count_key]}; one episode is collected per seed")
    if cfg.collection.test.num_envs != 1:
        raise ValueError(f"evaluation.{key} needs collection.test.num_envs=1 (one seeded episode per collect)")
    return seeds


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
