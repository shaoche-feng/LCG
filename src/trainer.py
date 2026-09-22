from functools import partial
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from hydra.utils import instantiate
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm, trange
import wandb

from agent import Agent, get_action_space_kwargs
from coroutines.collector import make_collector, NumToCollect
from coroutines.env_loop import EnvResetSeedState, RolloutHxCxState
from data import BatchSampler, collate_segments_to_batch, Dataset, DatasetTraverser
from data.batch_sampler import COMPONENT_SEED_ID
from envs import make_atari_env, make_dm_control_env, WorldModelEnv
from lcg import LCGConfig, LCGLifecycle
from models.drq_actor_critic import DrQActorCritic
from utils import (
    broadcast_if_needed,
    build_ddp_wrapper,
    CheckpointableGroup,
    CommonTools,
    configure_opt,
    count_parameters,
    derive_component_seed,
    derive_torch_generator,
    get_git_commit_hash,
    get_lr_sched,
    keep_agent_copies_every,
    Logs,
    process_confusion_matrices_if_any_and_compute_classification_metrics,
    RNGState,
    save_info_for_import_script,
    save_with_backup,
    set_seed,
    StateDictMixin,
    try_until_no_except,
    wandb_log,
)


class ResumeFidelityState:
    """Groups every piece of state needed for the actor-critic's imagined-rollout machinery
    to resume EXACTLY (not just approximately) -- distinct from self.opt/self.lr_sched/
    self.rng_state, which Trainer already checkpoints independently:

      - Each component's OWN BatchSampler RNG stream (data.batch_sampler.BatchSampler +
        utils.derive_component_seed): denoiser/rew_end_model/actor_critic each draw from an
        independent numpy.random.Generator instead of sharing global np.random state, so one
        component's sampling can never desync another's.
      - WorldModelEnv's preload-block replay state (see WorldModelEnv.make_generator_init):
        which exact 256 SegmentIds are in the current in-flight preload block and how far
        into it we've yielded, so resuming replays that block instead of drawing a fresh one.
      - WorldModelEnv's live rollout buffers (obs_buffer/act_buffer/hx_rew_end/cx_rew_end/
        ep_len) -- absent (None) until the first real use.
      - The actor-critic's own rollout LSTM hx/cx (coroutines.env_loop.RolloutHxCxState).
      - For DrQActorCritic specifically: its three independent exploration-noise RNG streams
        (imagined-training/real-collection/eval, see DrQExplorationState/DrQGeneratorState's
        docstrings) -- checkpointed explicitly here for the exact same reason rollout_hx_cx_state
        is: nn.Module.load_state_dict()'s recursion would silently never invoke an override on
        a nested module (see DrQExplorationState's docstring for the full explanation).
      - The real train/test collectors' deterministic env.reset() seed streams
        (coroutines.env_loop.EnvResetSeedState) -- actor-critic-agnostic (both ActorCritic and
        DrQActorCritic runs use these). Together with make_collector's flush_before_reset mode
        (see Trainer.__init__'s train-collector construction), this is what makes ongoing real
        train collection resume exactly: the collector is always rebuilt from a fully known,
        freshly-reset state at every checkpoint boundary, so no OTHER collector-local state
        (buffer, episode_ids, frame-stack/LSTM hx/cx, or the real env's own internal state)
        needs to be separately checkpointed at all -- only this seed stream's position does.

    Auto-discovered and checkpointed by Trainer's StateDictMixin machinery exactly like
    self.opt/self.lr_sched/self.rng_state already are, via a single
    `self.resume_fidelity_state = ResumeFidelityState(...)` attribute (see Trainer.__init__).
    Every piece is optional/gracefully-absent (model_free runs have no WorldModelEnv; a
    checkpoint saved before actor_critic ever trained has no rollout buffers yet; a non-DrQ run
    has no drq_exploration_states; a static-dataset run has no collectors at all), so a
    checkpoint taken at ANY point round-trips without special-casing by the caller.
    """

    def __init__(
        self,
        batch_samplers: Dict[str, BatchSampler],
        world_model_env: Optional[WorldModelEnv],
        rollout_hx_cx_state: Optional[RolloutHxCxState],
        drq_exploration_states: Optional[Dict[str, Any]] = None,
        collector_reset_states: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.batch_samplers = batch_samplers
        self.world_model_env = world_model_env
        self.rollout_hx_cx_state = rollout_hx_cx_state
        # name -> object exposing state_dict()/load_state_dict() (DrQExplorationState or
        # DrQGeneratorState); None for a non-DrQ actor-critic. See Trainer.__init__.
        self.drq_exploration_states = drq_exploration_states
        # name -> EnvResetSeedState (e.g. "train"/"test"); None for a static-dataset run.
        self.collector_reset_states = collector_reset_states

    def state_dict(self) -> Dict[str, Any]:
        sd: Dict[str, Any] = {"batch_samplers": {name: bs.state_dict() for name, bs in self.batch_samplers.items()}}
        if self.world_model_env is not None:
            sd["world_model_env_preload"] = self.world_model_env.preload_state_dict()
            sd["world_model_env_rollout"] = self.world_model_env.rollout_state_dict()
        if self.rollout_hx_cx_state is not None:
            sd["rollout_hx_cx"] = self.rollout_hx_cx_state.state_dict()
        if self.drq_exploration_states is not None:
            sd["drq_exploration_states"] = {k: v.state_dict() for k, v in self.drq_exploration_states.items()}
        if self.collector_reset_states is not None:
            sd["collector_reset_states"] = {k: v.state_dict() for k, v in self.collector_reset_states.items()}
        return sd

    def load_state_dict(self, state_dict: Dict[str, Any]) -> None:
        for name, sub_sd in state_dict.get("batch_samplers", {}).items():
            if name in self.batch_samplers:
                self.batch_samplers[name].load_state_dict(sub_sd)
        if self.world_model_env is not None and "world_model_env_preload" in state_dict:
            # Must happen before the imagined-rollout env_loop's first real .send() in this
            # process -- see WorldModelEnv.make_generator_init's docstring. Trainer calls
            # load_state_dict() (hence this) during __init__, well before trainer.run()
            # starts any actual training, so that ordering holds.
            self.world_model_env.load_preload_state_dict(state_dict["world_model_env_preload"])
            rollout_sd = state_dict.get("world_model_env_rollout")
            if rollout_sd is not None:
                self.world_model_env.load_rollout_state_dict(rollout_sd)
        if self.rollout_hx_cx_state is not None and "rollout_hx_cx" in state_dict:
            self.rollout_hx_cx_state.load_state_dict(state_dict["rollout_hx_cx"])
        if self.drq_exploration_states is not None and "drq_exploration_states" in state_dict:
            for k, v in self.drq_exploration_states.items():
                if k in state_dict["drq_exploration_states"]:
                    v.load_state_dict(state_dict["drq_exploration_states"][k])
        if self.collector_reset_states is not None and "collector_reset_states" in state_dict:
            for k, v in self.collector_reset_states.items():
                if k in state_dict["collector_reset_states"]:
                    v.load_state_dict(state_dict["collector_reset_states"][k])


class Trainer(StateDictMixin):
    def __init__(self, cfg: DictConfig, root_dir: Path) -> None:
        torch.backends.cuda.matmul.allow_tf32 = True
        OmegaConf.resolve(cfg)
        self._cfg = cfg
        self._rank = dist.get_rank() if dist.is_initialized() else 0
        self._world_size = dist.get_world_size() if dist.is_initialized() else 1

        # Pick a random seed, unless an explicit reproducible seed was requested (needed for
        # the resume-fidelity integration test and the A-E state-effect experiment; defaults
        # to null, so this preserves the exact prior behavior -- a fresh OS-entropy seed
        # every launch/resume -- for every existing config/run).
        if cfg.common.seed is not None:
            self._resolved_seed = cfg.common.seed
        else:
            self._resolved_seed = torch.seed() % 10 ** 9
        set_seed(self._resolved_seed)

        # `self.rng_state`'s presence (no leading underscore) means StateDictMixin picks it
        # up automatically in state_dict()/load_state_dict(), exactly like self.opt and
        # self.lr_sched already are -- no other code path needs to change for RNG to be
        # captured/restored at every checkpoint save/load. See load_state_checkpoint() for
        # the backward-compatibility handling of pre-existing state.pt files that predate
        # this and have no "rng_state" key.
        self.rng_state = RNGState()

        # Opt-in deterministic-CUDA mode (default False, unchanged behavior otherwise). Not a
        # full determinism guarantee: cuDNN's deterministic mode still has known caveats for
        # some backward kernels (e.g. certain LSTM/conv backward paths), and
        # torch.use_deterministic_algorithms(True) is deliberately NOT forced here since it
        # can raise on ops the diffusion sampler / LSTM path use without a deterministic
        # implementation. The resume-fidelity integration test documents what this does and
        # does not guarantee in practice.
        if getattr(cfg.common, "deterministic_cuda", False):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        # Device
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu", self._rank)
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
        self._is_model_free = cfg.training.model_free

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
        make_env = {"atari": make_atari_env, "dm_control": make_dm_control_env}[cfg.env.train.type]
        env_kwargs_train = {k: v for k, v in cfg.env.train.items() if k != "type"}
        env_kwargs_test = {k: v for k, v in cfg.env.test.items() if k != "type"}

        if self._rank == 0:
            train_env = make_env(num_envs=cfg.collection.train.num_envs, device=self._device, **env_kwargs_train)
            test_env = make_env(num_envs=cfg.collection.test.num_envs, device=self._device, **env_kwargs_test)
            action_kwargs = get_action_space_kwargs(test_env)
        else:
            action_kwargs = None
        action_kwargs, = broadcast_if_needed(action_kwargs)

        # Create models
        self.agent = Agent(instantiate(cfg.agent, **action_kwargs)).to(self._device)
        self._agent = build_ddp_wrapper(**self.agent._modules) if dist.is_initialized() else self.agent

        if cfg.initialization.path_to_ckpt is not None:
            self.agent.load(**cfg.initialization)

        # Whether this run's actor-critic is the DrQ-v2-style continuous-action module (selected
        # via config/agent/*.yaml's actor_critic._target_) rather than the original ActorCritic
        # -- read once here and reused everywhere below that needs a different code path (its
        # own optimizer pair, its own collector bindings/RNG streams, its own train_agent()
        # call sequence) instead of a single shared one.
        self._is_drq = isinstance(self.agent.actor_critic, DrQActorCritic)

        if self._is_drq:
            # Three independent RNG streams for DrQ's exploration noise (see
            # data.batch_sampler.COMPONENT_SEED_ID and models.drq_actor_critic's module
            # docstring, RNG-isolation section): imagined-training, real-env train collection,
            # and eval/test collection each get their OWN generator and must never perturb one
            # another. Constructed on self._device (not CPU) since DrQActorCritic's
            # sample_action draws noise via torch.randn(..., device=..., generator=...) at
            # wherever the model itself lives. Assigned onto the actor-critic's own state
            # objects (not kept as local variables) so a later cfg.common.resume load_state_dict
            # call (which mutates these generators IN PLACE via generator.set_state(...), see
            # DrQExplorationState/DrQGeneratorState) transparently updates every reference that
            # was handed out below (collector bindings, setup_training) too.
            self.agent.actor_critic.real_collection_exploration_state.generator = derive_torch_generator(
                self._resolved_seed, COMPONENT_SEED_ID["drq_real_collection_noise"], device=self._device
            )
            self.agent.actor_critic.eval_exploration_state.generator = derive_torch_generator(
                self._resolved_seed, COMPONENT_SEED_ID["drq_eval_noise"], device=self._device
            )

        # Collectors
        if not self._is_static_dataset and self._rank == 0:
            if self._is_drq:
                # Bound to this specific env AND this specific noise stream (see
                # DrQActorCritic.make_collector_binding's docstring) -- the train and test
                # collectors get their own independent cold-start/frame-stack state and their
                # own independent RNG stream, never shared with each other or with the
                # imagined-training loop below, even though all three drive the SAME learned
                # weights.
                train_model = self.agent.actor_critic.make_collector_binding(
                    env=train_env,
                    noise_generator=self.agent.actor_critic.real_collection_exploration_state.generator,
                    deterministic=False,
                )
                # deterministic=True: evaluation must be observation -> actor -> mu (rescaled)
                # -> env action, with NO exploration Gaussian noise -- cfg.collection.test.epsilon
                # defaulting to 0.0 already suppresses env_loop's SEPARATE random-action-
                # replacement, but that does nothing about DrQActorCritic's own noise, which is
                # only actually skipped by sample_action's deterministic branch. See
                # DrQPolicyBinding's docstring.
                test_model = self.agent.actor_critic.make_collector_binding(
                    env=test_env,
                    noise_generator=self.agent.actor_critic.eval_exploration_state.generator,
                    deterministic=True,
                )
            else:
                train_model = self.agent.actor_critic
                test_model = self.agent.actor_critic

            # Deterministic, checkpointable env.reset() seed streams -- independent of every
            # other RNG stream, actor-critic-agnostic (both ActorCritic and DrQActorCritic use
            # these) -- see EnvResetSeedState's docstring. Wired into ResumeFidelityState below.
            self._train_collector_reset_state = EnvResetSeedState(
                np.random.default_rng(derive_component_seed(self._resolved_seed, COMPONENT_SEED_ID["train_collector_reset"]))
            )
            self._test_collector_reset_state = EnvResetSeedState(
                np.random.default_rng(derive_component_seed(self._resolved_seed, COMPONENT_SEED_ID["test_collector_reset"]))
            )

            self._train_collector = make_collector(
                train_env,
                train_model,
                self.train_dataset,
                cfg.collection.train.epsilon,
                # reset_every_collect+flush_before_reset together: a checkpoint-safe collection
                # boundary at the end of EVERY epoch's collection batch (right before
                # Trainer.save_checkpoint() runs) -- whatever's been collected so far is always
                # persisted, then the whole collector (env, frame-stack/LSTM state) is rebuilt
                # from a deterministic seed, identically whether or not a checkpoint/resume
                # actually happens there. See make_collector's own docstring for the full
                # rationale and the tradeoff (episodes now cut off at steps_per_epoch rather
                # than running to their natural length, in exchange for exact resume fidelity
                # of real-env collection -- required because this Trainer's checkpoint/resume
                # path otherwise has no way to reproduce hidden env_loop/collector/environment
                # state that isn't part of any checkpointed field).
                reset_every_collect=True,
                flush_before_reset=True,
                reset_seed_state=self._train_collector_reset_state,
            )
            self._test_collector = make_collector(
                test_env,
                test_model,
                self.test_dataset,
                cfg.collection.test.epsilon,
                reset_every_collect=True,
                reset_seed_state=self._test_collector_reset_state,
            )

        ######################################################

        # Optimizers and LR schedulers

        def build_opt(name: str) -> torch.optim.AdamW:
            return configure_opt(getattr(self.agent, name), **getattr(cfg, name).optimizer)

        def build_lr_sched(opt: torch.optim.Optimizer, num_warmup_steps: int) -> torch.optim.lr_scheduler.LambdaLR:
            return get_lr_sched(opt, num_warmup_steps)

        self._model_names = ["denoiser", "rew_end_model", "actor_critic"]

        opt_denoiser = build_opt("denoiser")
        opt_rew_end_model = build_opt("rew_end_model")

        if self._is_drq:
            # DrQActorCritic trains via two disjoint-parameter optimizers (opt_critic: encoder +
            # critic's own trunk + Q1/Q2; opt_actor: actor's own trunk + MLP -- see
            # DrQActorCritic.critic_update/actor_update's docstrings), not the single-optimizer
            # pattern the rest of Trainer uses -- wrapped in one CheckpointableGroup so
            # self.opt.actor_critic still checkpoints/round-trips transparently through
            # CommonTools/StateDictMixin exactly like a plain AdamW does for the other
            # components and for the original ActorCritic.
            ac = self.agent.actor_critic
            opt_critic_ac = configure_opt(nn.ModuleList([ac.encoder, ac.critic]), **cfg.drq.critic_optimizer)
            opt_actor_ac = configure_opt(ac.actor, **cfg.drq.actor_optimizer)
            opt_actor_critic = CheckpointableGroup(critic=opt_critic_ac, actor=opt_actor_ac)
            # Single shared warmup step count (cfg.actor_critic.training.lr_warmup_steps, the
            # same training-cadence config both actor-critic implementations read their
            # steps_per_epoch/batch_size/etc from) applied independently to each optimizer --
            # DrQ has no single combined optimizer for get_lr_sched to wrap.
            warmup_steps = cfg.actor_critic.training.lr_warmup_steps
            lr_sched_actor_critic = CheckpointableGroup(
                critic=build_lr_sched(opt_critic_ac, warmup_steps), actor=build_lr_sched(opt_actor_ac, warmup_steps)
            )
        else:
            opt_actor_critic = build_opt("actor_critic")
            lr_sched_actor_critic = build_lr_sched(opt_actor_critic, cfg.actor_critic.training.lr_warmup_steps)

        self.opt = CommonTools(opt_denoiser, opt_rew_end_model, opt_actor_critic)
        self.lr_sched = CommonTools(
            build_lr_sched(opt_denoiser, cfg.denoiser.training.lr_warmup_steps),
            build_lr_sched(opt_rew_end_model, cfg.rew_end_model.training.lr_warmup_steps),
            lr_sched_actor_critic,
        )

        # Data loaders

        make_data_loader = partial(
            DataLoader,
            dataset=self.train_dataset,
            collate_fn=collate_segments_to_batch,
            num_workers=num_workers,
            persistent_workers=(num_workers > 0),
            pin_memory=self._use_cuda,
            pin_memory_device=str(self._device) if self._use_cuda else "",
        )

        # rng=derive_component_seed(...): each component's BatchSampler draws from its OWN
        # independent numpy Generator instead of sharing global np.random state (see
        # ResumeFidelityState's docstring) -- required so that, e.g., actor_critic's
        # WorldModelEnv preload burst can never desync denoiser/rew_end_model's sampling.
        make_batch_sampler = lambda *args, **kwargs: BatchSampler(
            self.train_dataset, self._rank, self._world_size, *args,
            rng=derive_component_seed(self._resolved_seed, COMPONENT_SEED_ID[kwargs.pop("_component_name")]),
            **kwargs,
        )

        def get_sample_weights(sample_weights: List[float]) -> Optional[List[float]]:
            return None if (self._is_static_dataset and cfg.static_dataset.ignore_sample_weights) else sample_weights

        c = cfg.denoiser.training
        seq_length = cfg.agent.denoiser.inner_model.num_steps_conditioning + 1 + c.num_autoregressive_steps
        bs_denoiser = make_batch_sampler(c.batch_size, seq_length, get_sample_weights(c.sample_weights), _component_name="denoiser")
        dl_denoiser_train = make_data_loader(
            batch_sampler=bs_denoiser,
            generator=derive_torch_generator(self._resolved_seed, COMPONENT_SEED_ID["denoiser"]),
        )
        dl_denoiser_test = DatasetTraverser(self.test_dataset, c.batch_size, seq_length)

        c = cfg.rew_end_model.training
        bs_rew_end_model = make_batch_sampler(
            c.batch_size, c.seq_length, get_sample_weights(c.sample_weights), can_sample_beyond_end=True,
            _component_name="rew_end_model",
        )
        dl_rew_end_model_train = make_data_loader(
            batch_sampler=bs_rew_end_model,
            generator=derive_torch_generator(self._resolved_seed, COMPONENT_SEED_ID["rew_end_model"]),
        )
        dl_rew_end_model_test = DatasetTraverser(self.test_dataset, c.batch_size, c.seq_length)

        self._data_loader_train = CommonTools(dl_denoiser_train, dl_rew_end_model_train, None)
        self._data_loader_test = CommonTools(dl_denoiser_test, dl_rew_end_model_test, None)

        # RL env

        lcg_enabled = bool(getattr(cfg, "intrinsic_reward", None) is not None and cfg.intrinsic_reward.enabled)

        if self._is_model_free:
            rl_env = make_env(num_envs=cfg.actor_critic.training.batch_size, device=self._device, **env_kwargs_train)
            bs_actor_critic = None

        else:
            c = cfg.actor_critic.training
            sl = cfg.agent.denoiser.inner_model.num_steps_conditioning
            bs_actor_critic = make_batch_sampler(c.batch_size, sl, get_sample_weights(c.sample_weights), _component_name="actor_critic")
            dl_actor_critic = make_data_loader(
                batch_sampler=bs_actor_critic,
                generator=derive_torch_generator(self._resolved_seed, COMPONENT_SEED_ID["actor_critic"]),
            )
            wm_env_cfg = instantiate(cfg.world_model_env)
            rl_env = WorldModelEnv(
                self.agent.denoiser, self.agent.rew_end_model, dl_actor_critic, wm_env_cfg,
                return_imagined_candidate=lcg_enabled,
            )

            if cfg.training.compile_wm:
                rl_env.predict_next_obs = torch.compile(rl_env.predict_next_obs, mode="reduce-overhead")
                rl_env.predict_rew_end = torch.compile(rl_env.predict_rew_end, mode="reduce-overhead")

        # Setup training
        sigma_distribution_cfg = instantiate(cfg.denoiser.sigma_distribution)
        if self._is_drq:
            actor_critic_loss_cfg = instantiate(cfg.drq.loss)
            imagination_noise_generator = derive_torch_generator(
                self._resolved_seed, COMPONENT_SEED_ID["drq_imagination_noise"], device=self._device
            )
            self.agent.setup_training(
                sigma_distribution_cfg, actor_critic_loss_cfg, rl_env, imagination_noise_generator
            )
        else:
            actor_critic_loss_cfg = instantiate(cfg.actor_critic.actor_critic_loss)
            self.agent.setup_training(sigma_distribution_cfg, actor_critic_loss_cfg, rl_env)

        # See ResumeFidelityState's docstring: everything needed for the actor-critic's
        # imagined-rollout machinery to resume exactly, grouped into one StateDictMixin-
        # discovered field (non-underscore attribute name) alongside self.opt/self.lr_sched/
        # self.rng_state.
        batch_samplers = {"denoiser": bs_denoiser, "rew_end_model": bs_rew_end_model}
        if bs_actor_critic is not None:
            batch_samplers["actor_critic"] = bs_actor_critic
        drq_exploration_states = None
        if self._is_drq:
            drq_exploration_states = {
                "exploration_state": self.agent.actor_critic.exploration_state,
                "real_collection_exploration_state": self.agent.actor_critic.real_collection_exploration_state,
                "eval_exploration_state": self.agent.actor_critic.eval_exploration_state,
            }
        collector_reset_states = None
        if not self._is_static_dataset and self._rank == 0:
            collector_reset_states = {
                "train": self._train_collector_reset_state,
                "test": self._test_collector_reset_state,
            }
        self.resume_fidelity_state = ResumeFidelityState(
            batch_samplers=batch_samplers,
            world_model_env=rl_env if not self._is_model_free else None,
            rollout_hx_cx_state=self.agent.actor_critic.rollout_hx_cx_state,
            drq_exploration_states=drq_exploration_states,
            collector_reset_states=collector_reset_states,
        )

        # LCG intrinsic-reward lifecycle -- disabled unless
        # cfg.intrinsic_reward.enabled is True (default False, see
        # config/intrinsic_reward/lcg.yaml); only meaningful for the world-model
        # (non-model-free) path. When disabled, self._lcg_lifecycle stays None and
        # train_agent()/train_component() take their exact original code paths.
        if lcg_enabled and not self._is_model_free:
            lcg_cfg = instantiate(cfg.intrinsic_reward)
            self._lcg_lifecycle = LCGLifecycle(
                lcg_cfg, sigma_distribution_cfg,
                img_channels=cfg.agent.denoiser.inner_model.img_channels,
                img_size=cfg.env.train.size,
                device=self._device,
            )
        else:
            self._lcg_lifecycle = None

        # Training state (things to be saved/restored)
        self.epoch = 0
        self.num_epochs_collect = None
        self.num_episodes_test = 0
        self.num_batch_train = CommonTools(0, 0, 0)
        self.num_batch_test = CommonTools(0, 0, 0)

        if cfg.common.resume:
            self.load_state_checkpoint()
        else:
            self.save_checkpoint()

        if self._rank == 0:
            for name in self._model_names:
                print(f"{count_parameters(getattr(self.agent, name))} parameters in {name}")
            print(self.train_dataset)
            print(self.test_dataset)

    def run(self) -> None:
        to_log = []

        if self.epoch == 0:
            if self._is_model_free or self._is_static_dataset:
                self.num_epochs_collect = 0
            else:
                if self._rank == 0:
                    self.num_epochs_collect, to_log_ = self.collect_initial_dataset()
                    to_log += to_log_
                self.num_epochs_collect, sd_train_dataset = broadcast_if_needed(self.num_epochs_collect, self.train_dataset.state_dict())
                self.train_dataset.load_state_dict(sd_train_dataset)

        num_epochs = self.num_epochs_collect + self._cfg.training.num_final_epochs

        while self.epoch < num_epochs:
            self.epoch += 1
            start_time = time.time()

            if self._rank == 0:
                print(f"\nEpoch {self.epoch} / {num_epochs}\n")

            # Training
            should_collect_train = (self._rank == 0 and not self._is_model_free and not self._is_static_dataset and self.epoch <= self.num_epochs_collect)

            if should_collect_train:
                c = self._cfg.collection.train
                to_log += self._train_collector.send(NumToCollect(steps=c.steps_per_epoch))
            sd_train_dataset, = broadcast_if_needed(self.train_dataset.state_dict())  # update dataset for ranks > 0
            self.train_dataset.load_state_dict(sd_train_dataset)
            
            if self._cfg.training.should:
                to_log += self.train_agent()

            # Evaluation
            should_test = self._rank == 0 and self._cfg.evaluation.should and (self.epoch % self._cfg.evaluation.every == 0)
            should_collect_test = should_test and not self._is_static_dataset

            if should_collect_test:
                to_log += self.collect_test()

            if should_test and not self._is_model_free:
                to_log += self.test_agent()

            # Logging
            to_log.append({"duration": (time.time() - start_time) / 3600})
            if self._rank == 0:
                wandb_log(to_log, self.epoch)
            to_log = []

            # Checkpointing
            self.save_checkpoint()
            
            if dist.is_initialized():
                dist.barrier()

        # Last collect
        if self._rank == 0 and not self._is_static_dataset:
            wandb_log(self.collect_test(final=True), self.epoch)

    def collect_initial_dataset(self) -> Tuple[int, Logs]:
        print("\nInitial collect\n")
        to_log = []
        c = self._cfg.collection.train
        min_steps = c.first_epoch.min
        steps_per_epoch = c.steps_per_epoch
        max_steps = c.first_epoch.max
        threshold_rew = c.first_epoch.threshold_rew
        continuous_reward = self.agent.rew_end_model.continuous_reward
        assert min_steps % steps_per_epoch == 0

        steps = min_steps
        while True:
            to_log += self._train_collector.send(NumToCollect(steps=steps))
            num_steps = self.train_dataset.num_steps

            if continuous_reward:
                # Dense/continuous rewards have no meaningful "minority reward class" to wait for
                # (see Stage 7B report for the full trace/rationale): stop purely on the step
                # budget already represented by first_epoch.min, without ever consulting
                # reward-sign counts, thresholds, or .sign().
                if num_steps >= min_steps:
                    break
                print(f"Continuous reward collection: {num_steps}/{min_steps} steps -> keep collecting\n")
            else:
                total_minority_rew = sum(sorted(self.train_dataset.counts_rew)[:-1])
                if total_minority_rew >= threshold_rew:
                    break
                if (max_steps is not None) and num_steps >= max_steps:
                    print("Reached the specified maximum for initial collect")
                    break
                print(f"Minority reward: {total_minority_rew}/{threshold_rew} -> Keep collecting\n")

            steps = steps_per_epoch

        print("\nSummary of initial collect:")
        print(f"Num steps: {num_steps} / {c.num_steps_total}")
        if not continuous_reward:
            print(f"Reward counts: {dict(self.train_dataset.counter_rew)}")

        remaining_steps = c.num_steps_total - num_steps
        assert remaining_steps % c.steps_per_epoch == 0
        num_epochs_collect = remaining_steps // c.steps_per_epoch

        return num_epochs_collect, to_log

    def collect_test(self, final: bool = False) -> Logs:
        c = self._cfg.collection.test
        episodes = c.num_final_episodes if final else c.num_episodes
        td = self.test_dataset
        td.clear()
        to_log = self._test_collector.send(NumToCollect(episodes=episodes))
        key_ep_id = f"{td.name}/episode_id"
        to_log = [{k: v + self.num_episodes_test if k == key_ep_id else v for k, v in x.items()} for x in to_log]

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
        model_names = ["actor_critic"] if self._is_model_free else self._model_names
        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                if name == "actor_critic" and self._lcg_lifecycle is not None:
                    # World-model update for this round is complete (denoiser/rew_end_model
                    # already trained above) -- refresh h_D/CRN banks/RunningRMS once per
                    # round, then rewire the frozen hook before this round's ActorCritic
                    # updates begin. See lcg.lifecycle.LCGLifecycle.
                    #
                    # round_identifier=self.epoch (not an internal counter): self.epoch is
                    # already persisted/restored by Trainer's own checkpoint (StateDictMixin
                    # picks up every non-underscore-prefixed attribute, and self._lcg_lifecycle
                    # itself is NOT checkpointed), so deriving the LCG round seed from it means
                    # a resumed run continues the seed sequence rather than restarting it from 0
                    # and reusing seeds (and therefore historical subsets/candidate banks) an
                    # earlier, pre-resume round already used.
                    self._lcg_lifecycle.refresh(self.agent.denoiser, self.train_dataset, round_identifier=self.epoch)
                    self.agent.actor_critic.set_intrinsic_reward_fn(self._lcg_lifecycle.intrinsic_reward_fn)
                steps = cfg.steps_first_epoch if self.epoch == 1 else cfg.steps_per_epoch
                # DrQActorCritic uses its own collect_rollout()/critic_update()/actor_update()
                # call sequence (two disjoint-parameter optimizers, no single external loss) --
                # see train_drq_actor_critic's docstring -- instead of train_component's
                # generic single-model()-call/single-optimizer pattern.
                train_this_component = (
                    (lambda: self.train_drq_actor_critic(steps))
                    if name == "actor_critic" and self._is_drq
                    else (lambda: self.train_component(name, steps))
                )
                if self._lcg_lifecycle is not None:
                    t0 = time.time()
                    to_log += train_this_component()
                    # NOTE: self._lcg_lifecycle.round_id is only authoritative for `name ==
                    # "actor_critic"` (refresh() has just run this epoch); for
                    # denoiser/rew_end_model it would still show the *previous* round's id,
                    # so this line is labeled by epoch instead to avoid mislabeling.
                    round_label = self._lcg_lifecycle.round_id if name == "actor_critic" else f"pending(epoch={self.epoch})"
                    print(f"[LCG-TIMING] round={round_label} component={name} "
                          f"time={time.time() - t0:.2f}s", flush=True)
                else:
                    to_log += train_this_component()
        return to_log

    @torch.no_grad()
    def test_agent(self) -> Logs:
        self.agent.eval()
        to_log = []
        model_names = [] if self._is_model_free else self._model_names[:-1]
        for name in model_names:
            cfg = getattr(self._cfg, name).training
            if self.epoch > cfg.start_after_epochs:
                to_log += self.test_component(name)
        return to_log

    def train_component(self, name: str, steps: int) -> Logs:
        cfg = getattr(self._cfg, name).training
        model = getattr(self._agent, name)
        opt = self.opt.get(name)
        lr_sched = self.lr_sched.get(name)
        data_loader = self._data_loader_train.get(name)

        model.train()
        opt.zero_grad()
        data_iterator = iter(data_loader) if data_loader is not None else None
        to_log = []

        num_steps = cfg.grad_acc_steps * steps

        # Gated purely-additive timing split: only active for actor_critic when LCG is
        # enabled, so it costs nothing and changes nothing when LCG is disabled or for
        # the other components.
        lcg_timing = self._lcg_lifecycle is not None and name == "actor_critic"
        t_forward_total = 0.0
        t_backward_total = 0.0

        for i in trange(num_steps, desc=f"Training {name}", disable=self._rank > 0):
            batch = next(data_iterator).to(self._device) if data_iterator is not None else None
            if lcg_timing:
                t0 = time.time()
            loss, metrics = model(batch) if batch is not None else model()
            if lcg_timing:
                t_forward_total += time.time() - t0
                t0 = time.time()
            loss.backward()

            num_batch = self.num_batch_train.get(name)
            metrics[f"num_batch_train_{name}"] = num_batch
            self.num_batch_train.set(name, num_batch + 1)

            if (i + 1) % cfg.grad_acc_steps == 0:
                if cfg.max_grad_norm is not None:
                    grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_grad_norm)
                    metrics["grad_norm_before_clip"] = grad_norm

                opt.step()
                opt.zero_grad()

                if lr_sched is not None:
                    metrics["lr"] = lr_sched.get_last_lr()[0]
                    lr_sched.step()
            if lcg_timing:
                t_backward_total += time.time() - t0

            to_log.append(metrics)

        if lcg_timing:
            print(f"[LCG-TIMING] round={self._lcg_lifecycle.round_id} component=actor_critic "
                  f"forward(imagination+scoring)={t_forward_total:.2f}s backward+opt={t_backward_total:.2f}s", flush=True)

        process_confusion_matrices_if_any_and_compute_classification_metrics(to_log)
        to_log = [{f"{name}/train/{k}": v for k, v in d.items()} for d in to_log]
        return to_log

    def train_drq_actor_critic(self, steps: int) -> Logs:
        """DrQActorCritic's own training call sequence, used by train_agent() in place of
        train_component("actor_critic", steps) whenever self._is_drq: one fresh imagined
        rollout (model.collect_rollout(), drawn from the WorldModelEnv-backed env_loop set up
        in setup_training) then a critic update then an actor update, per step -- NOT
        train_component's single-model()-call/single-optimizer/grad_acc_steps pattern, which
        assumes one combined loss and one optimizer (see DrQActorCritic's module docstring for
        why critic_update/actor_update are two separate methods with two separate optimizers
        and an explicit critic-then-actor staging instead). cfg.actor_critic.training's
        grad_acc_steps and max_grad_norm fields are therefore NOT read here -- DrQ has no
        gradient-accumulation loop (one step is one full rollout+critic+actor cycle already),
        and its own gradient clipping is configured independently via cfg.drq.loss's
        critic_max_grad_norm/actor_max_grad_norm (see critic_update/actor_update)."""
        model = self.agent.actor_critic
        opt_pair = self.opt.actor_critic
        lr_sched_pair = self.lr_sched.actor_critic
        model.train()
        to_log = []

        for i in trange(steps, desc="Training actor_critic (DrQ)", disable=self._rank > 0):
            model.collect_rollout()
            metrics = {**model.critic_update(opt_pair.critic), **model.actor_update(opt_pair.actor)}

            metrics["lr_critic"] = lr_sched_pair.critic.get_last_lr()[0]
            metrics["lr_actor"] = lr_sched_pair.actor.get_last_lr()[0]
            lr_sched_pair.critic.step()
            lr_sched_pair.actor.step()

            num_batch = self.num_batch_train.get("actor_critic")
            metrics["num_batch_train_actor_critic"] = num_batch
            self.num_batch_train.set("actor_critic", num_batch + 1)

            to_log.append(metrics)

        to_log = [{f"actor_critic/train/{k}": v for k, v in d.items()} for d in to_log]
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

    def load_state_checkpoint(self) -> None:
        # Trusted, locally generated DIAMOND state (includes Dataset state: numpy arrays, Counters,
        # etc.), not just tensor weights -- weights_only=False is required for PyTorch >=2.6, whose
        # default changed to weights_only=True.
        state_dict = torch.load(self._path_state_ckpt, map_location=self._device, weights_only=False)
        if "rng_state" not in state_dict:
            # Backward compatibility: a state.pt saved before RNG checkpointing existed has no
            # "rng_state" key. StateDictMixin.load_state_dict() asserts the key set matches
            # exactly, so inject a placeholder (this process's own current, freshly-seeded RNG
            # state) rather than restoring anything -- resume proceeds exactly as it always did
            # for old checkpoints, just without RNG-trajectory fidelity, and says so clearly.
            print(
                f"WARNING: {self._path_state_ckpt} predates RNG checkpointing (no 'rng_state' "
                f"key). Resuming with a fresh RNG state, not a restored one -- the resumed run's "
                f"random draws (dataset sampling order, diffusion noise, etc.) will NOT match "
                f"what the original process would have drawn next."
            )
            state_dict["rng_state"] = self.rng_state.state_dict()
        if "resume_fidelity_state" not in state_dict:
            # Backward compatibility: a state.pt saved before component-local sampler RNG /
            # WorldModelEnv preload-replay / rollout-hx_cx checkpointing existed has no
            # "resume_fidelity_state" key. Inject this process's own current (freshly seeded)
            # state as a placeholder rather than restoring anything -- resume proceeds, just
            # without exact-resume fidelity for the imagined-rollout machinery specifically.
            print(
                f"WARNING: {self._path_state_ckpt} predates resume-fidelity checkpointing (no "
                f"'resume_fidelity_state' key). Exact resume is NOT available from this "
                f"checkpoint: each component's BatchSampler will start from a fresh RNG "
                f"stream, and the actor-critic's imagined-rollout WorldModelEnv will draw a "
                f"brand new preload block and reset its rollout buffers/LSTM state from "
                f"scratch (matching this codebase's ORIGINAL behavior, before this feature) "
                f"instead of continuing exactly where the original process left off."
            )
            state_dict["resume_fidelity_state"] = self.resume_fidelity_state.state_dict()
        self.load_state_dict(state_dict)

    def save_checkpoint(self) -> None:
        if self._rank == 0:
            save_with_backup(self.state_dict(), self._path_state_ckpt)
            self.train_dataset.save_to_default_path()
            self.test_dataset.save_to_default_path()
            self._keep_agent_copies(self.agent.state_dict(), self.epoch)
            self._save_info_for_import_script(self.epoch)
            self._save_provenance_and_manifests()

    def _save_provenance_and_manifests(self) -> None:
        # Side files, not part of the resumable state_dict: informational/diagnostic, meant
        # for auditing a checkpoint after the fact (what code, what config, is the dataset on
        # disk internally consistent) rather than for driving resume behavior. Provenance is
        # cheap (a few KB) and always written; the dataset manifest re-hashes every episode
        # file on disk, which is fine for this project's small dataset but would be a real
        # per-epoch cost for a much larger one (e.g. LCG-vs-Random) -- gated behind an
        # explicit opt-in flag, default off, so no existing or future run pays for it unless
        # asked.
        provenance = {
            "epoch": self.epoch,
            "resolved_seed": self._resolved_seed,
            "git_commit": get_git_commit_hash(),
            "config_yaml": OmegaConf.to_yaml(self._cfg),
        }
        torch.save(provenance, self._path_ckpt_dir / "provenance.pt")
        if getattr(self._cfg.checkpointing, "save_dataset_manifest", False):
            torch.save(
                {
                    "epoch": self.epoch,
                    "train_dataset": self.train_dataset.compute_manifest(),
                    "test_dataset": self.test_dataset.compute_manifest(),
                },
                self._path_ckpt_dir / "dataset_manifest.pt",
            )
