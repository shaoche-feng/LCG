"""Step 3 (exploration trainer): reward isolation, the exploration loop, no reward/end
model, explicit imagined endings, and the uniform baseline."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader, IterableDataset

import exploration_trainer as trainer_module
from coroutines.env_loop import make_env_loop
from data import Dataset, Episode, collate_segments_without_rewards
from data.segment import SegmentId
from data.utils import make_segment
from envs import WorldModelEnv, WorldModelEnvConfig
from exploration_trainer import ExplorationTrainer
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig
from models.diffusion.inner_model import InnerModelConfig
from models.pmpo_beta import PMPOBeta, PMPOBetaConfig
from models.rew_end_model import RewEndModel

ROOT = Path(__file__).resolve().parents[2]
TASKS = ("stand", "walk")

# Tiny models and a tiny LCG scorer for the one-level test denoiser.
TINY = [
    "common.seed=7", "common.devices=cpu", "env.train.size=8", "env.train.task_names=[stand,walk]",
    "agent.denoiser.inner_model.num_steps_conditioning=2", "agent.denoiser.inner_model.cond_channels=16",
    "agent.denoiser.inner_model.depths=[1]", "agent.denoiser.inner_model.channels=[8]",
    "agent.denoiser.inner_model.attn_depths=[0]", "agent.actor_critic.channels=[8]", "agent.actor_critic.down=[0]",
    "agent.actor_critic.actor_hidden_dims=[16]", "agent.actor_critic.value_hidden_dims=[16]",
    "world_model_env.horizon=2", "world_model_env.num_batches_to_preload=1",
    "world_model_env.diffusion_sampler.num_steps_denoising=2", "training.compile_wm=false",
    "denoiser.training.batch_size=2", "actor_critic.training.batch_size=2",
    "denoiser.training.steps_first_epoch=2", "denoiser.training.steps_per_epoch=1",
    "actor_critic.training.steps_first_epoch=2", "actor_critic.training.steps_per_epoch=1",
    "intrinsic_reward.theta_s.include=[conv_out.*]", "intrinsic_reward.precision_reference_size=8",
    "intrinsic_reward.precision_num_mc=1", "intrinsic_reward.candidate_num_mc=2", "intrinsic_reward.candidate_chunk_size=4",
]
# Real collection with the fake env: 12 uniform steps, then two epochs of 6.
COLLECT = [
    "collection.train.first_epoch.min=12", "collection.train.steps_per_epoch=6",
    "collection.train.num_steps_total=24", "collection.test.num_final_episodes=1",
    "training.num_final_epochs=0", "evaluation.should=false",
]
UNIFORM = ["collection.train.mode=uniform", "intrinsic_reward.enabled=false"]


def compose(overrides):
    from hydra import compose as hydra_compose, initialize_config_dir
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    with initialize_config_dir(version_base="1.3", config_dir=str(ROOT / "config")):
        return hydra_compose(config_name="trainer", overrides=list(overrides))


class FakeEnv:
    """TorchEnv-like labeled env: 8x8 frames, 6-step episodes ending by time limit."""
    num_envs = 1
    action_dim = 2
    task_names = TASKS

    def __init__(self, device, terminate_at=None):
        self.device, self.terminate_at = device, terminate_at
        self.action_low = torch.tensor([-1., -1.], device=device)
        self.action_high = torch.tensor([1., 1.], device=device)

    def frame(self, action=None):
        value = (self.t % 6) / 3 - 1 + (0 if action is None else 0.1 * float(action.sum()))
        return torch.full((1, 3, 8, 8), value, device=self.device).clamp(-1, 1)

    def reset(self, **kwargs):
        self.t = 0
        return self.frame(), {}

    def get_env_states(self):
        return [dict(t=self.t)]

    def set_env_states(self, states):
        self.t = states[0]["t"]

    def step(self, action):
        self.t += 1
        obs = self.frame(action)
        rew = action.sum(1)
        end = torch.tensor([int(self.t == self.terminate_at)], device=self.device)
        trunc = torch.tensor([int(self.t == 6)], device=self.device)
        info = {"task_rewards": torch.stack([rew, 3 * rew + 1], 1)}
        if end.item() or trunc.item():
            info["final_observation"] = obs
            obs = self.reset()[0]
        return obs, rew, end, trunc, info


def run_trainer(tmp_path, monkeypatch, overrides, env_kwargs=None, build_only=False):
    """ExplorationTrainer on CPU in its own run folder; returns (trainer, logs)."""
    cfg = compose(TINY + list(overrides))
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(trainer_module, "make_dm_control_env",
                        lambda num_envs, device, **kwargs: FakeEnv(device, **(env_kwargs or {})))
    for method in ("init", "save"):
        monkeypatch.setattr(trainer_module.wandb, method, lambda *args, **kwargs: None)
    logs = []
    monkeypatch.setattr(trainer_module, "wandb_log", lambda to_log, epoch: logs.extend(to_log))
    run_dir = tmp_path / "run"
    (run_dir / ".hydra").mkdir(parents=True)
    OmegaConf.save(cfg, run_dir / ".hydra/config.yaml")
    monkeypatch.chdir(run_dir)
    trainer = ExplorationTrainer(cfg, ROOT)
    if not build_only:
        trainer._run()
    return trainer, logs


def write_static_dataset(root, reward_scale):
    """Identical observations and actions; only rew and task_rew depend on reward_scale."""
    generator = torch.Generator().manual_seed(0)
    for split in ("train", "test"):
        dataset = Dataset(root / split, cache_in_ram=True)
        for _ in range(2):
            obs = torch.rand(10, 3, 8, 8, generator=generator) * 2 - 1
            act = torch.rand(10, 2, generator=generator) * 2 - 1
            rew = reward_scale * torch.rand(10, generator=generator)
            dataset.add_episode(Episode(obs, act, rew, torch.zeros(10, dtype=torch.long), torch.zeros(10, dtype=torch.long),
                                        {}, task_rew=torch.stack([rew, -7 * rew], 1), task_names=TASKS))
        dataset.save_to_default_path()


def numeric_logs(logs):
    rows = []
    for row in logs:
        rows.append({k: float(v) for k, v in row.items()
                     if k != "duration" and not k.startswith("timing/") and np.ndim(v) == 0})
    return rows


# ---------------------------------------------------------------- reward isolation

def test_changing_task_rewards_does_not_change_exploration(tmp_path, monkeypatch):
    runs = []
    for scale in (0.0, 1e6):
        root = tmp_path / f"scale{scale:g}"
        write_static_dataset(root / "static", scale)
        overrides = [f"static_dataset.path={(root / 'static').as_posix()}", "training.num_final_epochs=2", "evaluation.every=1"]
        trainer, logs = run_trainer(root, monkeypatch, overrides)
        runs.append((trainer, logs))
    (a, logs_a), (b, logs_b) = runs
    rewards = [t.train_dataset.load_episode(0).rew.abs().max().item() for t in (a, b)]
    assert rewards[0] == 0 and rewards[1] > 1  # the stored task rewards really differ
    assert a.agent.actor_critic.updates.item() == b.agent.actor_critic.updates.item() == 3
    for key, value in a.agent.state_dict().items():
        if torch.is_tensor(value):
            torch.testing.assert_close(b.agent.state_dict()[key], value, rtol=0, atol=0)
    torch.testing.assert_close(a._lcg_lifecycle.h_D, b._lcg_lifecycle.h_D, rtol=0, atol=0)
    rows = numeric_logs(logs_a)
    assert any("actor_critic/train/loss_actor" in row and "actor_critic/train/action_mean_0" in row for row in rows)
    assert any("denoiser/test/loss_denoising" in row for row in rows)
    # Only the test return summaries (read from the labels for logging) may differ.
    def without_returns(rows):
        return [{k: v for k, v in row.items() if "/return_" not in k} for row in rows]
    assert without_returns(rows) == without_returns(numeric_logs(logs_b))  # losses, PMPO metrics, sampled actions
    returns = [[row["test/return_walk/mean"] for row in numeric_logs(logs) if "test/return_walk/mean" in row]
               for logs in (logs_a, logs_b)]
    assert returns[0] and returns[0] != returns[1]


def test_exploration_learner_inputs_have_no_task_rewards(tmp_path, monkeypatch):
    write_static_dataset(tmp_path / "static", 5.0)
    seen = []
    original = Denoiser.forward

    def spy(self, batch):
        seen.append((batch.task_rew, batch.rew.clone()))
        return original(self, batch)

    monkeypatch.setattr(Denoiser, "forward", spy)
    trainer, _ = run_trainer(tmp_path, monkeypatch, [f"static_dataset.path={(tmp_path / 'static').as_posix()}",
                                                     "training.num_final_epochs=1", "evaluation.every=1"])
    assert seen and all(task_rew is None and (rew == 0).all() for task_rew, rew in seen)
    # Denoiser train/test batches and imagination prompts all drop task rewards.
    loaders = [trainer._data_loader_train.denoiser, trainer._data_loader_test.denoiser,
               trainer.agent.actor_critic.rl_env.data_loader]
    assert all(loader.collate_fn is collate_segments_without_rewards for loader in loaders)


def test_reward_free_collate_drops_labels_and_zeroes_rewards():
    rew = torch.arange(1., 6.)
    episode = Episode(torch.zeros(5, 3, 8, 8), torch.zeros(5, 2), rew, torch.zeros(5, dtype=torch.long),
                      torch.zeros(5, dtype=torch.long), {}, task_rew=torch.stack([rew, rew], 1), task_names=TASKS)
    segment = make_segment(episode, SegmentId(0, 0, 5))
    batch = collate_segments_without_rewards([segment, segment])
    assert batch.task_rew is None and (batch.rew == 0).all() and batch.rew.shape == (2, 5)
    torch.testing.assert_close(batch.obs, torch.stack([segment.obs, segment.obs]), rtol=0, atol=0)
    assert segment.rew.tolist() == rew.tolist()  # the dataset keeps its rewards


def test_reward_free_collate_drops_reward_bearing_metadata_without_touching_the_source(tmp_path):
    from copy import deepcopy
    rew = torch.arange(1., 6.)
    info = {"final_observation": torch.ones(3, 8, 8), "episode_return": rew.sum(),
            "nested": {"task_returns": {"stand": torch.tensor(4.), "walk": [1., 2.]}}}
    dataset = Dataset(tmp_path / "d", cache_in_ram=True)
    dataset.add_episode(Episode(torch.zeros(5, 3, 8, 8), torch.zeros(5, 2), rew, torch.zeros(5, dtype=torch.long),
                                torch.ones(5, dtype=torch.long), deepcopy(info),
                                task_rew=torch.stack([rew, -rew], 1), task_names=TASKS))
    segments = [dataset[SegmentId(0, 0, 5)], dataset[SegmentId(0, -2, 3)]]
    batch = collate_segments_without_rewards(segments)
    assert batch.info == [{}, {}] and batch.task_rew is None and (batch.rew == 0).all()
    assert batch.to("cpu").info == [{}, {}]
    # Source segments, the cached episode and the saved file keep every reward and label.
    for episode in (dataset.load_episode(0), Episode.load(dataset._get_episode_path(0))):
        assert episode.info["episode_return"] == rew.sum()
        assert episode.info["nested"]["task_returns"]["stand"] == 4 and episode.info["nested"]["task_returns"]["walk"] == [1., 2.]
        torch.testing.assert_close(episode.info["final_observation"], info["final_observation"], rtol=0, atol=0)
        torch.testing.assert_close(episode.task_rew, torch.stack([rew, -rew], 1), rtol=0, atol=0)
        torch.testing.assert_close(episode.rew, rew, rtol=0, atol=0)
    assert all(segment.info is dataset.load_episode(0).info for segment in segments)
    assert segments[0].rew.tolist() == rew.tolist() and segments[0].task_rew is not None


# ---------------------------------------------------------------- learned loop, no reward/end model

def test_learned_exploration_loop_without_reward_end_model(tmp_path, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("reward/end model used during exploration")

    for name in ("__init__", "forward", "predict_rew_end"):
        monkeypatch.setattr(RewEndModel, name, forbidden)
    trainer, logs = run_trainer(tmp_path, monkeypatch, COLLECT)
    assert trainer.agent.rew_end_model is None
    assert trainer.opt.rew_end_model is None and trainer.lr_sched.rew_end_model is None
    assert trainer.agent.actor_critic.rl_env.rew_end_model is None
    assert not any(key.startswith("rew_end_model") for row in logs for key in row)
    # Initial collection plus two epochs: denoiser and PMPO trained every epoch.
    assert trainer.epoch == 2 and trainer.train_dataset.num_steps == 24
    assert trainer.num_batch_train.denoiser == 3 and trainer.agent.actor_critic.updates.item() == 3
    assert trainer.agent.actor_critic.intrinsic_reward_fn is trainer._lcg_lifecycle.intrinsic_reward_fn
    assert trainer.train_dataset.task_names == TASKS  # labels recorded for downstream training
    assert Path("checkpoints/state.pt").is_file()


@pytest.mark.parametrize("overrides,match", [
    (["intrinsic_reward.enabled=false"], "needs intrinsic_reward.enabled=true"),
    (["collection.train.mode=uniform"], "set intrinsic_reward.enabled=false"),
    (UNIFORM + ["+pmpo_diagnostic=bounded"], "pmpo_diagnostic needs PMPO updates"),
    (["env.train.domain_name=lqr", "env.train.task_names=[lqr_2_1]"], "not verified to end only at its time limit"),
])
def test_invalid_exploration_settings_are_rejected_before_any_work(tmp_path, monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        run_trainer(tmp_path, monkeypatch, COLLECT + overrides, build_only=True)
    assert not (tmp_path / "run" / "checkpoints").exists()


def test_learned_exploration_fails_without_intrinsic_hook(tmp_path, monkeypatch):
    # An LCG refresh that installs no hook must stop training, never fall back to task reward.
    monkeypatch.setattr(trainer_module.LCGLifecycle, "refresh", lambda self, *args, **kwargs: None)
    with pytest.raises(RuntimeError, match="no reward source"):
        run_trainer(tmp_path, monkeypatch, COLLECT)


def test_real_true_termination_is_rejected(tmp_path, monkeypatch):
    with pytest.raises(RuntimeError, match="true termination"):
        run_trainer(tmp_path, monkeypatch, COLLECT, env_kwargs={"terminate_at": 3})


# ---------------------------------------------------------------- uniform baseline

def test_uniform_baseline_trains_denoiser_only(tmp_path, monkeypatch):
    calls = []
    original = PMPOBeta.predict_act_value
    monkeypatch.setattr(PMPOBeta, "predict_act_value", lambda self, obs: calls.append(1) or original(self, obs))
    trainer, logs = run_trainer(tmp_path, monkeypatch, COLLECT + UNIFORM, build_only=True)
    initial = {k: v.clone() for k, v in trainer.agent.actor_critic.state_dict().items() if torch.is_tensor(v)}
    trainer._run()
    assert trainer.agent.actor_critic.updates.item() == 0 and not calls  # policy never evaluated or updated
    for key, value in trainer.agent.actor_critic.state_dict().items():
        if torch.is_tensor(value) and key != "updates":
            torch.testing.assert_close(value, initial[key], rtol=0, atol=0)
    assert trainer.opt.actor_critic is None and trainer.agent.actor_critic.rl_env is None
    assert trainer._lcg_lifecycle is None and trainer.num_batch_train.denoiser == 3
    assert trainer.train_dataset.num_steps == 24 and trainer.train_dataset.task_names == TASKS
    assert not any(key.startswith("actor_critic") for row in logs for key in row)


# ---------------------------------------------------------------- imagined endings

class Prompts(IterableDataset):
    def __iter__(self):
        generator = torch.Generator().manual_seed(1)
        while True:
            # num_steps_conditioning=2 frames per prompt, as the trainer samples them.
            yield torch.rand(2, 3, 8, 8, generator=generator) * 2 - 1, torch.rand(2, 2, generator=generator) * 2 - 1


def collate_prompts(samples):
    from types import SimpleNamespace
    obs = torch.stack([s[0] for s in samples])
    # A huge reward on the prompts: it must never reach imagination.
    return SimpleNamespace(obs=obs, act=torch.stack([s[1] for s in samples]), rew=torch.full(obs.shape[:2], 1e9),
                           mask_padding=torch.ones(obs.shape[:2], dtype=torch.bool))


def make_reward_free_system(horizon=3):
    torch.manual_seed(3)
    model = PMPOBeta(PMPOBetaConfig(img_channels=3, img_size=8, channels=[8], down=[0], continuous_action_dim=2,
                                    action_low=[-1., -1.], action_high=[1., 1.], actor_hidden_dims=(16,),
                                    value_hidden_dims=(16,), imagination_horizon=horizon))
    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16, depths=[1],
                                     channels=[8], attn_depths=[False], continuous_action_dim=2),
        sigma_data=0.5, sigma_offset_noise=0.3))
    loader = DataLoader(Prompts(), batch_size=2, collate_fn=collate_prompts, generator=torch.Generator().manual_seed(0))
    env = WorldModelEnv(denoiser, None, loader, WorldModelEnvConfig(
        horizon=horizon, num_batches_to_preload=1, diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)))
    model.setup_training(env)
    return model, env


def test_imagined_endings_are_horizon_truncations_with_correct_bootstrap_and_history():
    model, env = make_reward_free_system(horizon=3)
    prompts = []
    original_reset_dead = env.reset_dead

    def reset_dead(dead):
        prompts.append(original_reset_dead(dead))
        return prompts[-1]

    env.reset_dead = reset_dead
    loop = make_env_loop(env, model, store_policy_observations=True)
    with torch.no_grad():
        states, act, rew, end, trunc, _, values, bootstrap, infos = loop.send(4)
    loop.close()
    assert not hasattr(env, "hx_rew_end")  # no reward/end model burn-in
    assert (rew == 0).all() and (end == 0).all()  # placeholder reward, no true endings
    assert trunc.tolist() == [[0, 0, 1, 0]] * 2  # truncated exactly at the horizon
    final = infos[2]["final_observation"]
    with torch.no_grad():
        # The truncated step bootstraps from the successor of the final observation...
        successor = torch.cat((states[:, 2][:, 3:], final), 1)
        torch.testing.assert_close(bootstrap[:, 2], model.value(successor).squeeze(-1), rtol=0, atol=0)
        # ...and the next policy state is the new replay prompt, not the old history.
    (frames, valid), = prompts
    assert valid.all()
    # Two prompt frames: the four-frame stack repeats the earliest, as a real reset does.
    torch.testing.assert_close(states[:, 3], frames[:, [0, 0, 0, 1]].flatten(1, 2), rtol=0, atol=0)


def test_pmpo_needs_a_reward_source_without_reward_end_model():
    model, _ = make_reward_free_system()
    with pytest.raises(RuntimeError, match="no reward source"):
        model()
    rewards = []

    def hook(infos, env_rew):
        assert (env_rew == 0).all()  # only the placeholder reaches the hook
        rewards.append(torch.ones_like(env_rew))
        return rewards[-1]

    model.set_intrinsic_reward_fn(hook)
    loss, metrics = model()
    assert torch.isfinite(loss) and rewards and metrics["return_mean"] > 0
