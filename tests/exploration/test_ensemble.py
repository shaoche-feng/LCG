"""Full diffusion-world-model ensemble baseline (MAX-inspired ensemble disagreement):
configuration, independent members, bootstrap, disagreement, member selection, frozen
ensemble during PMPO, reward isolation, checkpoints/milestones and resume."""
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

import exploration_trainer as trainer_module
from data import BatchSampler, Dataset, Episode
from data.milestone import load_milestone
from ensemble import EnsembleTraining, episode_multiplicity, EnsembleDisagreementConfig
from envs import WorldModelEnv, WorldModelEnvConfig
from envs.world_model_env import ensemble_disagreement
from exploration_trainer import ExplorationTrainer
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSampler, DiffusionSamplerConfig
from models.diffusion.inner_model import InnerModelConfig
from test_exploration_trainer import COLLECT, compose, Prompts, collate_prompts, run_trainer, TINY, write_static_dataset
from test_test_diagnostics import BASE, Crash, start, same
from utils import rng_state

ENS = [o for o in TINY if not o.startswith("intrinsic_reward.")] + [
    "intrinsic_reward=ensemble_disagreement", "intrinsic_reward.num_members=3", "intrinsic_reward.bootstrap=true", "training.compile_wm=false",
    "agent.actor_critic.actor_objective=advantage_weighted"]
OFF = ["evaluation.should=false"]


def ens_overrides(extra=()):
    return [*COLLECT, *extra]


def run_ens(tmp_path, monkeypatch, extra=(), build_only=False):
    """run_trainer with the tiny model and an ensemble instead of LCG."""
    import test_exploration_trainer as h
    original = h.TINY
    h.TINY = ENS
    try:
        return run_trainer(tmp_path, monkeypatch, ens_overrides(extra), build_only=build_only)
    finally:
        h.TINY = original


def tiny_denoiser(seed):
    torch.manual_seed(seed)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=2, cond_channels=16, depths=[1], channels=[8],
                             attn_depths=[False], continuous_action_dim=2)
    return Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3)).eval()


# ---------------------------------------------------------------- configuration

def test_defaults_presets_and_single_denoiser_methods():
    cfg = compose([])
    assert "ensemble" not in cfg  # the ensemble lives in intrinsic_reward
    group = compose(["intrinsic_reward=ensemble_disagreement"])
    assert (group.intrinsic_reward.num_members, group.intrinsic_reward.bootstrap) == (5, True)
    assert OmegaConf.select(cfg, "intrinsic_reward.method", default="lcg") == "lcg"
    preset = compose(["+experiment=walker_ensemble_100k"])
    OmegaConf.resolve(preset)
    assert preset.intrinsic_reward.method == "ensemble_disagreement"
    assert "ensemble" not in preset
    assert (preset.intrinsic_reward.num_members, preset.intrinsic_reward.bootstrap, preset.training.compile_wm) == (
        5, True, False)
    assert (preset.agent.actor_critic.actor_objective, preset.agent.actor_critic.advantage_weight_cap) == (
        "advantage_weighted", 5.0)
    assert preset.intrinsic_reward.fixed_divisor == 1.0 and preset.intrinsic_reward.reward_scaling == "fixed"
    # The global default stays sign-only; every learned exploration preset (LCG included) uses
    # advantage-weighted PMPO; uniform trains no policy (its unused setting is the default).
    assert compose([]).agent.actor_critic.actor_objective == "sign"
    for name, objective in (("walker_lcg_100k", "advantage_weighted"), ("walker_uniform_100k", "sign")):
        other = compose([f"+experiment={name}"])
        assert other.agent.actor_critic.actor_objective == objective
        assert OmegaConf.select(other, "intrinsic_reward.method", default="lcg") == "lcg"
    # Same budget and per-member denoiser schedule as the LCG preset.
    lcg = compose(["+experiment=walker_lcg_100k"])
    OmegaConf.resolve(lcg)
    for key in ("collection", "denoiser", "milestones", "evaluation", "env"):
        assert OmegaConf.to_container(preset[key]) == OmegaConf.to_container(lcg[key]), key


@pytest.mark.parametrize("extra", [[], ["collection.train.mode=uniform", "intrinsic_reward.enabled=false"]])
def test_uniform_and_lcg_keep_one_denoiser(tmp_path, monkeypatch, extra):
    trainer, _ = run_trainer(tmp_path, monkeypatch, [*COLLECT, *extra], build_only=True)
    assert len(trainer.agent.denoisers) == 1 and trainer.ensemble is None
    assert not any(k.startswith("ensemble_members") for k in trainer.agent.state_dict())


@pytest.mark.parametrize("extra,match", [
    (["intrinsic_reward.num_members=1"], "intrinsic_reward.num_members must be an integer >= 2"),
    (["intrinsic_reward.num_members=0"], "intrinsic_reward.num_members must be an integer >= 2"),
    (["intrinsic_reward.bootstrap=1"], "intrinsic_reward.bootstrap must be true or false"),
    (["training.compile_wm=true"], "not compiled"),
    (["world_model_env.diffusion_sampler.s_churn=0.5"], "s_churn=0"),
    (["intrinsic_reward.num_noise_samples=0"], "num_noise_samples"),
    (["intrinsic_reward.fixed_divisor=0"], "fixed_divisor"),
    (["intrinsic_reward.member_selection=argmax"], "member_selection"),
])
def test_invalid_ensemble_settings_are_rejected(tmp_path, monkeypatch, extra, match):
    with pytest.raises(ValueError, match=match):
        run_ens(tmp_path, monkeypatch, extra, build_only=True)


# ---------------------------------------------------------------- members and bootstrap

def test_members_are_independent_and_train_their_own_schedule(tmp_path, monkeypatch):
    trainer, logs = run_ens(tmp_path, monkeypatch, OFF)
    members = trainer.agent.denoisers
    assert len(members) == 3 and len(trainer.ensemble.optimizers) == 2
    params = [torch.cat([p.detach().flatten() for p in m.parameters()]) for m in members]
    assert all(not torch.equal(params[i], params[j]) for i in range(3) for j in range(i + 1, 3))
    opts = [trainer.opt.denoiser, *trainer.ensemble.optimizers]
    assert len({id(o) for o in opts}) == 3
    for o, m in zip(opts, members):
        assert {id(p) for g in o.param_groups for p in g["params"]} == {id(p) for p in m.parameters()}
    # Per-member schedule: 2 (first epoch) + 1 + 1 updates each; member 0 also counts as denoiser.
    assert trainer.ensemble.updates == [3, 3, 3] and trainer.num_batch_train.denoiser == 3
    rows = [r for r in logs if "ensemble/total_denoiser_updates" in r]
    assert rows[-1]["ensemble/total_denoiser_updates"] == 9
    assert all(f"timing/denoiser_member_{k}_seconds" in rows[-1] for k in range(3))
    assert any("denoiser/member_2/train/loss_denoising" in r for r in logs)


def test_bootstrap_multiplicities_are_reproducible_fixed_and_grow():
    a = EnsembleTraining(3, True, seed=5, optimizers=[], schedulers=[])
    b = EnsembleTraining(3, True, seed=5, optimizers=[], schedulers=[])
    first = a.multiplicity(1, 4).copy()
    grown = a.multiplicity(1, 9)
    assert np.array_equal(grown[:4], first)  # earlier episodes keep their multiplicity
    assert np.array_equal(b.multiplicity(1, 9), grown)  # same seed, same assignment
    assert not np.array_equal(a.multiplicity(0, 200), a.multiplicity(1, 200))  # members differ
    assert grown.tolist() == [episode_multiplicity(5, 1, e) for e in range(9)]
    m = a.multiplicity(2, 4000)
    assert abs(m.mean() - 1) < 0.05 and abs((m == 0).mean() - np.exp(-1)) < 0.03  # Poisson(1)
    assert EnsembleTraining(3, True, seed=6, optimizers=[], schedulers=[]).multiplicity(1, 9).tolist() != grown.tolist()


def test_bootstrap_sampling_follows_multiplicities_and_handles_empty(tmp_path):
    dataset = Dataset(tmp_path / "d", cache_in_ram=True, save_on_disk=False)
    for _ in range(4):
        dataset.add_episode(Episode(torch.zeros(5, 3, 8, 8), torch.zeros(5, 2), torch.zeros(5),
                                    torch.zeros(5, dtype=torch.long), torch.zeros(5, dtype=torch.long), {}))
    mult = np.array([0., 2., 0., 1.])
    sampler = BatchSampler(dataset, 0, 1, 3000, 3, None, episode_multiplicity=lambda n: mult[:n])
    np.random.seed(0)
    episodes = np.array([s.episode_id for s in sampler.sample()])
    assert set(episodes) == {1, 3} and abs((episodes == 1).mean() - 2 / 3) < 0.03
    empty = BatchSampler(dataset, 0, 1, 50, 3, None, episode_multiplicity=lambda n: np.zeros(n))
    assert len(empty.sample()) == 50 and empty.empty_bootstrap_batches == 1  # ordinary weights, counted


# ---------------------------------------------------------------- disagreement and member selection

def test_disagreement_reduction():
    p = torch.zeros(3, 2, 1, 1, 2)
    p[:, 0, 0, 0, 0] = torch.tensor([0., 1., 2.])  # trajectory 0, one pixel varies: var = 2/3
    p[:, 1] = 0.5  # trajectory 1: all members agree
    torch.testing.assert_close(ensemble_disagreement(p), torch.tensor([2 / 3 / 2, 0.]))
    x = torch.randn(5, 4, 3, 8, 8)
    expected = ((x - x.mean(0)) ** 2).mean(0).mean(dim=(1, 2, 3))
    torch.testing.assert_close(ensemble_disagreement(x), expected)


def make_env(members, batch=4, horizon=3, samples=1):
    from torch.utils.data import DataLoader

    class Loader(DataLoader):
        pass

    loader = DataLoader(Prompts(), batch_size=batch, collate_fn=collate_prompts, generator=torch.Generator().manual_seed(0))
    cfg = WorldModelEnvConfig(horizon=horizon, num_batches_to_preload=1,
                              diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2))
    return WorldModelEnv(members[0], None, loader, cfg, ensemble=members, ensemble_noise_samples=samples)


def test_identical_members_with_shared_noise_disagree_by_zero():
    base = tiny_denoiser(0)
    clone = tiny_denoiser(1)
    clone.load_state_dict(base.state_dict())
    env = make_env([base, clone, tiny_denoiser(2)])
    env.reset()
    env.ensemble_samplers = env.ensemble_samplers[:2]  # two identical members only
    env.step(torch.zeros(4, 2))
    assert torch.equal(env.last_disagreement, torch.zeros(4))
    env3 = make_env([base, tiny_denoiser(1), tiny_denoiser(2)], samples=2)
    env3.reset()
    _, _, _, _, info = env3.step(torch.zeros(4, 2))
    assert (info["ensemble_disagreement"] > 0).all() and info["ensemble_disagreement"].shape == (4,)


def test_member_selection_is_uniform_per_trajectory_and_transition(monkeypatch):
    members = [tiny_denoiser(k) for k in range(3)]
    env = make_env(members, batch=8, horizon=50)
    for k, smp in enumerate(env.ensemble_samplers):  # member k predicts the constant frame k / 10
        smp.sample = (lambda value: lambda obs, act, noise=None: (torch.full_like(noise, value), []))(k / 10)
    env.reset()
    torch.manual_seed(0)
    chosen = []
    for _ in range(40):
        obs, _, _, _, info = env.step(torch.zeros(8, 2))
        member = info["ensemble_member"]
        # The next observation is exactly the selected member's prediction, per trajectory.
        torch.testing.assert_close(env.obs_buffer[:, -1].mean(dim=(1, 2, 3)), member.float() / 10)
        torch.testing.assert_close(info["ensemble_disagreement"], torch.full((8,), float(np.var([0, .1, .2]))))
        chosen.append(member)
    chosen = torch.stack(chosen)  # (steps, trajectories)
    assert len(set(chosen[:, 0].tolist())) == 3 and len(set(chosen[0].tolist())) > 1  # varies in time and across
    counts = torch.bincount(chosen.flatten(), minlength=3).float() / chosen.numel()
    assert (counts - 1 / 3).abs().max() < 0.08


# ---------------------------------------------------------------- PMPO, isolation, checkpoints

def test_ensemble_is_frozen_and_in_eval_mode_during_pmpo(tmp_path, monkeypatch):
    seen = []
    import ensemble as ensemble_module
    original = ensemble_module.EnsembleScorer.__call__

    def spy(self, infos, env_rew):
        trainer = state["trainer"]
        seen.append(([m.training for m in trainer.agent.denoisers],
                     [torch.cat([p.detach().flatten() for p in m.parameters()]).clone() for m in trainer.agent.denoisers]))
        return original(self, infos, env_rew)

    state = {}
    original_init = ExplorationTrainer.__init__

    def init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        state["trainer"] = self

    monkeypatch.setattr(ExplorationTrainer, "__init__", init)
    monkeypatch.setattr(ensemble_module.EnsembleScorer, "__call__", spy)
    trainer, logs = run_ens(tmp_path, monkeypatch, OFF)
    assert len(seen) == trainer.agent.actor_critic.updates.item() == 3
    assert all(not any(modes) for modes, _ in seen)
    # Within an epoch's PMPO phase the members never change (epoch 1 has two updates).
    assert all(torch.equal(a, b) for a, b in zip(seen[0][1], seen[1][1]))
    assert any("actor_critic/train/intrinsic/raw_reward_mean" in r for r in logs)
    for r in logs:
        if "actor_critic/train/intrinsic/raw_reward_mean" in r:
            assert r["actor_critic/train/intrinsic/scaled_reward_mean"] == pytest.approx(r["actor_critic/train/intrinsic/raw_reward_mean"])


def test_task_rewards_do_not_change_ensemble_exploration(tmp_path, monkeypatch):
    runs = []
    for scale in (0.0, 1e6):
        root = tmp_path / f"s{scale:g}"
        write_static_dataset(root / "static", scale)
        trainer, logs = run_ens(root, monkeypatch, [f"static_dataset.path={(root / 'static').as_posix()}",
                                                    "training.num_final_epochs=2", "evaluation.every=1"])
        runs.append((trainer, logs))
    (a, la), (b, lb) = runs
    assert same(a.agent.state_dict(), b.agent.state_dict())
    assert same(a.ensemble.state_dict(), b.ensemble.state_dict())

    def learning(rows):
        return [{k: float(v) for k, v in r.items() if np.ndim(v) == 0 and not k.startswith(("timing/", "test"))
                 and k != "duration" and "/return_" not in k} for r in rows]

    assert learning(la) == learning(lb)


def test_milestones_hold_every_member_and_load_by_index(tmp_path, monkeypatch):
    trainer, _ = run_ens(tmp_path, monkeypatch, OFF + ["milestones.at_steps=[24]"])
    path = Path("milestones/steps_000024")
    manifest = load_milestone(path, "walk").manifest
    assert manifest["ensemble"]["members"] == 3 and len(manifest["identity"]["member_sha256"]) == 3
    assert manifest["counters"]["member_denoiser_batches"] == [3, 3, 3]
    for k, member in enumerate(trainer.agent.denoisers):
        m = load_milestone(path, "walk", member_index=k)
        assert m.member_index == k and same(m.denoiser_state, member.state_dict())
    with pytest.raises(ValueError, match="member_index 3 is not one of them"):
        load_milestone(path, "walk", member_index=3)
    (path / "denoiser_member_2.pt").write_bytes(b"corrupt")
    with pytest.raises(ValueError, match="does not match its checksum"):
        load_milestone(path, "walk", member_index=0)


def test_resume_matches_uninterrupted_and_rejects_changed_ensembles(tmp_path, monkeypatch):
    import test_test_diagnostics as h
    original_tiny = h.TINY
    monkeypatch.setattr(h, "TINY", ENS)
    full, logs_full = start(tmp_path / "full", monkeypatch, BASE + OFF)
    full._run()
    full_rng = rng_state()

    original = ExplorationTrainer.train_agent

    def crash(self):
        if self.epoch == 2:
            raise Crash()
        return original(self)

    monkeypatch.setattr(ExplorationTrainer, "train_agent", crash)
    crashed, _ = start(tmp_path / "split", monkeypatch, BASE + OFF)
    with pytest.raises(Crash):
        crashed._run()
    monkeypatch.setattr(ExplorationTrainer, "train_agent", original)
    for overrides, match in ((["intrinsic_reward.num_members=4"], "ensemble settings"),
                             (["intrinsic_reward.bootstrap=false"], "ensemble settings"),
                             (["intrinsic_reward.fixed_divisor=2.0"], "must not switch")):
        with pytest.raises(ValueError, match=match):
            start(tmp_path / "split", monkeypatch, overrides, resume=True)
    resumed, logs_resumed = start(tmp_path / "split", monkeypatch, [], resume=True)
    resumed._run()
    assert same(full.agent.state_dict(), resumed.agent.state_dict())  # all members and PMPO
    assert same(full.ensemble.state_dict(), resumed.ensemble.state_dict())  # optimizers, RNG, bootstrap, counters
    assert same(full.opt.state_dict(), resumed.opt.state_dict())
    for k in ("python", "numpy", "torch"):
        assert same(full_rng[k], rng_state()[k])
    assert full.train_dataset.lengths.tolist() == resumed.train_dataset.lengths.tolist()
    for i in range(full.train_dataset.num_episodes):
        assert torch.equal(full.train_dataset.load_episode(i).obs, resumed.train_dataset.load_episode(i).obs)
    monkeypatch.setattr(h, "TINY", original_tiny)


# ---------------------------------------------------------------- Agent.load and ensembles

def make_agent(members):
    from hydra.utils import instantiate
    from agent import Agent
    cfg = compose(ENS)
    torch.manual_seed(members)
    return Agent(instantiate(cfg.agent, rew_end_model=None, ensemble_members=members, continuous_action_dim=2,
                             action_low=[-1.0, -1.0], action_high=[1.0, 1.0]))


def tensors(agent):
    return {k: v.clone() for k, v in agent.state_dict().items() if torch.is_tensor(v)}


@pytest.mark.parametrize("source_members,destination_members", [(3, 1), (1, 3), (3, 3)])
def test_agent_load_refuses_partial_ensemble_initialization(tmp_path, source_members, destination_members):
    path = tmp_path / "snapshot.pt"
    torch.save(make_agent(source_members).state_dict(), path)
    destination = make_agent(destination_members)
    before = tensors(destination)
    with pytest.raises(ValueError, match="not warm-started through initialization.path_to_ckpt"):
        destination.load(path, load_denoiser=True, load_rew_end_model=False, load_actor_critic=True)
    assert same(tensors(destination), before)  # validated before any weights were loaded


def test_agent_load_single_model_still_works(tmp_path):
    source = make_agent(1)
    path = tmp_path / "snapshot.pt"
    torch.save(source.state_dict(), path)
    destination = make_agent(1)
    destination.load(path, load_denoiser=True, load_rew_end_model=False, load_actor_critic=True)
    assert same(tensors(destination), tensors(source))
    # Without denoiser initialization, an ensemble snapshot's PMPO weights may still be loaded.
    ensemble_path = tmp_path / "ensemble.pt"
    ensemble = make_agent(3)
    torch.save(ensemble.state_dict(), ensemble_path)
    single = make_agent(1)
    denoiser_before = {k: v.clone() for k, v in single.denoiser.state_dict().items()}
    single.load(ensemble_path, load_denoiser=False, load_rew_end_model=False, load_actor_critic=True)
    assert same(single.actor_critic.state_dict(), ensemble.actor_critic.state_dict())
    assert same(single.denoiser.state_dict(), denoiser_before)
