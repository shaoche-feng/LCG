"""Plan2Explore-style lightweight ensemble disagreement: feature extraction (layers, shapes,
sigma/preconditioning, probe, pooling, early exit), causal context and transition alignment,
replay-vs-imagination features, disagreement, head fitting and freezing, bootstrap and RNG,
cache invalidation, task-reward isolation, checkpoints/resume, milestones and downstream
loading, and the other methods staying unchanged."""
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import exploration_trainer as trainer_module
import plan2explore as p2e_module
import test_exploration_trainer as base
import test_test_diagnostics as diag
from data import BatchSampler, collate_segments_without_rewards, Dataset, Episode
from data.milestone import load_milestone, publish_milestone, read_manifest
from envs import WorldModelEnv, WorldModelEnvConfig
from exploration_trainer import ExplorationTrainer
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig
from models.diffusion.inner_model import InnerModelConfig
from plan2explore import (build_feature_cache, FeatureCache, FeatureExtractor, Plan2Explore, Plan2ExploreConfig,
                          predictor_disagreement)
from test_exploration_trainer import COLLECT, compose, run_trainer, TASKS, write_static_dataset
from test_test_diagnostics import BASE, Crash, same, start
from utils import rng_state

# Two UNet levels (8x8 and 4x4 maps) so that two downsampling resolutions exist.
P2E = [o for o in base.TINY if not o.startswith("intrinsic_reward.") and not o.startswith("agent.denoiser.inner_model.")] + [
    "agent.denoiser.inner_model.num_steps_conditioning=2", "agent.denoiser.inner_model.cond_channels=16",
    "agent.denoiser.inner_model.depths=[1,1]", "agent.denoiser.inner_model.channels=[8,8]",
    "agent.denoiser.inner_model.attn_depths=[0,0]",
    "intrinsic_reward=plan2explore_disagreement", "intrinsic_reward.num_members=3",
    "intrinsic_reward.feature_layers=[unet.d_blocks.0,unet.d_blocks.1]", "intrinsic_reward.pool_size=2",
    "intrinsic_reward.hidden_dims=[16]", "intrinsic_reward.batch_size=8", "intrinsic_reward.fit_updates=5",
    "intrinsic_reward.encode_batch_size=7", "agent.actor_critic.actor_objective=advantage_weighted"]
OFF = ["evaluation.should=false"]
base.TINY_LCG = base.TINY  # the default (LCG) tiny settings, for tests that switch methods


def run_p2e(tmp_path, monkeypatch, extra=(), build_only=False, env_kwargs=None):
    monkeypatch.setattr(base, "TINY", P2E)
    return run_trainer(tmp_path, monkeypatch, [*COLLECT, *extra], env_kwargs=env_kwargs, build_only=build_only)


def tiny_denoiser(seed=0, n=2):
    torch.manual_seed(seed)
    inner = InnerModelConfig(img_channels=3, num_steps_conditioning=n, cond_channels=16, depths=[1, 1], channels=[8, 8],
                             attn_depths=[False, False], continuous_action_dim=2)
    denoiser = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3))
    # Fresh ResBlocks have zero conv2 weights, so noise level and actions would not reach the
    # activations yet; random weights make the conditioning (and its alignment) observable.
    with torch.no_grad():
        for parameter in denoiser.parameters():
            if parameter.ndim > 1 and not parameter.any():
                parameter.normal_(0, 0.2)
    return denoiser


def make_p2e(denoiser=None, **overrides):
    settings = dict(num_members=3, feature_layers=["unet.d_blocks.0", "unet.d_blocks.1"], pool_size=2,
                    hidden_dims=[16], batch_size=8, fit_updates=5, encode_batch_size=5)
    settings.update(overrides)
    denoiser = denoiser or tiny_denoiser()
    return Plan2Explore(Plan2ExploreConfig(**settings), denoiser, 8, [-1., -1.], [1., 1.], seed=3,
                        device=torch.device("cpu"))


def random_dataset(path, lengths=(9, 7, 5), finished=(True, False, False), seed=0):
    g = torch.Generator().manual_seed(seed)
    dataset = Dataset(path, "train_dataset", cache_in_ram=True)
    for length, done in zip(lengths, finished):
        obs = torch.randint(0, 256, (length, 3, 8, 8), generator=g).float().div(255).mul(2).sub(1)
        act = torch.rand(length, 2, generator=g) * 2 - 1
        trunc = torch.zeros(length, dtype=torch.long)
        trunc[-1] = int(done)
        info = {"final_observation": torch.rand(3, 8, 8, generator=g) * 2 - 1} if done else {}
        rew = torch.rand(length, generator=g)
        dataset.add_episode(Episode(obs, act, rew, torch.zeros(length, dtype=torch.long), trunc, info,
                                    task_rew=torch.stack([rew, -rew], 1), task_names=TASKS))
    return dataset


def state_of(p):
    return {k: v for k, v in p.state_dict().items()}


# ---------------------------------------------------------------- configuration

def test_config_group_and_preset():
    group = compose(["intrinsic_reward=plan2explore_disagreement"]).intrinsic_reward
    cfg = Plan2ExploreConfig(**{k: v for k, v in OmegaConf.to_container(group).items() if k != "_target_"})
    assert (cfg.num_members, cfg.bootstrap, cfg.feature_layers, cfg.pool_size, cfg.feature_sigma) == (
        5, True, ["unet.d_blocks.2", "unet.d_blocks.3"], 4, 0.1)
    assert (cfg.hidden_dims, cfg.lr, cfg.batch_size, cfg.fit_updates, cfg.reward_scaling, cfg.fixed_divisor) == (
        [256, 256], 1e-3, 256, 500, "fixed", 1.0)
    preset = compose(["+experiment=walker_plan2explore_100k"])
    OmegaConf.resolve(preset)
    assert preset.intrinsic_reward.method == "plan2explore_disagreement" and "ensemble" not in preset
    assert preset.training.compile_wm is False
    ens = compose(["+experiment=walker_ensemble_100k"])
    OmegaConf.resolve(ens)
    # Same budget, schedules, milestones, diagnostics, env, controller and models as the ensemble preset
    # (whose budget/schedules equal the LCG and uniform presets'), only the method differs.
    for key in ("collection", "denoiser", "milestones", "evaluation", "env", "actor_critic", "agent",
                "world_model_env", "common"):
        assert OmegaConf.to_container(preset[key]) == OmegaConf.to_container(ens[key]), key


def test_default_layers_on_the_real_architecture():
    """The default selectors on the repository's UNet (64x64 frames): 2 x 64 x 4 x 4 features."""
    cfg = compose(["intrinsic_reward=plan2explore_disagreement"])
    inner = InnerModelConfig(**{k: v for k, v in OmegaConf.to_container(cfg.agent.denoiser.inner_model).items()
                                if k != "_target_"}, )
    inner.continuous_action_dim = 6
    denoiser = Denoiser(DenoiserConfig(inner_model=inner, sigma_data=0.5, sigma_offset_noise=0.3))
    shapes = {}
    for layers in (["unet.d_blocks.2", "unet.d_blocks.3"], ["unet.mid_blocks"], ["unet.u_blocks.2"], ["unet.u_blocks.1"]):
        e = FeatureExtractor(denoiser, layers, 4, 0.1, 0, 64)
        shapes.update(e.layer_shapes)
        assert e.feature_dim == 64 * 16 * len(layers)
    assert shapes == {"unet.d_blocks.2": (64, 16, 16), "unet.d_blocks.3": (64, 8, 8), "unet.mid_blocks": (64, 8, 8),
                      "unet.u_blocks.2": (64, 32, 32), "unet.u_blocks.1": (64, 16, 16)}


@pytest.mark.parametrize("overrides,match", [
    (dict(num_members=1), "num_members must be an integer >= 2"),
    (dict(bootstrap=1), "bootstrap must be true or false"),
    (dict(feature_sigma=0.0), "feature_sigma must be a finite positive"),
    (dict(feature_sigma=float("nan")), "feature_sigma must be a finite positive"),
    (dict(feature_std_floor=0), "feature_std_floor must be a finite positive"),
    (dict(pool_size=0), "pool_size must be a positive integer"),
    (dict(feature_layers=[]), "feature_layers must be distinct"),
    (dict(feature_layers=["unet.d_blocks.0", "unet.d_blocks.0"]), "feature_layers must be distinct"),
    (dict(activation="tanh"), "activation must be one of"),
    (dict(reward_scaling="rms"), "reward_scaling must be one of"),
    (dict(fixed_divisor=0.0), "fixed_divisor must be a finite positive"),
    (dict(hidden_dims=[0]), "hidden_dims must be positive"),
])
def test_invalid_settings_are_rejected(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_p2e(**overrides)


@pytest.mark.parametrize("overrides,match", [
    (dict(feature_layers=["unet.d_blocks.7"]), "are not InnerModel modules"),
    (dict(feature_layers=["unet.d_blocks.1"], pool_size=8), "at least pool_size=8 wide"),
])
def test_layer_and_pooling_shape_check(overrides, match):
    with pytest.raises(ValueError, match=match):
        make_p2e(**overrides)


@pytest.mark.parametrize("overrides,match", [
    (["intrinsic_reward.num_members=1"], "num_members must be an integer >= 2"),
    (["intrinsic_reward.feature_layers=[unet.nope]"], "are not InnerModel modules"),
    (["training.compile_wm=true"], "validated uncompiled"),
])
def test_trainer_rejects_bad_settings_before_any_work(tmp_path, monkeypatch, overrides, match):
    with pytest.raises(ValueError, match=match):
        run_p2e(tmp_path, monkeypatch, overrides, build_only=True)
    assert not (tmp_path / "run" / "checkpoints").exists()


# ---------------------------------------------------------------- feature extraction

def test_extraction_uses_the_normal_corruption_preconditioning_and_a_fixed_probe():
    denoiser = tiny_denoiser()
    e = FeatureExtractor(denoiser, ["unet.d_blocks.0", "unet.d_blocks.1"], 2, 0.1, 5, 8)
    g = torch.Generator().manual_seed(1)
    frames, ctx, act = torch.rand(4, 3, 8, 8, generator=g) * 2 - 1, torch.rand(4, 2, 3, 8, 8, generator=g), torch.rand(4, 2, 2, generator=g)
    # Reference: o_t + sigma * eps + 0.3 * eps_offset with the seeded probe, EDM conditioners at sigma.
    probe = torch.Generator().manual_seed(5)
    eps, offset = torch.randn(3, 8, 8, generator=probe), torch.randn(3, 1, 1, generator=probe)
    sigma = torch.full((4,), 0.1)
    noisy = frames + 0.1 * eps + 0.3 * offset
    cs = denoiser.compute_conditioners(sigma)
    assert torch.allclose(cs.c_noise, torch.full((4,), (0.1 ** 2 + 0.3 ** 2) ** 0.5).log() / 4)
    captured = []
    hooks = [dict(denoiser.inner_model.named_modules())[n].register_forward_hook(
        lambda m, i, o: captured.append(o[0])) for n in ("unet.d_blocks.0", "unet.d_blocks.1")]
    with torch.no_grad():
        denoiser.inner_model(noisy * cs.c_in, cs.c_noise, ctx.reshape(4, 6, 8, 8) / 0.5, act)
    for h in hooks:
        h.remove()
    expected = torch.cat([torch.nn.functional.adaptive_avg_pool2d(c, 2).flatten(1) for c in captured], 1)
    features = e(frames, ctx, act)
    assert features.shape == (4, 8 * 4 + 8 * 4) and e.feature_dim == 64
    torch.testing.assert_close(features, expected, rtol=0, atol=0)
    # The probe is reused: repeated calls, and a new extractor with the same seed, agree exactly.
    torch.testing.assert_close(e(frames, ctx, act), features, rtol=0, atol=0)
    again = FeatureExtractor(denoiser, ["unet.d_blocks.0", "unet.d_blocks.1"], 2, 0.1, 5, 8)
    assert again.probe_sha256 == e.probe_sha256
    torch.testing.assert_close(again(frames, ctx, act), features, rtol=0, atol=0)
    other = FeatureExtractor(denoiser, ["unet.d_blocks.0", "unet.d_blocks.1"], 2, 0.1, 6, 8)
    assert other.probe_sha256 != e.probe_sha256 and not torch.equal(other(frames, ctx, act), features)
    assert not torch.equal(FeatureExtractor(denoiser, ["unet.d_blocks.0", "unet.d_blocks.1"], 2, 0.5, 5, 8)(
        frames, ctx, act), features)
    # The conditioning is live: past actions and context frames both change the feature.
    assert not torch.allclose(e(frames, ctx, act.flip(1)), features)
    assert not torch.allclose(e(frames, ctx.flip(1), act), features)


def test_early_exit_equals_a_full_pass_and_denoising_is_untouched():
    denoiser = tiny_denoiser().train()
    denoiser.inner_model.unet.eval()  # a mixed mode that must survive extraction
    e = FeatureExtractor(denoiser, ["unet.d_blocks.1"], 2, 0.1, 0, 8)
    g = torch.Generator().manual_seed(2)
    frames, ctx, act = torch.rand(3, 3, 8, 8, generator=g), torch.rand(3, 2, 3, 8, 8, generator=g), torch.rand(3, 2, 2, generator=g)
    obs, sigma = ctx.reshape(3, 6, 8, 8), torch.full((3,), 0.7)
    before = denoiser.denoise(frames, sigma, obs, act)
    modes = {n: m.training for n, m in denoiser.named_modules()}
    early = e.activations(frames, ctx, act, early_exit=True)
    full = e.activations(frames, ctx, act, early_exit=False)
    torch.testing.assert_close(early[0], full[0], rtol=0, atol=0)
    e(frames, ctx, act)
    assert {n: m.training for n, m in denoiser.named_modules()} == modes
    assert all(not m._forward_hooks for m in denoiser.modules())
    torch.testing.assert_close(denoiser.denoise(frames, sigma, obs, act), before, rtol=0, atol=0)


# ---------------------------------------------------------------- context and transitions

class SpyExtractor:
    """Records encoder inputs: features = [frame value, context frame values, context actions]
    for 1x1 single-channel frames whose value identifies them."""
    num_conditioning = 2
    feature_dim = 1 + 2 + 2

    def __call__(self, frames, ctx, act):
        return torch.cat([frames.flatten(1), ctx.flatten(1), act[..., 0]], 1)


def labeled_episode(values, done, final=None):
    t = len(values)
    obs = torch.tensor(values, dtype=torch.float).view(t, 1, 1, 1)
    act = torch.stack([torch.tensor(values, dtype=torch.float) + 0.5, torch.zeros(t)], 1)
    trunc = torch.zeros(t, dtype=torch.long)
    trunc[-1] = int(done)
    info = {"final_observation": torch.full((1, 1, 1), float(final))} if final is not None else {}
    rew = torch.full((t,), 1e9)
    return Episode(obs, act, rew, torch.zeros(t, dtype=torch.long), trunc, info,
                   task_rew=torch.stack([rew, rew], 1), task_names=TASKS)


def test_replay_transitions_context_boundaries_final_and_cut_successors(tmp_path):
    dataset = Dataset(tmp_path / "d", cache_in_ram=True, save_on_disk=False)
    dataset.add_episode(labeled_episode([10, 11, 12, 13], True, final=14))  # time-limit end: final observation
    dataset.add_episode(labeled_episode([20, 21, 22], False))  # unfinished, successor recorded at the cut
    dataset.add_episode(labeled_episode([30, 31], False))  # unfinished, no successor: last step excluded
    cache = build_feature_cache(SpyExtractor(), dataset, {1: torch.full((1, 1, 1), 23.)}, 3, {})
    f = cache.features
    assert cache.counts == dict(episodes=3, encoded_frames=5 + 4 + 2, eligible_transitions=4 + 3 + 1,
                                excluded_no_successor=1, successor_transitions=2, padded_context_transitions=2 + 2 + 1)
    # Frame rows: [value, o_{t-2}, o_{t-1}, a_{t-2}, a_{t-1}] (a = value + 0.5), zero before episode starts.
    assert f[0].tolist() == [10, 0, 0, 0, 0] and f[1].tolist() == [11, 0, 10, 0, 10.5]
    assert f[3].tolist() == [13, 11, 12, 11.5, 12.5]
    assert f[4].tolist() == [14, 12, 13, 12.5, 13.5]  # final observation: context includes a_t of the last step
    assert f[5].tolist() == [20, 0, 0, 0, 0]  # never context from the previous episode
    assert f[8].tolist() == [23, 21, 22, 21.5, 22.5]  # cut successor
    assert f[9].tolist() == [30, 0, 0, 0, 0] and f.shape[0] == 11
    values = f[:, 0]
    pairs = [(values[c].item(), values[t].item(), a[0].item()) for c, t, a in zip(cache.current, cache.target, cache.actions)]
    assert pairs == [(10, 11, 10.5), (11, 12, 11.5), (12, 13, 12.5), (13, 14, 13.5), (20, 21, 20.5), (21, 22, 21.5),
                     (22, 23, 22.5), (30, 31, 30.5)]
    assert cache.episode.tolist() == [0] * 4 + [1] * 3 + [2]


def test_a_future_frame_cannot_change_the_current_feature_or_its_reward(tmp_path):
    p = make_p2e()
    original = random_dataset(tmp_path / "a")
    changed = random_dataset(tmp_path / "b")
    episode = changed.load_episode(0)
    episode.obs[5] = -episode.obs[5]  # the frame after transition t = 4
    cache_a = build_feature_cache(p.extractor, original, {}, 4, {})
    cache_b = build_feature_cache(p.extractor, changed, {}, 4, {})
    # Frames 0..4 of episode 0 (and their transitions' current features) are unchanged; frame 5 differs.
    torch.testing.assert_close(cache_a.features[:5], cache_b.features[:5], rtol=0, atol=0)
    assert not torch.equal(cache_a.features[5], cache_b.features[5])
    p.cache = cache_a
    p.fit_standardization()
    p.fit_heads(epoch=1)
    t = 4
    reward = [predictor_disagreement(p.predict(c.features[c.current[t:t + 1]], c.actions[t:t + 1])) for c in (cache_a, cache_b)]
    torch.testing.assert_close(reward[0], reward[1], rtol=0, atol=0)


def prompt_env(dataset, extractor, seq_length=2, batch=4, horizon=3, seed=0):
    sampler = BatchSampler(dataset, 0, 1, batch, seq_length)
    loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_segments_without_rewards)
    env = WorldModelEnv(extractor.denoiser if hasattr(extractor, "denoiser") else tiny_denoiser(), None, loader,
                        WorldModelEnvConfig(horizon=horizon, num_batches_to_preload=1,
                                            diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)),
                        feature_encoder=extractor)
    batches = []
    original = env.preceding_step
    env.preceding_step = lambda b: batches.append(b) or original(b)
    np.random.seed(seed)
    return env, batches


def test_imagination_context_tracks_the_preceding_frame_and_actions(tmp_path):
    dataset = random_dataset(tmp_path / "d", lengths=(12, 12), finished=(True, True))
    calls = []
    e = FeatureExtractor(tiny_denoiser(), ["unet.d_blocks.0"], 2, 0.1, 0, 8)

    def spy(frames, ctx, act):
        calls.append((frames.clone(), ctx.clone(), act.clone()))
        return e(frames, ctx, act)
    spy.denoiser = e.denoiser
    env, batches = prompt_env(dataset, spy)
    obs, _ = env.reset()
    (batch,) = batches
    prompt, prompt_act = batch.obs.clone(), batch.act.clone()
    prev_obs, prev_act = env.prev_obs.clone(), env.prev_act.clone()
    for i, sid in enumerate(batch.segment_ids):
        ep = dataset.load_episode(sid.episode_id)
        if batch.mask_padding[i, 0] and sid.start > 0:
            assert torch.equal(prev_obs[i], ep.obs[sid.start - 1]) and torch.equal(prev_act[i], ep.act[sid.start - 1])
        else:
            assert not prev_obs[i].any() and not prev_act[i].any()
    actions = [torch.rand(4, 2) * 2 - 1 for _ in range(3)]
    generated, infos = [], []
    for a in actions:
        next_obs, *_, info = env.step(a)
        generated.append(next_obs.clone())
        infos.append(info)
    (f0, c0, a0), (f1, c1, a1), (f2, c2, a2) = calls
    # Step 0: o_t = last prompt frame, context (prev, prompt[0]), actions (prev_act, prompt_act[0]).
    assert torch.equal(f0, prompt[:, 1]) and torch.equal(c0, torch.stack([prev_obs, prompt[:, 0]], 1))
    assert torch.equal(a0, torch.stack([prev_act, prompt_act[:, 0]], 1))
    # Step 1: the generated frame, context = the prompt, actions (prompt_act[0], a_0) -- a_1 excluded.
    assert torch.equal(f1, generated[0]) and torch.equal(c1, prompt) and torch.equal(a1, torch.stack([prompt_act[:, 0], actions[0]], 1))
    assert torch.equal(f2, generated[1]) and torch.equal(c2, torch.stack([prompt[:, 1], generated[0]], 1))
    assert torch.equal(a2, torch.stack([actions[0], actions[1]], 1))
    assert all(torch.equal(info["plan2explore_action"], a) for info, a in zip(infos, actions))


def test_replay_cache_and_online_imagination_features_agree(tmp_path):
    dataset = random_dataset(tmp_path / "d", lengths=(12, 3, 12), finished=(True, True, False))
    p = make_p2e(encode_batch_size=4)
    cache = build_feature_cache(p.extractor, dataset, {}, 4, {})
    env, batches = prompt_env(dataset, p.extractor, batch=6, seed=4)
    env.reset()
    _, *_, info = env.step(torch.zeros(6, 2))
    (batch,) = batches
    starts = np.concatenate([[0], np.cumsum([len(dataset.load_episode(i)) + int(i == 0 or i == 1) for i in range(3)])])
    rows = [int(starts[s.episode_id]) + s.stop - 1 for s in batch.segment_ids]
    # Same function and inputs; only the batch composition differs (CPU reductions): 1e-6 relative.
    torch.testing.assert_close(info["plan2explore_feature"], cache.features[rows], rtol=1e-6, atol=1e-6)
    # Both kinds of prompt occur: with a real preceding step, and at an episode start (zero padding).
    padded = [not bool(m[0]) or s.start == 0 for m, s in zip(batch.mask_padding, batch.segment_ids)]
    assert any(padded) and not all(padded)


def test_feature_encoder_draws_no_random_numbers_and_keeps_imagination_unchanged(tmp_path):
    dataset = random_dataset(tmp_path / "d", lengths=(12, 12), finished=(True, True))
    p = make_p2e()
    outputs = []
    for encoder in (p.extractor, None):
        sampler = BatchSampler(dataset, 0, 1, 3, 2)
        loader = DataLoader(dataset, batch_sampler=sampler, collate_fn=collate_segments_without_rewards)
        env = WorldModelEnv(p.extractor.denoiser, None, loader, WorldModelEnvConfig(
            horizon=3, num_batches_to_preload=1, diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)),
            feature_encoder=encoder)
        np.random.seed(0)
        torch.manual_seed(0)
        frames = [env.reset()[0]]
        for _ in range(4):
            frames.append(env.step(torch.full((3, 2), 0.3))[0])
        outputs.append((torch.stack(frames), rng_state()))
    assert torch.equal(outputs[0][0], outputs[1][0]) and same(outputs[0][1], outputs[1][1])


def test_imagined_successor_cannot_change_that_steps_feature_or_reward(tmp_path, monkeypatch):
    dataset = random_dataset(tmp_path / "d", lengths=(12, 12), finished=(True, True))
    p = make_p2e()
    p.cache = build_feature_cache(p.extractor, dataset, {}, 5, {})
    p.fit_standardization()
    p.fit_heads(epoch=1)
    runs = []
    for scale in (1.0, -1.0):
        env, _ = prompt_env(dataset, p.extractor, batch=3, seed=1)
        original = env.predict_next_obs
        env.predict_next_obs = lambda: (scale * original()[0], [])  # a different imagined next frame
        env.reset()
        infos = [env.step(torch.full((3, 2), 0.2))[-1] for _ in range(2)]
        runs.append(p(infos, torch.zeros(3, 2)))
    assert torch.equal(runs[0][:, 0], runs[1][:, 0])  # o_t's reward does not see o_{t+1}
    assert not torch.equal(runs[0][:, 1], runs[1][:, 1])  # it does once o_{t+1} is the current frame


# ---------------------------------------------------------------- disagreement and fitting

def test_disagreement_hand_computed_sample_variance():
    predictions = torch.tensor([[[1., 0.]], [[2., 0.]], [[3., 3.]]])  # (K=3, batch 1, D=2)
    # Feature 0: mean 2, squares 1+0+1 = 2, / (K-1) = 1. Feature 1: mean 1, 1+1+4 = 6, / 2 = 3. Mean: 2.
    assert predictor_disagreement(predictions).tolist() == [2.0]
    assert predictor_disagreement(torch.ones(4, 5, 7)).tolist() == [0.0] * 5
    with pytest.raises(ValueError):
        predictor_disagreement(torch.ones(1, 2, 3))


def synthetic_cache(n=600, d=6, seed=0):
    """A learnable fixture: z_{t+1} = A z_t + B a_t exactly."""
    g = torch.Generator().manual_seed(seed)
    z = torch.randn(n, d, generator=g)
    act = torch.rand(n, 2, generator=g) * 2 - 1
    a_mat, b_mat = torch.randn(d, d, generator=g) * 0.5, torch.randn(2, d, generator=g)
    nxt = z @ a_mat + act @ b_mat
    features = torch.cat([z, nxt])
    return FeatureCache(features, torch.arange(n), torch.arange(n, 2 * n), act, torch.arange(n) % 20, {}, {})


def test_heads_learn_only_their_parameters_and_are_frozen_afterwards():
    p = make_p2e(fit_updates=300, batch_size=64, hidden_dims=[64], lr=3e-3)
    p.feature_dim = 6
    p.cache = synthetic_cache()
    p.fit_standardization()
    denoiser_before = {k: v.clone() for k, v in p.extractor.denoiser.state_dict().items()}
    logs = p.fit_heads(epoch=1)
    for k in range(3):
        assert logs[f"plan2explore/member_{k}_loss_last10_mean"] < 0.1 * logs[f"plan2explore/member_{k}_loss_first"]
        assert logs[f"plan2explore/member_{k}_fit_updates"] == 300
    assert all(torch.equal(v, denoiser_before[k]) for k, v in p.extractor.denoiser.state_dict().items())
    assert all(param.grad is None for param in p.extractor.denoiser.parameters())
    assert all(not param.requires_grad for param in p.heads.parameters()) and not p.heads[0].training
    # Trained heads beat copying the current feature (persistence) on this fixture.
    c = p.cache
    pred = p.predict(c.features[c.current], c.actions).mean(0)
    target = p.standardize(c.features[c.target])
    assert (pred - target).square().mean() < 0.1 * (p.standardize(c.features[c.current]) - target).square().mean()


def test_heads_do_not_change_during_pmpo_and_only_the_controller_trains(tmp_path, monkeypatch):
    snapshots = []
    original = Plan2Explore.refresh

    def refresh(self, *args, **kwargs):
        logs = original(self, *args, **kwargs)
        snapshots.append(({k: v.clone() for k, v in self.heads.state_dict().items()},
                          {k: v.clone() for k, v in trainer_ref[0].agent.denoiser.state_dict().items()},
                          {k: v.clone() for k, v in trainer_ref[0].agent.actor_critic.actor.state_dict().items()}))
        return logs

    monkeypatch.setattr(Plan2Explore, "refresh", refresh)
    trainer_ref = []
    trainer, _ = run_p2e(tmp_path, monkeypatch, OFF + ["training.num_final_epochs=0"], build_only=True)
    trainer_ref.append(trainer)
    original_train_component = ExplorationTrainer.train_component
    after_pmpo = []

    def train_component(self, name, steps, member=None):
        out = original_train_component(self, name, steps, member)
        if name == "actor_critic":
            after_pmpo.append(({k: v.clone() for k, v in self.plan2explore.heads.state_dict().items()},
                               {k: v.clone() for k, v in self.agent.denoiser.state_dict().items()},
                               {k: v.clone() for k, v in self.agent.actor_critic.actor.state_dict().items()}))
        return out

    monkeypatch.setattr(ExplorationTrainer, "train_component", train_component)
    trainer._run()
    assert len(snapshots) == len(after_pmpo) == 2
    for (heads, denoiser, actor), (heads_after, denoiser_after, actor_after) in zip(snapshots, after_pmpo):
        assert same(heads, heads_after) and same(denoiser, denoiser_after)
        assert not same(actor, actor_after)  # the controller did update
    assert trainer.agent.actor_critic.intrinsic_reward_fn is trainer.plan2explore


# ---------------------------------------------------------------- bootstrap, RNG, refits

def test_bootstrap_weights_fallback_and_deterministic_independent_refits(monkeypatch):
    p = make_p2e()
    p.feature_dim = 6
    p.cache = synthetic_cache(n=60)
    p.fit_standardization()
    m = p.multiplicities(epoch=2, member=1, num_episodes=20)
    assert np.array_equal(m, p.multiplicities(epoch=2, member=1, num_episodes=20))  # fixed per refresh and member
    assert not np.array_equal(m, p.multiplicities(epoch=3, member=1, num_episodes=20))
    assert not np.array_equal(m, p.multiplicities(epoch=2, member=2, num_episodes=20))
    p.fit_heads(epoch=1)
    first = [{k: v.clone() for k, v in h.state_dict().items()} for h in p.heads]
    p.fit_heads(epoch=1)
    assert same(first, [h.state_dict() for h in p.heads])  # deterministic fresh refit
    assert not same(first[0], first[1])  # independent members
    p.fit_heads(epoch=2)
    assert not same(first[0], p.heads[0].state_dict())
    # Transitions are weighted by their episode's multiplicity (uniform within the episode).
    probs, zero_fraction, fallback = p.sampling_probabilities(epoch=2, member=1, num_episodes=20)
    expected = m[p.cache.episode.numpy()]
    assert np.allclose(probs, expected / expected.sum()) and not fallback and zero_fraction == float((m == 0).mean())
    monkeypatch.setattr(Plan2Explore, "multiplicities",
                        lambda self, epoch, member, n: np.array([float(e == member) for e in range(n)]))
    for k in range(3):
        probs, _, fallback = p.sampling_probabilities(1, k, 20)
        assert set(p.cache.episode[torch.from_numpy(probs > 0)].tolist()) == {k} and not fallback
    # Empty support: uniform sampling, counted once per refit and member.
    monkeypatch.setattr(Plan2Explore, "multiplicities", lambda self, epoch, member, n: np.zeros(n))
    probs, zero_fraction, fallback = p.sampling_probabilities(1, 0, 20)
    assert fallback and zero_fraction == 1.0 and np.allclose(probs, 1 / 60)
    before = list(p.empty_bootstrap_fallbacks)
    logs = p.fit_heads(epoch=1)
    assert [a - b for a, b in zip(p.empty_bootstrap_fallbacks, before)] == [1, 1, 1]
    assert logs["plan2explore/member_2_empty_bootstrap_fallbacks"] == p.empty_bootstrap_fallbacks[2]
    # Bootstrap off: shared uniform support, still independent initializations.
    q = make_p2e(bootstrap=False)
    q.feature_dim = 6
    q.cache = p.cache
    q.fit_standardization()
    probs, zero_fraction, fallback = q.sampling_probabilities(1, 0, 20)
    assert np.allclose(probs, 1 / 60) and zero_fraction is None and not fallback
    q.fit_heads(epoch=1)
    assert q.empty_bootstrap_fallbacks == [0, 0, 0]
    assert not same(q.heads[0].state_dict(), q.heads[1].state_dict())


def test_refresh_uses_private_rng_only(tmp_path):
    dataset = random_dataset(tmp_path / "d")
    results = []
    for global_seed in (0, 123):
        p = make_p2e()
        torch.manual_seed(global_seed)
        np.random.seed(global_seed)
        before = rng_state()
        p.refresh(p.extractor.denoiser, dataset, {}, epoch=1)
        assert same(before, rng_state())  # no global stream consumed
        results.append(state_of(p))
    assert same(results[0], results[1])  # nor read


# ---------------------------------------------------------------- cache lifecycle

def test_cache_is_reused_only_for_the_same_denoiser_replay_and_settings(tmp_path, monkeypatch):
    dataset = random_dataset(tmp_path / "d")
    p = make_p2e()
    denoiser = p.extractor.denoiser
    logs = p.refresh(denoiser, dataset, {}, epoch=1)[0]
    assert logs["plan2explore/cache_reused"] == 0
    features = p.cache.features.clone()
    assert p.refresh(denoiser, dataset, {}, epoch=2)[0]["plan2explore/cache_reused"] == 1
    with torch.no_grad():
        denoiser.inner_model.conv_in.weight.add_(0.01)  # a denoiser update
    assert p.refresh(denoiser, dataset, {}, epoch=3)[0]["plan2explore/cache_reused"] == 0
    assert not torch.equal(p.cache.features, features)
    dataset.add_episode(random_dataset(tmp_path / "e", lengths=(4,), finished=(True,), seed=9).load_episode(0))
    assert p.refresh(denoiser, dataset, {}, epoch=4)[0]["plan2explore/cache_reused"] == 0  # replay changed
    assert p.refresh(denoiser, dataset, {2: torch.zeros(3, 8, 8)}, epoch=5)[0]["plan2explore/cache_reused"] == 0  # successor
    other = make_p2e(denoiser, feature_sigma=0.2)  # other extraction settings never take this cache
    other.cache = p.cache
    assert other.refresh(denoiser, dataset, {2: torch.zeros(3, 8, 8)}, epoch=5)[0]["plan2explore/cache_reused"] == 0
    # A failed build leaves no cache behind.
    monkeypatch.setattr(p2e_module, "build_feature_cache", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("boom")))
    with torch.no_grad():
        denoiser.inner_model.conv_in.weight.add_(0.01)
    with pytest.raises(RuntimeError, match="boom"):
        p.refresh(denoiser, dataset, {}, epoch=6)
    assert p.cache is None


def copy_dataset(src, path, edit=None):
    """A new replay with src's episodes (same ids, lengths and version filenames); edit(i, fields)
    may change episode i before it is written."""
    out = Dataset(path, "train_dataset", cache_in_ram=True)
    for i in range(src.num_episodes):
        ep = src.load_episode(i)
        fields = dict(obs=ep.obs.clone(), act=ep.act.clone(), rew=ep.rew.clone(), end=ep.end.clone(),
                      trunc=ep.trunc.clone(), info={k: v.clone() for k, v in ep.info.items()},
                      task_rew=ep.task_rew.clone(), task_names=ep.task_names)
        if edit is not None:
            edit(i, fields)
        out.add_episode(Episode(**fields))
    return out


def test_replay_identity_follows_reward_free_content_not_filenames(tmp_path):
    base_ds = random_dataset(tmp_path / "base")
    names = [p.name for p in base_ds.episode_files()]

    def changed(name, edit):
        ds = copy_dataset(base_ds, tmp_path / name, edit)
        assert [p.name for p in ds.episode_files()] == names and ds.lengths.tolist() == base_ds.lengths.tolist()
        return ds

    def negate_obs(i, f):
        if i == 1:
            f["obs"][2] = -f["obs"][2]

    def shift_act(i, f):
        if i == 1:
            f["act"][3] = f["act"][3] * 0.5

    def new_final(i, f):
        if i == 0:
            f["info"]["final_observation"] = -f["info"]["final_observation"]

    p = make_p2e()
    denoiser = p.extractor.denoiser
    successor = {1: torch.zeros(3, 8, 8)}
    assert p.refresh(denoiser, base_ds, successor, epoch=1)[0]["plan2explore/cache_reused"] == 0
    base_identity = dict(p.cache.identity)
    # Same filenames and lengths, different observations / actions / final observation: rebuilt.
    for name, edit in (("obs", negate_obs), ("act", shift_act), ("final", new_final)):
        q = make_p2e(denoiser)
        q.cache = p.cache  # a cache of the base replay offered for reuse
        logs = q.refresh(denoiser, changed(name, edit), successor, epoch=1)[0]
        assert logs["plan2explore/cache_reused"] == 0, name
        assert q.cache.identity["replay_sha256"] != base_identity["replay_sha256"], name
    # Changed cut successor: rebuilt. Unchanged content (even in another directory): reused.
    assert p.refresh(denoiser, base_ds, {1: torch.ones(3, 8, 8)}, epoch=2)[0]["plan2explore/cache_reused"] == 0
    assert p.refresh(denoiser, base_ds, successor, epoch=3)[0]["plan2explore/cache_reused"] == 0
    assert p.refresh(denoiser, base_ds, successor, epoch=4)[0]["plan2explore/cache_reused"] == 1
    assert p.refresh(denoiser, changed("copy", None), successor, epoch=5)[0]["plan2explore/cache_reused"] == 1
    # A replay rewritten in the same directory with the same filenames: the memo is keyed by the
    # versions' write-time checksums, so the new contents are hashed, not the remembered digests.
    same_dir = changed("rewrite", None)
    p.refresh(denoiser, same_dir, successor, epoch=6)
    same_dir.clear()
    rewritten = copy_dataset(base_ds, tmp_path / "rewrite", negate_obs)
    assert [p_.name for p_ in rewritten.episode_files()] == names
    assert p.refresh(denoiser, rewritten, successor, epoch=7)[0]["plan2explore/cache_reused"] == 0
    # Unchanged versions reuse their digests without reloading episodes.
    rewritten.load_episode = lambda i: (_ for _ in ()).throw(AssertionError("episode reloaded"))
    assert p.identity(denoiser, rewritten, successor)["replay_sha256"] == p.cache.identity["replay_sha256"]


def test_task_rewards_do_not_change_identity_features_heads_or_rewards(tmp_path):
    base_ds = random_dataset(tmp_path / "base")

    def task_only(i, f):
        f["rew"] = f["rew"] * 1e6 + 3
        f["task_rew"] = -f["task_rew"] * 1e6
        f["info"]["episode_return"] = torch.tensor(1e9)  # unrelated reward-bearing metadata

    other = copy_dataset(base_ds, tmp_path / "other", task_only)
    assert not torch.equal(other.load_episode(0).task_rew, base_ds.load_episode(0).task_rew)
    assert other.file_sha256 != base_ds.file_sha256  # the files differ, the reward-free content does not
    runs = []
    for ds in (base_ds, other):
        p = make_p2e()
        logs = p.refresh(p.extractor.denoiser, ds, {1: torch.zeros(3, 8, 8)}, epoch=1)[0]
        c = p.cache
        rewards = predictor_disagreement(p.predict(c.features[c.current], c.actions))
        runs.append((p, logs, rewards))
    (a, logs_a, rew_a), (b, logs_b, rew_b) = runs
    assert a.cache.identity == b.cache.identity
    torch.testing.assert_close(a.cache.features, b.cache.features, rtol=0, atol=0)
    assert same(a.state_dict(), b.state_dict())  # heads (predictor updates), statistics, counters
    assert {k: v for k, v in logs_a.items() if not k.startswith("timing/")} == \
           {k: v for k, v in logs_b.items() if not k.startswith("timing/")}
    torch.testing.assert_close(rew_a, rew_b, rtol=0, atol=0)


# ---------------------------------------------------------------- trainer integration

def test_trainer_runs_one_denoiser_and_logs(tmp_path, monkeypatch):
    trainer, logs = run_p2e(tmp_path, monkeypatch, OFF)
    assert len(trainer.agent.denoisers) == 1 and trainer.ensemble is None
    assert trainer.plan2explore.refreshes == 2 and trainer.plan2explore.last_refresh_epoch == 2
    assert trainer.agent.actor_critic.updates.item() == 3
    keys = set().union(*logs)
    for key in ("actor_critic/train/intrinsic/raw_reward_mean", "actor_critic/train/intrinsic/scaled_reward_max"):
        assert key in keys, key
    # Refresh and feature diagnostics and method timers are not logged; the timers still run.
    assert not [k for k in keys if k.startswith(("plan2explore/", "timing/p2e_"))]
    assert trainer.plan2explore.scoring_seconds > 0 and trainer.plan2explore.extractor.seconds > 0
    rewards = [row["actor_critic/train/intrinsic/raw_reward_mean"] for row in logs if "actor_critic/train/intrinsic/raw_reward_mean" in row]
    assert all(torch.isfinite(torch.as_tensor(r)) and r > 0 for r in rewards)


def test_unfinished_episode_uses_the_collectors_successor(tmp_path, monkeypatch):
    """Epochs of 4 steps with 6-step episodes: refreshes see an unfinished episode whose next
    frame is the collector's current observation."""
    seen = []
    original = p2e_module.build_feature_cache

    def spy(extractor, dataset, successors, batch_size, identity):
        cache = original(extractor, dataset, successors, batch_size, identity)
        seen.append((dict(successors), cache))
        return cache

    monkeypatch.setattr(p2e_module, "build_feature_cache", spy)
    collect = ["collection.train.first_epoch.min=12", "collection.train.steps_per_epoch=4",
               "collection.train.num_steps_total=20", "training.num_final_epochs=0", "evaluation.should=false"]
    monkeypatch.setattr(base, "TINY", P2E)
    trainer, _ = run_trainer(tmp_path, monkeypatch, collect)
    (succ1, cache1), (succ2, cache2) = seen
    assert list(succ1) == [2] and cache1.counts["excluded_no_successor"] == 0
    assert cache1.counts["successor_transitions"] == 3 and cache1.counts["eligible_transitions"] == 16
    # Epoch 2 finishes episode 2 (final observation) and leaves episode 3 unfinished at the cut.
    assert list(succ2) == [3] and cache2.counts["eligible_transitions"] == 20
    assert cache2.counts["successor_transitions"] == 4 and cache2.counts["excluded_no_successor"] == 0
    # The cut successor is encoded with the episode's own last frames and actions as context.
    episode = trainer.train_dataset.load_episode(3)
    e = trainer.plan2explore.extractor
    expected = e(succ2[3].unsqueeze(0).cpu(), episode.obs[-2:].unsqueeze(0), episode.act[-2:].unsqueeze(0))
    torch.testing.assert_close(cache2.features[-1:], expected, rtol=1e-6, atol=1e-6)


def test_changing_task_rewards_does_not_change_plan2explore(tmp_path, monkeypatch):
    runs = []
    for scale in (0.0, 1e6):
        root = tmp_path / f"scale{scale:g}"
        write_static_dataset(root / "static", scale)
        overrides = [f"static_dataset.path={(root / 'static').as_posix()}", "training.num_final_epochs=2",
                     "evaluation.every=1"]
        monkeypatch.setattr(base, "TINY", P2E)
        runs.append(run_trainer(root, monkeypatch, overrides))
    (a, logs_a), (b, logs_b) = runs
    assert a.train_dataset.load_episode(0).rew.abs().max() == 0 and b.train_dataset.load_episode(0).rew.abs().max() > 1
    assert same(a.plan2explore.state_dict(), b.plan2explore.state_dict())
    assert same(a.agent.state_dict(), b.agent.state_dict())
    torch.testing.assert_close(a.plan2explore.cache.features, b.plan2explore.cache.features, rtol=0, atol=0)
    def strip(rows):
        return [{k: v for k, v in r.items() if "/return_" not in k} for r in base.numeric_logs(rows)]
    assert strip(logs_a) == strip(logs_b)  # losses, disagreement, PMPO metrics, sampled actions


def test_test_diagnostics_do_not_touch_plan2explore(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", P2E)
    off, _ = start(tmp_path / "off", monkeypatch, BASE + OFF)
    off._run()
    on, _ = start(tmp_path / "on", monkeypatch, BASE + diag.FRESH)
    on._run()
    assert same(off.plan2explore.state_dict(), on.plan2explore.state_dict())
    assert same(off.agent.state_dict(), on.agent.state_dict())


def test_crash_resume_is_identical_and_changed_settings_are_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(diag, "TINY", P2E)
    full, _ = start(tmp_path / "full", monkeypatch, BASE + OFF)
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
    saved = torch.load(tmp_path / "split/checkpoints/state.pt", weights_only=False)["plan2explore"]
    assert saved["heads"] is not None and saved["refreshes"] == 1 and saved["cache_identity"] is not None
    loads = []
    monkeypatch.setattr(torch.nn.Module, "load_state_dict",
                        (lambda f: lambda self, *a, **k: loads.append(1) or f(self, *a, **k))(torch.nn.Module.load_state_dict))
    for override, match in (("intrinsic_reward.num_members=4", "Plan2Explore settings"),
                            ("intrinsic_reward.bootstrap=false", "Plan2Explore settings"),
                            ("intrinsic_reward.feature_layers=[unet.d_blocks.1]", "Plan2Explore settings"),
                            ("intrinsic_reward.feature_sigma=0.2", "Plan2Explore settings"),
                            ("intrinsic_reward.pool_size=1", "Plan2Explore settings"),
                            ("intrinsic_reward.probe_seed=1", "Plan2Explore settings"),
                            ("intrinsic_reward.feature_std_floor=0.1", "Plan2Explore settings"),
                            ("intrinsic_reward.hidden_dims=[8]", "Plan2Explore settings"),
                            ("intrinsic_reward.lr=0.01", "Plan2Explore settings"),
                            ("intrinsic_reward.batch_size=4", "Plan2Explore settings"),
                            ("intrinsic_reward.fit_updates=6", "Plan2Explore settings"),
                            ("intrinsic_reward.activation=elu", "Plan2Explore settings"),
                            ("intrinsic_reward.encode_batch_size=3", "Plan2Explore settings"),
                            ("intrinsic_reward.fixed_divisor=2.0", "must not switch")):
        loads.clear()
        with pytest.raises(ValueError, match=match):
            start(tmp_path / "split", monkeypatch, [override], resume=True)
        assert not loads, override  # refused before any weights were loaded
    resumed, _ = start(tmp_path / "split", monkeypatch, [], resume=True)
    resumed._run()
    assert same(full.agent.state_dict(), resumed.agent.state_dict())
    assert same(full.opt.state_dict(), resumed.opt.state_dict())
    assert same(full.plan2explore.state_dict(), resumed.plan2explore.state_dict())
    for k in ("python", "numpy", "torch"):
        assert same(full_rng[k], rng_state()[k])


def test_other_methods_refuse_a_plan2explore_checkpoint_and_build_no_heads(tmp_path, monkeypatch):
    p2e, _ = run_p2e(tmp_path / "p2e", monkeypatch, OFF, build_only=True)
    p2e_state = torch.load(tmp_path / "p2e/run/checkpoints/state.pt", weights_only=False)
    monkeypatch.setattr(base, "TINY", base.TINY_LCG)
    lcg, _ = run_trainer(tmp_path / "lcg", monkeypatch, COLLECT, build_only=True)
    lcg_state = torch.load(tmp_path / "lcg/run/checkpoints/state.pt", weights_only=False)
    with pytest.raises(ValueError, match="must not switch"):  # reward settings differ first
        lcg.load_state_dict(p2e_state)
    p2e_state["lcg_reward_scaling"] = lcg_state["lcg_reward_scaling"]
    with pytest.raises(ValueError, match="Plan2Explore feature predictors"):
        lcg.load_state_dict(p2e_state)
    lcg_state["lcg_reward_scaling"] = p2e._lcg_reward_scaling
    with pytest.raises(ValueError, match="Plan2Explore settings None"):
        p2e.load_state_dict(lcg_state)
    for name, extra in (("lcg2", []), ("uniform", ["collection.train.mode=uniform", "intrinsic_reward.enabled=false"])):
        trainer, _ = run_trainer(tmp_path / name, monkeypatch, [*COLLECT, *extra], build_only=True)
        assert trainer.plan2explore is None and len(trainer.agent.denoisers) == 1
        assert trainer.agent.actor_critic.rl_env is None or trainer.agent.actor_critic.rl_env.feature_encoder is None
    import test_ensemble
    monkeypatch.setattr(base, "TINY", test_ensemble.ENS)
    ens, _ = run_trainer(tmp_path / "ens", monkeypatch, COLLECT, build_only=True)
    assert ens.plan2explore is None and ens.agent.actor_critic.rl_env.feature_encoder is None


def test_older_checkpoints_without_the_key_still_resume(tmp_path, monkeypatch):
    run, _ = start(tmp_path / "lcg", monkeypatch, BASE + OFF)
    run._run()
    path = tmp_path / "lcg/checkpoints/state.pt"
    state = torch.load(path, weights_only=False)
    assert state["plan2explore"] is None
    del state["plan2explore"]  # as saved before this method existed
    torch.save(state, path)
    resumed, _ = start(tmp_path / "lcg", monkeypatch, [], resume=True)
    assert resumed.plan2explore is None and resumed.epoch == run.epoch


def test_milestones_are_single_denoiser_with_provenance_and_load_downstream(tmp_path, monkeypatch):
    trainer, _ = run_p2e(tmp_path, monkeypatch, OFF + ["milestones.at_steps=[24]"])
    path = Path("milestones/steps_000024")
    milestone = load_milestone(path, "walk")
    manifest = milestone.manifest
    assert sorted(manifest["files"]) == ["dataset", "denoiser"] and "ensemble" not in manifest
    assert manifest["exploration"]["method"] == "plan2explore_disagreement"
    assert manifest["plan2explore"]["num_members"] == 3 and manifest["plan2explore"]["feature_dim"] == 64
    assert same(milestone.denoiser_state, trainer.agent.denoiser.state_dict())
    assert not any(p.name.startswith("denoiser_member") for p in path.iterdir())


def test_downstream_trainer_loads_a_plan2explore_milestone(tmp_path, monkeypatch):
    import sys
    monkeypatch.syspath_prepend(str(base.ROOT / "tests" / "downstream"))
    import test_downstream_trainer as down
    single = down.build_milestone(tmp_path)
    manifest = read_manifest(single)
    keep = {k: manifest[k] for k in ("created_by", "domain", "task_names", "collected_steps", "num_episodes", "epoch",
                                     "counters", "model", "run_config", "source")}
    keep["exploration"] = dict(collection_mode="policy", intrinsic_reward_enabled=True, method="plan2explore_disagreement")
    keep["plan2explore"] = dict(num_members=5, feature_layers=["unet.d_blocks.2", "unet.d_blocks.3"], feature_dim=2048,
                                feature_sigma=0.1, pool_size=4, probe_sha256="x", last_refresh_epoch=2, refreshes=2,
                                note="single denoiser; feature predictors are not part of the milestone")
    m = load_milestone(single, "walk")
    path = publish_milestone(single.parent, "steps_p2e", m.denoiser_state, m.dataset, m.successor_observations, keep)
    trainer = down.make_trainer(path, monkeypatch)
    assert down.states_equal(m.denoiser_state, trainer.agent.denoiser.state_dict())
    out = trainer.run()
    assert (out / "summary.json").is_file()
