"""DreamerV3-style controller (agent=dreamerv3): reference calculations, gradient paths, replay
targets, slow critic, return normalization and checkpoint round trips. CPU only."""
from copy import deepcopy

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

from data import BatchSampler, Dataset, Episode, collate_segments_to_batch
from envs import WorldModelEnv
from envs.world_model_env import WorldModelEnvConfig
from models.actor_critic import compute_lambda_returns
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig, InnerModelConfig
from models.dreamerv3_controller import (
    DreamerV3Controller, DreamerV3ControllerConfig, DreamerV3Optimizers, dreamer_actor_loss,
    imagination_weights, replay_lambda_return, symexp, symexp_bins, twohot_encode, twohot_loss, twohot_mean,
    unfold_windows)
from models.pmpo_beta import PMPOBeta
from models.rew_end_model import RewEndModel, RewEndModelConfig

torch.set_num_threads(2)
LOW, HIGH = [-3., 0.5], [2., 7.]


def make_dreamer(**kwargs):
    cfg = dict(img_channels=3, img_size=8, channels=[8], down=[0], continuous_action_dim=2, action_low=LOW,
               action_high=HIGH, actor_hidden_dims=(16,), value_hidden_dims=(16,), imagination_horizon=3)
    cfg.update(kwargs)
    return DreamerV3Controller(DreamerV3ControllerConfig(**cfg))


def make_dataset(tmp_path, episodes=3, length=20, seed=0):
    g = torch.Generator().manual_seed(seed)
    ds = Dataset(tmp_path / "replay", cache_in_ram=True)
    low, high = torch.tensor(LOW), torch.tensor(HIGH)
    for _ in range(episodes):
        trunc = torch.zeros(length, dtype=torch.long)
        trunc[-1] = 1
        ds.add_episode(Episode(torch.rand(length, 3, 8, 8, generator=g) * 2 - 1,
                               low + torch.rand(length, 2, generator=g) * (high - low),
                               torch.rand(length, generator=g), torch.zeros(length, dtype=torch.long), trunc, {}))
    return ds


def make_system(tmp_path, prompts=8, **kwargs):
    torch.manual_seed(7)
    np.random.seed(7)
    model = make_dreamer(**kwargs)
    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=4, cond_channels=16, depths=[1],
                                     channels=[8], attn_depths=[False], continuous_action_dim=2),
        sigma_data=0.5, sigma_offset_noise=0.3))
    reward = RewEndModel(RewEndModelConfig(lstm_dim=16, img_channels=3, img_size=8, cond_channels=16, depths=[1],
                                         channels=[8], attn_depths=[False], continuous_action_dim=2))
    ds = make_dataset(tmp_path)
    loader = DataLoader(ds, batch_sampler=BatchSampler(ds, 0, 1, prompts, 4), collate_fn=collate_segments_to_batch)
    env = WorldModelEnv(denoiser, reward, loader, WorldModelEnvConfig(
        horizon=3, num_batches_to_preload=1, diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)))
    model.setup_training(env)
    return model, env, ds


# ---------------------------------------------------------------- actor loss -------------------
def test_actor_loss_and_gradients_match_explicit_calculation():
    logp = torch.tensor([-1.0, 0.5, -2.0, 0.25], requires_grad=True)
    ent = torch.tensor([1.5, -0.5, 2.0, 0.0], requires_grad=True)
    adv = torch.tensor([0.3, -1.2, 2.0, 0.0], requires_grad=True)
    w = torch.tensor([0.997, 0.994, 0.5, 0.0], requires_grad=True)
    coef = 3e-4
    loss = dreamer_actor_loss(logp, ent, adv, w, coef)
    explicit = sum(-wi * (li * ai + coef * ei) for wi, li, ai, ei in zip(w.tolist(), logp.tolist(), adv.tolist(), ent.tolist())) / 4
    assert loss.item() == pytest.approx(explicit, rel=1e-6)
    loss.backward()
    torch.testing.assert_close(logp.grad, -(w * adv).detach() / 4)
    torch.testing.assert_close(ent.grad, -(w * coef).detach() / 4)
    assert adv.grad is None and w.grad is None  # stop-gradients on advantage and weight


def test_actor_loss_uses_no_pmpo_groups_magnitude_matters():
    logp = torch.zeros(2, requires_grad=True)
    small = dreamer_actor_loss(logp, torch.zeros(2), torch.tensor([0.1, -0.1]), torch.ones(2), 0.0)
    (g_small,) = torch.autograd.grad(small, logp)
    large = dreamer_actor_loss(logp, torch.zeros(2), torch.tensor([10.0, -0.1]), torch.ones(2), 0.0)
    (g_large,) = torch.autograd.grad(large, logp)
    assert g_large[0] == pytest.approx(100 * g_small[0])  # magnitude-weighted, not sign-only


# ---------------------------------------------------------------- return normalization --------
def test_return_percentiles_ema_and_denominator_floor():
    model = make_dreamer()
    x = torch.arange(101, dtype=torch.float32)  # 5th/95th percentiles exactly 5 and 95
    assert np.percentile(x.numpy(), 5) == 5 and np.percentile(x.numpy(), 95) == 95
    model.update_return_norm(x)
    assert model.retnorm_lo.item() == pytest.approx(0.05) and model.retnorm_hi.item() == pytest.approx(0.95)
    assert model.return_scale().item() == 1.0  # floor: 0.9 < 1 is not amplified
    for n in range(2, 301):
        model.update_return_norm(x)
    assert model.retnorm_lo.item() == pytest.approx(5 * (1 - 0.99 ** 300), rel=1e-4)
    assert model.retnorm_hi.item() == pytest.approx(95 * (1 - 0.99 ** 300), rel=1e-4)
    assert model.return_scale().item() == pytest.approx(90 * (1 - 0.99 ** 300), rel=1e-4)
    y = torch.randn(480) * 7
    lo = model.retnorm_lo.clone()
    model.update_return_norm(y)
    assert model.retnorm_lo.item() == pytest.approx(0.99 * lo.item() + 0.01 * np.percentile(y.numpy(), 5), rel=1e-5)


# ---------------------------------------------------------------- two-hot critic ---------------
def test_symexp_bins_match_reference_construction():
    bins = symexp_bins(255)
    assert bins.shape == (255,) and bins.dtype == torch.float32
    assert bins[127].item() == 0 and torch.equal(bins, -bins.flip(0))
    assert bins[-1].item() == pytest.approx(float(np.expm1(20.0)), rel=1e-6)
    assert (bins[1:] > bins[:-1]).all()
    half = symexp(torch.linspace(-20, 0, 128))
    torch.testing.assert_close(bins[:128], half)


def test_twohot_encoding_interpolation_readout_and_gradients():
    bins = symexp_bins(255)
    t = torch.tensor([0.0, bins[140].item(), 0.37, -12.5, 3e3, 1e12, -1e12])
    enc = twohot_encode(t, bins)
    torch.testing.assert_close(enc.sum(-1), torch.ones(len(t)))
    assert ((enc > 0).sum(-1) <= 2).all()
    assert enc[0, 127] == 1 and enc[1, 140] == 1  # exactly on a bin -> one-hot
    torch.testing.assert_close((enc * bins).sum(-1)[:5], t[:5], rtol=1e-5, atol=1e-5)  # linear interpolation
    assert enc[5, -1] == 1 and enc[6, 0] == 1  # beyond range clamps to the end bins
    logits = torch.zeros(4, 255, requires_grad=True)
    assert torch.equal(twohot_mean(logits, bins), torch.zeros(4))  # symmetric readout is exactly 0
    sharp = torch.log(enc[:5].clamp_min(1e-30))
    torch.testing.assert_close(twohot_mean(sharp, bins), t[:5], rtol=1e-4, atol=1e-4)
    loss = twohot_loss(logits, torch.tensor([1.0, -50.0, 1e4, 0.0]), bins).sum()
    (g,) = torch.autograd.grad(loss, logits)
    assert torch.isfinite(g).all() and g.abs().sum() > 0


def test_value_output_zero_initialized_and_reads_zero():
    model = make_dreamer()
    assert torch.equal(model.value.head[-1].weight, torch.zeros_like(model.value.head[-1].weight))
    out = model.predict_act_value(torch.randn(5, 12, 8, 8))
    assert torch.equal(out[1], torch.zeros(5))


# ---------------------------------------------------------------- slow critic -------------------
def test_slow_critic_initialization_ema_and_exclusion_from_optimizers(tmp_path):
    model, env, _ = make_system(tmp_path)
    for (k, s), (_, v) in zip(model.slow_value.state_dict().items(), model.value.state_dict().items()):
        assert torch.equal(s, v), k
    assert not any(p.requires_grad for p in model.slow_value.parameters())
    opt = DreamerV3Optimizers(model, warmup_steps=0)
    optimized = {id(p) for g in (opt.actor.param_groups + opt.value.param_groups) for p in g["params"]}
    assert not optimized & {id(p) for p in model.slow_value.parameters()}
    slow_before = deepcopy(model.slow_value.state_dict())
    loss, _ = model()
    loss.backward()
    assert all(p.grad is None for p in model.slow_value.parameters())
    opt.step()
    for k, v in model.value.state_dict().items():
        torch.testing.assert_close(model.slow_value.state_dict()[k], 0.98 * slow_before[k] + 0.02 * v)
    assert model.slow_value_count.item() == 1 and model.updates.item() == 1


# ---------------------------------------------------------------- returns and weights -----------
def test_imagination_weights_and_horizon_bootstrap():
    g = 0.997
    end = torch.tensor([[0, 0, 0], [0, 1, 0]])
    w = imagination_weights(end, g)
    torch.testing.assert_close(w, torch.tensor([[g, g ** 2, g ** 3], [g, g ** 2, 0.0]]))
    rew = torch.tensor([[1.0, 2.0, 3.0]])
    boot = torch.tensor([[10.0, 20.0, 30.0]])
    trunc = torch.tensor([[0, 0, 1]])
    ret = compute_lambda_returns(rew, torch.zeros_like(trunc), trunc, boot, g, 0.95)
    assert ret[0, 2].item() == pytest.approx(3 + g * 30)  # truncation bootstraps from the critic
    assert ret[0, 1].item() == pytest.approx(2 + g * ((1 - 0.95) * 20 + 0.95 * ret[0, 2].item()))
    term = compute_lambda_returns(rew, torch.tensor([[0, 0, 1]]), torch.zeros_like(trunc), boot, g, 0.95)
    assert term[0, 2].item() == pytest.approx(3.0)  # a predicted end is terminal


def reference_lambda_return(last, term, rew, boot, disc, lam):
    """Line-for-line numpy port of dreamerv3/agent.py lambda_return (reference indexing)."""
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


def test_replay_targets_match_reference_lambda_return():
    rng = np.random.default_rng(0)
    S, K, disc, lam = 5, 6, 1 - 1 / 333, 0.95
    rew = rng.normal(size=(S, K)).astype(np.float32)
    term = (rng.random((S, K)) < 0.2).astype(np.float32)
    last = np.maximum(term, (rng.random((S, K)) < 0.2)).astype(np.float32)
    boot = rng.normal(size=(S, K)).astype(np.float32) * 5
    # reference arrays: rew/term/last at index t+1 describe the transition leaving state t
    shift = lambda a: np.concatenate([np.zeros((S, 1), np.float32), a[:, :-1]], 1)
    ref = reference_lambda_return(shift(last), shift(term), shift(rew), boot, disc, lam)
    ours = replay_lambda_return(*map(torch.from_numpy, (rew, term, last, boot)), disc, lam)
    np.testing.assert_allclose(ours.numpy(), ref, rtol=1e-5, atol=1e-5)


def test_replay_windows_align_with_imagination_starts(tmp_path):
    model, env, ds = make_system(tmp_path)
    obs, *_ = model.collect_imagination()
    replay = model.replay_loader.replay
    S, K, W = 2, 4, 4
    assert replay.obs.shape[:2] == (S, W + K - 1)
    # every imagination start (row s*K+k, step 0) is the replay frame W-1+k of segment s,
    # repeated over the four stack slots (repeated-final-frame initialization)
    starts = replay.obs[:, W - 1:W - 1 + K].flatten(0, 1)
    torch.testing.assert_close(obs[:, 0, -3:], starts)
    torch.testing.assert_close(obs[:, 0, :3], starts)
    assert replay.mask_padding[:, W - 1:].all()  # imagination starts are never padding
    windows = unfold_windows(replay, W, K)
    torch.testing.assert_close(windows.obs[:, -1], starts)


# ---------------------------------------------------------------- full update / gradient paths ---
def test_full_update_finite_and_gradient_paths(tmp_path):
    model, env, _ = make_system(tmp_path)
    opt = DreamerV3Optimizers(model, warmup_steps=0)
    loss, metrics = model()
    for key in ("loss_policy", "loss_value_imag", "loss_repval", "return_scale", "advantage_mean",
                "replay_return_mean", "positive_fraction", "value_abs_max", "policy_entropy", "kl_initial"):
        assert np.isfinite(float(metrics[key])), key
    assert "kl_prior" not in metrics
    loss.backward()
    for module in (env.sampler.denoiser, env.rew_end_model, model.slow_value, model.initial_actor):
        assert all(p.grad is None for p in module.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.actor.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value.parameters())
    metrics.update(opt.step())
    assert np.isfinite(float(metrics["actor_grad_norm"])) and np.isfinite(float(metrics["value_grad_norm"]))


@pytest.mark.parametrize("which", ["actor_only", "critic_only"])
def test_actor_and_critic_losses_touch_only_their_own_parameters(tmp_path, which):
    scales = dict(policy_loss_scale=1.0, value_loss_scale=0.0, repval_loss_scale=0.0) if which == "actor_only" \
        else dict(policy_loss_scale=0.0, value_loss_scale=1.0, repval_loss_scale=0.3)
    model, env, _ = make_system(tmp_path, **scales)
    loss, _ = model()
    loss.backward()
    zero = lambda m: all(p.grad is None or p.grad.abs().sum() == 0 for p in m.parameters())
    assert zero(model.value) if which == "actor_only" else zero(model.actor)
    assert not zero(model.actor) if which == "actor_only" else not zero(model.value)


# ---------------------------------------------------------------- checkpoint / resume -----------
def test_checkpoint_round_trip_preserves_optimizer_slow_critic_normalizer_and_rng(tmp_path):
    model, env, ds = make_system(tmp_path)
    opt = DreamerV3Optimizers(model, warmup_steps=5)
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model()
        loss.backward()
        opt.step()
    state = deepcopy(dict(model=model.state_dict(), opt=opt.state_dict()))
    assert {"slow_value.head.1.weight", "retnorm_lo", "retnorm_hi", "slow_value_count", "updates"} <= set(state["model"])
    assert "_extra_state" in state["model"]
    # continue the original
    opt.zero_grad()
    loss_a, metrics_a = model()
    loss_a.backward()
    opt.step()
    after_a = deepcopy(model.state_dict())
    # restore into a fresh system with different initialization and replay the same update
    model2, env2, _ = make_system(tmp_path / "b")
    env2.data_loader.dataset.__dict__.update(ds.__dict__)  # same replay contents
    torch.manual_seed(123)
    model2.load_state_dict(state["model"])
    opt2 = DreamerV3Optimizers(model2, warmup_steps=5)
    opt2.load_state_dict(state["opt"])
    env2.sampler.denoiser.load_state_dict(env.sampler.denoiser.state_dict())
    env2.rew_end_model.load_state_dict(env.rew_end_model.state_dict())
    opt2.zero_grad()
    loss_b, metrics_b = model2()
    loss_b.backward()
    opt2.step()
    assert loss_a.item() == loss_b.item()
    for k in ("return_scale", "retnorm_lo", "loss_repval", "slow_value_mean"):
        assert float(metrics_a[k]) == float(metrics_b[k]), k
    for k, v in after_a.items():
        if torch.is_tensor(v):
            assert torch.equal(v, model2.state_dict()[k]), k


# ---------------------------------------------------------------- PMPO path unchanged -----------
def test_pmpo_path_is_still_selected_and_unchanged():
    from test_pmpo_beta import make_controller
    model = make_controller(actor_distribution="tanh_gaussian")
    assert type(model) is PMPOBeta and hasattr(model, "prior_actor") and not hasattr(model, "slow_value")
    assert getattr(model, "optimizers_class", None) is None


def test_hydra_agent_selection(tmp_path):
    from pathlib import Path
    from hydra import compose, initialize_config_dir
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    from agent import Agent
    root = Path(__file__).resolve().parents[2]
    OmegaConf.register_new_resolver("eval", eval, replace=True)
    kinds = {}
    for agent in ("pmpo_beta", "dreamerv3"):
        with initialize_config_dir(version_base="1.3", config_dir=str(root / "config")):
            cfg = compose(config_name="trainer", overrides=[f"agent={agent}", "env.train.size=8"])
        OmegaConf.resolve(cfg)
        kinds[agent] = type(Agent(instantiate(cfg.agent, continuous_action_dim=2, action_low=LOW,
                                              action_high=HIGH)).actor_critic).__name__
    assert kinds == {"pmpo_beta": "PMPOBeta", "dreamerv3": "DreamerV3Controller"}


# ---------------------------------------------------------------- replay-value input identity ---
def test_replay_value_inputs_match_imagination_start_states_with_motion(tmp_path, monkeypatch):
    """Whole stacked critic input and bootstrap-to-replay index mapping, on replay with motion."""
    import models.dreamerv3_controller as mod
    model, env, ds = make_system(tmp_path)
    captured = {}
    lambda_returns = mod.compute_lambda_returns

    def spy_returns(*args, **kwargs):
        captured["targets"] = lambda_returns(*args, **kwargs)
        return captured["targets"]
    monkeypatch.setattr(mod, "compute_lambda_returns", spy_returns)
    replay_value_loss = model.replay_value_loss

    def spy_replay(replay, boot, start_states=None):
        captured.update(replay=replay, boot=boot.clone(), start=start_states.clone())
        return replay_value_loss(replay, boot, start_states)
    model.replay_value_loss = spy_replay
    model()
    S, K, W, F = 2, 4, 4, model.frame_stack
    replay, start, targets = captured["replay"], captured["start"], captured["targets"]
    stacks = model.replay_inputs(replay)
    rows = model.replay_start_rows(S)
    assert stacks.shape == (S * (K - 1), 3 * F, 8, 8)
    for s in range(S):
        for t in range(K - 1):
            i = s * (K - 1) + t
            # entire 4-frame input = the stored imagination state at step 0 of rollout s*K+t ...
            torch.testing.assert_close(stacks[i], start[rows[s, t]], rtol=0, atol=0)
            # ... = that replay state's frame repeated (repeated-final-frame initialization) ...
            torch.testing.assert_close(stacks[i], replay.obs[s, W - 1 + t].repeat(F, 1, 1), rtol=0, atol=0)
            # ... and NOT the real-history stack, which carries motion here
            real_history = replay.obs[s, t:t + W].flatten(0, 1)
            assert not torch.equal(stacks[i], real_history)
            assert (replay.obs[s, W - 1 + t] - replay.obs[s, W - 2 + t]).abs().max() > 0
            # bootstrap of replay state t = lambda-return at step 0 of the rollout started there
            assert captured["boot"].view(S, K)[s, t] == targets[rows[s, t], 0]
        assert captured["boot"].view(S, K)[s, K - 1] == targets[rows[s, K - 1], 0]
    with torch.no_grad():  # same input and parameters -> the replay critic equals the imagination critic
        replay_value = model.value_mean(model.value(stacks))
        imagined_value = model.value_mean(model.value(start[rows[:, :K - 1].flatten()]))
    torch.testing.assert_close(replay_value, imagined_value, rtol=0, atol=0)
    wrong = start.clone()
    wrong[[0, 1]] = wrong[[1, 0]]  # off-by-one row mapping must be rejected
    with pytest.raises(RuntimeError, match="differ from the imagination start states"):
        replay_value_loss(replay, captured["boot"], wrong)
