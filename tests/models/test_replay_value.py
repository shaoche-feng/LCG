"""Replay-value learning for the scalar-EMA Gaussian-PMPO controller (agent.actor_critic.replay_value_coef > 0).
CPU only. Several tests are ports of the reviewed DreamerV3-controller tests (test_dreamerv3_controller.py)."""
import inspect
from copy import deepcopy
from pathlib import Path

import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader

import models.pmpo_beta as pmpo_module
import models.replay_value as rv
from data import Batch, BatchSampler, Dataset, Episode, collate_segments_to_batch
from data.segment import SegmentId
from envs import WorldModelEnv
from envs.world_model_env import WorldModelEnvConfig
from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig, InnerModelConfig
from models.pmpo_beta import PMPOOptimizers
from models.replay_value import ReplayStartSampler, replay_lambda_return, unfold_windows
from models.rew_end_model import RewEndModel, RewEndModelConfig
from test_pmpo_beta import make_controller

torch.set_num_threads(2)
LOW, HIGH = [-3., 0.5], [2., 7.]
RV = dict(actor_distribution="tanh_gaussian", slow_critic_coef=1.0, replay_value_coef=0.3)
DREAMER = Path("/home/bigking/scfeng/LCG-stage3-dreamerv3-controller/src/models/dreamerv3_controller.py")


def make_dataset(path, episodes=3, length=20, seed=0, encoded=False):
    """encoded=True: frame t of episode e is the constant (e + 1) / 10 + t / 1000, reward e * 1000 + t."""
    g = torch.Generator().manual_seed(seed)
    ds = Dataset(path, cache_in_ram=True)
    low, high = torch.tensor(LOW), torch.tensor(HIGH)
    for e in range(episodes):
        trunc = torch.zeros(length, dtype=torch.long)
        trunc[-1] = 1
        if encoded:
            obs = ((e + 1) / 10 + torch.arange(length).float() / 1000)[:, None, None, None].expand(length, 3, 8, 8).clone()
            rew = e * 1000 + torch.arange(length).float()
        else:
            obs = torch.rand(length, 3, 8, 8, generator=g) * 2 - 1
            rew = torch.rand(length, generator=g)
        ds.add_episode(Episode(obs, low + torch.rand(length, 2, generator=g) * (high - low), rew,
                               torch.zeros(length, dtype=torch.long), trunc, {}))
    return ds


def make_system(tmp_path, prompts=8, encoded=False, **kwargs):
    torch.manual_seed(7)
    np.random.seed(7)
    model = make_controller(**{**RV, **kwargs})
    denoiser = Denoiser(DenoiserConfig(
        inner_model=InnerModelConfig(img_channels=3, num_steps_conditioning=4, cond_channels=16, depths=[1],
                                     channels=[8], attn_depths=[False], continuous_action_dim=2),
        sigma_data=0.5, sigma_offset_noise=0.3))
    reward = RewEndModel(RewEndModelConfig(lstm_dim=16, img_channels=3, img_size=8, cond_channels=16, depths=[1],
                                         channels=[8], attn_depths=[False], continuous_action_dim=2))
    ds = make_dataset(tmp_path / "replay", encoded=encoded)
    loader = DataLoader(ds, batch_sampler=BatchSampler(ds, 0, 1, prompts, 4), collate_fn=collate_segments_to_batch)
    env = WorldModelEnv(denoiser, reward, loader, WorldModelEnvConfig(
        horizon=3, num_batches_to_preload=1, diffusion_sampler=DiffusionSamplerConfig(num_steps_denoising=2)))
    model.setup_training(env)
    return model, env, ds


def capture(model, monkeypatch):
    rec = {}
    lam = pmpo_module.compute_lambda_returns

    def spy_returns(*a, **k):
        rec["targets"] = lam(*a, **k)
        return rec["targets"]
    monkeypatch.setattr(pmpo_module, "compute_lambda_returns", spy_returns)
    original = model.replay_value_loss

    def spy(replay, boot, start_states):
        rec.update(replay=replay, boot=boot.clone(), start=start_states.clone())
        return original(replay, boot, start_states)
    model.replay_value_loss = spy
    return rec, original


# ---------------------------------------------------------------- reference provenance and arithmetic
@pytest.mark.skipif(not DREAMER.exists(), reason="reviewed Dreamer controller worktree not present")
def test_module_code_is_verbatim_reference():
    ref = DREAMER.read_text().replace("\r\n", "\n")
    for obj in (rv.replay_lambda_return, rv.ReplayStartSampler, rv.unfold_windows, rv.ReplayWindowLoader):
        assert inspect.getsource(obj) in ref, obj.__name__


def reference_lambda_return(last, term, rew, boot, disc, lam):
    """Line-for-line numpy port of dreamerv3/agent.py lambda_return (reference indexing), from the reviewed tests."""
    rets = [boot[:, -1]]
    live = (1 - term)[:, 1:] * disc
    cont = (1 - last)[:, 1:] * lam
    interm = rew[:, 1:] + (1 - cont) * live * boot[:, 1:]
    for t in reversed(range(live.shape[1])):
        rets.append(interm[:, t] + live[:, t] * cont[:, t] * rets[-1])
    return np.stack(list(reversed(rets))[:-1], 1)


def test_replay_targets_match_reference_lambda_return():
    rng = np.random.default_rng(0)
    S, K, disc, lam = 5, 6, 0.985, 0.95
    rew = rng.normal(size=(S, K)).astype(np.float32)
    term = (rng.random((S, K)) < 0.2).astype(np.float32)
    last = np.maximum(term, (rng.random((S, K)) < 0.2)).astype(np.float32)
    boot = rng.normal(size=(S, K)).astype(np.float32) * 5
    shift = lambda a: np.concatenate([np.zeros((S, 1), np.float32), a[:, :-1]], 1)
    ref = reference_lambda_return(shift(last), shift(term), shift(rew), boot, disc, lam)
    ours = replay_lambda_return(*map(torch.from_numpy, (rew, term, last, boot)), disc, lam)
    np.testing.assert_allclose(ours.numpy(), ref, rtol=1e-5, atol=1e-5)


def test_replay_targets_hand_calculated():
    g, lam = 0.9, 0.5
    rew = torch.tensor([[1., 2., 3., 4.]])
    boot = torch.tensor([[10., 20., 30., 40.]])
    z = torch.zeros(1, 4)
    R = replay_lambda_return(rew, z, z, boot, g, lam)[0]
    r2 = 3 + g * (0.5 * 40 + 0.5 * 40)            # 39
    r1 = 2 + g * (0.5 * 30 + 0.5 * r2)            # 33.05
    r0 = 1 + g * (0.5 * 20 + 0.5 * r1)            # 24.8725
    torch.testing.assert_close(R, torch.tensor([r0, r1, r2]))
    last = torch.tensor([[0., 1., 0., 0.]])       # transition leaving state 1 is the last one: bootstrap there
    R = replay_lambda_return(rew, z, last, boot, g, lam)[0]
    torch.testing.assert_close(R[1], torch.tensor(2 + g * 30.))
    term = torch.tensor([[0., 1., 0., 0.]])       # true terminal leaving state 1: no continuation value
    R = replay_lambda_return(rew, term, term, boot, g, lam)[0]
    torch.testing.assert_close(R[1], torch.tensor(2.))
    torch.testing.assert_close(R[0], torch.tensor(1 + g * (0.5 * 20 + 0.5 * 2.)))


# ---------------------------------------------------------------- sampling, timing, alignment
def test_window_sampler_boundaries_and_short_episodes(tmp_path):
    ds = make_dataset(tmp_path / "d", episodes=4, length=9)
    sampler = ReplayStartSampler(ds, segments=50, window=4, starts=4, sample_weights=None)
    np.random.seed(0)
    for _ in range(20):
        for sid in sampler.sample():
            assert sid.stop - sid.start == 7 and 4 <= sid.stop <= ds.lengths[sid.episode_id]
            seg = ds[sid]
            assert bool(seg.mask_padding[3:].all())  # imagination starts never padding
            assert not seg.mask_padding[:max(0, -sid.start)].any() and seg.mask_padding[max(0, -sid.start):].all()
    short = make_dataset(tmp_path / "s", episodes=1, length=3)
    with pytest.raises(ValueError, match="shorter than the number of imagination starts"):
        ReplayStartSampler(short, segments=2, window=4, starts=4).sample()


def test_timing_and_rollout_to_replay_mapping(tmp_path, monkeypatch):
    model, env, ds = make_system(tmp_path, encoded=True)
    rec, original = capture(model, monkeypatch)
    _, metrics = model()
    replay, start, boot, targets = rec["replay"], rec["start"], rec["boot"], rec["targets"]
    S, K, W, F = 2, 4, 4, 4
    rows = model.replay_start_rows(S)
    frame = lambda e, t: (e + 1) / 10 + t / 1000
    for s, sid in enumerate(replay.segment_ids):
        e = sid.episode_id
        # Batch segment ids hold the clipped real range; the sampled window ends at `stop` and spans W + K - 1 frames.
        for j in range(W + K - 1):
            t = sid.stop - (W + K - 1) + j
            assert bool(replay.mask_padding[s, j]) == (t >= 0)
            if t >= 0:  # real frame (padding only before the start of the episode)
                assert replay.obs[s, j].flatten()[0].item() == pytest.approx(frame(e, t), abs=1e-6)
                assert replay.rew[s, j].item() == e * 1000 + t
        for k in range(K):
            t = sid.stop - K + k  # imagination start k of segment s = episode frame stop-K+k
            torch.testing.assert_close(start[rows[s, k]], torch.full((3 * F, 8, 8), frame(e, t)), rtol=0, atol=1e-6)
            assert boot[rows[s, k]] == targets[rows[s, k], 0]  # annotation = lambda-return at step 0 of that rollout
    # independent recomputation of the replay loss from the dataset indices
    total = 0.0
    with torch.no_grad():
        for s, sid in enumerate(replay.segment_ids):
            e, t0 = sid.episode_id, sid.stop - K
            b = [boot[rows[s, k]].item() for k in range(K)]
            R = [None] * K
            R[K - 1] = b[K - 1]
            for t in reversed(range(K - 1)):
                r = e * 1000 + t0 + t
                last = 1.0 if t0 + t == ds.lengths[e] - 1 else 0.0
                R[t] = r + 0.985 * ((1 - 0.95 * (1 - last)) * b[t + 1] + 0.95 * (1 - last) * R[t + 1])
            for t in range(K - 1):
                x = torch.full((1, 3 * F, 8, 8), frame(e, t0 + t))
                v = model.value(x).squeeze().item()
                vs = model.slow_value(x).squeeze().item()
                w = 0.0 if t0 + t == ds.lengths[e] - 1 else 1.0
                total += w * ((v - R[t]) ** 2 + 1.0 * (v - vs) ** 2)
    assert float(metrics["loss_repval"]) == pytest.approx(total / (S * (K - 1)), rel=1e-4)


def test_replay_inputs_match_imagination_starts_with_motion(tmp_path, monkeypatch):
    model, env, ds = make_system(tmp_path)
    rec, original = capture(model, monkeypatch)
    model()
    S, K, W, F = 2, 4, 4, model.frame_stack
    replay, start = rec["replay"], rec["start"]
    stacks = model.replay_inputs(replay)
    rows = model.replay_start_rows(S)
    assert stacks.shape == (S * (K - 1), 3 * F, 8, 8)
    for s in range(S):
        for t in range(K - 1):
            i = s * (K - 1) + t
            torch.testing.assert_close(stacks[i], start[rows[s, t]], rtol=0, atol=0)
            torch.testing.assert_close(stacks[i], replay.obs[s, W - 1 + t].repeat(F, 1, 1), rtol=0, atol=0)
            assert not torch.equal(stacks[i], replay.obs[s, t:t + W].flatten(0, 1))  # never the real-history stack
            assert (replay.obs[s, W - 1 + t] - replay.obs[s, W - 2 + t]).abs().max() > 0  # the frames do move
    with torch.no_grad():  # same input and parameters: replay critic = imagination critic at that start
        torch.testing.assert_close(model.value(stacks), model.value(start[rows[:, :K - 1].flatten()]), rtol=0, atol=0)
    wrong = start.clone()
    wrong[[0, 1]] = wrong[[1, 0]]  # off-by-one row mapping must be rejected
    with pytest.raises(RuntimeError, match="differ from the imagination start states"):
        original(replay, rec["boot"], wrong)


def test_terminal_truncation_and_padding_masks(tmp_path):
    model, env, ds = make_system(tmp_path)
    S, K, W, F = 2, 4, 4, 4
    g = torch.Generator().manual_seed(3)
    obs = torch.rand(S, W + K - 1, 3, 8, 8, generator=g) * 2 - 1
    rew = torch.arange(S * (W + K - 1), dtype=torch.float).view(S, -1)
    end = torch.zeros(S, W + K - 1, dtype=torch.long)
    trunc = torch.zeros(S, W + K - 1, dtype=torch.long)
    mask = torch.ones(S, W + K - 1, dtype=torch.bool)
    trunc[0, W - 1 + 1] = 1   # segment 0: transition leaving state 1 is a time-limit truncation
    end[1, W - 1 + 0] = 1     # segment 1: transition leaving state 0 is a true terminal
    mask[1, :2] = False       # segment 1: padding before the episode start (allowed, outside the states)
    batch = Batch(obs, torch.zeros(S, W + K - 1, 2), rew, end, trunc, mask, [{}] * S,
                  [SegmentId(0, 0, W + K - 1), SegmentId(1, -2, W + K - 3)])
    stacks = obs[:, W - 1:W - 1 + K].flatten(0, 1).repeat(1, F, 1, 1)
    boot = torch.arange(S * K, dtype=torch.float) * 3
    _, stats = model.replay_value_loss(batch, boot, stacks)
    sl = slice(W - 1, W - 1 + K)
    R = replay_lambda_return(rew[:, sl], end[:, sl].float(), torch.clamp(end + trunc, max=1)[:, sl].float(),
                             boot.view(S, K), model.cfg.gamma, model.cfg.lambda_)
    # hand checks of the boundary cases
    b = boot.view(S, K)
    assert R[0, 1].item() == pytest.approx(rew[0, W].item() + model.cfg.gamma * b[0, 2].item())   # truncation: bootstrap
    assert R[1, 0].item() == pytest.approx(rew[1, W - 1].item())                                  # terminal: no continuation
    w = torch.ones(S, K - 1)
    w[0, 1] = 0.0  # truncated transition has no stored successor: no target
    with torch.no_grad():
        v = model.value(stacks[torch.tensor([0, 1, 2, 4, 5, 6])]).squeeze(-1).view(S, K - 1)
        vs = model.slow_value(stacks[torch.tensor([0, 1, 2, 4, 5, 6])]).squeeze(-1).view(S, K - 1)
    expected = (w * ((v - R) ** 2 + (v - vs) ** 2)).mean()
    assert float(stats["loss_repval"]) == pytest.approx(float(expected), rel=1e-5)
    assert float(stats["replay_weight_mean"]) == pytest.approx(5 / 6)
    mask[1, W - 1 + 1] = False  # padding inside the states (cannot occur in sampling) is also masked
    batch2 = Batch(obs, torch.zeros(S, W + K - 1, 2), rew, end, trunc, mask, [{}] * S, batch.segment_ids)
    _, stats2 = model.replay_value_loss(batch2, boot, stacks)
    assert float(stats2["replay_weight_mean"]) == pytest.approx(3 / 6)


# ---------------------------------------------------------------- gradients, loss formula, EMA timing
def test_targets_detached_and_gradient_paths(tmp_path, monkeypatch):
    model, env, ds = make_system(tmp_path)
    rec, original = capture(model, monkeypatch)
    model()
    assert not rec["boot"].requires_grad and not rec["targets"].requires_grad
    for p in list(model.parameters()) + list(env.sampler.denoiser.parameters()) + list(env.rew_end_model.parameters()):
        p.grad = None
    loss, stats = original(rec["replay"], rec["boot"], rec["start"])
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.value.parameters())
    assert all(p.grad is None for p in model.actor.parameters())
    assert all(p.grad is None for p in model.slow_value.parameters())
    assert all(p.grad is None for p in model.prior_actor.parameters())
    assert all(p.grad is None for p in env.sampler.denoiser.parameters())
    assert all(p.grad is None for p in env.rew_end_model.parameters())


def test_total_critic_loss_formula_and_coefficients(tmp_path, monkeypatch):
    model, env, ds = make_system(tmp_path)
    loss, m = model()
    f = lambda k: float(m[k])
    assert f("loss_value_imag") == pytest.approx(f("loss_value_target") + 1.0 * f("loss_slowreg"), rel=1e-5)
    assert f("loss_repval") == pytest.approx(f("loss_repval_target") + 1.0 * f("loss_repval_slowreg"), rel=1e-5)
    assert f("loss_repval_weighted") == pytest.approx(0.3 * f("loss_repval"), rel=1e-6)
    assert f("loss_value") == pytest.approx(f("loss_value_imag") + 0.3 * f("loss_repval"), rel=1e-5)
    assert loss.item() == pytest.approx(f("loss_actor") + f("loss_value"), rel=1e-5)
    assert f("loss_repval") > 0 and f("replay_weight_mean") > 0
    for k in ("replay_pred_mean", "replay_target_mean", "replay_bias", "replay_mae", "replay_rmse", "replay_slow_gap"):
        assert np.isfinite(f(k)), k
    assert f("replay_rmse") >= f("replay_mae") - 1e-6


def test_ema_update_once_after_critic_step(tmp_path):
    model, env, ds = make_system(tmp_path)
    opt = PMPOOptimizers(model, warmup_steps=0)
    order = []
    value_step, update = opt.value.step, model.update_slow_critic
    opt.value.step = lambda *a, **k: (order.append("value_step"), value_step(*a, **k))[1]
    model.update_slow_critic = lambda: (order.append("slow_update"), update())[1]
    before = deepcopy(model.slow_value.state_dict())
    loss, _ = model()
    loss.backward()
    assert order == [] and all(torch.equal(v, model.slow_value.state_dict()[k]) for k, v in before.items())
    m = opt.step()
    assert order == ["value_step", "slow_update"]
    assert "value_grad_clipped" in m and "value_grad_norm" in m
    for k, online in model.value.state_dict().items():
        torch.testing.assert_close(model.slow_value.state_dict()[k], 0.98 * before[k] + 0.02 * online, rtol=0, atol=1e-7)


def test_imagination_budget_unchanged_and_prompt_divisibility(tmp_path):
    model, env, ds = make_system(tmp_path, prompts=8)
    obs, act, *_ = model.collect_imagination()
    assert obs.shape[:2] == (8, 3)  # prompts x horizon, unchanged
    assert model.replay_loader.replay.obs.shape[:2] == (2, 7)  # 2 segments x (4 + 4 - 1) frames
    with pytest.raises(ValueError, match="multiple of replay_starts"):
        make_system(tmp_path / "x", prompts=6)


# ---------------------------------------------------------------- disabled mode, checkpoint/resume
def test_disabled_mode_keeps_loader_state_and_training_identical(tmp_path):
    torch.set_num_threads(1)
    off = dict(replay_value_coef=0.0)
    a, env_a, _ = make_system(tmp_path / "a", **off)
    b, env_b, _ = make_system(tmp_path / "b", replay_value_coef=0.0, replay_starts=8)
    assert a.replay_loader is None and isinstance(env_a.data_loader, DataLoader)
    assert "replay_loader" not in dict(a.named_modules())
    oa, ob = PMPOOptimizers(a, warmup_steps=2), PMPOOptimizers(b, warmup_steps=2)
    for _ in range(3):
        for m, o in ((a, oa), (b, ob)):
            o.zero_grad()
            loss, metrics = m()
            loss.backward()
            o.step()
    assert not any(k.startswith(("replay_", "loss_repval", "loss_value_imag")) for k in metrics) and "value_grad_clipped" not in metrics
    for k, v in a.state_dict().items():
        if torch.is_tensor(v):
            assert torch.equal(v, b.state_dict()[k]), k
    assert torch.equal(a._rng["cpu"], b._rng["cpu"]) and a._rng["python"] == b._rng["python"]
    na, nb = a._rng["numpy"], b._rng["numpy"]
    assert na[0] == nb[0] and np.array_equal(na[1], nb[1]) and tuple(na[2:]) == tuple(nb[2:])


def test_checkpoint_round_trip_with_replay_value(tmp_path):
    torch.set_num_threads(1)
    model, env, ds = make_system(tmp_path)
    opt = PMPOOptimizers(model, warmup_steps=5)
    for _ in range(2):
        opt.zero_grad()
        loss, _ = model()
        loss.backward()
        opt.step()
    state = deepcopy(dict(model=model.state_dict(), opt=opt.state_dict()))
    assert any(k.startswith("slow_value.") for k in state["model"]) and "_extra_state" in state["model"]
    opt.zero_grad()
    loss_a, m_a = model()
    loss_a.backward()
    opt.step()
    after = deepcopy(model.state_dict())
    model2, env2, _ = make_system(tmp_path / "b")
    env2.data_loader.dataset.__dict__.update(ds.__dict__)
    torch.manual_seed(123)
    model2.load_state_dict(state["model"])
    opt2 = PMPOOptimizers(model2, warmup_steps=5)
    opt2.load_state_dict(state["opt"])
    env2.sampler.denoiser.load_state_dict(env.sampler.denoiser.state_dict())
    env2.rew_end_model.load_state_dict(env.rew_end_model.state_dict())
    opt2.zero_grad()
    loss_b, m_b = model2()
    loss_b.backward()
    opt2.step()
    assert loss_a.item() == loss_b.item()
    for k in ("loss_repval", "replay_target_mean", "replay_boot_mean", "loss_slowreg"):
        assert float(m_a[k]) == float(m_b[k]), k
    for k, v in after.items():
        if torch.is_tensor(v):
            assert torch.equal(v, model2.state_dict()[k]), k
