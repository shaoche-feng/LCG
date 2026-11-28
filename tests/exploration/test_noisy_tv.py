"""Cartpole noisy TV (docs/implement/AlEATRORISTIC_EXPERIMENT_DESIGN.md): the observation wrapper, its factory hook,
the per-frame diagnostics in episodes, the run-config checks and the noisy-tv analysis metrics. The wrapper and
vector-env tests render the real Cartpole (any MuJoCo backend: the default on Windows, EGL on Linux)."""
import copy
import random
from pathlib import Path

import numpy as np
import pytest
import torch
from omegaconf import OmegaConf

from data import Dataset, Episode
from data.episode import TV_KEYS, join_physical_states, stack_physical_states
from envs import DMControlEnv, make_dm_control_env
from envs.cartpole_noisy_tv import CartpoleNoisyTV, check_noisy_tv_config, in_region, panel_box
import exploration_analysis as ea
from exploration_trainer import check_noisy_tv_settings
from test_exploration_trainer import compose

INTERVAL = [0.3, 0.6]
TASKS = ["balance", "balance_sparse"]


def tv(enabled=True, size=16, seed=0, location="top_left", margin=0):
    return dict(enabled=enabled, size=size, position_interval=list(INTERVAL), panel_location=location, margin=margin,
                seed=seed)


def actions(n=160):
    """Fixed actions that push the cart right into the interval and back out again (balance start near 0)."""
    return [np.array([1.0 if (t // 35) % 2 == 0 else -0.4], dtype=np.float32) for t in range(n)]


def rollout(noisy_tv, reset_seed=5, steps=160, record=True):
    env = DMControlEnv("cartpole", TASKS, record_physical_state=record)
    env = env if noisy_tv is None else CartpoleNoisyTV(env, noisy_tv)
    obs, info = env.reset(seed=reset_seed)
    out = dict(obs=[obs], info=[info], rew=[], term=[], trunc=[], task=[], x=[float(info["qpos"][0])] if record else [])
    for a in actions(steps):
        obs, r, term, trunc, info = env.step(a)
        out["obs"].append(obs), out["info"].append(info), out["rew"].append(r)
        out["term"].append(term), out["trunc"].append(trunc), out["task"].append(info["task_rewards"].copy())
        if record:
            out["x"].append(float(info["qpos"][0]))
    out["env"] = env
    return out


# ---------------------------------------------------------------------------------------------- configuration

def test_config_validation():
    assert check_noisy_tv_config(None, "walker", 64) is None
    assert check_noisy_tv_config(tv(), "cartpole", 64)["position_interval"] == INTERVAL
    with pytest.raises(ValueError, match="cartpole only"):
        check_noisy_tv_config(tv(), "walker", 64)
    with pytest.raises(ValueError, match="cartpole only"):  # Clean logging needs the cart position too
        check_noisy_tv_config(tv(enabled=False), "hopper", 64)
    with pytest.raises(ValueError, match="does not fit"):
        check_noisy_tv_config(tv(size=60, margin=8), "cartpole", 64)
    with pytest.raises(ValueError, match="exactly the keys"):
        check_noisy_tv_config({**tv(), "extra": 1}, "cartpole", 64)
    with pytest.raises(ValueError, match="x_L < x_R"):
        check_noisy_tv_config({**tv(), "position_interval": [0.6, 0.3]}, "cartpole", 64)
    with pytest.raises(ValueError, match="panel_location"):
        check_noisy_tv_config(tv(location="center"), "cartpole", 64)
    with pytest.raises(ValueError, match="cartpole only"):
        make_dm_control_env(1, torch.device("cpu"), "walker", ["stand"], 64, 0, 2, None, noisy_tv=tv())


def test_panel_boxes_share_the_corner_and_margin():
    assert panel_box("top_left", 16, 0, 64) == (slice(0, 16), slice(0, 16))
    assert panel_box("top_left", 24, 0, 64) == (slice(0, 24), slice(0, 24))
    assert panel_box("top_right", 24, 2, 64) == (slice(2, 26), slice(38, 62))


@pytest.mark.parametrize("condition,enabled,size", [("clean", False, 16), ("tv16", True, 16), ("tv24", True, 24)])
def test_condition_configs_resolve(condition, enabled, size):
    cfg = compose(["+experiment=lcg_predictor_100k", f"+noisy_tv={condition}", "env.train.domain_name=cartpole",
                   "env.train.task_names=[balance,balance_sparse]", "common.seed=0"])
    OmegaConf.resolve(cfg)
    train, test = cfg.env.train.noisy_tv, cfg.env.test.noisy_tv
    assert (train.enabled, train.size, list(train.position_interval), train.panel_location, train.margin) == (
        enabled, size, INTERVAL, "top_left", 0)
    assert (test.enabled, test.size, test.seed, train.seed) == (enabled, size, 1000, 0)
    assert cfg.collection.train.record_physical_state and cfg.collection.test.record_physical_state
    checked = check_noisy_tv_settings(cfg)
    assert ea.condition_name(checked["train"]) == condition
    # The method's settings are untouched by the condition: identical to the preset without it.
    plain = compose(["+experiment=lcg_predictor_100k", "env.train.domain_name=cartpole",
                     "env.train.task_names=[balance,balance_sparse]", "common.seed=0"])
    for key in ("intrinsic_reward", "agent", "actor_critic", "denoiser", "world_model_env", "milestones"):
        assert OmegaConf.to_container(cfg[key], resolve=True) == OmegaConf.to_container(plain[key], resolve=True), key
    assert check_noisy_tv_settings(plain) is None


def test_run_config_checks():
    base = ["+experiment=uniform_100k", "+noisy_tv=tv16", "env.train.domain_name=cartpole",
            "env.train.task_names=[balance,balance_sparse]", "common.seed=0"]
    for extra, match in ((["env.test.noisy_tv.seed=0"], "must differ"),
                         (["env.test.noisy_tv.size=24"], "differs in"),
                         (["collection.test.record_physical_state=false"], "record_physical_state"),
                         (["~env.test.noisy_tv"], "given together")):
        cfg = compose(base + extra)
        with pytest.raises(ValueError, match=match):
            check_noisy_tv_settings(cfg)
    walker = compose(["+experiment=uniform_100k", "+noisy_tv=clean", "common.seed=0"])  # walker default domain
    with pytest.raises(ValueError, match="cartpole only"):
        check_noisy_tv_settings(walker)


# ---------------------------------------------------------------------------------------------- wrapper

@pytest.fixture(scope="module")
def runs():
    return {name: rollout(cfg) for name, cfg in (("plain", None), ("clean", tv(False)), ("tv16", tv(True, 16)),
                                                 ("tv24", tv(True, 24)))}


def test_fixed_actions_give_identical_physics_rewards_and_boundaries(runs):
    base = runs["plain"]
    for name in ("clean", "tv16", "tv24"):
        r = runs[name]
        assert r["x"] == base["x"] and r["rew"] == base["rew"] and r["term"] == base["term"]
        assert r["trunc"] == base["trunc"]
        assert all(np.array_equal(a, b) for a, b in zip(r["task"], base["task"]))
        for a, b in zip(r["info"], base["info"]):
            assert np.array_equal(a["qpos"], b["qpos"]) and np.array_equal(a["qvel"], b["qvel"])
        assert all(o.shape == (64, 64, 3) and o.dtype == np.uint8 for o in r["obs"])
    inside = [in_region(x, INTERVAL) for x in base["x"]]
    assert 0 < sum(inside) < len(inside) and not inside[0]  # the actions enter and leave the interval


def test_clean_reproduces_the_original_observations_and_records_membership(runs):
    base, clean = runs["plain"], runs["clean"]
    assert all(np.array_equal(a, b) for a, b in zip(clean["obs"], base["obs"]))
    assert [i["tv_in_region"] for i in clean["info"]] == [in_region(x, INTERVAL) for x in base["x"]]
    assert not any(i["tv_active"] for i in clean["info"]) and all(i["tv_panel_size"] == 0 for i in clean["info"])
    assert clean["env"].draws == 0


@pytest.mark.parametrize("name,size", [("tv16", 16), ("tv24", 24)])
def test_trigger_timing_panel_size_and_untouched_pixels(runs, name, size):
    base, r = runs["plain"], runs[name]
    rows, cols = panel_box("top_left", size, 0, 64)
    outside = np.ones((64, 64), bool)
    outside[rows, cols] = False
    for t, (o, b, info, x) in enumerate(zip(r["obs"], base["obs"], r["info"], base["x"])):
        active = in_region(x, INTERVAL)  # I_t from the cart position of the state o_t was rendered from
        assert info["tv_in_region"] == info["tv_active"] == active, t
        assert info["tv_panel_size"] == (size if active else 0)
        assert np.array_equal(o[outside], b[outside]), t  # pixels outside the panel never change
        if active:
            assert not np.array_equal(o[rows, cols], b[rows, cols])
        else:
            assert np.array_equal(o, b), t  # inactive: the original rendering
    assert r["env"].draws == sum(in_region(x, INTERVAL) for x in base["x"])
    # Leaving the interval restores the background immediately (no permanent activation).
    flags = [in_region(x, INTERVAL) for x in base["x"]]
    exit_t = next(t for t in range(1, len(flags)) if flags[t - 1] and not flags[t])
    assert np.array_equal(r["obs"][exit_t], base["obs"][exit_t])


def test_textures_are_fresh_each_step_and_uniform(runs):
    r = runs["tv24"]
    rows, cols = panel_box("top_left", 24, 0, 64)
    panels = [o[rows, cols] for o, i in zip(r["obs"], r["info"]) if i["tv_active"]]
    assert len(panels) >= 2
    assert all(not np.array_equal(a, b) for a, b in zip(panels, panels[1:]))
    values = np.concatenate([p.ravel() for p in panels])
    assert values.min() == 0 and values.max() == 255 and abs(values.mean() - 127.5) < 6


def test_reads_do_not_draw_and_global_rngs_are_untouched():
    env = CartpoleNoisyTV(DMControlEnv("cartpole", TASKS, record_physical_state=True), tv())
    env.reset(seed=5)
    for a in actions(40):
        env.step(a)
    states = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    rng, draws = copy.deepcopy(env.rng.bit_generator.state), env.draws
    for _ in range(3):
        _ = env.physical_state, env.collection_state, env.last_observation
    assert env.rng.bit_generator.state == rng and env.draws == draws
    for a in actions(120)[40:]:
        env.step(a)
    assert env.draws > draws  # the region was visited, so textures were drawn
    after = (random.getstate(), np.random.get_state(), torch.get_rng_state())
    assert after[0] == states[0] and np.array_equal(after[1][1], states[1][1]) and torch.equal(after[2], states[2])


def test_reset_seed_restarts_the_tv_stream_and_pairs_sizes():
    a, b = rollout(tv(True, 16, seed=1000)), rollout(tv(True, 16, seed=1000))
    assert all(np.array_equal(x, y) for x, y in zip(a["obs"], b["obs"]))  # same reset seed: same textures
    c = rollout(tv(True, 16, seed=1001))
    assert any(not np.array_equal(x, y) for x, y in zip(a["obs"], c["obs"]))  # another TV seed: other textures
    e16 = CartpoleNoisyTV(DMControlEnv("cartpole", TASKS), tv(True, 16, seed=1000))
    e24 = CartpoleNoisyTV(DMControlEnv("cartpole", TASKS), tv(True, 24, seed=1000))
    e16.reset(seed=3001), e24.reset(seed=3001)
    assert e16.rng.bit_generator.state == e24.rng.bit_generator.state  # TV16/TV24 evaluation streams are paired


def test_checkpoint_restore_continues_exactly():
    env = CartpoleNoisyTV(DMControlEnv("cartpole", TASKS, record_physical_state=True), tv(True, 24))
    obs, _ = env.reset(seed=5)
    acts = actions(160)
    for a in acts[:50]:
        obs, *_ = env.step(a)
    saved = copy.deepcopy(env.collection_state)
    assert np.array_equal(saved["noisy_tv"]["last_observation"], obs)
    tail = [env.step(a) for a in acts[50:]]
    restored = CartpoleNoisyTV(DMControlEnv("cartpole", TASKS, record_physical_state=True), tv(True, 24))
    restored.reset(seed=99)  # another episode and TV stream, then overwritten by the saved state
    restored.collection_state = saved
    assert np.array_equal(restored.last_observation, obs)  # the cached observation, not a new draw
    assert restored.physical_state["tv_active"] == saved["noisy_tv"]["last_tv_state"]["tv_active"]
    again = [restored.step(a) for a in acts[50:]]
    for (o1, r1, *_ , i1), (o2, r2, *_, i2) in zip(tail, again):
        assert np.array_equal(o1, o2) and r1 == r2
        assert all(i1[k] == i2[k] for k in ("tv_in_region", "tv_active", "tv_panel_size"))
    other = CartpoleNoisyTV(DMControlEnv("cartpole", TASKS, record_physical_state=True), tv(True, 16))
    with pytest.raises(ValueError, match="differ"):
        other.collection_state = saved
    plain = DMControlEnv("cartpole", TASKS)
    with pytest.raises(ValueError, match="noisy-TV state"):
        CartpoleNoisyTV(plain, tv()).collection_state = {"env": plain.collection_state}


# ---------------------------------------------------------------------------------------------- vector env, replay

class RecordingPolicy(torch.nn.Module):
    """Stands in for the exploration policy: records the input of every action it samples (not the bootstrap-value
    calls) and returns fixed actions."""
    frame_stack = 4

    def __init__(self):
        super().__init__()
        self.inputs, self.t, self.pending = [], 0, None

    def predict_act_value(self, obs):
        self.pending = obs.clone()
        return obs.new_zeros(obs.size(0), 1), obs.new_zeros(obs.size(0))

    def sample_action(self, logits):
        if self.pending is not None:
            self.inputs.append(self.pending)
        a = actions(1000)[self.t]
        self.t += 1
        return torch.tensor(a).reshape(1, 1)


def test_policy_inputs_replay_restoration_and_videos_use_the_emitted_observations(tmp_path):
    from coroutines.collector import CollectorState, make_collector, NumToCollect
    env = make_dm_control_env(1, torch.device("cpu"), "cartpole", TASKS, 64, 0, 2, None, True, noisy_tv=tv(True, 24))
    try:
        policy, state = RecordingPolicy(), CollectorState()
        dataset = Dataset(tmp_path / "train", "train", cache_in_ram=True)
        collector = make_collector(env, policy, dataset, 0.0, verbose=False, state=state)
        collector.send(NumToCollect(steps=120))
        saved_envs, saved_obs = env.get_env_states(), state.loop.obs.clone()
        episode = dataset.load_episode(0)
        ps = episode.physical_state
        assert set(TV_KEYS) <= set(ps) and bool(ps["tv_active"].any()) and not bool(ps["tv_active"].all())
        # Policy input: the last frame of each frame stack is the current emitted observation (replay row t).
        last = torch.cat([x[:, -3:] for x in policy.inputs])  # (T, 3, 64, 64)
        assert torch.equal(last[:len(episode)], episode.obs.to(last))
        # The replay file holds the same frames (uint8 round trip) and the diagnostics survive save/load.
        reloaded = Dataset(tmp_path / "train", "reload", save_on_disk=False)
        dataset.save_to_default_path()
        reloaded.load_from_default_path()
        ep2 = reloaded.load_episode(0)
        assert torch.equal(ep2.obs, episode.obs) and all(torch.equal(ep2.physical_state[k], ps[k]) for k in TV_KEYS)
        # Active rows differ from inactive rendering only inside the panel: the panel region has no constant color.
        rows, cols = panel_box("top_left", 24, 0, 64)
        active = ps["tv_active"]
        panel_std = episode.obs[:, :, rows, cols].flatten(1).std(1)
        assert bool((panel_std[active] > 0.3).all()) and bool((panel_std[~active] < 0.3).all())
        # Restoring the saved env states continues the episode with the same frames as an uninterrupted env.
        collector.send(NumToCollect(steps=40))
        uninterrupted = torch.cat([x[:, -3:] for x in policy.inputs[-40:]])
    finally:
        env.close()
    env2 = make_dm_control_env(1, torch.device("cpu"), "cartpole", TASKS, 64, 0, 2, None, True, noisy_tv=tv(True, 24))
    try:
        env2.reset(seed=[123])
        env2.set_env_states(saved_envs)  # restored physics, TV stream and cached observation
        policy2 = RecordingPolicy()
        policy2.t = 120  # the actions the uninterrupted collector takes next: o_120 -> o_121, ...
        frames = []
        for _ in range(40):
            obs, *_ = env2.step(policy2.sample_action(None))
            frames.append(obs)
    finally:
        env2.close()
    assert torch.equal(uninterrupted[0:1], saved_obs)  # the next policy input is the saved current observation
    assert torch.equal(torch.cat(frames)[:-1], uninterrupted[1:])
    # Videos and PNG strips are written from the replay frames; the PNG strip is lossless.
    rows_ = ea.noisy_tv_condition_rows("tv24", "uniform", [ep2], [5])[0]
    written = ea.export_noisy_tv_videos({"tv24": [ep2]}, {"tv24": rows_}, tmp_path / "videos", scale=1)
    import cv2
    strip = cv2.imread([w for w in written if w.endswith(".png")][0])[..., ::-1]
    frames_u8 = ep2.obs.add(1).div(2).mul(255).round().byte().permute(0, 2, 3, 1).numpy()
    start = int(np.argmax(ep2.physical_state["tv_in_region"].numpy()))
    picks = list(range(max(0, start - 3), min(len(frames_u8), start + 5)))
    assert np.array_equal(strip, np.concatenate([frames_u8[t] for t in picks], axis=1))
    video = cv2.VideoCapture([w for w in written if w.endswith(".mp4")][0])
    assert int(video.get(cv2.CAP_PROP_FRAME_COUNT)) == len(ep2)


# ---------------------------------------------------------------------------------------------- episode format

def state(x, inside, active, nq=2):
    return dict(qpos=torch.tensor([x, 0.0], dtype=torch.float64), qvel=torch.zeros(nq, dtype=torch.float64),
                tv_in_region=torch.tensor(inside), tv_active=torch.tensor(active),
                tv_panel_size=torch.tensor(16 if active else 0, dtype=torch.int16))


def test_episode_tv_diagnostics_stack_join_and_validate():
    s = stack_physical_states([state(0.0, False, False), None, state(0.4, True, True)], final=state(0.5, True, True))
    assert s["tv_in_region"].tolist() == [False, False, True] and s["tv_panel_size"].tolist() == [0, 0, 16]
    assert s["recorded"].tolist() == [True, False, True] and bool(s["final_tv_active"])
    first = stack_physical_states([state(0.0, False, False)])
    joined = join_physical_states(first, 1, stack_physical_states([state(0.4, True, True)]), 1)
    assert joined["tv_active"].tolist() == [False, True]
    with pytest.raises(ValueError, match="mix"):
        stack_physical_states([state(0.0, False, False), dict(qpos=torch.zeros(2, dtype=torch.float64),
                                                                qvel=torch.zeros(2, dtype=torch.float64))])
    bad = dict(s)
    bad["tv_active"] = torch.tensor([True, False, True])  # active outside the region
    from data.episode import check_physical_state
    with pytest.raises(ValueError, match="outside the trigger region"):
        check_physical_state(bad, 3)


# ---------------------------------------------------------------------------------------------- metrics

def test_metrics_on_a_known_trajectory():
    inside = np.array([1, 1, 0, 0, 1, 1, 1, 0, 1, 0], bool)  # starts inside; visits of 2, 3 and 1
    x = np.where(inside, 0.45, -0.5)
    qpos = np.stack([x, np.zeros(10)], 1)
    qvel = np.stack([np.arange(10.0) - 4.5, np.full(10, 2.0)], 1)
    m = ea.noisy_tv_episode_metrics(qpos, qvel, inside, inside)
    assert (m["occupancy"], m["initial_in_region"], m["region_entries"], m["visits"], m["reached"]) == (0.6, 1, 2, 3, 1)
    assert (m["mean_dwell"], m["max_dwell"]) == (2.0, 3) and m["tv_active_fraction"] == 0.6
    assert m["position_angle_bins_visited"] == 2 and m["cart_velocity_max_abs"] == 4.5
    assert m["pole_angular_velocity_mean_abs"] == 2.0 and m["pole_angular_velocity_std"] == 0.0
    none = ea.noisy_tv_episode_metrics(qpos, qvel, np.zeros(10, bool), np.zeros(10, bool))
    assert (none["reached"], none["region_entries"], none["mean_dwell"], none["max_dwell"]) == (0, 0, "", "")
    with pytest.raises(ValueError, match="outside the trigger region"):
        ea.noisy_tv_episode_metrics(qpos, qvel, np.zeros(10, bool), inside)


def test_coverage_bins_handle_range_and_angle_periodicity():
    pos, ang = ea.coverage_bins(np.array([-2.0, -1.8, 0.05, 1.8, 2.1]), np.array([0.0, np.pi, -np.pi, 3 * np.pi, -0.01]))
    assert pos.tolist() == [0, 1, 10, 18, 19]
    assert ang[1] == ang[2] == ang[3] == 0  # pi, -pi and 3 pi are one angle
    assert ang[0] == 12 and ang[4] == 11
    assert ea.NUM_COVERAGE_BINS == 20 * 24


def test_paired_differences():
    row = lambda c, s, occ, dwell, cov: dict(condition=c, method="m", reset_seed=s, occupancy=occ, max_dwell=dwell,
                                            position_angle_coverage=cov)
    rows = {"clean": [row("clean", 1, 0.1, 5, 0.2), row("clean", 2, 0.0, "", 0.3)],
            "tv16": [row("tv16", 1, 0.4, 10, 0.1), row("tv16", 2, 0.2, 4, 0.3)]}
    paired, summary = ea.noisy_tv_paired(rows)
    assert [p["delta_occupancy"] for p in paired] == pytest.approx([0.3, 0.2])
    assert [p["delta_max_dwell"] for p in paired] == [5, 4]
    assert summary[0]["delta_occupancy_mean"] == pytest.approx(0.25) and summary[0]["num_pairs"] == 2
    with pytest.raises(ValueError, match="no Clean episode"):
        ea.noisy_tv_paired({"clean": rows["clean"][:1], "tv16": rows["tv16"]})


def test_cli_rejects_mixed_options(tmp_path):
    with pytest.raises(SystemExit):
        ea.main(["--test", "noisy-tv", "--milestone", str(tmp_path), "--output", str(tmp_path)])
    with pytest.raises(SystemExit):
        ea.main(["--test", "task-reward-coverage", "--milestone", str(tmp_path), "--condition", f"clean={tmp_path}",
                 "--output", str(tmp_path)])
    with pytest.raises(ValueError, match="--condition takes"):
        ea.noisy_tv_analysis({"tv32": tmp_path}, tmp_path / "out")


# ---------------------------------------------------------------------------------------------- downstream guard

@pytest.mark.parametrize("enabled", [True, False])
def test_downstream_refuses_an_enabled_noisy_tv_milestone(tmp_path, monkeypatch, enabled):
    """Downstream evaluation builds a plain environment, so a milestone explored with an enabled TV is refused;
    Clean (enabled false) emits the original rendering and is accepted."""
    import json
    sys_path = str(Path(__file__).resolve().parents[1] / "downstream")
    monkeypatch.syspath_prepend(sys_path)
    from test_downstream_trainer import build_milestone, make_trainer
    milestone = build_milestone(tmp_path)
    path = Path(milestone) / "manifest.json"
    manifest = json.loads(path.read_text())
    manifest["model"]["env"]["noisy_tv"] = tv(enabled=enabled)
    path.write_text(json.dumps(manifest, indent=2) + "\n")
    if enabled:
        with pytest.raises(ValueError, match="enabled noisy TV"):
            make_trainer(milestone, monkeypatch)
    else:
        make_trainer(milestone, monkeypatch)  # builds (nothing trained here)
