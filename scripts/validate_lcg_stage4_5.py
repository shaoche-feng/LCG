#! /usr/bin/env python
"""
LCG Stage 4.5: exact batched per-candidate VJP acceleration -- two-phase benchmark.

Rewritten after the first (single-process) version hung for ~2 hours on an 8GB RTX 5060
during the exhaustive 480-candidate x 6-chunk-size sweep, with no diagnosable partial
output (Python's stdout was block-buffered when redirected to a file, so nothing had been
flushed to disk despite real GPU activity). Two structural fixes:

  1. Every chunk-size test now runs in an isolated subprocess (`--worker`) launched via
     subprocess.Popen(...).communicate(timeout=...); a config that hangs or thrashes gets
     killed after a bounded timeout instead of blocking the whole benchmark indefinitely.
     After each worker exits (normally, by timeout, or by OOM), the driver polls
     `nvidia-smi` until GPU memory returns near its pre-worker baseline before starting the
     next configuration.
  2. The expensive one-time setup (denoiser, h_D, 480 candidates, CRN banks) is built once
     by the driver and cached to disk as plain CPU tensors (no live CUDA objects are ever
     pickled). Each worker subprocess reloads the denoiser from its existing checkpoint
     file and moves only its own CPU-loaded tensors to CUDA, so killing a worker reliably
     tears down all of its CUDA state.

Phase A (this run): 32 candidates, chunk_size in {1,2,4,8,16,32}, one run each, compared
against a sequential reference computed on the same 32 candidates. Phase B (480 candidates,
winning chunk size only) is intentionally not run in this invocation -- per instruction,
Phase A's results are reviewed first.

Does not change the LCG mathematics, CRN semantics, K=2, num_strata=3, h_D, or
lcg.batched_vjp's accelerated implementation -- this stage only benchmarks it.

Usage:
    python scripts/validate_lcg_stage4_5.py             # driver, Phase A
    python scripts/validate_lcg_stage4_5.py --worker ... # internal, invoked by the driver
"""
import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import numpy as np
import torch
from torch.utils.data import DataLoader

SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parent.parent
SCRATCH_BASE = Path(
    r"C:\Users\jerry\AppData\Local\Temp\claude\c--Users-jerry-Project-LCG"
    r"\974bd621-91fa-4766-822e-267b547a92c5\scratchpad"
)
DIAG_CHECKPOINT_PATH = SCRATCH_BASE / "lcg_diag_denoiser.pt"
TRAIN_DATASET_DIR = SCRATCH_BASE / "lcg_diag_train_dataset"
HELDOUT_DATASET_DIR = SCRATCH_BASE / "lcg_stage4_5_heldout_dataset"
CACHE_PATH = SCRATCH_BASE / "lcg_stage4_5_cache.pt"

DOMAIN_NAME, TASK_NAME = "cheetah", "run"
ENV_KWARGS = dict(size=64, camera_id=0, action_repeat=2, time_limit=1.0)
NUM_STEPS_CONDITIONING = 4
H_D_B = 40
DAMPING = 1e-4
BETA = 1.0
NUM_CRN_BANKS = 2
B_PROFILE, H_PROFILE = 32, 15
CHUNK_SIZES = [1, 2, 4, 8, 16, 32]
PHASE_A_TIMEOUT = 60  # seconds, per worker
PHASE_B_TIMEOUT = 180  # seconds
BOTTLENECK_TIMEOUT = 60  # seconds
STAGE4_BASELINE_TIME_S = 64.75  # Stage 4's sequential 480-candidate result, for reference
STAGE4_BASELINE_MS_PER_CAND = 134.9


# --------------------------------------------------------------------------------------
# Driver-only imports (workers re-import lazily inside run_worker_main to keep --worker
# startup minimal, but importing here too is harmless since the driver needs them for
# prepare_cache anyway).
# --------------------------------------------------------------------------------------


def _driver_imports():
    global Dataset, Episode, BatchSampler, collate_segments_to_batch
    global ImaginedCandidate, WorldModelEnv, WorldModelEnvConfig, make_dm_control_env
    global imagined_candidates_from_batch, score_candidates_with_banks, make_crn_bank_set
    global historical_precision, selected_parameters, CRNBank
    global Denoiser, DenoiserConfig, DiffusionSamplerConfig, SigmaDistributionConfig, InnerModelConfig
    global RewEndModel, RewEndModelConfig

    from data import BatchSampler, Dataset, Episode, collate_segments_to_batch
    from envs import ImaginedCandidate, WorldModelEnv, WorldModelEnvConfig
    from envs.dm_control_env import make_dm_control_env
    from lcg.batched_scoring import imagined_candidates_from_batch, score_candidates_with_banks
    from lcg.crn import CRNBank, make_crn_bank_set
    from lcg.precision import historical_precision
    from lcg.theta_s import selected_parameters
    from models.diffusion import Denoiser, DenoiserConfig, DiffusionSamplerConfig, SigmaDistributionConfig
    from models.diffusion.inner_model import InnerModelConfig
    from models.rew_end_model import RewEndModel, RewEndModelConfig


SIGMA_CFG = None  # set in _driver_imports's caller (needs SigmaDistributionConfig class)


def collect_episode(env, rng: np.random.Generator):
    obs_frames, act_list, rew_list, end_list, trunc_list = [], [], [], [], []
    raw_obs, _ = env.reset(seed=int(rng.integers(0, 2**31 - 1)))
    done = False
    while not done:
        action = rng.uniform(env.action_low, env.action_high).astype(np.float32)
        next_raw_obs, rew, terminated, truncated, _ = env.step(action)
        obs_frames.append(raw_obs)
        act_list.append(action)
        rew_list.append(rew)
        end_list.append(int(terminated))
        trunc_list.append(int(truncated))
        raw_obs = next_raw_obs
        done = terminated or truncated
    obs = torch.from_numpy(np.stack(obs_frames)).float().div(255).mul(2).sub(1).permute(0, 3, 1, 2).contiguous()
    act = torch.from_numpy(np.stack(act_list)).float()
    rew = torch.tensor(rew_list, dtype=torch.float32)
    end = torch.tensor(end_list, dtype=torch.uint8)
    trunc = torch.tensor(trunc_list, dtype=torch.uint8)
    return Episode(obs=obs, act=act, rew=rew, end=end, trunc=trunc, info={})


def build_dataset(directory, name, target_num_steps, seed, include_stray):
    if directory.exists():
        shutil.rmtree(directory)
    dataset = Dataset(directory, name, cache_in_ram=True)
    if include_stray:
        stray_path = REPO_ROOT / "outputs" / "2026-08-17" / "15-35-46" / "dataset" / "train" / "000" / "00" / "0" / "0.pt"
        if stray_path.is_file():
            dataset.add_episode(Episode.load(stray_path))
    env = make_dm_control_env(domain_name=DOMAIN_NAME, task_name=TASK_NAME, **ENV_KWARGS)
    rng = np.random.default_rng(seed)
    while dataset.num_steps < target_num_steps:
        dataset.add_episode(collect_episode(env, rng))
    dataset.save_to_default_path()
    return dataset


def load_diagnostic_denoiser(device):
    ckpt = torch.load(DIAG_CHECKPOINT_PATH, map_location=device, weights_only=False)
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=ckpt["action_dim"],
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    return denoiser, ckpt["action_dim"]


def build_rew_end_model(device, action_dim):
    cfg = RewEndModelConfig(
        lstm_dim=512, img_channels=3, img_size=64, cond_channels=128,
        depths=[2, 2, 2, 2], channels=[32, 32, 32, 32], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim, continuous_reward=True,
    )
    model = RewEndModel(cfg).to(device)
    model.eval()
    return model


def build_world_model_env(denoiser, rew_end_model, dataset, num_envs, horizon, sampler_cfg):
    cfg = WorldModelEnvConfig(horizon=horizon, num_batches_to_preload=4, diffusion_sampler=sampler_cfg)
    bs = BatchSampler(dataset, 0, 1, num_envs, NUM_STEPS_CONDITIONING, sample_weights=None)
    loader = DataLoader(dataset=dataset, batch_sampler=bs, collate_fn=collate_segments_to_batch, num_workers=0)
    return WorldModelEnv(denoiser, rew_end_model, loader, cfg, return_imagined_candidate=True)


def collect_profile_candidates(denoiser, rew_end_model, dataset, num_envs, horizon, action_dim, device, seed, sampler_cfg):
    env = build_world_model_env(denoiser, rew_end_model, dataset, num_envs, horizon, sampler_cfg)
    torch.manual_seed(seed)
    np.random.seed(seed)
    env.reset()
    tuples = []
    for _ in range(horizon):
        act = torch.rand(num_envs, action_dim, device=device) * 2 - 1
        _, _, _, _, info = env.step(act)
        batch = info["imagined_candidate"]
        tuples.extend(imagined_candidates_from_batch(batch.x_obs, batch.x_act, batch.y_star))
    return tuples


def pearson_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    a, b = a - a.mean(), b - b.mean()
    return ((a * b).sum() / (a.norm() * b.norm() + 1e-12)).item()


def spearman_corr(a, b):
    a, b = torch.as_tensor(a, dtype=torch.float64), torch.as_tensor(b, dtype=torch.float64)
    ra = torch.argsort(torch.argsort(a)).double()
    rb = torch.argsort(torch.argsort(b)).double()
    return pearson_corr(ra, rb)


def cuda_sync(device):
    if device.type == "cuda":
        torch.cuda.synchronize(device)


# --------------------------------------------------------------------------------------
# Driver: one-time cache preparation (CPU-only artifacts on disk)
# --------------------------------------------------------------------------------------


def prepare_cache(device):
    _driver_imports()
    global SIGMA_CFG
    SIGMA_CFG = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
    sampler_cfg = DiffusionSamplerConfig(num_steps_denoising=3, sigma_min=2e-3, sigma_max=5.0, rho=7, order=1)

    if CACHE_PATH.exists():
        print(f"Using existing cache at {CACHE_PATH}", flush=True)
        return

    print("Building Stage 4.5 benchmark cache (one-time)...", flush=True)
    denoiser, action_dim = load_diagnostic_denoiser(device)
    rew_end_model = build_rew_end_model(device, action_dim)
    params = selected_parameters(denoiser)
    d_S = sum(p.numel() for p in params)

    train_dataset = build_dataset(TRAIN_DATASET_DIR, "lcg_stage4_5_train_for_hD", 1500, seed=0, include_stray=True)
    heldout_dataset = build_dataset(HELDOUT_DATASET_DIR, "lcg_stage4_5_heldout", 700, seed=555, include_stray=False)

    h_D = historical_precision(
        denoiser, params, train_dataset, SIGMA_CFG, B=H_D_B, num_strata=3,
        beta=BETA, damping=DAMPING, seed=0,
    )
    assert h_D.shape == (d_S,) and torch.all(h_D > 0)
    print(f"  h_D: shape={tuple(h_D.shape)}, min={h_D.min().item():.4g}, max={h_D.max().item():.4g}", flush=True)

    candidates = collect_profile_candidates(
        denoiser, rew_end_model, heldout_dataset, B_PROFILE, H_PROFILE, action_dim, device, seed=999, sampler_cfg=sampler_cfg
    )
    print(f"  collected {len(candidates)} candidates (B={B_PROFILE} x H={H_PROFILE})", flush=True)

    y_shape = candidates[0][2].shape
    banks = make_crn_bank_set(SIGMA_CFG, y_shape, device, num_crn_banks=NUM_CRN_BANKS, num_strata=3, seed=7000)

    print("  computing 32-candidate sequential reference (Stage-4 scorer, known-safe/fast)...", flush=True)
    cuda_sync(device)
    t0 = time.perf_counter()
    reference_32 = score_candidates_with_banks(denoiser, params, h_D, banks, candidates[:32])
    cuda_sync(device)
    t_ref_32 = time.perf_counter() - t0
    print(f"  reference_32: {t_ref_32:.2f}s ({t_ref_32 / 32 * 1000:.2f} ms/candidate)", flush=True)

    cache = {
        "action_dim": action_dim,
        "h_D": h_D.detach().cpu(),
        "obs": torch.cat([c[0] for c in candidates], dim=0).detach().cpu(),
        "act": torch.cat([c[1] for c in candidates], dim=0).detach().cpu(),
        "y": torch.cat([c[2] for c in candidates], dim=0).detach().cpu(),
        "bank_sigmas": [[s.detach().cpu() for s in b.sigmas] for b in banks],
        "bank_epsilons": [[e.detach().cpu() for e in b.epsilons] for b in banks],
        "bank_xis": [[x.detach().cpu() for x in b.xis] for b in banks],
        "reference_32": reference_32.detach().cpu(),
        "t_ref_32": t_ref_32,
    }
    torch.save(cache, CACHE_PATH)
    print(f"  cache saved to {CACHE_PATH} (CPU tensors only)", flush=True)

    del denoiser, rew_end_model, params, h_D, candidates, banks
    if device.type == "cuda":
        torch.cuda.empty_cache()
    print("  driver released its CUDA context/memory.", flush=True)


# --------------------------------------------------------------------------------------
# Worker: single chunk-size test, isolated subprocess
# --------------------------------------------------------------------------------------


def timed_score_chunk(denoiser, params, h_D, banks, chunk, device, timers):
    """Same math as lcg.batched_vjp.compute_vjp_batched/score_candidates_batched, just with
    forward / batched-backward / v^2-h_D-reduction timed separately for the bottleneck
    breakdown. Not a different implementation."""
    from lcg.gauss_newton import differentiable_denoise

    obs_batch = torch.cat([c[0] for c in chunk], dim=0)
    act_batch = torch.cat([c[1] for c in chunk], dim=0)
    y_batch = torch.cat([c[2] for c in chunk], dim=0)
    B = len(chunk)
    num_strata = banks[0].num_strata
    num_banks = len(banks)
    score_accum = torch.zeros(B, device=device)
    for bank in banks:
        for sigma, eps, xi in zip(bank.sigmas, bank.epsilons, bank.xis):
            y_sigma_batch = (y_batch + sigma.view(-1, 1, 1, 1) * eps).detach()

            cuda_sync(device)
            t0 = time.perf_counter()
            d_theta, w = differentiable_denoise(denoiser, y_sigma_batch, sigma, obs_batch, act_batch)
            cuda_sync(device)
            timers["forward"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            per_example = (torch.sqrt(2 * w) * xi * d_theta).reshape(B, -1).sum(dim=1)
            grad_outputs = torch.eye(B, device=per_example.device, dtype=per_example.dtype)
            grads = torch.autograd.grad(
                per_example, params, grad_outputs=grad_outputs, is_grads_batched=True,
                retain_graph=False, create_graph=False,
            )
            v_batch = torch.cat([g.reshape(B, -1) for g in grads], dim=1)
            cuda_sync(device)
            timers["backward"] += time.perf_counter() - t0

            t0 = time.perf_counter()
            contribution = (v_batch.square() / h_D.unsqueeze(0)).sum(dim=1)
            score_accum = score_accum + contribution / (num_strata * num_banks)
            cuda_sync(device)
            timers["reduce"] += time.perf_counter() - t0
            del v_batch, contribution
    return score_accum.detach().cpu()


def run_worker_main(args):
    from lcg.batched_vjp import score_candidates_batched
    from lcg.crn import CRNBank
    from lcg.theta_s import selected_parameters

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    try:
        cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    except Exception as e:
        print("RESULT " + json.dumps({"status": "error", "message": f"cache load failed: {e}"}), flush=True)
        return

    try:
        # Reload the denoiser fresh in THIS process (module-level imports needed here
        # since run_worker_main is invoked directly by __main__, before _driver_imports).
        ckpt = torch.load(DIAG_CHECKPOINT_PATH, map_location=device, weights_only=False)
        from models.diffusion import Denoiser, DenoiserConfig
        from models.diffusion.inner_model import InnerModelConfig

        inner_cfg = InnerModelConfig(
            img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
            depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
            continuous_action_dim=ckpt["action_dim"],
        )
        cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
        denoiser = Denoiser(cfg).to(device)
        denoiser.load_state_dict(ckpt["denoiser"])
        denoiser.eval()

        params = selected_parameters(denoiser)
        h_D = cache["h_D"].to(device)
        obs, act, y = cache["obs"].to(device), cache["act"].to(device), cache["y"].to(device)
        n = min(args.num_candidates, obs.size(0))
        candidates = [(obs[i : i + 1], act[i : i + 1], y[i : i + 1]) for i in range(n)]
        banks = tuple(
            CRNBank(
                tuple(s.to(device) for s in cache["bank_sigmas"][b]),
                tuple(e.to(device) for e in cache["bank_epsilons"][b]),
                tuple(x.to(device) for x in cache["bank_xis"][b]),
            )
            for b in range(len(cache["bank_sigmas"]))
        )

        if device.type == "cuda":
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)

        if args.bottleneck:
            timers = {"forward": 0.0, "backward": 0.0, "reduce": 0.0}
            for start in range(0, len(candidates), args.chunk_size):
                chunk = candidates[start : start + args.chunk_size]
                timed_score_chunk(denoiser, params, h_D, banks, chunk, device, timers)
            print("RESULT " + json.dumps({
                "status": "ok", "mode": "bottleneck", "chunk_size": args.chunk_size,
                "num_candidates": n, **timers,
            }), flush=True)
            return

        cuda_sync(device)
        t0 = time.perf_counter()
        scores = score_candidates_batched(denoiser, params, h_D, banks, candidates, args.chunk_size)
        cuda_sync(device)
        dt = time.perf_counter() - t0
        peak_mem = torch.cuda.max_memory_allocated(device) / 1e6 if device.type == "cuda" else None

        result = {
            "status": "ok", "chunk_size": args.chunk_size, "num_candidates": n,
            "time": dt, "peak_mem_mb": peak_mem, "scores": scores.tolist(),
        }

        if args.check_determinism:
            repeat = score_candidates_batched(denoiser, params, h_D, banks, candidates, args.chunk_size)
            result["deterministic"] = bool(torch.equal(scores, repeat))

        print("RESULT " + json.dumps(result), flush=True)

    except RuntimeError as e:
        status = "oom" if "out of memory" in str(e).lower() else "error"
        print("RESULT " + json.dumps({"status": status, "chunk_size": args.chunk_size, "message": str(e)[:300]}), flush=True)
    except Exception as e:
        print("RESULT " + json.dumps({
            "status": "error", "chunk_size": args.chunk_size, "message": f"{type(e).__name__}: {e}"[:300]
        }), flush=True)


# --------------------------------------------------------------------------------------
# Driver: subprocess launch + GPU-release verification
# --------------------------------------------------------------------------------------


def get_gpu_used_mb():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=10,
        )
        return float(out.stdout.strip().splitlines()[0])
    except Exception:
        return None


def wait_for_gpu_release(baseline_mb, extra_mb=500, timeout=20, poll=0.5):
    if baseline_mb is None:
        time.sleep(1.0)
        return None
    deadline = time.time() + timeout
    last = baseline_mb
    while time.time() < deadline:
        used = get_gpu_used_mb()
        last = used
        if used is None or used <= baseline_mb + extra_mb:
            return used
        time.sleep(poll)
    return last


def launch_worker(extra_args, timeout):
    cmd = [sys.executable, "-u", str(SCRIPT_PATH), "--worker"] + extra_args
    env = dict(os.environ)
    env["PYTHONUNBUFFERED"] = "1"
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, env=env)
    timed_out = False
    try:
        stdout, _ = proc.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        try:
            stdout, _ = proc.communicate(timeout=10)
        except Exception:
            stdout = ""
        timed_out = True
    proc.wait()
    return timed_out, proc.returncode, stdout


def parse_result(stdout):
    for line in (stdout or "").splitlines():
        if line.startswith("RESULT "):
            try:
                return json.loads(line[len("RESULT "):])
            except json.JSONDecodeError:
                return None
    return None


# --------------------------------------------------------------------------------------
# Phase A
# --------------------------------------------------------------------------------------


def phase_a(gpu_baseline):
    print("\n" + "=" * 88)
    print("PHASE A: fast screening -- 32 candidates, chunk_size in {1,2,4,8,16,32}, one run each")
    print("=" * 88, flush=True)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    reference_32 = cache["reference_32"]
    t_ref_32 = cache["t_ref_32"]
    print(f"  sequential reference (32 candidates, precomputed during cache build): "
          f"{t_ref_32:.2f}s ({t_ref_32 / 32 * 1000:.2f} ms/candidate)", flush=True)

    results = {}
    for chunk_size in CHUNK_SIZES:
        print(f"\n  >>> starting chunk_size={chunk_size} (worker timeout={PHASE_A_TIMEOUT}s)...", flush=True)
        t_wall0 = time.perf_counter()
        timed_out, rc, stdout = launch_worker(["--chunk-size", str(chunk_size), "--num-candidates", "32"], PHASE_A_TIMEOUT)
        wall = time.perf_counter() - t_wall0
        after_mem = wait_for_gpu_release(gpu_baseline)

        if timed_out:
            print(f"  <<< chunk_size={chunk_size}: TIMED OUT after {wall:.1f}s -- worker killed. "
                  f"GPU memory after cleanup: {after_mem} MiB (baseline {gpu_baseline} MiB).", flush=True)
            results[chunk_size] = {"status": "timeout"}
            continue

        result = parse_result(stdout)
        if result is None or result.get("status") != "ok":
            reason = (result or {}).get("message") or (stdout[-300:] if stdout else f"exit code {rc}, no RESULT line")
            status = (result or {}).get("status", "unknown")
            print(f"  <<< chunk_size={chunk_size}: {status.upper()}: {reason}. "
                  f"GPU memory after cleanup: {after_mem} MiB.", flush=True)
            results[chunk_size] = result or {"status": "failed", "message": reason}
            continue

        scores = torch.tensor(result["scores"])
        abs_err = (scores - reference_32).abs()
        rel_err = abs_err / reference_32.abs().clamp_min(1e-12)
        pear = pearson_corr(scores, reference_32)
        spear = spearman_corr(scores, reference_32)
        speedup = t_ref_32 / result["time"]
        results[chunk_size] = dict(
            status="ok", time=result["time"], ms_per_candidate=result["time"] / 32 * 1000,
            speedup=speedup, peak_mem=result["peak_mem_mb"],
            max_abs_err=abs_err.max().item(), max_rel_err=rel_err.max().item(),
            pearson=pear, spearman=spear,
        )
        r = results[chunk_size]
        print(f"  <<< chunk_size={chunk_size}: {r['time']:.3f}s  {r['ms_per_candidate']:.2f} ms/cand  "
              f"speedup={r['speedup']:.2f}x  peak_mem={r['peak_mem']:.1f}MB  "
              f"max_rel_err={r['max_rel_err']:.2e}  Pearson={r['pearson']:.6f}  Spearman={r['spearman']:.6f}  "
              f"(GPU after cleanup: {after_mem} MiB, baseline {gpu_baseline} MiB)", flush=True)

    print("\n" + "=" * 88)
    print("PHASE A SUMMARY")
    print("=" * 88)
    for chunk_size in CHUNK_SIZES:
        r = results[chunk_size]
        if r.get("status") != "ok":
            print(f"  chunk_size={chunk_size:<3}: {r.get('status', '?').upper()}")
        else:
            print(f"  chunk_size={chunk_size:<3}: {r['time']:>7.3f}s  {r['ms_per_candidate']:>7.2f} ms/cand  "
                  f"{r['speedup']:>6.2f}x speedup  peak_mem={r['peak_mem']:>6.1f}MB  max_rel_err={r['max_rel_err']:.2e}")
    return results


# --------------------------------------------------------------------------------------
# Phase B: one full-scale run at the winning chunk size
# --------------------------------------------------------------------------------------


def phase_b(gpu_baseline, chunk_size):
    print("\n" + "=" * 88)
    print(f"PHASE B: full-scale run -- 480 candidates, chunk_size={chunk_size} (winner from Phase A)")
    print("=" * 88, flush=True)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=False)
    reference_32 = cache["reference_32"]
    num_candidates = cache["obs"].size(0)

    print(f"  >>> starting chunk_size={chunk_size} on {num_candidates} candidates "
          f"(worker timeout={PHASE_B_TIMEOUT}s)...", flush=True)
    t_wall0 = time.perf_counter()
    timed_out, rc, stdout = launch_worker(
        ["--chunk-size", str(chunk_size), "--num-candidates", str(num_candidates)], PHASE_B_TIMEOUT
    )
    wall = time.perf_counter() - t_wall0
    after_mem = wait_for_gpu_release(gpu_baseline)

    if timed_out:
        print(f"  <<< TIMED OUT after {wall:.1f}s -- worker killed. "
              f"GPU memory after cleanup: {after_mem} MiB (baseline {gpu_baseline} MiB).", flush=True)
        return None

    result = parse_result(stdout)
    if result is None or result.get("status") != "ok":
        reason = (result or {}).get("message") or (stdout[-300:] if stdout else f"exit code {rc}, no RESULT line")
        status = (result or {}).get("status", "unknown")
        print(f"  <<< {status.upper()}: {reason}. GPU memory after cleanup: {after_mem} MiB.", flush=True)
        return None

    scores = torch.tensor(result["scores"])
    dt = result["time"]
    ms_per_candidate = dt / num_candidates * 1000
    cand_per_sec = num_candidates / dt
    peak_mem = result["peak_mem_mb"]
    speedup = STAGE4_BASELINE_TIME_S / dt

    # correctness where available: candidates[:32] are the same 32 used for reference_32
    abs_err_32 = (scores[:32] - reference_32).abs()
    rel_err_32 = abs_err_32 / reference_32.abs().clamp_min(1e-12)
    pear_32 = pearson_corr(scores[:32], reference_32)
    spear_32 = spearman_corr(scores[:32], reference_32)

    print(f"  <<< done in {wall:.1f}s wall (worker-reported: {dt:.2f}s). "
          f"GPU memory after cleanup: {after_mem} MiB (baseline {gpu_baseline} MiB).", flush=True)

    print("\n" + "=" * 88)
    print("PHASE B RESULT")
    print("=" * 88)
    print(f"  total scoring time:        {dt:.2f} s")
    print(f"  ms/candidate:              {ms_per_candidate:.2f}")
    print(f"  candidates/sec:            {cand_per_sec:.1f}")
    print(f"  peak GPU memory:           {peak_mem:.1f} MB")
    print(f"  speedup vs Stage-4 baseline ({STAGE4_BASELINE_TIME_S}s / {STAGE4_BASELINE_MS_PER_CAND} ms/cand): "
          f"{speedup:.2f}x")
    print(f"  correctness vs existing reference (first 32 of {num_candidates} candidates, "
          f"same reference_32 used in Phase A):")
    print(f"    max_abs_err={abs_err_32.max().item():.3e}  max_rel_err={rel_err_32.max().item():.3e}  "
          f"Pearson={pear_32:.6f}  Spearman={spear_32:.6f}")
    print(f"  all {num_candidates} scores finite={torch.isfinite(scores).all().item()}  "
          f"nonneg={(scores >= 0).all().item()}  min={scores.min().item():.4g}  max={scores.max().item():.4g}")
    return dict(time=dt, ms_per_candidate=ms_per_candidate, cand_per_sec=cand_per_sec, peak_mem=peak_mem,
                speedup=speedup, max_rel_err_32=rel_err_32.max().item(), pearson_32=pear_32, spearman_32=spear_32)


# --------------------------------------------------------------------------------------
# Small bottleneck breakdown
# --------------------------------------------------------------------------------------


def bottleneck_breakdown(gpu_baseline, chunk_size, num_candidates=16):
    print("\n" + "=" * 88)
    print(f"BOTTLENECK BREAKDOWN -- {num_candidates} candidates, chunk_size={chunk_size}")
    print("=" * 88, flush=True)

    print(f"  >>> starting (worker timeout={BOTTLENECK_TIMEOUT}s)...", flush=True)
    t_wall0 = time.perf_counter()
    timed_out, rc, stdout = launch_worker(
        ["--chunk-size", str(chunk_size), "--num-candidates", str(num_candidates), "--bottleneck"],
        BOTTLENECK_TIMEOUT,
    )
    wall = time.perf_counter() - t_wall0
    after_mem = wait_for_gpu_release(gpu_baseline)

    if timed_out:
        print(f"  <<< TIMED OUT after {wall:.1f}s -- worker killed. "
              f"GPU memory after cleanup: {after_mem} MiB.", flush=True)
        return None

    result = parse_result(stdout)
    if result is None or result.get("status") != "ok":
        reason = (result or {}).get("message") or (stdout[-300:] if stdout else f"exit code {rc}, no RESULT line")
        print(f"  <<< FAILED: {reason}. GPU memory after cleanup: {after_mem} MiB.", flush=True)
        return None

    forward, backward, reduce_ = result["forward"], result["backward"], result["reduce"]
    total = forward + backward + reduce_
    print(f"  <<< done in {wall:.1f}s wall. GPU memory after cleanup: {after_mem} MiB.", flush=True)
    print(f"\n  forward (denoiser, batched):        {forward:.3f}s  ({forward / total * 100:.1f}%)")
    print(f"  batched backward (is_grads_batched): {backward:.3f}s  ({backward / total * 100:.1f}%)")
    print(f"  v^2/h_D reduction:                   {reduce_:.3f}s  ({reduce_ / total * 100:.1f}%)")
    return dict(forward=forward, backward=backward, reduce=reduce_)


# --------------------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--chunk-size", type=int, default=1)
    parser.add_argument("--num-candidates", type=int, default=32)
    parser.add_argument("--check-determinism", action="store_true")
    parser.add_argument("--bottleneck", action="store_true")
    parser.add_argument("--phase", type=str, default="b", choices=["a", "b"])
    args = parser.parse_args()

    if args.worker:
        run_worker_main(args)
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)
    prepare_cache(device)
    gpu_baseline = get_gpu_used_mb()
    print(f"GPU baseline: {gpu_baseline} MiB", flush=True)

    if args.phase == "a":
        phase_a(gpu_baseline)
        print("\nPhase A complete. Not proceeding to Phase B in this invocation.")
    else:
        phase_b(gpu_baseline, chunk_size=4)
        bottleneck_breakdown(gpu_baseline, chunk_size=4, num_candidates=16)
        print("\nPhase B and bottleneck breakdown complete. Stopping here.")


if __name__ == "__main__":
    main()
