"""Phase 4: train the denoiser ONLY (no rew_end_model, no actor_critic) on
each of the six Phase-3 mixture datasets, using the REAL DIAMOND Trainer
class (src/trainer.py, unmodified on disk) via Hydra's static-dataset path.

Three things this script does WITHOUT touching src/*.py, all isolated here:

1. Directory-layout adapter: static_dataset.path must contain a `train/`
   subfolder (Trainer.__init__: `p / "train"`, `p / "test"`), but our Phase-3
   mixture datasets are flat `dataset/` folders. We create a `train/` subfolder
   containing NTFS hardlinks (not copies -- no admin privilege needed, unlike
   symlinks which crashed wandb.save() in Phase 1) to the real episode files,
   so no data is duplicated on disk.

2. Runtime monkey-patch of `trainer.set_seed`: cfg.common.seed is dead code
   upstream (Trainer.__init__ always calls `set_seed(torch.seed() % 10**9)`,
   ignoring cfg.common.seed entirely -- discovered during Phase 1). Patching
   the name Trainer's module resolves at call time makes the documented fixed
   TRAINING_SEED actually take effect, without editing trainer.py.

3. Runtime monkey-patch of `trainer.wandb_log`: wandb is kept disabled (its
   non-disabled `.save()` call crashes on this Windows machine trying to
   symlink -- also a Phase 1 finding), so nothing is otherwise persisted;
   this captures every per-step training metrics dict trainer.py would have
   sent to wandb, so we get real loss history without enabling wandb.

Also bypasses Hydra's automatic per-run job directory/chdir machinery (since
we call `compose()` directly rather than the `@hydra.main` decorator) by
manually chdir-ing into the run directory and writing the `.hydra/config.yaml`
file trainer.py's `__init__` expects to move into `config/trainer.yaml`.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/train_denoiser_only.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import sys
import time
from pathlib import Path

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
from types import SimpleNamespace

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))

# main.py normally registers this before any config is resolved (trainer.yaml uses
# ${eval:...} interpolations, e.g. rew_end_model.training.seq_length). We bypass main.py
# entirely (no @hydra.main decorator), so it must be registered here instead.
OmegaConf.register_new_resolver("eval", eval)

PROBE_ROOT = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic"
MIXTURE_ROOT = PROBE_ROOT / "mixtures"
MODELS_ROOT = PROBE_ROOT / "models"

TRAINING_SEED = 42  # documented fixed seed, applied via monkey-patch (see module docstring)
DENOISER_OPTIMIZER_STEPS = 3000
NEVER_TRAIN_EPOCH_THRESHOLD = 1_000_000  # start_after_epochs value that guarantees a component never trains

DOMAINS = ["walker", "quadruped"]
CONDITIONS = ["run_scarce", "balanced", "walk_scarce"]


def fingerprint(state_dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def make_static_dataset_dir(domain: str, condition: str) -> Path:
    """Create <mixture>/for_denoiser_training/train/ as hardlinks to the real
    dataset files (no duplication). Idempotent: rebuilt fresh each call."""
    src_dir = MIXTURE_ROOT / domain / condition / "dataset"
    wrapper_dir = MIXTURE_ROOT / domain / condition / "for_denoiser_training"
    train_dir = wrapper_dir / "train"
    if wrapper_dir.exists():
        shutil.rmtree(wrapper_dir)
    train_dir.mkdir(parents=True)

    for path in src_dir.rglob("*"):
        rel = path.relative_to(src_dir)
        dest = train_dir / rel
        if path.is_dir():
            dest.mkdir(parents=True, exist_ok=True)
        else:
            dest.parent.mkdir(parents=True, exist_ok=True)
            os.link(path, dest)  # NTFS hardlink: same data, no extra disk space, no admin needed
    return wrapper_dir


def validate_checkpoint(domain: str, checkpoint_path: Path, expected_fingerprint: str, expected_action_dim: int) -> dict:
    """Reload the final checkpoint into a FRESH Agent instance (fresh process
    state, not the trained Trainer object) and verify it end to end: shapes,
    finiteness, a real forward pass, action-conditioning dim, and that the
    reloaded parameters reproduce the exact fingerprint recorded at save time."""
    from agent import Agent, get_action_space_kwargs
    sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))
    from dm_control_env import DMControlEnv

    probe = DMControlEnv(domain_name=domain, task_name="walk", size=64, camera_id=0, action_repeat=2)
    fake_env = SimpleNamespace(
        is_discrete=False, action_dim=probe.action_dim,
        action_low=torch.as_tensor(probe.action_low), action_high=torch.as_tensor(probe.action_high),
    )
    action_kwargs = get_action_space_kwargs(fake_env)

    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])
    fresh_agent = Agent(instantiate(cfg.agent, **action_kwargs))
    fresh_agent.load(checkpoint_path, load_denoiser=True, load_rew_end_model=False, load_actor_critic=False)

    shapes_ok = fresh_agent.denoiser.cfg.inner_model.continuous_action_dim == expected_action_dim
    params_finite = all(torch.isfinite(p).all() for p in fresh_agent.denoiser.parameters())
    reload_fp = fingerprint(fresh_agent.denoiser.state_dict())
    fingerprint_matches = reload_fp == expected_fingerprint

    forward_ok, output_finite = False, False
    try:
        n = fresh_agent.denoiser.cfg.inner_model.num_steps_conditioning
        b, c, h, w = 2, 3, 64, 64
        obs = torch.randn(b, n * c, h, w)
        act = torch.rand(b, n, expected_action_dim) * 2 - 1
        sigma = torch.ones(b) * 0.5
        noisy_next_obs = torch.randn(b, c, h, w)
        with torch.no_grad():
            out = fresh_agent.denoiser.denoise(noisy_next_obs, sigma, obs, act)
        forward_ok = True
        output_finite = bool(torch.isfinite(out).all())
    except Exception as e:  # noqa: BLE001
        print(f"    forward pass FAILED: {e}")

    result = {
        "reload_fingerprint": reload_fp,
        "expected_fingerprint": expected_fingerprint,
        "fingerprint_matches": fingerprint_matches,
        "action_dim_correct": shapes_ok,
        "params_finite": params_finite,
        "forward_pass_ok": forward_ok,
        "forward_output_finite": output_finite,
        "checkpoint_reload_passed": fingerprint_matches and shapes_ok and params_finite and forward_ok and output_finite,
    }
    print(f"    checkpoint reload: fp_match={fingerprint_matches} action_dim_ok={shapes_ok} "
          f"params_finite={params_finite} forward_ok={forward_ok} output_finite={output_finite}")
    return result


def train_one(domain: str, condition: str, seed: int = TRAINING_SEED, theta0_path: Path = None, run_dir: Path = None) -> dict:
    """seed/theta0_path/run_dir default to Seed A's exact original values/paths when not
    passed, so existing call sites (and Seed A reproducibility) are unaffected. Multi-seed
    callers pass an explicit seed (used for BOTH which theta_0 to load AND the training RNG
    -- one unified per-realization seed, matching the paired-seed experimental design) and
    seed-suffixed theta0_path/run_dir."""
    print(f"\n{'=' * 60}\n{domain}/{condition} (seed={seed})\n{'=' * 60}")
    static_dataset_dir = make_static_dataset_dir(domain, condition)
    theta0_path = theta0_path if theta0_path is not None else (MODELS_ROOT / f"theta_0_{domain}.pt")
    run_dir = run_dir if run_dir is not None else (MODELS_ROOT / domain / condition)
    if run_dir.exists():
        shutil.rmtree(run_dir)
    run_dir.mkdir(parents=True)

    overrides = [
        "env=dm_control",
        f"env.train.domain_name={domain}",
        "env.train.task_name=walk",  # nominal only: static_dataset skips all live collection
        "common.devices=0",
        "intrinsic_reward.enabled=false",
        "training.model_free=false",
        "training.compile_wm=false",  # avoid requiring Triton (unavailable on Windows here)
        f"static_dataset.path={static_dataset_dir.as_posix()}",
        f"initialization.path_to_ckpt={theta0_path.as_posix()}",
        "initialization.load_denoiser=true",
        "initialization.load_rew_end_model=false",
        "initialization.load_actor_critic=false",
        f"denoiser.training.steps_first_epoch={DENOISER_OPTIMIZER_STEPS}",
        "training.num_final_epochs=1",
        f"rew_end_model.training.start_after_epochs={NEVER_TRAIN_EPOCH_THRESHOLD}",
        f"actor_critic.training.start_after_epochs={NEVER_TRAIN_EPOCH_THRESHOLD}",
        "evaluation.should=false",
        "checkpointing.save_agent_every=1",
        "checkpointing.num_to_keep=1",
        "wandb.mode=disabled",
    ]

    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=overrides)
    resolved_config_yaml = OmegaConf.to_yaml(cfg)

    orig_cwd = os.getcwd()
    os.chdir(run_dir)
    (run_dir / ".hydra").mkdir()
    (run_dir / ".hydra" / "config.yaml").write_text(resolved_config_yaml)

    captured_logs = []

    def capture_wandb_log(logs, epoch):
        for d in logs:
            captured_logs.append({"epoch": epoch, **{k: (v.item() if torch.is_tensor(v) else v) for k, v in d.items()}})

    from utils import set_seed as real_set_seed

    def fixed_set_seed(_ignored_seed):
        real_set_seed(seed)

    # trainer.py unconditionally copies root_dir/src and root_dir/scripts into every fresh
    # run directory for crash-recovery (see scripts/resume.sh). root_dir/scripts here also
    # contains this diagnostic's checkpoints/ folder (~121MB) -- copying that into all 6 run
    # dirs would waste >700MB and slow every run's startup for no benefit in this diagnostic
    # (we never resume these runs via resume.sh). No-op it globally for this process only.
    shutil.copytree = lambda *a, **k: None

    try:
        import trainer as trainer_module
        trainer_module.wandb_log = capture_wandb_log
        trainer_module.set_seed = fixed_set_seed

        t0 = time.time()
        t = trainer_module.Trainer(cfg, root_dir=_LCG_ROOT)
        init_denoiser_fp = fingerprint(t.agent.denoiser.state_dict())
        action_dim = t.agent.denoiser.cfg.inner_model.continuous_action_dim
        t.run()
        wall_clock_s = time.time() - t0
        final_denoiser_fp_in_memory = fingerprint(t.agent.denoiser.state_dict())
    finally:
        os.chdir(orig_cwd)
        sys.modules.pop("trainer", None)  # drop the module so the next domain/condition re-imports cleanly

    denoiser_logs = [d for d in captured_logs if "denoiser/train/loss_denoising" in d]
    losses = [d["denoiser/train/loss_denoising"] for d in denoiser_logs]
    n_recent = min(200, len(losses))
    rolling_mean_end = float(sum(losses[-n_recent:]) / n_recent) if losses else float("nan")
    min_loss = float(min(losses)) if losses else float("nan")
    all_finite = all(torch.isfinite(torch.tensor(losses))) if losses else False

    final_ckpt = run_dir / "checkpoints" / "agent_versions" / "agent_epoch_00001.pt"
    if not final_ckpt.exists():
        candidates = sorted((run_dir / "checkpoints" / "agent_versions").glob("agent_epoch_*.pt"))
        final_ckpt = candidates[-1] if candidates else None

    (run_dir / "training_loss_history.json").write_text(json.dumps(losses))

    checkpoint_validation = validate_checkpoint(domain, final_ckpt, final_denoiser_fp_in_memory, action_dim)

    result = {
        "checkpoint_validation": checkpoint_validation,
        "domain": domain, "condition": condition,
        "training_transitions": 5000,
        "denoiser_optimizer_steps_requested": DENOISER_OPTIMIZER_STEPS,
        "denoiser_optimizer_steps_logged": len(losses),
        "initialization_fingerprint_before_training": init_denoiser_fp,
        "training_seed": seed,
        "initial_loss": losses[0] if losses else None,
        "final_loss": losses[-1] if losses else None,
        "final_loss_rolling_mean": rolling_mean_end,
        "min_observed_loss": min_loss,
        "all_losses_finite": all_finite,
        "wall_clock_seconds": wall_clock_s,
        "checkpoint_path": str(final_ckpt) if final_ckpt else None,
        "static_dataset_dir": str(static_dataset_dir),
        "theta0_path": str(theta0_path),
        "run_dir": str(run_dir),
    }
    print(f"  optimizer_steps={len(losses)} initial_loss={result['initial_loss']:.4f} "
          f"final_loss={result['final_loss']:.4f} rolling_mean={rolling_mean_end:.4f} "
          f"wall_clock={wall_clock_s:.1f}s all_finite={all_finite}")
    return result


def main() -> None:
    all_results = {}
    for domain in DOMAINS:
        for condition in CONDITIONS:
            key = f"{domain}/{condition}"
            all_results[key] = train_one(domain, condition)
            (MODELS_ROOT / "phase4_training_results.json").write_text(json.dumps(all_results, indent=2))

    print("\n\n=== PHASE 4 TRAINING SUMMARY ===")
    for key, r in all_results.items():
        print(f"{key:25s} steps={r['denoiser_optimizer_steps_logged']:5d} "
              f"init_loss={r['initial_loss']:.4f} final_loss={r['final_loss']:.4f} "
              f"wall_clock={r['wall_clock_seconds']:.1f}s")


if __name__ == "__main__":
    main()
