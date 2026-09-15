"""Phase 4 step 1: create ONE shared random denoiser initialization per domain
(theta_0_walker, theta_0_quadruped), so all 3 mixture conditions within a
domain start from an identical initial parameter state.

Builds a real Agent (src/agent.py, unmodified) via Hydra config composition of
config/agent/default.yaml, with action-space kwargs read from a throwaway
DMControlEnv instance (no gymnasium/AsyncVectorEnv machinery needed for this).
Fixes torch's global RNG to THETA0_SEED immediately before construction, since
nn.Module weight init consumes the global RNG stream.

Run from the LCG/ project root:
    python scripts/lcg_diagnostic/pretrained_collectors/create_theta0.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch
from hydra import compose, initialize_config_dir
from hydra.utils import instantiate

_THIS_DIR = Path(__file__).resolve().parent
_LCG_ROOT = _THIS_DIR.parent.parent.parent
_PROJECT_ROOT = _LCG_ROOT.parent
sys.path.insert(0, str(_LCG_ROOT / "src"))
sys.path.insert(0, str(_LCG_ROOT / "src" / "envs"))

from agent import Agent, get_action_space_kwargs  # noqa: E402
from dm_control_env import DMControlEnv  # noqa: E402

THETA0_SEED = 0  # documented fixed seed, used only for this shared initialization
OUT_DIR = _PROJECT_ROOT / "docs" / "lcg_undersample_diagnostic" / "models"
DOMAINS = ["walker", "quadruped"]


def fingerprint(state_dict) -> str:
    h = hashlib.sha256()
    for k in sorted(state_dict.keys()):
        t = state_dict[k].detach().cpu().contiguous()
        h.update(k.encode())
        h.update(t.numpy().tobytes())
    return h.hexdigest()


def build_theta0(domain: str, seed: int = THETA0_SEED, out_path: Path = None) -> dict:
    """seed/out_path default to Seed A's exact original values/filename when not passed,
    so existing call sites (and Seed A reproducibility) are unaffected. Multi-seed callers
    pass an explicit seed and a seed-suffixed out_path (e.g. theta_0_walker_seed43.pt)."""
    probe = DMControlEnv(domain_name=domain, task_name="walk", size=64, camera_id=0, action_repeat=2)
    fake_env = SimpleNamespace(
        is_discrete=False,
        action_dim=probe.action_dim,
        action_low=torch.as_tensor(probe.action_low),
        action_high=torch.as_tensor(probe.action_high),
    )
    action_kwargs = get_action_space_kwargs(fake_env)

    with initialize_config_dir(version_base="1.3", config_dir=str(_LCG_ROOT / "config")):
        cfg = compose(config_name="trainer", overrides=["env=dm_control"])

    torch.manual_seed(seed)
    agent = Agent(instantiate(cfg.agent, **action_kwargs))

    out_path = out_path if out_path is not None else (OUT_DIR / f"theta_0_{domain}.pt")
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(agent.state_dict(), out_path)

    denoiser_fp = fingerprint(agent.denoiser.state_dict())
    full_fp = fingerprint(agent.state_dict())
    result = {
        "domain": domain,
        "theta0_seed": seed,
        "checkpoint_path": str(out_path),
        "denoiser_fingerprint_sha256": denoiser_fp,
        "full_agent_fingerprint_sha256": full_fp,
        "action_dim": action_kwargs["continuous_action_dim"],
    }
    print(f"{domain}: theta_0 saved to {out_path}")
    print(f"  denoiser fingerprint: {denoiser_fp}")
    print(f"  action_dim: {result['action_dim']}")
    return result


def main() -> None:
    results = {d: build_theta0(d) for d in DOMAINS}
    (OUT_DIR / "theta0_manifest.json").write_text(json.dumps(results, indent=2))
    print(f"\nSaved {OUT_DIR / 'theta0_manifest.json'}")


if __name__ == "__main__":
    main()
