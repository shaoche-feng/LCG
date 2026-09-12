#! /usr/bin/env python
"""
The one retained manual LCG smoke script (per the repository cleanup, see
docs/lcg_diagnostic/corruption_fix/ and the cleanup report): a lightweight real-model,
real-GPU-if-available check that the production forward-JVP scorer runs end-to-end on
real (not tiny-synthetic) data. This is NOT a substitute for tests/lcg/ (which is fast,
deterministic, and CPU-only) -- it exists only for spot-checking on real hardware with a
real checkpoint, which pytest deliberately avoids depending on.

Uses the durable checkpoint/candidate backups under docs/lcg_diagnostic/checkpoint/
(NOT the ephemeral scratchpad, which has been wiped by OS Temp cleanup more than once
this project) so this script keeps working across sessions without needing a retrain.

h_D_full.pt here is used only as a structurally-valid (correct shape, positive) diagonal
precision tensor to run scoring against -- NOT as a scientific fixture. It was computed
under the pre-offset-noise-fix corruption law (see PRE_OFFSET_NOISE_FIX_README.txt) and
this script does not assert anything about the resulting scores' scientific validity,
only that they are finite/nonnegative/correctly shaped and the whole pipeline runs.

Usage:
    python scripts/lcg/smoke_test.py
"""
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))

DOCS_CHECKPOINT_DIR = _REPO_ROOT.parent / "docs" / "lcg_diagnostic" / "checkpoint"
DENOISER_PATH = DOCS_CHECKPOINT_DIR / "lcg_diag_denoiser_converged.pt"
H_D_PATH = DOCS_CHECKPOINT_DIR / "h_D_full.pt"
CANDIDATES_PATH = DOCS_CHECKPOINT_DIR / "frozen_candidates_480.pt"

NUM_STEPS_CONDITIONING = 4
NUM_SMOKE_CANDIDATES = 16  # a handful, not the full 480 -- this is a smoke test, not a benchmark
M = 4
CHUNK_SIZE = 4


def main():
    import torch

    from lcg.forward_jvp import (
        make_jvp_bank,
        score_one_jvp_bank,
    )
    from lcg.theta_s import ThetaSConfig, frozen_named_parameters, selected_named_parameters
    from models.diffusion import Denoiser, DenoiserConfig, SigmaDistributionConfig
    from models.diffusion.inner_model import InnerModelConfig

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device={device}", flush=True)

    if not (DENOISER_PATH.is_file() and H_D_PATH.is_file() and CANDIDATES_PATH.is_file()):
        print(f"SKIPPED: checkpoint/fixtures not found under {DOCS_CHECKPOINT_DIR}")
        return

    ckpt = torch.load(DENOISER_PATH, map_location=device, weights_only=False)
    action_dim = ckpt["action_dim"]
    inner_cfg = InnerModelConfig(
        img_channels=3, num_steps_conditioning=NUM_STEPS_CONDITIONING, cond_channels=256,
        depths=[2, 2, 2, 2], channels=[64, 64, 64, 64], attn_depths=[0, 0, 0, 0],
        continuous_action_dim=action_dim,
    )
    cfg = DenoiserConfig(inner_model=inner_cfg, sigma_data=0.5, sigma_offset_noise=0.3)
    denoiser = Denoiser(cfg).to(device)
    denoiser.load_state_dict(ckpt["denoiser"])
    denoiser.eval()
    print(f"loaded checkpoint (total_step={ckpt['step']}) from {DENOISER_PATH}", flush=True)

    h_D = torch.load(H_D_PATH, map_location=device, weights_only=True)
    candidates_all = torch.load(CANDIDATES_PATH, map_location=device, weights_only=True)
    candidates = [(c[0].to(device), c[1].to(device), c[2].to(device)) for c in candidates_all[:NUM_SMOKE_CANDIDATES]]
    print(f"h_D shape={tuple(h_D.shape)}  using {len(candidates)}/{len(candidates_all)} candidates", flush=True)

    # matches config/intrinsic_reward/lcg.yaml's production default for this
    # depths=[2,2,2,2] architecture (last u_block index 3)
    theta_s_cfg = ThetaSConfig(include=("unet.u_blocks.3.*", "norm_out.*", "conv_out.*"), exclude=())
    theta_s_named = selected_named_parameters(denoiser, theta_s_cfg)
    frozen_named = frozen_named_parameters(denoiser, theta_s_named)
    h_D_inv_sqrt = h_D.rsqrt()

    sigma_cfg = SigmaDistributionConfig(loc=-0.4, scale=1.2, sigma_min=0.002, sigma_max=20.0)
    bank = make_jvp_bank(
        sigma_cfg, torch.Size([1, 3, 64, 64]), h_D.numel(), device, num_samples=M, seed=0
    )

    scores = score_one_jvp_bank(
        denoiser, theta_s_named, frozen_named, theta_s_named, h_D, h_D_inv_sqrt, bank, candidates, CHUNK_SIZE
    )

    finite_ok = torch.isfinite(scores).all().item()
    nonneg_ok = (scores >= 0).all().item()
    shape_ok = scores.shape == (len(candidates),)
    print(f"scores: shape={tuple(scores.shape)} finite={finite_ok} nonneg={nonneg_ok} "
          f"mean={scores.mean().item():.3f} min={scores.min().item():.3f} max={scores.max().item():.3f}")

    passed = finite_ok and nonneg_ok and shape_ok
    print(f"\nSMOKE TEST {'PASSED' if passed else 'FAILED'}")
    if not passed:
        sys.exit(1)


if __name__ == "__main__":
    main()
