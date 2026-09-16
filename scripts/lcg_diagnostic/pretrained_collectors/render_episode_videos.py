"""Ad-hoc visualization utility (not part of the diagnostic pipeline): renders
EVERY episode of every Phase 2 source pool to an .mp4, for visual inspection.
Read-only -- does not touch the datasets.

Run from anywhere (paths are absolute):
    python scripts/lcg_diagnostic/pretrained_collectors/render_episode_videos.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import cv2

_LCG_ROOT = Path(r"C:\Users\jerry\Project_LCG\LCG")
sys.path.insert(0, str(_LCG_ROOT / "src"))
from data import Dataset  # noqa: E402

SOURCE_ROOT = Path(r"C:\Users\jerry\Project_LCG\docs\lcg_undersample_diagnostic\source_pools")

FPS = 20
SCALE = 4  # upscale 64x64 -> 256x256 for visibility


def episode_to_video(domain: str, source: str, ds: Dataset, episode_id: int, out_dir: Path) -> Path:
    ep = ds.load_episode(episode_id)
    obs = ep.obs  # (T, 3, 64, 64), float in [-1, 1]

    frames = ((obs + 1) / 2 * 255).clamp(0, 255).byte().numpy()  # (T,3,64,64) uint8
    frames = frames.transpose(0, 2, 3, 1)  # (T,64,64,3) RGB

    out_path = out_dir / f"ep{episode_id:02d}.mp4"
    h, w = frames.shape[1] * SCALE, frames.shape[2] * SCALE
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), FPS, (w, h))
    for f in frames:
        bgr = cv2.cvtColor(f, cv2.COLOR_RGB2BGR)
        bgr = cv2.resize(bgr, (w, h), interpolation=cv2.INTER_NEAREST)
        writer.write(bgr)
    writer.release()
    return out_path


def main() -> None:
    for domain in ("walker", "quadruped"):
        for source in ("random", "walk", "run"):
            ds = Dataset(SOURCE_ROOT / domain / source / "dataset", name=f"{domain}_{source}", cache_in_ram=True)
            ds.load_from_default_path()
            out_dir = SOURCE_ROOT / domain / source / "episodes"
            out_dir.mkdir(parents=True, exist_ok=True)
            print(f"=== {domain}/{source}: {ds.num_episodes} episodes ===")
            for eid in range(ds.num_episodes):
                out_path = episode_to_video(domain, source, ds, eid, out_dir)
                print(f"  saved {out_path.name}")


if __name__ == "__main__":
    main()
