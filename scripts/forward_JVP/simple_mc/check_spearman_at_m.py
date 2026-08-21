#! /usr/bin/env python
"""
Lightweight checkpoint checker for the forward simple-MC JVP M-search: loads the cached
(seed -> (num_samples, 480)) score tensor and reports cross-seed mean pairwise Spearman
(and top10/top20) for a given M. CPU-only, no GPU, no new JVPs -- pure bookkeeping so the
adaptive M schedule can decide whether to extend further without rerunning the full
analysis suite each time.

Usage:
    python scripts/forward_JVP/simple_mc/check_spearman_at_m.py <M> [<M2> ...]
"""
import itertools
import sys
from pathlib import Path


def _find_repo_root(start: Path) -> Path:
    p = start.resolve()
    while not ((p / "src").is_dir() and (p / "scripts").is_dir()):
        if p.parent == p:
            raise RuntimeError("could not locate LCG repo root (no ancestor has both src/ and scripts/)")
        p = p.parent
    return p


_REPO_ROOT = _find_repo_root(Path(__file__).parent)
sys.path.insert(0, str(_REPO_ROOT / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import numpy as np
import torch

import diagnose_lcg_backward_variance_setup as setup

NUM_OUTER_SEEDS = 8
CACHE_PATH = setup.DIAG_DIR / "forward_simple_mc_raw_scores.pt"


def main():
    m_list = [int(x) for x in sys.argv[1:]]
    if not m_list:
        print("usage: check_spearman_at_m.py <M> [<M2> ...]")
        sys.exit(1)

    cache = torch.load(CACHE_PATH, map_location="cpu", weights_only=True)
    min_available = min(v.shape[0] for v in cache.values())
    print(f"cache: {len(cache)} seeds, min samples available = {min_available}")

    for M in m_list:
        if M > min_available:
            print(f"M={M:>4}: SKIPPED (only {min_available} samples cached)")
            continue
        vectors = [cache[s][:M].mean(dim=0) for s in range(NUM_OUTER_SEEDS)]
        sps, t10s, t20s = [], [], []
        for s1, s2 in itertools.combinations(range(NUM_OUTER_SEEDS), 2):
            a, b = vectors[s1], vectors[s2]
            sps.append(setup.spearman_corr(a, b))
            t10s.append(setup.top_q_overlap(a, b, 0.10))
            t20s.append(setup.top_q_overlap(a, b, 0.20))
        sp_mean = float(np.mean(sps))
        print(f"M={M:>4}: Spearman_mean={sp_mean:.4f}  top10={np.mean(t10s):.4f}  top20={np.mean(t20s):.4f}  "
              f"{'>=0.90' if sp_mean >= 0.90 else ''}{' >=0.95' if sp_mean >= 0.95 else ''}")


if __name__ == "__main__":
    main()
