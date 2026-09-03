#! /usr/bin/env python
"""
Shared forward-mode JVP scorer for the LCG forward/JVP diagnostics (Stage F1 onward).

This module is now a thin re-export over the production implementation moved to
src/lcg/forward_jvp.py (Stage 6 integration): the diagnostic scripts under
scripts/forward_JVP/ keep importing `diagnose_lcg_forward_jvp_common as jvp_common` and
calling `jvp_common.jvp_through_F` / `jvp_common.make_jvp_bank` / `jvp_common.
score_one_jvp_bank` / etc. exactly as before -- only the underlying math now lives in one
place (src/lcg/forward_jvp.py) instead of being duplicated between production and
diagnostic code. No mathematical change: this is the exact Stage F1-F6 validated
implementation (genuine torch.func.jvp + torch.func.functional_call forward-mode AD,
EDM-cancelled J_F formulation, torch.no_grad()-wrapped to avoid spurious reverse-mode
graph construction), byte-for-byte relocated.
"""
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
# scripts/ was reorganized into topic subfolders after this file was first written; this
# extra entry lets this module find diagnose_lcg_backward_variance_setup regardless of
# exactly where this file itself ends up nested.
sys.path.insert(0, str(_REPO_ROOT / "scripts" / "backward_VJP" / "3-stratum"))

import diagnose_lcg_backward_variance_setup as setup
from lcg.forward_jvp import (
    JVPBank,
    assert_setup_valid,
    flatten_dict,
    frozen_named_parameters,
    jvp_through_F,
    make_forward_jvp_simple_mc_bank,
    make_jvp_bank,
    score_one_jvp_bank,
    selected_named_parameters,
    unflatten_to_dict,
)

NUM_STRATA = setup.NUM_STRATA

__all__ = [
    "JVPBank",
    "NUM_STRATA",
    "assert_setup_valid",
    "flatten_dict",
    "frozen_named_parameters",
    "jvp_through_F",
    "make_forward_jvp_simple_mc_bank",
    "make_jvp_bank",
    "score_one_jvp_bank",
    "selected_named_parameters",
    "unflatten_to_dict",
]
