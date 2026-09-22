"""Filesystem/label names for the hopper domain.

Hopper has no walk/run tasks in dm_control (only stand/hop). Every pipeline script
keeps using the INTERNAL slot names "walk" (=stand) and "run" (=hop) and the internal
condition names run_scarce / balanced / walk_scarce -- these are dict keys, CSV
column values and JSON result keys, and changing them would break resume-skip keys
and joins between phases. Only where a name becomes a FOLDER or FILE name, or a figure
label, is it translated through this module. For walker/quadruped every function here
is the identity, so their existing paths are untouched.
"""
from __future__ import annotations

_SLOT_NAMES = {"hopper": {"walk": "stand", "run": "hop"}}
_CONDITION_NAMES = {"hopper": {"run_scarce": "hop_scarce", "walk_scarce": "stand_scarce"}}


def slot_dir(domain: str, slot: str) -> str:
    """Folder/label name for a behavior slot ("walk"/"run"); "random" and non-hopper pass through."""
    return _SLOT_NAMES.get(domain, {}).get(slot, slot)


def cond_dir(domain: str, condition: str) -> str:
    """Folder/label name for a condition (run_scarce/balanced/walk_scarce)."""
    return _CONDITION_NAMES.get(domain, {}).get(condition, condition)


slot_label = slot_dir
cond_label = cond_dir
