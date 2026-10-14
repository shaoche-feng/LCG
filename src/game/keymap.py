from typing import Dict, List, Tuple

import pygame


ActionNames = List[str]
Keymap = Dict[Tuple[int], int]


def get_keymap_and_action_names(name: str) -> Tuple[Keymap, ActionNames]:
    if name == "empty":
        return EMPTY_KEYMAP, EMPTY_ACTION_NAMES

    if name == "dataset_mode":
        return DATASET_MODE_KEYMAP, DATASET_MODE_ACTION_NAMES

    raise ValueError(f"Unknown keymap: {name}")


DATASET_MODE_ACTION_NAMES = [
    "noop",
    "previous",
    "next",
    "previous_10",
    "next_10",
]

DATASET_MODE_KEYMAP = {
    (pygame.K_LEFT,): 1,
    (pygame.K_RIGHT,): 2,
    (pygame.K_PAGEDOWN,): 3,
    (pygame.K_PAGEUP,): 4,
}

EMPTY_ACTION_NAMES = [
    "noop",
]

EMPTY_KEYMAP = {}
