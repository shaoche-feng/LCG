"""Downstream milestones: an immutable denoiser + replay index published during exploration.

Layout inside a run folder:

    milestones/steps_NNNNNN/denoiser.pt    denoiser weights (ensemble: member 0)
    milestones/steps_NNNNNN/denoiser_member_K.pt  ensemble member K >= 1 (ensemble runs only)
    milestones/steps_NNNNNN/dataset.pt     replay index (episode versions, checksums, lengths, labels)
    milestones/steps_NNNNNN/manifest.yaml  what the milestone is, with file checksums

All paths are relative to the milestone folder, so the whole run folder can be moved.
Episode files are the dataset's immutable versions (`<id>_<length>.pt`), shared with
checkpoints and other milestones rather than copied.
"""
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

import torch
import yaml

from .dataset import Dataset
from utils import file_sha256

SCHEMA_VERSION = 1
DENOISER_FILE = "denoiser.pt"
DATASET_FILE = "dataset.pt"
MANIFEST_FILE = "manifest.yaml"


def milestone_name(num_steps: int) -> str:
    return f"steps_{num_steps:06d}"


def downstream_output_dir(milestone_path: Path, task_name: str, seed: int, member_index: int = 0) -> Path:
    """Reserved: <run>/downstream/<milestone>/<task>/seed_<seed>/ (not created here); an
    ensemble member other than 0 adds `_member_<k>`."""
    milestone_path = Path(milestone_path)
    name = f"seed_{seed}" + (f"_member_{member_index}" if member_index else "")
    return milestone_path.parent.parent / "downstream" / milestone_path.name / task_name / name


def member_file(member_index: int) -> str:
    return DENOISER_FILE if member_index == 0 else f"denoiser_member_{member_index}.pt"


def state_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """Content hash of tensors, independent of torch.save's file format."""
    digest = hashlib.sha256()
    for key, value in state_dict.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def publish_milestone(
    milestones_dir: Path,
    name: str,
    denoiser_state: Dict[str, torch.Tensor],
    dataset: Dataset,
    successor_observations: Dict[int, torch.Tensor],
    manifest: Dict[str, Any],
    member_states: Optional[List[Dict[str, torch.Tensor]]] = None,
) -> Path:
    """Publish a complete milestone folder atomically (written aside, then renamed). An
    existing milestone is kept if it holds the same training state, otherwise rejected.
    member_states: ensemble members 1..K-1 (member 0 is denoiser_state)."""
    milestones_dir = Path(milestones_dir)
    final = milestones_dir / name
    member_states = member_states or []
    identity = dict(denoiser_sha256=state_sha256(denoiser_state), dataset_sha256=index_sha256(dataset),
                    **{k: manifest[k] for k in ("epoch", "collected_steps", "counters")})
    if member_states:
        identity["member_sha256"] = [identity["denoiser_sha256"], *(state_sha256(s) for s in member_states)]
    if final.exists():
        existing = read_manifest(final)
        if existing.get("identity") != identity:
            raise ValueError(f"Milestone {final} already exists with a different training state")
        return final

    tmp = milestones_dir / f".{name}.tmp"
    if tmp.exists():
        shutil.rmtree(tmp)  # left by an interrupted publish
    tmp.mkdir(parents=True)
    torch.save(denoiser_state, tmp / DENOISER_FILE)
    for k, state in enumerate(member_states, start=1):
        torch.save(state, tmp / member_file(k))
    if None in dataset.file_sha256:
        raise ValueError(f"{dataset.name} has episode files without checksums; cannot publish a milestone")
    files = []
    for path, sha256 in zip(dataset.episode_files(), dataset.file_sha256):
        # Checksums were taken when each immutable version was written: no re-hashing here.
        files.append(dict(path=Path(os.path.relpath(path, final)).as_posix(), sha256=sha256))
    torch.save(dict(
        schema_version=SCHEMA_VERSION,
        directory=Path(os.path.relpath(dataset.directory, final)).as_posix(),
        index=dataset.state_dict(),
        files=files,
        # Observation after the last transition of each unfinished episode at the cut.
        successor_observations={k: v.cpu() for k, v in successor_observations.items()},
    ), tmp / DATASET_FILE)
    manifest = dict(
        schema_version=SCHEMA_VERSION,
        **manifest,
        files=dict(
            denoiser=dict(path=DENOISER_FILE, sha256=file_sha256(tmp / DENOISER_FILE)),
            dataset=dict(path=DATASET_FILE, sha256=file_sha256(tmp / DATASET_FILE)),
            **{f"denoiser_member_{k}": dict(path=member_file(k), sha256=file_sha256(tmp / member_file(k)))
               for k in range(1, len(member_states) + 1)},
        ),
        identity=identity,
    )
    with open(tmp / MANIFEST_FILE, "w", encoding="utf-8") as f:
        yaml.safe_dump(manifest, f, sort_keys=False)
    replace_directory(tmp, final)
    return final


def replace_directory(src: Path, dst: Path, attempts: int = 20, delay: float = 0.5) -> None:
    """os.replace, retried on PermissionError: on Windows a scanner or indexer can briefly hold a
    freshly written file, which makes renaming its directory fail (WinError 5)."""
    for attempt in range(attempts):
        try:
            os.replace(src, dst)
            return
        except PermissionError:
            if attempt == attempts - 1:
                raise
            time.sleep(delay)


def index_sha256(dataset: Dataset) -> str:
    digest = hashlib.sha256()
    digest.update(repr((dataset.num_episodes, dataset.num_steps, dataset.task_names)).encode())
    digest.update(dataset.lengths.astype("int64").tobytes())
    return digest.hexdigest()


def read_manifest(milestone_path: Path) -> Dict[str, Any]:
    path = Path(milestone_path) / MANIFEST_FILE
    if not path.is_file():
        raise FileNotFoundError(f"No milestone manifest at {path}")
    with open(path, encoding="utf-8") as f:
        manifest = yaml.safe_load(f)
    if manifest.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"{path}: schema_version {manifest.get('schema_version')}, expected {SCHEMA_VERSION}")
    return manifest


def discover_milestones(milestones_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    """(path, manifest) of every immediate child milestone of a folder, ordered by the manifests'
    collected_steps (not by folder name). Every entry must be a published milestone: an empty
    folder, any other entry (including an interrupted publish `.<name>.tmp`) and two milestones
    recording the same step count are rejected. Only manifests are read; load_milestone checks files."""
    milestones_dir = Path(milestones_dir)
    if not milestones_dir.is_dir():
        raise FileNotFoundError(f"Milestones folder {milestones_dir} does not exist")
    found, problems = [], []
    for path in sorted(milestones_dir.iterdir()):
        if not path.is_dir():
            problems.append(f"{path.name}: not a milestone folder")
        elif path.name.startswith("."):
            problems.append(f"{path.name}: an interrupted or in-progress milestone publish")
        else:
            try:
                manifest = read_manifest(path)
            except (OSError, ValueError, yaml.YAMLError) as error:
                problems.append(f"{path.name}: {error}")
                continue
            steps = manifest.get("collected_steps")
            if isinstance(steps, bool) or not isinstance(steps, int):
                problems.append(f"{path.name}: manifest has no integer collected_steps ({steps!r})")
            else:
                found.append((path, manifest))
    if problems:
        raise ValueError(f"Malformed milestones folder {milestones_dir}:\n  " + "\n  ".join(problems))
    if not found:
        raise ValueError(f"Milestones folder {milestones_dir} contains no milestones")
    by_steps = {}
    for path, manifest in found:
        by_steps.setdefault(manifest["collected_steps"], []).append(path.name)
    duplicates = {steps: names for steps, names in by_steps.items() if len(names) > 1}
    if duplicates:
        raise ValueError(f"Milestones folder {milestones_dir}: ambiguous duplicate step counts "
                         + "; ".join(f"{steps} recorded by {', '.join(names)}" for steps, names in sorted(duplicates.items())))
    return sorted(found, key=lambda item: item[1]["collected_steps"])


def referenced_files(milestone_path: Path) -> List[Path]:
    """Episode versions a milestone refers to (for retention), resolved on disk."""
    milestone_path = Path(milestone_path)
    index = torch.load(milestone_path / DATASET_FILE, weights_only=False)
    return [(milestone_path / f["path"]).resolve() for f in index["files"]]


@dataclass
class Milestone:
    path: Path
    manifest: Dict[str, Any]
    denoiser_state: Dict[str, torch.Tensor]
    dataset: Dataset  # read-only, refers to the milestone's exact episode versions
    task_name: str
    task_index: int  # column of Episode.task_rew for task_name
    successor_observations: Dict[int, torch.Tensor]
    member_index: int = 0  # ensemble member loaded as denoiser_state (0 for single-model milestones)


def milestone_members(manifest: Dict[str, Any]) -> int:
    """Number of denoisers a milestone holds (1 for single-model milestones)."""
    return int((manifest.get("ensemble") or {}).get("members", 1))


def load_milestone(milestone_path: Path, task_name: str, map_location: Optional[torch.device] = None,
                   cache_in_ram: bool = False, member_index: int = 0) -> Milestone:
    """Resolve and validate a milestone for downstream training on task_name; denoiser_state
    is ensemble member `member_index` (0, the only one, for single-model milestones)."""
    path = Path(milestone_path)
    manifest = read_manifest(path)
    members = milestone_members(manifest)
    if isinstance(member_index, bool) or not isinstance(member_index, int) or not 0 <= member_index < members:
        raise ValueError(f"Milestone {path} has {members} denoiser(s); member_index {member_index!r} is not one of them")
    for key, entry in manifest["files"].items():
        file = path / entry["path"]
        if not file.is_file():
            raise FileNotFoundError(f"Milestone {path}: missing {key} file {file}")
        if file_sha256(file) != entry["sha256"]:
            raise ValueError(f"Milestone {path}: {key} file {file} does not match its checksum")

    # Trusted, locally generated index (numpy arrays, Counter): weights_only=False.
    saved = torch.load(path / manifest["files"]["dataset"]["path"], weights_only=False)
    dataset = Dataset(path / saved["directory"], "milestone_dataset", cache_in_ram=cache_in_ram, save_on_disk=False)
    dataset.load_state_dict(saved["index"])
    dataset.is_static = True  # reads only; never written

    if int(dataset.lengths.sum()) != dataset.num_steps or dataset.num_steps != manifest["collected_steps"]:
        raise ValueError(f"Milestone {path}: dataset has {dataset.num_steps} steps "
                         f"(episodes sum to {int(dataset.lengths.sum())}), manifest says {manifest['collected_steps']}")
    if dataset.num_episodes != len(saved["files"]) or dataset.num_episodes != manifest["num_episodes"]:
        raise ValueError(f"Milestone {path}: inconsistent episode counts")
    for episode_path, entry in zip(dataset.episode_files(), saved["files"]):
        file = path / entry["path"]
        if episode_path.resolve() != file.resolve():
            raise ValueError(f"Milestone {path}: index and file list disagree on {entry['path']}")
        if not file.is_file():
            raise FileNotFoundError(f"Milestone {path}: missing episode file {file}")
        if file_sha256(file) != entry["sha256"]:
            raise ValueError(f"Milestone {path}: episode file {file} does not match its checksum")
    task_names = list(dataset.task_names or [])
    if task_names != list(manifest["task_names"]):
        raise ValueError(f"Milestone {path}: dataset labels {task_names} differ from manifest {manifest['task_names']}")
    if task_name not in task_names:
        raise ValueError(f"Milestone {path} has no reward labels for task {task_name!r}; recorded: {task_names}")

    key = "denoiser" if member_index == 0 else f"denoiser_member_{member_index}"
    denoiser_state = torch.load(path / manifest["files"][key]["path"], map_location=map_location, weights_only=True)
    if members > 1 and state_sha256(denoiser_state) != manifest["identity"]["member_sha256"][member_index]:
        raise ValueError(f"Milestone {path}: member {member_index} weights do not match the recorded identity")
    return Milestone(path, manifest, denoiser_state, dataset, task_name, task_names.index(task_name),
                     saved["successor_observations"], member_index)
