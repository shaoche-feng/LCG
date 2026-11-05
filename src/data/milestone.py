"""Downstream milestones: an immutable denoiser + replay index published during exploration.

Layout inside a run folder:

    milestones/steps_NNNNNN/denoiser.pt    denoiser weights (ensemble: member 0)
    milestones/steps_NNNNNN/denoiser_member_K.pt  ensemble member K >= 1 (ensemble runs only)
    milestones/steps_NNNNNN/dataset.pt     replay index (episode versions, checksums, lengths, labels)
    milestones/steps_NNNNNN/manifest.json  what the milestone is, with file checksums

All paths are relative to the milestone folder, so the whole run folder can be moved.
Episode files are the dataset's immutable versions (`<id>_<length>.pt`), shared with
checkpoints and other milestones rather than copied.

File formats: configuration is YAML, generated metadata (milestone manifests, downstream
summaries and evaluation reports) is JSON only, append-only training metrics are JSONL.
"""
from dataclasses import dataclass, field
import hashlib
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any, Dict, List, Optional, Tuple

from omegaconf import OmegaConf
import torch

from .dataset import Dataset
from utils import file_sha256

SCHEMA_VERSION = 1
DENOISER_FILE = "denoiser.pt"
DATASET_FILE = "dataset.pt"
MANIFEST_FILE = "manifest.json"
SUMMARY_FILE = "summary.json"  # a completed downstream output
RETURN_CONVENTION = ("sum over environment steps of the requested task's reward; each environment step "
                     "sums the reward of its executed action-repeat substeps")
# Output folder label of each exploration method (default run folder outputs/<label>/<domain>/seed_<seed>);
# the keys are the intrinsic_reward.method identifiers (None: intrinsic reward disabled, uniform random).
OUTPUT_LABELS = {None: "uniform", "lcg": "lcg", "ensemble_disagreement": "full-ensemble",
                 "laplace_disagreement": "laplace-ensemble", "plan2explore_disagreement": "plan2explore"}


def write_json(path: Path, data: Dict[str, Any]) -> None:
    """Generated metadata as standard JSON: a non-finite number raises instead of being written as
    NaN/Infinity. Written aside and renamed, so a reader never sees a partial file."""
    path = Path(path)
    text = json.dumps(data, indent=2, allow_nan=False) + "\n"
    tmp = path.with_name(f"{path.name}.tmp")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, path)


def reject_constant(name: str) -> None:
    raise ValueError(f"non-standard JSON constant {name}")


def read_json(path: Path, what: str) -> Dict[str, Any]:
    """Generated metadata: standard JSON holding a mapping (NaN/Infinity are rejected). A missing file
    is reported as such, naming a generated YAML file left by older code if there is one (not read)."""
    path = Path(path)
    if not path.is_file():
        old = path.with_suffix(".yaml")
        raise FileNotFoundError(f"No {what} at {path}" + (
            f" (only {old.name}, generated YAML metadata from before 2026-10-29, which is no longer read)"
            if old.is_file() else ""))
    try:
        data = json.loads(path.read_text(encoding="utf-8"), parse_constant=reject_constant)
    except ValueError as error:
        raise ValueError(f"{path} is not valid JSON ({error})") from None
    if not isinstance(data, dict):
        raise ValueError(f"{path} does not hold a JSON object")
    return data


def downstream_summary(output_dir: Path) -> Optional[Dict[str, Any]]:
    """The completion summary (summary.json) of a downstream output; None if it was not completed."""
    path = Path(output_dir) / SUMMARY_FILE
    return read_json(path, "downstream summary") if path.is_file() else None


def method_label(intrinsic_enabled: bool, method: Optional[str]) -> str:
    """The output label of an exploration configuration (intrinsic_reward.enabled and .method)."""
    key = (method or "lcg") if intrinsic_enabled else None
    if key not in OUTPUT_LABELS:
        raise ValueError(f"Unknown intrinsic_reward.method {method!r}; known: {sorted(k for k in OUTPUT_LABELS if k)}")
    return OUTPUT_LABELS[key]


def exploration_identity(run_dir: Path) -> Dict[str, Any]:
    """Method label, domain and exploration seed of an exploration run, from its saved configuration
    (config/exploration.yaml), never from folder names."""
    path = Path(run_dir) / "config" / "exploration.yaml"
    if not path.is_file():
        raise FileNotFoundError(f"{path} is missing: not an exploration run folder")
    cfg = OmegaConf.load(path)
    seed = OmegaConf.select(cfg, "common.seed")
    return dict(method=method_label(bool(OmegaConf.select(cfg, "intrinsic_reward.enabled", default=False)),
                                    OmegaConf.select(cfg, "intrinsic_reward.method", default=None)),
                domain=OmegaConf.select(cfg, "env.train.domain_name"), exploration_seed=seed)


def check_evaluation_config(c: Any) -> None:
    seeds = list(c.seeds)
    if not seeds or not all(isinstance(s, int) for s in seeds) or len(set(seeds)) != len(seeds):
        raise ValueError(f"evaluation.seeds must be a non-empty list of distinct integers, got {seeds}")
    if not isinstance(c.max_steps, int) or c.max_steps < 1:
        raise ValueError(f"evaluation.max_steps must be a positive integer, got {c.max_steps!r}")


def deterministic_policy_description(actor_distribution: str) -> str:
    return {"tanh_gaussian": "deterministic (tanh(mu))", "beta": "deterministic (Beta mean)"}[actor_distribution]


def evaluation_path(output_dir: Path, task_name: str, seeds: List[int], max_steps: int,
                    policy_state: Dict[str, Any]) -> Path:
    """evaluations/eval_<task>_policy-<weights hash>_protocol-<seeds and step limit hash>.json"""
    policy = state_sha256({k: v for k, v in policy_state.items() if torch.is_tensor(v)})[:12]
    return Path(output_dir) / "evaluations" / f"eval_{task_name}_policy-{policy}_protocol-{protocol_id(seeds, max_steps)}.json"


def protocol_id(seeds: List[int], max_steps: int) -> str:
    """The evaluation protocol identifier: a hash of the seeds and the step limit (12 hex digits)."""
    return hashlib.sha256(repr(([int(s) for s in seeds], int(max_steps))).encode()).hexdigest()[:12]


def milestone_name(num_steps: int) -> str:
    return f"steps_{num_steps:06d}"


def downstream_output_dir(milestone_path: Path, task_name: str, seed: int, member_index: int = 0) -> Path:
    """Reserved: <run>/downstream/<milestone>/<task>/seed_<seed>/ (not created here); an
    ensemble member other than 0 adds `_member_<k>`."""
    milestone_path = Path(milestone_path)
    name = f"seed_{seed}" + (f"_member_{member_index}" if member_index else "")
    return milestone_path.parent.parent / "downstream" / milestone_path.name / task_name / name


def shared_run_path(milestone_path: Path, task_name: str, seed: int, member_index: int = 0) -> Path:
    """<run>/downstream/wandb_<task>_seed_<seed>[_member_<k>].json: the W&B run shared by every milestone of
    the exploration run adapted for one task, adaptation seed and member (identity and publication state)."""
    name = f"wandb_{task_name}_seed_{seed}" + (f"_member_{member_index}" if member_index else "")
    return Path(milestone_path).parent.parent / "downstream" / f"{name}.json"


def member_file(member_index: int) -> str:
    return DENOISER_FILE if member_index == 0 else f"denoiser_member_{member_index}.pt"


def state_sha256(state_dict: Dict[str, torch.Tensor]) -> str:
    """Content hash of tensors, independent of torch.save's file format."""
    digest = hashlib.sha256()
    for key, value in state_dict.items():
        digest.update(key.encode())
        digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def physical_state_manifest(recording: bool) -> Dict[str, Any]:
    """Manifest entry describing the analysis-only physical states of the milestone's training
    episodes (data.episode.PHYSICAL_STATE_KEYS). recording: the run's collection.train setting
    when the milestone was published; episodes from other settings say so themselves."""
    return dict(
        recording=recording,
        episode_field="physical_state",
        keys=dict(qpos="(T, nq) float64", qvel="(T, nv) float64", recorded="(T,) bool",
                  final_qpos="(nq,) float64, ended episodes only", final_qvel="(nv,) float64, ended episodes only"),
        alignment="row t is the MuJoCo state at which obs[t] was rendered (before act[t]); final_* belong to "
                  "info['final_observation']; dataset.pt successor_physical_states belong to successor_observations",
        missing="episode field absent (None): not recorded; recorded=False rows are NaN; no successor entry: unknown",
        use="analysis only; not read by exploration training, intrinsic scoring or downstream adaptation",
    )


def publish_milestone(
    milestones_dir: Path,
    name: str,
    denoiser_state: Dict[str, torch.Tensor],
    dataset: Dataset,
    successor_observations: Dict[int, torch.Tensor],
    manifest: Dict[str, Any],
    member_states: Optional[List[Dict[str, torch.Tensor]]] = None,
    successor_physical_states: Optional[Dict[int, Dict[str, torch.Tensor]]] = None,
) -> Path:
    """Publish a complete milestone folder atomically (written aside, then renamed). An
    existing milestone is kept if it holds the same training state, otherwise rejected.
    member_states: ensemble members 1..K-1 (member 0 is denoiser_state).
    successor_physical_states: analysis only, the physical state of successor observations."""
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
    saved = dict(
        schema_version=SCHEMA_VERSION,
        directory=Path(os.path.relpath(dataset.directory, final)).as_posix(),
        index=dataset.state_dict(),
        files=files,
        # Observation after the last transition of each unfinished episode at the cut.
        successor_observations={k: v.cpu() for k, v in successor_observations.items()},
    )
    if successor_physical_states:  # absent when nothing was recorded: the file is as before
        unknown = set(successor_physical_states) - set(successor_observations)
        if unknown:
            raise ValueError(f"Successor physical states for episodes {sorted(unknown)} without a successor observation")
        saved["successor_physical_states"] = {k: {n: v.cpu().clone() for n, v in s.items()}
                                              for k, s in successor_physical_states.items()}
    torch.save(saved, tmp / DATASET_FILE)
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
    write_json(tmp / MANIFEST_FILE, manifest)
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
    manifest = read_json(path, "milestone manifest")
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
            except (OSError, ValueError) as error:
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
    # Analysis only: {"qpos", "qvel"} of successor observations whose state was recorded (none in
    # milestones published before recording existed). Downstream training never reads it.
    successor_physical_states: Dict[int, Dict[str, torch.Tensor]] = field(default_factory=dict)


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
                     saved["successor_observations"], member_index, saved.get("successor_physical_states", {}))
