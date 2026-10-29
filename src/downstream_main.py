import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Callable, List, Optional, Tuple

import hydra
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf, open_dict
import torch

from data.milestone import discover_milestones, downstream_output_dir, load_milestone
from downstream_trainer import DownstreamTrainer, evaluate_saved_output
from utils import setup_visible_cuda_devices


OmegaConf.register_new_resolver("eval", eval, replace=True)  # replace: importable after another registration (tests)

# Not passed on to a child run: the launcher-level milestone source and the mode (set per command).
LAUNCHER_KEYS = ("initialization.milestones_dir", "initialization.milestone_path", "downstream.mode")


@hydra.main(config_path="../config", config_name="downstream", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mode = cfg.downstream.mode
    if mode not in ("train", "resume", "evaluate"):
        raise ValueError(f"downstream.mode must be train, resume or evaluate, got {mode!r}")
    milestones_dir = pop_milestones_dir(cfg)
    if milestones_dir is not None:
        code = run_milestones(cfg, Path(milestones_dir), child_command_base(HydraConfig.get()))
        if code:
            raise SystemExit(code)
        return
    setup_visible_cuda_devices(cfg.common.devices)
    if mode == "evaluate":
        evaluate_saved_output(cfg)
    else:
        DownstreamTrainer(cfg, resume=mode == "resume").run()


def pop_milestones_dir(cfg: DictConfig) -> Optional[str]:
    """Exactly one of initialization.milestone_path (one milestone) and initialization.milestones_dir
    (every milestone in a folder). The folder setting is removed from cfg, so a trainer always
    receives a normal single-milestone configuration and saved configs (resume checks) are unchanged."""
    init = cfg.initialization
    folder = init.get("milestones_dir")
    single = not OmegaConf.is_missing(init, "milestone_path") and init.milestone_path is not None
    if (folder is not None) == single:
        raise ValueError("Give exactly one of initialization.milestone_path (one milestone) and "
                         "initialization.milestones_dir (every milestone in a folder)"
                         + (", not both" if single else ""))
    with open_dict(cfg):
        init.pop("milestones_dir", None)
    return folder


# ---------------------------------------------------------------------- folder mode

def child_command_base(hydra_cfg: DictConfig) -> List[str]:
    """The parent's own command for a child run, without the milestone and mode: same Python
    executable and entry point, same configuration source (primary config path, every --config-dir,
    config name, hydra.* overrides) and the same overrides, minus LAUNCHER_KEYS. The folder setting is
    cleared explicitly, since it may come from a config file rather than from the command line."""
    command = [sys.executable, str(Path(__file__).resolve())]
    for source in hydra_cfg.runtime.config_sources:
        location = str(Path(source.path).resolve()) if source.schema == "file" else f"{source.schema}://{source.path}"
        if source.provider == "main":
            command.append(f"--config-path={location}")
        elif source.provider == "command-line":
            command.append(f"--config-dir={location}")
    command.append(f"--config-name={hydra_cfg.job.config_name}")
    command += [o for o in hydra_cfg.overrides.hydra if o != "hydra.mode=RUN"]  # Hydra adds hydra.mode itself
    command += [o for o in hydra_cfg.overrides.task if o.lstrip("+~").split("=", 1)[0] not in LAUNCHER_KEYS]
    return command + ["++initialization.milestones_dir=null"]


def milestone_command(base: List[str], path: Path, mode: str) -> List[str]:
    """One milestone's single-milestone command: the child run (train), or its recovery (resume, evaluate)."""
    return [*base, f"initialization.milestone_path={hydra_quote(path)}", f"downstream.mode={mode}"]


def run_milestones(cfg: DictConfig, milestones_dir: Path, base: List[str],
                   run: Callable[..., Any] = subprocess.run) -> int:
    """Train one independent downstream run per milestone of the folder, in collection order, one
    after another. Each is a child process of this entry point (milestone_command: same Python
    executable, configuration and settings, its own milestone), so no GPU memory or RNG state carries
    over and every run starts from a fresh reward model and controller. Stops at the first failure,
    deleting nothing; returns the exit status (0: all completed)."""
    mode = cfg.downstream.mode
    if mode != "train":
        raise ValueError(f"initialization.milestones_dir supports only downstream.mode=train, got {mode!r}. "
                         f"{mode} takes one milestone: initialization.milestone_path=<run>/milestones/steps_N")
    task, seed = cfg.downstream.task_name, cfg.downstream.seed
    member = OmegaConf.select(cfg, "initialization.member_index", default=0)
    selected = preflight(milestones_dir, task, seed, member, base)

    print(f"Downstream batch: {len(selected)} milestones from {milestones_dir}")
    print(f"  task {task} | adaptation seed {seed} | member {member} | mode train | sequential, one process per milestone")
    for i, (path, steps, output) in enumerate(selected, start=1):
        print(f"  {i}. {path.name} ({steps:,} collected steps) -> {output}")

    results = []
    for i, (path, steps, output) in enumerate(selected, start=1):
        print(f"\n=== Downstream {i}/{len(selected)}: {path.name} ({steps:,} collected steps) ===", flush=True)
        start = time.time()
        code = run(milestone_command(base, path, "train")).returncode
        results.append((code, time.time() - start))
        if code != 0:
            break

    print(f"\nDownstream batch summary (task {task}, adaptation seed {seed}):")
    for i, (path, steps, output) in enumerate(selected):
        if i >= len(results):
            print(f"  {path.name}: not started")
            continue
        code, seconds = results[i]
        if code == 0:
            print(f"  {path.name}: completed in {seconds / 60:.1f} min -> {output}")
        else:
            print(f"  {path.name}: FAILED (exit code {code}) after {seconds / 60:.1f} min; "
                  f"{failure_state(path, output, base)}")
    failed = results[-1][0] != 0
    print("Stopped at the failure; later milestones were not started." if failed else f"All {len(selected)} completed.")
    return 1 if failed else 0


def failure_state(path: Path, output: Path, base: List[str]) -> str:
    """What a failed child left behind (read only) and the single-milestone command that continues it."""
    work = output.with_name(f".{output.name}.tmp")
    if (output / "summary.yaml").is_file():  # published: training completed, the failure came after it
        return (f"training outputs were published at {output} (training completed; the failure came after it, "
                f"e.g. in the evaluation); evaluate them with:\n      {shell(milestone_command(base, path, 'evaluate'))}")
    if (work / "checkpoints" / "state.pt").is_file():
        return f"resumable work folder {work} kept; resume it with:\n      {shell(milestone_command(base, path, 'resume'))}"
    if work.exists():
        return f"work folder {work} kept (no checkpoint yet: delete it to rerun)"
    return "no work folder was created"


def preflight(milestones_dir: Path, task: str, seed: int, member: int, base: List[str]) -> List[Tuple[Path, int, Path]]:
    """(milestone, collected steps, output folder) in collection order, after checking every
    milestone's files and task labels (load_milestone) and that no output or incomplete attempt
    exists for any of them, so nothing fails on these grounds after training has started."""
    problems, selected = [], []
    for path, manifest in discover_milestones(milestones_dir):
        try:
            load_milestone(path, task, map_location=torch.device("cpu"), member_index=member)
        except (OSError, ValueError) as error:
            problems.append(str(error))
        output = downstream_output_dir(path, task, seed, member)
        work = output.with_name(f".{output.name}.tmp")
        if output.exists():
            problems.append(f"{path.name}: downstream output {output} already exists; not overwriting it")
        if work.exists():
            problems.append(f"{path.name}: incomplete downstream output {work} from an earlier attempt; delete it to "
                            f"rerun, or resume it with: {shell(milestone_command(base, path, 'resume'))}")
        selected.append((path, manifest["collected_steps"], output))
    if problems:
        raise ValueError(f"Nothing was started. {len(problems)} problem(s) in {milestones_dir}:\n  " + "\n  ".join(problems))
    return selected


def hydra_quote(path: Path) -> str:
    """A path as a Hydra override value: as is when plain, else quoted (spaces etc. stay literal)."""
    text = Path(path).resolve().as_posix()
    if re.fullmatch(r"[\w./:-]+", text):
        return text
    return "'" + text.replace("\\", "\\\\").replace("'", "\\'") + "'"


def shell(command: List[str]) -> str:
    """A command as it would be typed in this platform's shell."""
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


if __name__ == "__main__":
    main()
