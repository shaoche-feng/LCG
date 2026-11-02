"""Results of completed downstream outputs (CPU only). Nothing is trained or evaluated; source files are only read.

downstream.mode=collect: one exploration run's completed downstream outputs for one task, adaptation seed and
denoiser member -> <run>/result/<task>/seed_<s>[_member_<k>]/result.csv (one row per milestone).
downstream.mode=compare: the result.csv of every method's run of one domain and exploration seed -> comparison.csv,
analysis.json and plots.

This module imports neither trainer: with MUJOCO_GL=egl, importing dm_control opens the GPU devices.
"""
import csv
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
from typing import Any, Dict, List, Optional, Tuple

import hydra
import numpy as np
from omegaconf import DictConfig, OmegaConf
import torch

from data.milestone import (RETURN_CONVENTION, check_evaluation_config, deterministic_policy_description,
                            discover_milestones, downstream_output_dir, downstream_summary, evaluation_path,
                            exploration_identity, method_label, read_json, read_manifest, write_json)
from utils import file_sha256, say


OmegaConf.register_new_resolver("eval", eval, replace=True)  # replace: importable after another registration (tests)


@hydra.main(config_path="../config", config_name="downstream", version_base="1.3")
def main(cfg: DictConfig) -> None:
    mode = cfg.downstream.mode
    if mode not in ("collect", "compare"):
        raise ValueError(f"src/downstream_analysis.py takes downstream.mode=collect or compare, got {mode!r}. "
                         "Training, resuming and evaluating use:\n"
                         f"  python src/downstream_main.py downstream.mode={mode if mode in ('resume', 'evaluate') else 'train'} "
                         "initialization.milestone_path=<run>/milestones/steps_N downstream.task_name=<task> "
                         "downstream.seed=<adaptation seed>\n"
                         "Results use:\n"
                         "  python src/downstream_analysis.py downstream.mode=collect analysis.run_dir=<exploration run> "
                         "downstream.task_name=<task> downstream.seed=<adaptation seed>\n"
                         "  python src/downstream_analysis.py downstream.mode=compare analysis.domain=<domain> "
                         "analysis.exploration_seed=<seed> downstream.task_name=<task> downstream.seed=<adaptation seed>")
    analyze(cfg)


def analyze(cfg: DictConfig) -> None:
    """collect: one exploration run's result.csv; compare: every method's results of one domain and seed."""
    a, task, seed = cfg.analysis, cfg.downstream.task_name, cfg.downstream.seed
    member = OmegaConf.select(cfg, "initialization.member_index", default=0)
    if cfg.downstream.mode == "collect":
        if a.run_dir is None:
            raise ValueError("downstream.mode=collect needs analysis.run_dir=<exploration run folder>")
        path, rows, missing = collect_results(Path(a.run_dir), task, seed, member, a.protocol)
        for entry in missing:
            say(f"Missing: {entry['milestone']} ({entry['train_env_steps']:,} training env steps): {entry['reason']}")
        if path is None:
            raise ValueError(f"No completed downstream results in {a.run_dir} for {task}, adaptation seed {seed}, "
                             f"member {member}")
        say(f"Result: {len(rows)} milestones ({rows[0]['method']}, {rows[0]['domain']}, exploration seed "
            f"{rows[0]['exploration_seed']}; {task}, adaptation seed {seed}); {len(missing)} missing")
        say(f"Result saved: > {path.resolve()}")
        return
    if a.domain is None or a.exploration_seed is None:
        raise ValueError("downstream.mode=compare needs analysis.domain and analysis.exploration_seed")
    out = compare_methods(Path(a.output_root), a.domain, a.exploration_seed, task, seed, member, a.protocol)
    analysis = json.loads((out / "analysis.json").read_text(encoding="utf-8"))
    say(f"Comparison: {', '.join(analysis['methods'])} | {len(analysis['missing'])} missing results")
    for entry in analysis["missing"]:
        say(f"Missing: {entry}")
    say(f"Comparison saved: > {out.resolve()}")


THRESHOLDS = (200, 400, 500, 600)  # fractions of evaluation episodes with return strictly above each
METHOD_COLORS = {"uniform": "#7f7f7f", "lcg": "#d62728", "full-ensemble": "#1f77b4", "laplace-ensemble": "#9467bd",
                 "plan2explore": "#2ca02c"}
# Marker and line style per method too, so curves that coincide (e.g. all at 0) stay distinguishable.
METHOD_STYLES = {"uniform": ("o", "-"), "lcg": ("s", "--"), "full-ensemble": ("^", "-."),
                 "laplace-ensemble": ("D", ":"), "plan2explore": ("v", "--")}
RESULT_COLUMNS = ["method", "domain", "exploration_seed", "task", "adaptation_seed", "member_index", "milestone",
                  "train_env_steps", "num_eval_episodes", "return_mean", "return_median", "return_std", "return_min",
                  "return_max", *(f"frac_over_{t}" for t in THRESHOLDS), "protocol", "evaluation_max_steps",
                  "compatibility_sha256"]  # then return_seed_<s> per evaluation seed, then source_evaluation_file
# What must match for results to be compared (compatibility_settings), as effective (resolved) values: the
# environment the policy was trained and evaluated in (domain, tasks, action repeat, time limit, camera, image
# size), the controller and its imagination, both training phases, and the evaluation's environment, policy and
# return convention. Not compared: paths, run identity (seeds and task are the selection itself), checkpoint
# cadence, W&B and the diagnostics-only switches below. The column was downstream_settings_sha256 before the
# environment and evaluation were included; tables with that column must be collected again.
COMPARED_SETTINGS = ("env.train", "agent", "world_model_env", "rew_end_model.training", "rew_end_model.optimizer",
                     "actor_critic.training")
NOT_COMPARED = ("agent.actor_critic.debug_metrics", "agent.actor_critic.log_diagnostics")
COMPARED_REPORT_FIELDS = ("domain_name", "task_name", "action_repeat", "policy", "return_convention")


def result_dir(run_dir: Path, task_name: str, seed: int, member_index: int = 0) -> Path:
    """<run>/result/<task>/seed_<s>[_member_<k>] (member suffix as for downstream outputs)."""
    return Path(run_dir) / "result" / task_name / (f"seed_{seed}" + (f"_member_{member_index}" if member_index else ""))


def protocol_id(path: Path) -> str:
    return Path(path).stem.rsplit("_protocol-", 1)[1]


def compatibility_settings(output_dir: Path, report: Dict[str, Any]) -> Dict[str, Any]:
    """The settings that make two results comparable (COMPARED_SETTINGS), resolved, from the output's saved
    config.yaml, and the evaluation report's environment, policy and return convention."""
    saved = OmegaConf.load(Path(output_dir) / "config.yaml")
    settings = {}
    for key in COMPARED_SETTINGS:
        value = OmegaConf.select(saved, key)
        settings[key] = OmegaConf.to_container(value, resolve=True) if OmegaConf.is_config(value) else value
    for dotted in NOT_COMPARED:
        parent, _, leaf = dotted.rpartition(".")
        node = settings.get(parent.split(".", 1)[0], {})
        for part in parent.split(".")[1:]:
            node = node.get(part, {}) if isinstance(node, dict) else {}
        if isinstance(node, dict):
            node.pop(leaf, None)
    settings["evaluation"] = {k: report.get(k) for k in COMPARED_REPORT_FIELDS}
    return settings


def compatibility_sha256(settings: Dict[str, Any]) -> str:
    return hashlib.sha256(json.dumps(settings, sort_keys=True, default=str).encode()).hexdigest()


def flatten(d: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
    out = {}
    for key, value in d.items():
        if isinstance(value, dict) and value:
            out.update(flatten(value, f"{prefix}{key}."))
        else:
            out[f"{prefix}{key}"] = value
    return out


def settings_difference(a: Dict[str, Any], b: Dict[str, Any]) -> str:
    fa, fb = flatten(a), flatten(b)
    keys = sorted(k for k in set(fa) | set(fb) if fa.get(k) != fb.get(k))
    return "; ".join(f"{k}: {fa.get(k)!r} vs {fb.get(k)!r}" for k in keys[:8]) + (" ..." if len(keys) > 8 else "")


def check_compatible(sources: List[Tuple[str, Dict[str, Any]]], what: str) -> None:
    """(label, compatibility settings) of every result: an error naming the differing settings unless all match."""
    first_label, first = sources[0]
    for label, settings in sources[1:]:
        if compatibility_sha256(settings) != compatibility_sha256(first):
            raise ValueError(f"{what}: not comparable: {first_label} and {label} differ in "
                             f"{settings_difference(first, settings)}")


def result_row(report_path: Path, report: Dict[str, Any], identity: Dict[str, Any], summary: Dict[str, Any],
               output_dir: Path, run_dir: Path) -> Dict[str, Any]:
    """One milestone's row: statistics recomputed from the episode returns (the saved aggregates are checked
    against them), strict threshold fractions and every evaluation seed's return."""
    saved = OmegaConf.load(output_dir / "config.yaml")
    milestone_path = run_dir / "milestones" / output_dir.parent.parent.name
    manifest = read_manifest(milestone_path)
    task, seed = summary["task_name"], summary["seed"]
    member = summary["milestone"].get("member_index", 0)
    if summary["milestone"]["identity"] != manifest["identity"] or \
            summary["milestone"]["collected_steps"] != manifest["collected_steps"] or \
            (output_dir / summary["milestone"]["path"]).resolve() != milestone_path.resolve():
        raise ValueError(f"{output_dir}: summary does not match its milestone")
    recorded = manifest["exploration"]
    if method_label(recorded["intrinsic_reward_enabled"], recorded.get("method", "lcg")) != identity["method"]:
        raise ValueError(f"{milestone_path}: exploration method differs from the saved run configuration")
    if manifest["domain"] != identity["domain"] or task not in manifest["task_names"]:
        raise ValueError(f"{milestone_path}: domain or task identity differs from the selected result")
    exploration_cfg = OmegaConf.load(run_dir / "config/exploration.yaml")
    for key, recorded_value in (("env.train", manifest["model"]["env"]),
                                ("agent.denoiser", manifest["model"]["denoiser"])):
        current_value = OmegaConf.to_container(OmegaConf.select(exploration_cfg, key), resolve=True)
        if current_value != recorded_value:
            raise ValueError(f"{milestone_path}: exploration configuration differs in "
                             f"{settings_difference({key: recorded_value}, {key: current_value})}")
    if (saved.downstream.task_name, saved.downstream.seed,
            OmegaConf.select(saved, "initialization.member_index", default=0)) != (task, seed, member):
        raise ValueError(f"{output_dir}: saved task, adaptation seed or member differs from its summary")
    if OmegaConf.to_container(saved.env.train, resolve=True) != manifest["model"]["env"]:
        difference = settings_difference({"env.train": manifest["model"]["env"]},
                                          {"env.train": OmegaConf.to_container(saved.env.train, resolve=True)})
        raise ValueError(f"not comparable: {output_dir} and its milestone differ in {difference}")
    if OmegaConf.to_container(saved.agent.denoiser, resolve=True) != manifest["model"]["denoiser"]:
        raise ValueError(f"{output_dir}: agent.denoiser differs from its milestone")
    expected_member = (manifest["identity"].get("member_sha256") or [manifest["identity"]["denoiser_sha256"]])
    if not 0 <= member < len(expected_member) or summary["milestone"]["member_sha256"] != expected_member[member]:
        raise ValueError(f"{output_dir}: denoiser member identity differs from its milestone")
    if report["task_name"] != task or report["domain_name"] != identity["domain"]:
        raise ValueError(f"{report_path}: evaluation task or domain differs from its policy")
    if report["action_repeat"] != saved.env.train.action_repeat or report["return_convention"] != RETURN_CONVENTION:
        difference = settings_difference(dict(evaluation=dict(action_repeat=saved.env.train.action_repeat,
                                                               return_convention=RETURN_CONVENTION)),
                                          dict(evaluation={k: report[k] for k in ("action_repeat", "return_convention")}))
        raise ValueError(f"not comparable: {report_path} and its saved environment differ in {difference}")
    distribution = OmegaConf.select(saved, "agent.actor_critic.actor_distribution", default="beta")
    if report["policy"] != deterministic_policy_description(distribution):
        raise ValueError(f"{report_path}: evaluation.policy differs from its actor distribution; re-evaluate the policy")
    seeds = list(report["seeds"])
    check_evaluation_config(OmegaConf.create(dict(seeds=seeds, max_steps=report["max_steps"])))
    episodes = report["episodes"]
    if [int(e["seed"]) for e in episodes] != seeds:
        raise ValueError(f"{report_path}: episode seeds {[e['seed'] for e in episodes]} differ from the protocol {seeds}")
    expected = hashlib.sha256(repr((seeds, int(report["max_steps"]))).encode()).hexdigest()[:12]
    if expected != protocol_id(report_path):
        raise ValueError(f"{report_path}: its seeds and step limit do not match the protocol in its name")
    policy_state = torch.load(output_dir / "actor_critic.pt", map_location="cpu", weights_only=False)
    expected_path = evaluation_path(output_dir, task, seeds, report["max_steps"], policy_state)
    policy_hash = expected_path.stem.split("_policy-", 1)[1].split("_protocol-", 1)[0]
    if report_path.resolve() != expected_path.resolve() or report.get("policy_sha256_prefix") != policy_hash:
        raise ValueError(f"{report_path}: report policy identity does not match actor_critic.pt")
    returns = np.array([float(e["episode_return"]) for e in episodes])
    if not np.isfinite(returns).all():
        raise ValueError(f"{report_path}: non-finite episode returns")
    stats = dict(return_mean=float(returns.mean()), return_median=float(np.median(returns)),
                 return_std=float(returns.std()), return_min=float(returns.min()), return_max=float(returns.max()))
    for key, value in stats.items():
        if not np.isclose(value, float(report[key]), rtol=1e-12, atol=1e-12):
            raise ValueError(f"{report_path}: saved {key} {report[key]} differs from the episodes' {value}")
    row = dict(identity, task=summary["task_name"], adaptation_seed=summary["seed"],
               member_index=summary["milestone"].get("member_index", 0), milestone=output_dir.parent.parent.name,
               train_env_steps=summary["milestone"]["collected_steps"], num_eval_episodes=len(returns), **stats,
               **{f"frac_over_{t}": float((returns > t).mean()) for t in THRESHOLDS},
               protocol=protocol_id(report_path), evaluation_max_steps=int(report["max_steps"]),
               compatibility_sha256=compatibility_sha256(compatibility_settings(output_dir, report)))
    row.update({f"return_seed_{s}": float(r) for s, r in zip(seeds, returns)})
    row["source_evaluation_file"] = Path(os.path.relpath(report_path, run_dir)).as_posix()
    return row


def collect_results(run_dir: Path, task_name: str, seed: int, member_index: int = 0,
                    protocol: Optional[str] = None, write: bool = True
                    ) -> Tuple[Optional[Path], List[Dict[str, Any]], List[Dict[str, Any]]]:
    """Every milestone of the run, in collected-step order: its completed downstream output's evaluation of
    the one (selected) protocol -> result.csv. Returns (csv path or None, rows, missing). Different
    protocols or downstream settings are never mixed; ambiguity is an error asking for a selection."""
    run_dir = Path(run_dir)
    identity = exploration_identity(run_dir)
    found, missing = [], []
    for path, manifest in discover_milestones(run_dir / "milestones"):
        output = downstream_output_dir(path, task_name, seed, member_index)
        summary = downstream_summary(output)
        entry = dict(milestone=path.name, train_env_steps=manifest["collected_steps"])
        if summary is None:
            work = output.with_name(f".{output.name}.tmp")
            missing.append(dict(entry, reason="incomplete (work folder " + work.name + " only)" if work.exists()
                                else "no downstream output"))
            continue
        if (summary["task_name"], summary["seed"], summary["milestone"].get("member_index", 0)) != (task_name, seed, member_index):
            raise ValueError(f"{output}: summary task, adaptation seed or member differs from the selection")
        if summary["milestone"]["collected_steps"] != manifest["collected_steps"] or \
                summary["milestone"].get("identity") != manifest["identity"]:
            raise ValueError(f"{output}: its summary does not match milestone {path.name}")
        reports = {}  # protocol -> every report of it (a completed output holds one policy)
        for p in sorted((output / "evaluations").glob("eval_*.json")):
            reports.setdefault(protocol_id(p), []).append(p)
        found.append((entry, output, summary, reports))
    protocols = sorted({p for *_, reports in found for p in reports})
    if protocol is None and len(protocols) > 1:
        raise ValueError(f"{run_dir}: several evaluation protocols ({', '.join(protocols)}); select one with "
                         "analysis.protocol=<id> (the eval_*_protocol-<id>.json suffix)")
    chosen = protocol if protocol is not None else (protocols[0] if protocols else None)
    selected = []
    for entry, output, summary, reports in found:
        if chosen not in reports:
            missing.append(dict(entry, reason=f"completed, but no evaluation with protocol {chosen}"))
            continue
        if len(reports[chosen]) > 1:  # never chosen by name, time or return
            raise ValueError(f"{output}: {len(reports[chosen])} evaluation reports of different policies with protocol "
                             f"{chosen}: {', '.join(p.name for p in reports[chosen])}; a completed output has one final "
                             "policy, so this is ambiguous. Nothing was collected.")
        selected.append((entry, output, summary, reports[chosen][0]))
    rows, sources = [], []
    for entry, output, summary, path in selected:
        report = read_json(path, "evaluation report")
        rows.append(result_row(path, report, identity, summary, output, run_dir))
        sources.append((Path(os.path.relpath(path, run_dir)).as_posix(), compatibility_settings(output, report)))
    if rows:
        check_compatible(sources, f"{run_dir}")
    if not rows:
        return None, rows, missing
    rows.sort(key=lambda r: r["train_env_steps"])
    seed_columns = [k for k in rows[0] if k.startswith("return_seed_")]
    path = result_dir(run_dir, task_name, seed, member_index) / "result.csv"
    if write:
        write_csv(path, rows, RESULT_COLUMNS + seed_columns + ["source_evaluation_file"])
    return path, rows, missing


def write_csv(path: Path, rows: List[Dict[str, Any]], columns: List[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f"{path.name}.tmp")
    with open(tmp, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        writer.writerows(rows)
    os.replace(tmp, path)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    with open(path, encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))
    for row in rows:
        for key, value in row.items():
            if key.startswith(("return_", "frac_over_")):
                row[key] = float(value)
            elif key in ("exploration_seed", "adaptation_seed", "member_index", "train_env_steps", "num_eval_episodes",
                         "evaluation_max_steps"):
                row[key] = int(value)
    return rows


def compare_methods(output_root: Path, domain: str, exploration_seed: int, task_name: str, seed: int,
                    member_index: int = 0, protocol: Optional[str] = None) -> Path:
    """Every method's run outputs/<method>/<domain>/seed_<exploration seed> with a result.csv for the task,
    adaptation seed and member: comparison.csv, analysis.json, task_return.{png,pdf} and
    fraction_over_thresholds.{png,pdf}. Methods come from each run's saved configuration. Rows of different
    evaluation protocols or downstream settings are not compared; missing milestones are recorded, never
    interpolated."""
    output_root = Path(output_root)
    comparison_dir = output_root / "comparisons" / domain / f"exploration_seed_{exploration_seed}" / task_name / \
        f"adaptation_seed_{seed}"
    runs, missing, inputs, rows = {}, [], [], []
    for run_dir in sorted(output_root.glob(f"*/{domain}/seed_{exploration_seed}")):
        if not (run_dir / "config" / "exploration.yaml").is_file():
            continue
        identity = exploration_identity(run_dir)
        if (identity["domain"], identity["exploration_seed"]) != (domain, exploration_seed):
            raise ValueError(f"{run_dir}: its configuration records {identity['domain']}, seed "
                             f"{identity['exploration_seed']}; not {domain}, seed {exploration_seed}")
        if identity["method"] in runs:
            raise ValueError(f"Two runs record method {identity['method']}: {runs[identity['method']]} and {run_dir}")
        runs[identity["method"]] = run_dir
        csv_path = result_dir(run_dir, task_name, seed, member_index) / "result.csv"
        if not csv_path.is_file():
            missing.append(dict(method=identity["method"], run=str(run_dir), reason=f"no {csv_path.relative_to(run_dir)} "
                                "(run downstream_analysis.py downstream.mode=collect)"))
            continue
        method_rows = read_csv(csv_path)
        if method_rows and "compatibility_sha256" not in method_rows[0]:
            raise ValueError(f"{csv_path} was collected before the compatibility fingerprint covered the environment and "
                             "evaluation settings; collect it again (downstream_analysis.py downstream.mode=collect)")
        for row in method_rows:
            if (row["method"], row["domain"], row["exploration_seed"], row["task"], row["adaptation_seed"],
                    row["member_index"]) != (identity["method"], domain, exploration_seed, task_name, seed, member_index):
                raise ValueError(f"{csv_path}: a row of {row['method']}/{row['task']}/seed {row['adaptation_seed']}/"
                                 f"member {row['member_index']} does not belong to this selection")
        steps = [r["train_env_steps"] for r in method_rows]
        if len(set(steps)) != len(steps):
            raise ValueError(f"{csv_path}: duplicate milestones {sorted(s for s in steps if steps.count(s) > 1)}")
        # Re-run collection read-only against current configs, manifests, policy weights and
        # reports. Cached fingerprints and statistics are evidence to verify, not authority.
        try:
            table_protocols = {r["protocol"] for r in method_rows}
            if len(table_protocols) != 1:
                raise ValueError("the table must contain exactly one evaluation protocol")
            _, current_rows, _ = collect_results(run_dir, task_name, seed, member_index,
                                                  next(iter(table_protocols)), write=False)
            current = {r["milestone"]: r for r in current_rows}
            if len(current) != len(method_rows):
                raise ValueError("the completed evaluation set changed after collection")
            for row in method_rows:
                fresh = current.get(row["milestone"])
                if fresh != row:
                    differences = settings_difference(row, fresh or {})
                    raise ValueError(f"{row['milestone']} differs from its current sources: {differences}")
        except (OSError, ValueError, KeyError, TypeError, RuntimeError) as error:
            raise ValueError(f"{csv_path}: stale or inconsistent result table ({error}); recollect with "
                             "downstream_analysis.py downstream.mode=collect analysis.run_dir=<run> "
                             "and the selected task, seed, member and protocol") from error
        evaluations = [run_dir / r["source_evaluation_file"] for r in method_rows]
        inputs.append(dict(method=identity["method"], run=str(run_dir.resolve()), result_csv=str(csv_path.resolve()),
                           result_csv_sha256=file_sha256(csv_path), evaluations={str(p.resolve()): file_sha256(p)
                                                                                for p in evaluations}))
        rows += method_rows
    if not rows:
        raise ValueError(f"No result.csv under {output_root}/*/{domain}/seed_{exploration_seed} for {task_name}, "
                         f"adaptation seed {seed}, member {member_index}; missing: {missing}")
    protocols = sorted({r["protocol"] for r in rows})
    if protocol is not None:
        rows = [r for r in rows if r["protocol"] == protocol]
        missing += [dict(method=m, reason=f"no rows with protocol {protocol}") for m in runs
                    if not any(r["method"] == m for r in rows)]
    elif len(protocols) > 1:
        raise ValueError(f"The methods' results use different evaluation protocols ({', '.join(protocols)}); "
                         "select one with analysis.protocol=<id>")
    if not rows:
        raise ValueError(f"No results for selected protocol {protocol}; collect that protocol first")
    if len({r["compatibility_sha256"] for r in rows}) > 1:  # name the differing settings
        sources = []
        for r in rows:
            report_path = runs[r["method"]] / r["source_evaluation_file"]
            sources.append((f"{r['method']} {r['milestone']}", compatibility_settings(
                report_path.parent.parent, read_json(report_path, "evaluation report"))))
        check_compatible(sources, "The methods' results")
        raise ValueError("The methods' results have different compatibility fingerprints; collect them again")
    seed_columns = sorted({k for r in rows for k in r if k.startswith("return_seed_")})
    if any(set(k for k in r if k.startswith("return_seed_")) != set(seed_columns) for r in rows):
        raise ValueError("Rows with different evaluation seeds; not comparing them")
    methods = [m for m in METHOD_COLORS if any(r["method"] == m for r in rows)]
    # Every known milestone of the selected methods and every observed row: a milestone that no method has a
    # result for still breaks the curves instead of being skipped.
    known = {m["collected_steps"] for method in methods for _, m in discover_milestones_safe(runs[method] / "milestones")}
    grid = sorted(known | {r["train_env_steps"] for r in rows})
    for method in methods:
        have = {r["train_env_steps"] for r in rows if r["method"] == method}
        for steps in sorted(set(grid) - have):
            missing.append(dict(method=method, train_env_steps=steps, reason="no result row"))
    rows.sort(key=lambda r: (methods.index(r["method"]), r["train_env_steps"]))
    comparison_dir.mkdir(parents=True, exist_ok=True)
    write_csv(comparison_dir / "comparison.csv", rows, RESULT_COLUMNS + seed_columns + ["source_evaluation_file"])
    first = rows[0]
    eval_seeds = [int(k.rsplit("_", 1)[1]) for k in seed_columns]
    analysis = dict(
        selection=dict(output_root=str(output_root.resolve()), domain=domain, exploration_seed=exploration_seed,
                       task=task_name, adaptation_seed=seed, member_index=member_index, protocol=protocol),
        evaluation_protocol=dict(protocol=first["protocol"], seeds=eval_seeds, max_steps=first["evaluation_max_steps"],
                                 num_eval_episodes=first["num_eval_episodes"]),
        compatibility_sha256=first["compatibility_sha256"], plot_grid=grid, methods=methods, inputs=inputs,
        missing=missing,
        note="One exploration seed and one adaptation seed per method. return_* and frac_over_* summarize the "
             "evaluation episodes (seeds) of one adapted policy per milestone; their spread is not uncertainty over "
             "training seeds. Missing milestones are not interpolated.",
        outputs=["comparison.csv", "task_return.png", "task_return.pdf", "fraction_over_thresholds.png",
                 "fraction_over_thresholds.pdf"])
    write_json(comparison_dir / "analysis.json", analysis)
    plot_comparison(comparison_plot_spec(rows, methods, grid, analysis, comparison_dir))
    return comparison_dir


def discover_milestones_safe(milestones_dir: Path) -> List[Tuple[Path, Dict[str, Any]]]:
    return discover_milestones(milestones_dir) if Path(milestones_dir).is_dir() else []


# The figures, drawn in a separate Python process that loads matplotlib but not torch (as utils.probe_egl's
# EGL probe runs in its own process): in this project's Windows conda environment torch's bundled OpenMP
# runtime and the one matplotlib's numpy work loads abort a process holding both ("OMP: Error #15").
PLOT_PROGRAM = r"""
import json, math, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
spec = json.loads(sys.stdin.read())
grid, out = spec["grid"], spec["out"]
nan = lambda values: [math.nan if v is None else v for v in values]
fig, ax = plt.subplots(figsize=(7.5, 4.5))
for method in spec["methods"]:
    marker, line = spec["styles"][method]
    ax.plot(grid, nan(spec["series"][method]["return_mean"]), marker=marker, linestyle=line, color=spec["colors"][method],
            label=method, alpha=0.85)
ax.set_xlabel("Exploration training environment steps")
ax.set_ylabel("Mean downstream task return")
ax.set_title(spec["title"] + "\n" + spec["subtitle"], fontsize=9)
ax.legend()
ax.grid(alpha=0.3)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(f"{out}/task_return.{suffix}", dpi=150)
plt.close(fig)
fig, axes = plt.subplots(2, 2, figsize=(9, 6.5), sharex=True, sharey=True)
for ax, t in zip(axes.flat, spec["thresholds"]):
    for method in spec["methods"]:
        marker, line = spec["styles"][method]
        ax.plot(grid, nan(spec["series"][method][f"frac_over_{t}"]), marker=marker, linestyle=line,
                color=spec["colors"][method], label=method, alpha=0.85)
    ax.set_title(f"Fraction of evaluation episodes with return > {t}", fontsize=9)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(alpha=0.3)
for ax in axes[1]:
    ax.set_xlabel("Exploration training environment steps")
for ax in axes[:, 0]:
    ax.set_ylabel("Fraction")
axes[0, 0].legend()
fig.suptitle(spec["subtitle"], fontsize=9)
fig.tight_layout()
for suffix in ("png", "pdf"):
    fig.savefig(f"{out}/fraction_over_thresholds.{suffix}", dpi=150)
"""


def comparison_plot_spec(rows: List[Dict[str, Any]], methods: List[str], grid: List[int], analysis: Dict[str, Any],
                         out: Path) -> Dict[str, Any]:
    """The data both figures plot: per method and value, one entry per grid step, None where the method has no
    result (a gap, never interpolated)."""
    s, p = analysis["selection"], analysis["evaluation_protocol"]
    seeds = p["seeds"]
    seed_text = f"{seeds[0]}-{seeds[-1]}" if seeds == list(range(seeds[0], seeds[-1] + 1)) else ", ".join(map(str, seeds))
    series = {}
    for method in methods:
        by_steps = {r["train_env_steps"]: r for r in rows if r["method"] == method}
        series[method] = {key: [by_steps[x][key] if x in by_steps else None for x in grid]
                          for key in ["return_mean", *(f"frac_over_{t}" for t in THRESHOLDS)]}
    spec = dict(grid=grid, out=str(Path(out).resolve()), methods=methods, colors=METHOD_COLORS, styles=METHOD_STYLES,
                series=series,
                thresholds=list(THRESHOLDS), title=f"Downstream {s['task']} return",
                subtitle=(f"{s['domain']} {s['task']} | exploration seed {s['exploration_seed']} | adaptation seed "
                          f"{s['adaptation_seed']}\n{p['num_eval_episodes']} evaluation episodes per point (seeds "
                          f"{seed_text}, <= {p['max_steps']} steps); no interpolation across missing milestones"))
    return spec


def plot_comparison(spec: Dict[str, Any]) -> None:
    """task_return: mean return per milestone, one curve per method; fraction_over_thresholds: four panels.
    Drawn by PLOT_PROGRAM from comparison_plot_spec."""
    result = subprocess.run([sys.executable, "-c", PLOT_PROGRAM], input=json.dumps(spec), text=True, capture_output=True)
    if result.returncode != 0:
        raise RuntimeError(f"Plotting failed:\n{result.stderr[-3000:]}")


if __name__ == "__main__":
    main()
