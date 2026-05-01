#!/usr/bin/env python3
"""Generate and optionally execute a generic continual-extension suite."""

from __future__ import annotations

import argparse
from datetime import datetime
import json
import shlex
import subprocess
import sys
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, List, Mapping, Optional

PACKAGE_ROOT = find_repo_root(__file__)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from continual_suite_lib import (
    VALID_TASKS,
    ensure_dir,
    collect_checkpoint_steps,
    expand_stage_plans,
    load_jsonl,
    load_suite_config,
    load_yaml,
    parse_stage_selector,
    resolve_seed_adapter_path,
    save_stage_manifest,
    select_mtl_checkpoint_record,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build and optionally run a continual experiment suite.")
    parser.add_argument("--suite-config", required=True, help="Path to continual suite YAML config.")
    parser.add_argument("--mode", default="all", choices=("merge", "mtl", "all"), help="Which backend(s) to prepare or run.")
    parser.add_argument("--path", dest="path_id", default="all", help="Path id to process, or 'all'.")
    parser.add_argument("--stages", default="all", help="Comma-separated stage ids, or 'all'.")
    parser.add_argument("--execute", action="store_true", help="Execute backend commands after generating configs/manifests.")
    return parser.parse_args()


def _dump_yaml(path: Path, payload: Mapping[str, Any]) -> None:
    import yaml

    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yaml.safe_dump(dict(payload), handle, sort_keys=False)


def _quote_cmd(parts: List[str]) -> str:
    return shlex.join(parts)


def _extras_for_final_eval(report_tasks: List[str], seen_tasks: List[str]) -> List[str]:
    extras = [task for task in report_tasks if task not in seen_tasks and task != "speech_qa"]
    return extras


def _load_template(path: Path) -> Dict[str, Any]:
    return load_yaml(path)


def _slugify_task_name(task: str) -> str:
    import re

    value = re.sub(r"[^a-z0-9]+", "_", str(task).strip().lower()).strip("_")
    return value or "task"


def _build_task_set_slug(
    task_names: List[str],
    *,
    mode: str = "sorted_names",
    base_task_names: Optional[List[str]] = None,
    added_task_names: Optional[List[str]] = None,
) -> str:
    if mode == "sorted_names":
        return "_".join(sorted(_slugify_task_name(task) for task in task_names)) or "empty_task_set"
    if mode == "base_then_added":
        base_names = [_slugify_task_name(task) for task in (base_task_names or [])]
        added_names = [_slugify_task_name(task) for task in (added_task_names or [])]
        base_part = "_".join(base_names) if base_names else "none"
        added_part = "_".join(added_names) if added_names else "none"
        return f"base_{base_part}__added_{added_part}"
    raise ValueError(f"Unsupported task_set_slug_mode '{mode}'.")


def _resolve_mtl_paths(
    *,
    adapter_subdir: str,
    task_names: List[str],
    layout: str = "task_set",
    task_set_slug_mode: str = "sorted_names",
    base_task_names: Optional[List[str]] = None,
    added_task_names: Optional[List[str]] = None,
    artifacts_root: Path,
) -> Dict[str, Path]:
    if layout != "task_set":
        raise ValueError(f"Unsupported MTL artifacts.layout '{layout}'.")
    base = ensure_dir(artifacts_root)
    task_set_slug = _build_task_set_slug(
        task_names,
        mode=task_set_slug_mode,
        base_task_names=base_task_names,
        added_task_names=added_task_names,
    )
    task_set_root = ensure_dir(base / f"{len(task_names)}_task" / task_set_slug)
    adapters = ensure_dir(task_set_root / "adapters")
    metrics = ensure_dir(task_set_root / "metrics")
    output_dir = ensure_dir(adapters / adapter_subdir)
    return {
        "base": task_set_root,
        "adapters": adapters,
        "metrics": metrics,
        "output_dir": output_dir,
    }


def _find_latest_sweep_summary(sweeps_dir: Path) -> Optional[Path]:
    summaries = sorted(sweeps_dir.glob("sweep_*.json"))
    return summaries[-1] if summaries else None


def _resolve_prior_merge_source(stage_dir: Path) -> Optional[Path]:
    sweeps_dir = stage_dir / "merge" / "sweeps"
    summary_path = _find_latest_sweep_summary(sweeps_dir)
    if summary_path is None:
        return None
    with summary_path.open("r", encoding="utf-8") as handle:
        summary = json.load(handle)
    post_eval = summary.get("post_sweep_eval")
    if isinstance(post_eval, Mapping) and post_eval.get("enabled"):
        artifact_dir = post_eval.get("artifact_dir")
        if artifact_dir:
            return Path(str(artifact_dir)).resolve()
    best_index = summary.get("best_index")
    runs = summary.get("runs") or []
    if isinstance(best_index, int) and 0 <= best_index < len(runs):
        continual = runs[best_index].get("continual") or {}
        artifact_dir = continual.get("artifact_dir")
        if artifact_dir:
            return Path(str(artifact_dir)).resolve()
    return None


def _resolve_prior_mtl_adapter(
    stage_dir: Path,
    *,
    adapter_subdir: str,
    previous_seen_tasks: List[str],
    previous_base_tasks: List[str],
    previous_added_task: Optional[str],
) -> Path:
    paths = _resolve_mtl_paths(
        adapter_subdir=adapter_subdir,
        task_names=previous_seen_tasks,
        layout="task_set",
        task_set_slug_mode="base_then_added",
        base_task_names=previous_base_tasks,
        added_task_names=([previous_added_task] if previous_added_task else []),
        artifacts_root=stage_dir / "mtl" / "artifacts",
    )
    return (paths["output_dir"] / "best").resolve()


def _run_command(parts: List[str]) -> None:
    print(_quote_cmd(parts))
    subprocess.run(parts, check=True)


def _run_timed_command(parts: List[str]) -> Dict[str, Any]:
    started_at = datetime.now().isoformat()
    start_ts = datetime.now().timestamp()
    _run_command(parts)
    end_dt = datetime.now()
    return {
        "command": _quote_cmd(parts),
        "started_at": started_at,
        "completed_at": end_dt.isoformat(),
        "runtime_seconds": max(0.0, end_dt.timestamp() - start_ts),
    }


def _write_stage_commands(path: Path, commands: Mapping[str, Any]) -> None:
    lines = ["# Auto-generated continual suite commands", ""]
    for label, command in commands.items():
        if isinstance(command, str) and command:
            lines.append(f"# {label}")
            lines.append(command)
            lines.append("")
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def _prepare_merge(plan, suite_cfg, *, prior_merge_source: Optional[Path]) -> Dict[str, Any]:
    stage_root = ensure_dir(plan.stage_dir / "merge")
    sweeps_dir = ensure_dir(stage_root / "sweeps")
    template = _load_template(suite_cfg.merge_template)
    x_source: Optional[str]
    if plan.stage_index == 2:
        x_source = str(resolve_seed_adapter_path(plan.prior_tasks[0], suite_cfg.seed_adapters))
    else:
        x_source = str(prior_merge_source) if prior_merge_source is not None else None
    y_source = str(resolve_seed_adapter_path(plan.added_task, suite_cfg.seed_adapters)) if plan.added_task else None

    template["adapters"] = [x_source or "__RESOLVE_PREVIOUS_STAGE__", y_source]
    merge_defaults = suite_cfg.defaults.get("merge", {}) if isinstance(suite_cfg.defaults, Mapping) else {}
    default_method = merge_defaults.get("method") if isinstance(merge_defaults, Mapping) else None
    template["method"] = str(default_method or template.get("method") or "continual")
    template["split"] = suite_cfg.selection_split
    template["eval_tasks"] = list(plan.selection_tasks)
    template["output_dir"] = str(sweeps_dir)
    template["post_sweep_eval"] = {
        "enabled": True,
        "split": suite_cfg.report_split,
        "save_merged": True,
        "eval_tasks": list(plan.report_tasks),
    }

    if isinstance(merge_defaults, Mapping):
        if isinstance(merge_defaults.get("method_params"), Mapping):
            template["method_params"] = dict(template.get("method_params", {}))
            template["method_params"].update(dict(merge_defaults["method_params"]))
        if isinstance(merge_defaults.get("search"), Mapping):
            template["search"] = dict(merge_defaults["search"])
        if isinstance(merge_defaults.get("optimizer"), Mapping):
            template["optimizer"] = dict(merge_defaults["optimizer"])
        if merge_defaults.get("eval_subset") is not None:
            template["eval_subset"] = dict(merge_defaults["eval_subset"])
        if merge_defaults.get("constraint_nonnegative") is not None:
            template["constraint_nonnegative"] = bool(merge_defaults["constraint_nonnegative"])

    if template["method"] == "continual_supermerge":
        template.setdefault("optimizer", {})
        template["optimizer"].setdefault("type", "continual_supermerge")
        template["optimizer"].setdefault("params", {})
        if isinstance(template["optimizer"]["params"], Mapping):
            template["optimizer"]["params"] = dict(template["optimizer"]["params"])
            template["optimizer"]["params"].setdefault("tasks", list(plan.selection_tasks))

    config_path = stage_root / "merge_config.yaml"
    _dump_yaml(config_path, template)

    raw_command = None
    if x_source and y_source:
        raw_command = _quote_cmd(["python3", "main.py", "merge-sweep", "--config", str(config_path)])

    return {
        "config_path": str(config_path),
        "sweeps_dir": str(sweeps_dir),
        "stage_root": str(stage_root),
        "x_source": x_source,
        "y_source": y_source,
        "raw_command": raw_command,
    }


def _prepare_mtl(plan, suite_cfg) -> Dict[str, Any]:
    stage_root = ensure_dir(plan.stage_dir / "mtl")
    template = _load_template(suite_cfg.mtl_template)
    adapter_subdir = f"qwen2_5_omni_lora_mtl_continual_{plan.path_id}_{plan.stage_name}"

    if plan.stage_index == 2:
        base_adapter = resolve_seed_adapter_path(plan.prior_tasks[0], suite_cfg.seed_adapters)
        base_tasks = list(plan.prior_tasks)
    else:
        prev_stage = plan.stage_dir.parent / f"stage_{plan.stage_index - 1:02d}"
        prev_adapter_subdir = f"qwen2_5_omni_lora_mtl_continual_{plan.path_id}_stage_{plan.stage_index - 1:02d}"
        base_adapter = _resolve_prior_mtl_adapter(
            prev_stage,
            adapter_subdir=prev_adapter_subdir,
            previous_seen_tasks=plan.prior_tasks,
            previous_base_tasks=plan.prior_tasks[:-1],
            previous_added_task=plan.prior_tasks[-1] if plan.prior_tasks else None,
        )
        base_tasks = list(plan.prior_tasks)

    template.setdefault("artifacts", {})
    template["artifacts"]["root"] = str(stage_root / "artifacts")
    template["artifacts"]["adapter_subdir"] = adapter_subdir
    template["artifacts"]["layout"] = "task_set"
    template["artifacts"]["task_set_slug_mode"] = "base_then_added"

    template.setdefault("continual", {})
    template["continual"]["enabled"] = True
    template["continual"]["base_adapter"] = str(base_adapter)
    template["continual"]["base_tasks_override"] = base_tasks
    template["continual"]["added_tasks"] = [plan.added_task] if plan.added_task else []
    template["continual"]["selection_mode"] = "added_task_metric"
    template["continual"]["selection_task_set"] = "base_plus_added"
    template["continual"]["final_eval_include_speech_qa"] = suite_cfg.include_speech_qa

    template.setdefault("training", {})
    template["training"]["final_eval_extra_tasks"] = _extras_for_final_eval(plan.report_tasks, plan.seen_tasks)

    template["tasks"] = [
        {"name": task_name, "config": f"{task_name}.yaml", "train_weight": 1.0}
        for task_name in sorted(VALID_TASKS)
        if task_name != "speech_qa"
    ]

    config_path = stage_root / "mtl_config.yaml"
    _dump_yaml(config_path, template)

    resolved_paths = _resolve_mtl_paths(
        adapter_subdir=adapter_subdir,
        task_names=plan.seen_tasks,
        layout="task_set",
        task_set_slug_mode="base_then_added",
        base_task_names=base_tasks,
        added_task_names=[plan.added_task] if plan.added_task else [],
        artifacts_root=Path(template["artifacts"]["root"]),
    )
    raw_command = _quote_cmd(["python3", "main.py", "mtl", "--config", str(config_path)])
    return {
        "config_path": str(config_path),
        "stage_root": str(stage_root),
        "adapter_subdir": adapter_subdir,
        "raw_command": raw_command,
        "base_adapter": str(base_adapter),
        "expected_task_set_root": str(resolved_paths["base"]),
        "expected_output_dir": str(resolved_paths["output_dir"]),
        "expected_metrics_dir": str(resolved_paths["metrics"]),
        "expected_best_adapter": str((resolved_paths["output_dir"] / "best").resolve()),
        "expected_best_metrics_dir": str((resolved_paths["metrics"] / "best").resolve()),
    }


def _posthoc_balanced_eval(plan, suite_cfg, mtl_meta: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    adapter_output_dir = Path(mtl_meta["expected_output_dir"])
    metrics_dir = Path(mtl_meta["expected_metrics_dir"])
    history_path = metrics_dir / "mtl_eval_history.jsonl"
    if not history_path.exists():
        return None

    rows = load_jsonl(history_path)
    checkpoint_steps = collect_checkpoint_steps(adapter_output_dir)
    record = select_mtl_checkpoint_record(
        rows,
        seen_tasks=plan.seen_tasks,
        mode="balanced_best",
        available_steps=checkpoint_steps,
    )
    if record is None:
        return None

    step = int(record["step"])
    checkpoint_dir = adapter_output_dir / f"checkpoint-{step}"
    if not checkpoint_dir.exists():
        return None

    output_metrics_dir = metrics_dir / "balanced_best"
    eval_cmd = [
        "python3",
        "scripts/eval/eval_mtl_adapter.py",
        "--adapter",
        str(checkpoint_dir),
        "--tasks",
        *plan.report_tasks,
        "--split",
        suite_cfg.report_split,
        "--metrics-dir",
        str(output_metrics_dir),
    ]
    timing = _run_timed_command(eval_cmd)

    summary = {
        "checkpoint_mode": "balanced_best",
        "selected_step": step,
        "selection_record": record,
        "checkpoint_dir": str(checkpoint_dir.resolve()),
        "metrics_dir": str(output_metrics_dir.resolve()),
        "eval_results_path": str((output_metrics_dir / f"eval_results_{suite_cfg.report_split}.json").resolve()),
        "runtime": {
            "eval_runtime_seconds": timing["runtime_seconds"],
            "eval_started_at": timing["started_at"],
            "eval_completed_at": timing["completed_at"],
        },
    }
    save_stage_manifest(metrics_dir / "balanced_best_selection.json", summary)
    return summary


def main() -> None:
    args = parse_args()
    suite_cfg = load_suite_config(args.suite_config)
    selected_paths = None if args.path_id == "all" else [args.path_id]
    selected_stages = parse_stage_selector(args.stages)
    plans = expand_stage_plans(
        suite_cfg,
        selected_paths=selected_paths,
        selected_stages=selected_stages,
        start_stage=2,
    )

    if not plans:
        raise SystemExit("No stages matched the requested filters.")

    for plan in plans:
        ensure_dir(plan.stage_dir)
        suite_runner_command = _quote_cmd(
            [
                "python3",
                "scripts/continual/run_continual_suite.py",
                "--suite-config",
                str(suite_cfg.config_path),
                "--mode",
                args.mode,
                "--path",
                plan.path_id,
                "--stages",
                str(plan.stage_index),
                *(["--execute"] if args.execute else []),
            ]
        )

        prior_merge_source = None
        if plan.stage_index > 2:
            prior_merge_source = _resolve_prior_merge_source(plan.stage_dir.parent / f"stage_{plan.stage_index - 1:02d}")

        manifest: Dict[str, Any] = {
            "suite_id": suite_cfg.suite_id,
            "suite_config_path": str(suite_cfg.config_path),
            "path_id": plan.path_id,
            "stage_index": plan.stage_index,
            "stage_name": plan.stage_name,
            "stage_dir": str(plan.stage_dir.resolve()),
            "seen_tasks": list(plan.seen_tasks),
            "prior_tasks": list(plan.prior_tasks),
            "added_task": plan.added_task,
            "eval_only_tasks": list(plan.eval_only_tasks),
            "report_tasks": list(plan.report_tasks),
            "selection_tasks": list(plan.selection_tasks),
            "selection_split": suite_cfg.selection_split,
            "report_split": suite_cfg.report_split,
            "commands": {
                "suite_runner": suite_runner_command,
            },
        }

        commands: Dict[str, str] = {
            "suite_runner": suite_runner_command,
        }

        if args.mode in {"merge", "all"}:
            merge_meta = _prepare_merge(plan, suite_cfg, prior_merge_source=prior_merge_source)
            manifest["merge"] = merge_meta
            if merge_meta.get("raw_command"):
                commands["merge_raw"] = str(merge_meta["raw_command"])

        if args.mode in {"mtl", "all"}:
            mtl_meta = _prepare_mtl(plan, suite_cfg)
            manifest["mtl"] = mtl_meta
            commands["mtl_raw"] = str(mtl_meta["raw_command"])

        manifest["commands"] = dict(commands)

        manifest_path = plan.stage_dir / "stage_manifest.json"
        save_stage_manifest(manifest_path, manifest)
        _write_stage_commands(plan.stage_dir / "commands.sh", commands)

        print(f"[{plan.path_id} {plan.stage_name}] wrote {manifest_path}")
        for label, command in commands.items():
            print(f"  {label}: {command}")

        if not args.execute:
            continue

        if args.mode in {"merge", "all"}:
            raw_command = manifest.get("merge", {}).get("raw_command")
            if raw_command:
                timing = _run_timed_command(["python3", "main.py", "merge-sweep", "--config", str(manifest["merge"]["config_path"])])
                manifest.setdefault("merge", {}).setdefault("runtime", {})
                manifest["merge"]["runtime"]["search_runtime_seconds"] = timing["runtime_seconds"]
                manifest["merge"]["runtime"]["search_started_at"] = timing["started_at"]
                manifest["merge"]["runtime"]["search_completed_at"] = timing["completed_at"]
                save_stage_manifest(manifest_path, manifest)

        if args.mode in {"mtl", "all"}:
            timing = _run_timed_command(["python3", "main.py", "mtl", "--config", str(manifest["mtl"]["config_path"])])
            manifest.setdefault("mtl", {}).setdefault("runtime", {})
            manifest["mtl"]["runtime"]["train_runtime_seconds"] = timing["runtime_seconds"]
            manifest["mtl"]["runtime"]["train_started_at"] = timing["started_at"]
            manifest["mtl"]["runtime"]["train_completed_at"] = timing["completed_at"]
            save_stage_manifest(manifest_path, manifest)
            balanced_summary = _posthoc_balanced_eval(plan, suite_cfg, manifest["mtl"])
            if balanced_summary is not None:
                manifest.setdefault("mtl", {})["balanced_best"] = balanced_summary
                save_stage_manifest(manifest_path, manifest)


if __name__ == "__main__":
    try:
        main()
    except subprocess.CalledProcessError as exc:
        print(f"Command failed with exit code {exc.returncode}", file=sys.stderr)
        raise
