"""Generic continual-suite planning and summary helpers."""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import yaml

PACKAGE_ROOT = Path(__file__).resolve().parents[2]
VALID_TASKS = {
    "asr",
    "emotion",
    "intent",
    "speaker_id",
    "speech_qa",
    "kws",
    "langid",
    "st",
    "speaker_ver",
    "vocalsound",
}
TASK_METRICS: Dict[str, Tuple[str, bool]] = {
    "asr": ("wer", False),
    "emotion": ("macro_f1", True),
    "intent": ("accuracy", True),
    "speech_qa": ("accuracy", True),
    "kws": ("macro_f1", True),
    "langid": ("accuracy", True),
    "speaker_id": ("accuracy", True),
    "speaker_ver": ("accuracy", True),
    "vocalsound": ("accuracy", True),
}


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def load_yaml(path: Path) -> Dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle) or {}


def dump_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(data, handle, indent=2, sort_keys=True)


def _dedupe_preserve_order(values: Iterable[str]) -> List[str]:
    ordered: List[str] = []
    seen = set()
    for raw in values:
        value = str(raw).strip().lower()
        if not value or value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def _validate_task_names(values: Sequence[str], *, field_name: str) -> List[str]:
    normalized: List[str] = []
    valid = set(VALID_TASKS)
    for raw in values:
        task = str(raw).strip().lower()
        if task not in valid:
            allowed = ", ".join(sorted(valid))
            raise ValueError(f"Unknown task '{raw}' in {field_name}. Valid tasks: {allowed}")
        normalized.append(task)
    duplicates = sorted({task for task in normalized if normalized.count(task) > 1})
    if duplicates:
        raise ValueError(f"Duplicate tasks are not allowed in {field_name}: {duplicates}")
    return normalized


def _resolve_path(path: str | Path) -> Path:
    resolved = Path(path).expanduser()
    if not resolved.is_absolute():
        resolved = (PACKAGE_ROOT / resolved).resolve()
    return resolved


def _safe_float(value: Any) -> Optional[float]:
    if isinstance(value, (int, float)):
        return float(value)
    return None


def _json_dumps(data: Mapping[str, Any]) -> str:
    return json.dumps(data, sort_keys=True)


@dataclass(frozen=True)
class ContinualSuitePath:
    path_id: str
    tasks: List[str]
    eval_only_tasks: List[str] = field(default_factory=list)
    report_tasks: List[str] = field(default_factory=list)
    stage_overrides: Dict[int, Dict[str, Any]] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinualSuiteConfig:
    suite_id: str
    config_path: Path
    output_root: Path
    defaults: Dict[str, Any]
    merge_template: Path
    mtl_template: Path
    selection_split: str
    report_split: str
    include_speech_qa: bool
    seed_adapters: Dict[str, str]
    paths: List[ContinualSuitePath]


@dataclass(frozen=True)
class StagePlan:
    suite_id: str
    path_id: str
    stage_index: int
    stage_name: str
    stage_dir: Path
    seen_tasks: List[str]
    prior_tasks: List[str]
    added_task: Optional[str]
    eval_only_tasks: List[str]
    report_tasks: List[str]
    selection_tasks: List[str]
    stage_overrides: Dict[str, Any]


@dataclass(frozen=True)
class StageResultRecord:
    suite_id: str
    path_id: str
    stage_index: int
    stage_name: str
    method: str
    checkpoint_view: str
    split: str
    seen_tasks: List[str]
    prior_tasks: List[str]
    added_task: Optional[str]
    eval_only_tasks: List[str]
    report_tasks: List[str]
    results: Dict[str, Dict[str, Any]]
    source_path: str
    runtime_info: Dict[str, Any] = field(default_factory=dict)


def load_suite_config(path: str | Path) -> ContinualSuiteConfig:
    config_path = _resolve_path(path)
    payload = load_yaml(config_path)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Suite config must be a mapping: {config_path}")

    suite_id = str(payload.get("suite_id", "")).strip()
    if not suite_id:
        raise ValueError("suite_id is required.")

    defaults = dict(payload.get("defaults", {}) or {})
    output_root = _resolve_path(defaults.get("output_root", "artifacts/continual_suite"))
    merge_template = _resolve_path(defaults.get("merge_template", "configs/merge/continual/merge_continual_6task_materialized_plus_asr.yaml"))
    mtl_template = _resolve_path(defaults.get("mtl_template", "configs/mtl/continual/continual_6task_plus_asr.yaml"))
    selection_split = str(defaults.get("selection_split", "validation")).strip().lower() or "validation"
    report_split = str(defaults.get("report_split", "test")).strip().lower() or "test"
    include_speech_qa = bool(defaults.get("include_speech_qa", True))
    seed_adapters_raw = defaults.get("seed_adapters", {}) or {}
    if not isinstance(seed_adapters_raw, Mapping):
        raise ValueError("defaults.seed_adapters must be a mapping.")
    seed_adapters = {str(k).strip().lower(): str(v) for k, v in seed_adapters_raw.items() if str(k).strip()}

    paths_raw = payload.get("paths")
    if not isinstance(paths_raw, list) or not paths_raw:
        raise ValueError("paths must be a non-empty list.")

    parsed_paths: List[ContinualSuitePath] = []
    seen_ids = set()
    for entry in paths_raw:
        if not isinstance(entry, Mapping):
            raise ValueError("Each paths entry must be a mapping.")
        path_id = str(entry.get("path_id", "")).strip()
        if not path_id:
            raise ValueError("Each path requires a non-empty path_id.")
        if path_id in seen_ids:
            raise ValueError(f"Duplicate path_id '{path_id}'.")
        seen_ids.add(path_id)

        tasks = _validate_task_names(entry.get("tasks", []), field_name=f"{path_id}.tasks")
        if len(tasks) < 1:
            raise ValueError(f"{path_id}.tasks must contain at least one task.")
        eval_only_tasks = _validate_task_names(
            entry.get("eval_only_tasks", []),
            field_name=f"{path_id}.eval_only_tasks",
        )
        report_tasks = _validate_task_names(
            entry.get("report_tasks", []),
            field_name=f"{path_id}.report_tasks",
        )

        stage_overrides_raw = entry.get("stage_overrides", {}) or {}
        if not isinstance(stage_overrides_raw, Mapping):
            raise ValueError(f"{path_id}.stage_overrides must be a mapping.")
        stage_overrides: Dict[int, Dict[str, Any]] = {}
        for raw_stage, override in stage_overrides_raw.items():
            stage_idx = int(raw_stage)
            if stage_idx < 1 or stage_idx > len(tasks):
                raise ValueError(
                    f"{path_id}.stage_overrides contains invalid stage '{raw_stage}'. "
                    f"Expected 1..{len(tasks)}."
                )
            if override is None:
                stage_overrides[stage_idx] = {}
            elif isinstance(override, Mapping):
                stage_overrides[stage_idx] = dict(override)
            else:
                raise ValueError(f"{path_id}.stage_overrides[{raw_stage}] must be a mapping.")

        parsed_paths.append(
            ContinualSuitePath(
                path_id=path_id,
                tasks=tasks,
                eval_only_tasks=eval_only_tasks,
                report_tasks=report_tasks,
                stage_overrides=stage_overrides,
            )
        )

    return ContinualSuiteConfig(
        suite_id=suite_id,
        config_path=config_path,
        output_root=output_root,
        defaults=defaults,
        merge_template=merge_template,
        mtl_template=mtl_template,
        selection_split=selection_split,
        report_split=report_split,
        include_speech_qa=include_speech_qa,
        seed_adapters=seed_adapters,
        paths=parsed_paths,
    )


def parse_stage_selector(raw: str | None) -> Optional[List[int]]:
    if raw is None:
        return None
    text = str(raw).strip().lower()
    if not text or text == "all":
        return None
    values: List[int] = []
    for token in text.split(","):
        token = token.strip()
        if not token:
            continue
        values.append(int(token))
    return values or None


def expand_stage_plans(
    config: ContinualSuiteConfig,
    *,
    selected_paths: Optional[Sequence[str]] = None,
    selected_stages: Optional[Sequence[int]] = None,
    start_stage: int = 2,
) -> List[StagePlan]:
    selected_path_ids = {str(path_id).strip() for path_id in (selected_paths or []) if str(path_id).strip()}
    selected_stage_ids = {int(stage) for stage in (selected_stages or [])}

    plans: List[StagePlan] = []
    for path in config.paths:
        if selected_path_ids and path.path_id not in selected_path_ids:
            continue
        for stage_index in range(max(1, int(start_stage)), len(path.tasks) + 1):
            if selected_stage_ids and stage_index not in selected_stage_ids:
                continue
            seen_tasks = list(path.tasks[:stage_index])
            prior_tasks = list(path.tasks[: stage_index - 1])
            added_task = seen_tasks[-1] if stage_index > 1 else None
            report_tasks = _dedupe_preserve_order(
                [*seen_tasks, *path.eval_only_tasks, *path.report_tasks, *(["speech_qa"] if config.include_speech_qa else [])]
            )
            stage_dir = (
                ensure_dir(config.output_root)
                / config.suite_id
                / path.path_id
                / f"stage_{stage_index:02d}"
            )
            plans.append(
                StagePlan(
                    suite_id=config.suite_id,
                    path_id=path.path_id,
                    stage_index=stage_index,
                    stage_name=f"stage_{stage_index:02d}",
                    stage_dir=stage_dir,
                    seen_tasks=seen_tasks,
                    prior_tasks=prior_tasks,
                    added_task=added_task,
                    eval_only_tasks=list(path.eval_only_tasks),
                    report_tasks=report_tasks,
                    selection_tasks=seen_tasks,
                    stage_overrides=dict(path.stage_overrides.get(stage_index, {})),
                )
            )
    return plans


def resolve_seed_adapter_path(task: str, seed_adapters: Mapping[str, str]) -> Path:
    task_key = str(task).strip().lower()
    explicit = seed_adapters.get(task_key)
    if explicit:
        return _resolve_path(explicit)

    adapter_root = PACKAGE_ROOT / "artifacts" / task_key / "adapters"
    candidates = sorted(
        path
        for path in adapter_root.glob("*/best")
        if path.exists() and (path / "adapter_config.json").exists()
    )
    if not candidates:
        raise FileNotFoundError(
            f"Could not resolve a seed adapter for task '{task_key}'. "
            "Add defaults.seed_adapters.<task> to the suite config."
        )
    if len(candidates) == 1:
        return candidates[0].resolve()
    candidates.sort(key=lambda path: path.stat().st_mtime, reverse=True)
    return candidates[0].resolve()


def load_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def select_mtl_checkpoint_record(
    history_rows: Sequence[Mapping[str, Any]],
    *,
    seen_tasks: Sequence[str],
    mode: str,
    available_steps: Optional[Sequence[int]] = None,
) -> Optional[Dict[str, Any]]:
    step_filter = {int(step) for step in (available_steps or [])}
    best_record: Optional[Dict[str, Any]] = None
    best_key: Optional[Tuple[float, float, int]] = None
    for row in history_rows:
        step = row.get("step")
        if not isinstance(step, (int, float)):
            continue
        step_int = int(step)
        if step_filter and step_int not in step_filter:
            continue

        if mode == "added_task_best":
            primary = _safe_float(row.get("eval_added_tasks_primary_oriented_mean"))
            if primary is None:
                continue
            key = (primary, primary, step_int)
        elif mode == "balanced_best":
            deltas = []
            for task in seen_tasks:
                value = _safe_float(row.get(f"eval_{task}_interference_delta"))
                if value is not None:
                    deltas.append(value)
            if not deltas:
                continue
            key = (min(deltas), sum(deltas) / len(deltas), step_int)
        else:
            raise ValueError(f"Unsupported MTL checkpoint mode '{mode}'.")

        if best_key is None or key > best_key:
            best_key = key
            best_record = dict(row)

    if best_record is None:
        return None
    return best_record


def collect_checkpoint_steps(adapter_output_dir: str | Path) -> List[int]:
    output_dir = Path(adapter_output_dir)
    steps: List[int] = []
    for path in output_dir.glob("checkpoint-*"):
        suffix = path.name.replace("checkpoint-", "", 1)
        if suffix.isdigit():
            steps.append(int(suffix))
    return sorted(set(steps))


def load_eval_results_payload(path: str | Path) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"Eval results payload must be a mapping: {path}")
    return dict(payload)


def build_stage_result_record(
    *,
    suite_id: str,
    path_id: str,
    stage_index: int,
    stage_name: str,
    method: str,
    checkpoint_view: str,
    split: str,
    seen_tasks: Sequence[str],
    prior_tasks: Sequence[str],
    added_task: Optional[str],
    eval_only_tasks: Sequence[str],
    report_tasks: Sequence[str],
    results_payload: Mapping[str, Any],
    source_path: str | Path,
    runtime_info: Optional[Mapping[str, Any]] = None,
) -> StageResultRecord:
    results = results_payload.get("results", {})
    if not isinstance(results, Mapping):
        raise ValueError(f"results payload is missing a mapping under 'results': {source_path}")
    normalized: Dict[str, Dict[str, Any]] = {}
    for task, metrics in results.items():
        if isinstance(metrics, Mapping):
            normalized[str(task)] = dict(metrics)
    return StageResultRecord(
        suite_id=suite_id,
        path_id=path_id,
        stage_index=int(stage_index),
        stage_name=str(stage_name),
        method=str(method),
        checkpoint_view=str(checkpoint_view),
        split=str(split),
        seen_tasks=list(seen_tasks),
        prior_tasks=list(prior_tasks),
        added_task=(None if added_task is None else str(added_task)),
        eval_only_tasks=list(eval_only_tasks),
        report_tasks=list(report_tasks),
        results=normalized,
        source_path=str(source_path),
        runtime_info=(dict(runtime_info) if isinstance(runtime_info, Mapping) else {}),
    )


def compute_stage_tables(records: Sequence[StageResultRecord]) -> Dict[str, List[Dict[str, Any]]]:
    stage_rows: List[Dict[str, Any]] = []
    growth_rows: List[Dict[str, Any]] = []
    forgetting_rows: List[Dict[str, Any]] = []

    ordered = sorted(
        records,
        key=lambda item: (item.path_id, item.method, item.checkpoint_view, item.stage_index),
    )
    best_by_group_task: Dict[Tuple[str, str, str], Dict[str, float]] = {}

    for record in ordered:
        group_key = (record.path_id, record.method, record.checkpoint_view)
        previous_best = best_by_group_task.setdefault(group_key, {})

        seen_deltas: Dict[str, float] = {}
        report_deltas: Dict[str, float] = {}
        report_primary: Dict[str, Dict[str, Any]] = {}
        for task in record.report_tasks:
            metrics = dict(record.results.get(task, {}))
            metric_spec = TASK_METRICS.get(task)
            metric_name = metric_spec[0] if metric_spec is not None else None
            report_deltas[task] = _safe_float(metrics.get("interference_delta")) if metrics else None  # type: ignore[assignment]
            if metric_name:
                report_primary[task] = {
                    "metric_name": metric_name,
                    "metric_value": _safe_float(metrics.get(metric_name)),
                }
            else:
                report_primary[task] = {
                    "metric_name": None,
                    "metric_value": None,
                }
        report_deltas = {task: value for task, value in report_deltas.items() if value is not None}

        for task in record.seen_tasks:
            value = report_deltas.get(task)
            if value is not None:
                seen_deltas[task] = value

        prior_deltas = {task: seen_deltas[task] for task in record.prior_tasks if task in seen_deltas}
        new_task_delta = seen_deltas.get(record.added_task or "", None)
        seen_values = list(seen_deltas.values())
        prior_values = list(prior_deltas.values())
        seen_mean_delta = (sum(seen_values) / len(seen_values)) if seen_values else None
        seen_min_delta = min(seen_values) if seen_values else None
        prior_avg_delta = (sum(prior_values) / len(prior_values)) if prior_values else None

        forgetting_values: List[float] = []
        for task, current_value in prior_deltas.items():
            best_previous = previous_best.get(task)
            if best_previous is None:
                continue
            forgetting = current_value - best_previous
            forgetting_values.append(forgetting)
            forgetting_rows.append(
                {
                    "suite_id": record.suite_id,
                    "path_id": record.path_id,
                    "method": record.method,
                    "checkpoint_view": record.checkpoint_view,
                    "stage_index": record.stage_index,
                    "stage_name": record.stage_name,
                    "prior_task": task,
                    "current_delta": current_value,
                    "best_previous_delta": best_previous,
                    "forgetting": forgetting,
                    "split": record.split,
                    "source_path": record.source_path,
                }
            )
        avg_prior_forgetting = (
            (sum(forgetting_values) / len(forgetting_values))
            if forgetting_values
            else None
        )

        row = {
            "suite_id": record.suite_id,
            "path_id": record.path_id,
            "method": record.method,
            "checkpoint_view": record.checkpoint_view,
            "stage_index": record.stage_index,
            "stage_name": record.stage_name,
            "split": record.split,
            "seen_tasks": ",".join(record.seen_tasks),
            "prior_tasks": ",".join(record.prior_tasks),
            "eval_only_tasks": ",".join(record.eval_only_tasks),
            "report_tasks": ",".join(record.report_tasks),
            "added_task": record.added_task,
            "new_task_delta": new_task_delta,
            "seen_mean_delta": seen_mean_delta,
            "seen_min_delta": seen_min_delta,
            "prior_avg_delta": prior_avg_delta,
            "avg_prior_forgetting": avg_prior_forgetting,
            "per_task_primary_json": _json_dumps(report_primary),
            "per_task_delta_json": _json_dumps(report_deltas),
            "source_path": record.source_path,
        }
        if record.runtime_info:
            row.update(dict(record.runtime_info))

        for task in record.report_tasks:
            primary_entry = report_primary.get(task, {})
            row[f"{task}__primary_metric_name"] = primary_entry.get("metric_name")
            row[f"{task}__primary_metric_value"] = primary_entry.get("metric_value")
            row[f"{task}__interference_delta"] = report_deltas.get(task)

        stage_rows.append(row)

        for metric_name, metric_value in (
            ("new_task_delta", new_task_delta),
            ("seen_mean_delta", seen_mean_delta),
            ("seen_min_delta", seen_min_delta),
            ("prior_avg_delta", prior_avg_delta),
            ("avg_prior_forgetting", avg_prior_forgetting),
        ):
            growth_rows.append(
                {
                    "suite_id": record.suite_id,
                    "path_id": record.path_id,
                    "method": record.method,
                    "checkpoint_view": record.checkpoint_view,
                    "stage_index": record.stage_index,
                    "stage_name": record.stage_name,
                    "added_task": record.added_task,
                    "metric_name": metric_name,
                    "metric_value": metric_value,
                    "split": record.split,
                    "source_path": record.source_path,
                    **dict(record.runtime_info),
                }
            )

        for task, current_value in seen_deltas.items():
            previous_best[task] = max(current_value, previous_best.get(task, float("-inf")))

    final_rows = []
    by_key: Dict[Tuple[str, str, str], Dict[str, Any]] = {}
    for row in stage_rows:
        key = (str(row["path_id"]), str(row["method"]), str(row["checkpoint_view"]))
        prev = by_key.get(key)
        if prev is None or int(row["stage_index"]) > int(prev["stage_index"]):
            by_key[key] = row
    final_rows.extend(by_key.values())

    return {
        "stage_rows": stage_rows,
        "growth_rows": growth_rows,
        "forgetting_rows": forgetting_rows,
        "final_rows": sorted(final_rows, key=lambda item: (item["path_id"], item["method"], item["checkpoint_view"])),
    }


def write_csv_rows(path: str | Path, rows: Sequence[Mapping[str, Any]]) -> None:
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row.keys():
            if key not in fieldnames:
                fieldnames.append(str(key))
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(dict(row))


def save_stage_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    dump_json(dict(payload), Path(path))


__all__ = [
    "VALID_TASKS",
    "ContinualSuiteConfig",
    "ContinualSuitePath",
    "StagePlan",
    "StageResultRecord",
    "build_stage_result_record",
    "collect_checkpoint_steps",
    "compute_stage_tables",
    "dump_json",
    "ensure_dir",
    "expand_stage_plans",
    "load_eval_results_payload",
    "load_jsonl",
    "load_suite_config",
    "load_yaml",
    "parse_stage_selector",
    "resolve_seed_adapter_path",
    "save_stage_manifest",
    "select_mtl_checkpoint_record",
    "write_csv_rows",
]
