"""Backfill ASR LibriSpeech test-other evaluations for existing test-clean runs.

Default mode is dry-run: print the commands that would be executed. Pass
``--execute`` to run them sequentially.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Iterable, Iterator, Optional

REPO_ROOT = Path(__file__).resolve().parents[1]
PYTHON = sys.executable
DEFAULT_MERGED_METHODS = {"weighted_delta_n", "uniform_scalar_delta", "continual"}
LEGACY_WEIGHTED_METHOD = "weighted"
KNOWN_SINGLE_ADAPTER_TASKS = {
    "asr",
    "emotion",
    "intent",
    "kws",
    "langid",
    "speaker_id",
    "speaker_ver",
    "vocalsound",
}


@dataclass(frozen=True)
class EvalCommand:
    family: str
    name: str
    command: tuple[str, ...]
    expected_path: Optional[Path] = None


def _load_json(path: Path) -> Optional[dict]:
    try:
        payload = json.loads(path.read_text())
    except Exception:
        return None
    return payload if isinstance(payload, dict) else None


def _has_asr_result(path: Path) -> bool:
    payload = _load_json(path)
    if not payload:
        return False
    if isinstance(payload.get("results"), dict) and isinstance(payload["results"].get("asr"), dict):
        return "wer" in payload["results"]["asr"]
    if "wer" in payload:
        return True
    return False


def _task_from_adapter_metric(filename: str) -> Optional[str]:
    if filename == "base_model.json":
        return None
    if filename == "best_asr_adapter.json":
        return "asr"
    prefix = "best_"
    suffix = "_adapter.json"
    if filename.startswith(prefix) and filename.endswith(suffix):
        return filename[len(prefix) : -len(suffix)]
    return None


def discover_single_task_commands() -> Iterator[EvalCommand]:
    eval_dir = REPO_ROOT / "artifacts" / "asr" / "metrics" / "eval" / "test"
    if not eval_dir.exists():
        return

    base_expected = REPO_ROOT / "artifacts" / "asr" / "metrics" / "eval" / "test_other" / "base_model.json"
    yield EvalCommand(
        family="single_task",
        name="base_model",
        command=(PYTHON, "main.py", "evaluate", "--task", "asr", "--split", "test-other", "--no-confusion-matrix"),
        expected_path=base_expected,
    )

    for path in sorted(eval_dir.glob("best_*_adapter.json")):
        task = _task_from_adapter_metric(path.name)
        if not task:
            continue
        if task not in KNOWN_SINGLE_ADAPTER_TASKS:
            continue
        expected = REPO_ROOT / "artifacts" / "asr" / "metrics" / "eval" / "test_other" / path.name
        yield EvalCommand(
            family="single_task",
            name=f"best_{task}_adapter_on_asr",
            command=(
                PYTHON,
                "main.py",
                "evaluate",
                "--task",
                "asr",
                "--adapter",
                task,
                "--split",
                "test-other",
                "--no-confusion-matrix",
            ),
            expected_path=expected,
        )


def discover_weighted_pairwise_commands() -> Iterator[EvalCommand]:
    """Discover legacy pairwise weighted adapters with saved ASR test metrics."""
    weighted_root = REPO_ROOT / "artifacts" / "merged" / LEGACY_WEIGHTED_METHOD
    if not weighted_root.exists():
        return

    for results_path in sorted(weighted_root.rglob("runs/*/eval_results_test.json")):
        payload = _load_json(results_path)
        if not payload or not _has_asr_result(results_path):
            continue
        merge_tag = payload.get("merge_tag")
        if not isinstance(merge_tag, str) or not merge_tag:
            continue
        run_path = results_path.parent
        if not (run_path / "adapter_config.json").exists():
            continue
        rel_run = run_path.relative_to(REPO_ROOT)
        expected = REPO_ROOT / "artifacts" / "asr" / "metrics" / "eval" / "test_other" / f"best_{merge_tag}_adapter.json"
        yield EvalCommand(
            family=f"merged/{LEGACY_WEIGHTED_METHOD}",
            name=merge_tag,
            command=(
                PYTHON,
                "main.py",
                "evaluate",
                "--task",
                "asr",
                "--adapter",
                str(rel_run),
                "--trained-on-task",
                merge_tag,
                "--split",
                "test-other",
                "--no-confusion-matrix",
            ),
            expected_path=expected,
        )


def discover_merged_commands(*, include_legacy: bool = False) -> Iterator[EvalCommand]:
    merged_root = REPO_ROOT / "artifacts" / "merged"
    if not merged_root.exists():
        return

    for index_path in sorted(merged_root.rglob("eval/index.json")):
        index = _load_json(index_path)
        entries = index.get("entries") if isinstance(index, dict) else None
        if not isinstance(entries, list):
            continue
        for entry in entries:
            if not isinstance(entry, dict):
                continue
            if str(entry.get("split")) != "test":
                continue
            results_path_raw = entry.get("results_path")
            if not isinstance(results_path_raw, str):
                continue
            results_path = Path(results_path_raw)
            if not results_path.is_absolute():
                results_path = REPO_ROOT / results_path
            if not results_path.exists() or not _has_asr_result(results_path):
                continue
            run_path_raw = entry.get("run_path")
            if not isinstance(run_path_raw, str) or not run_path_raw:
                continue
            run_path = Path(run_path_raw)
            if not run_path.is_absolute():
                run_path = REPO_ROOT / run_path
            if not run_path.exists():
                continue
            rel_run = run_path.relative_to(REPO_ROOT)
            method = str(entry.get("method") or entry.get("metadata", {}).get("merge_method") or index_path.parts[-4])
            if not include_legacy and method not in DEFAULT_MERGED_METHODS:
                continue
            name = str(entry.get("merge_tag") or run_path.name)
            expected = run_path / "eval_results_test_other.json"
            if method == "continual":
                command = (
                    PYTHON,
                    "main.py",
                    "evaluate-continual",
                    "--artifact-path",
                    str(rel_run),
                    "--eval-tasks",
                    "asr",
                    "--split",
                    "test-other",
                    "--no-compute-interference-baselines",
                )
            else:
                command = (
                    PYTHON,
                    "main.py",
                    "evaluate-merged",
                    "--adapter-path",
                    str(rel_run),
                    "--eval-tasks",
                    "asr",
                    "--split",
                    "test-other",
                    "--no-confusion-matrix",
                )
            yield EvalCommand(
                family=f"merged/{method}",
                name=name,
                command=command,
                expected_path=expected,
            )


def _candidate_mtl_adapters(task_root: Path) -> list[Path]:
    adapters_dir = task_root / "adapters"
    if not adapters_dir.exists():
        return []
    candidates: list[Path] = []
    for adapter_family in sorted(adapters_dir.iterdir()):
        if not adapter_family.is_dir():
            continue
        for label in ("best", "latest"):
            candidate = adapter_family / label
            if (candidate / "adapter_config.json").exists():
                candidates.append(candidate)
        runs_dir = adapter_family / "runs"
        if runs_dir.exists():
            runs = sorted((p for p in runs_dir.iterdir() if p.is_dir()), reverse=True)
            for run in runs[:1]:
                if (run / "adapter_config.json").exists():
                    candidates.append(run)
    return candidates


def discover_mtl_commands() -> Iterator[EvalCommand]:
    mtl_root = REPO_ROOT / "artifacts" / "mtl"
    if not mtl_root.exists():
        return
    for metrics_path in sorted(mtl_root.rglob("metrics/latest/test_metrics.json")):
        payload = _load_json(metrics_path) or {}
        per_task_asr = metrics_path.parent / "per_task" / "asr" / "eval_results_test.json"
        if "eval_asr_wer" not in payload and not per_task_asr.exists():
            continue
        task_root = metrics_path.parents[2]
        adapters = _candidate_mtl_adapters(task_root)
        if not adapters:
            continue
        adapter = adapters[0]
        rel_adapter = adapter.relative_to(REPO_ROOT)
        rel_metrics_dir = (task_root / "metrics" / adapter.name).relative_to(REPO_ROOT)
        expected = task_root / "metrics" / adapter.name / "eval_results_test_other.json"
        yield EvalCommand(
            family="mtl",
            name=str(task_root.relative_to(mtl_root)),
            command=(
                PYTHON,
                "scripts/eval_mtl_adapter.py",
                "--adapter",
                str(rel_adapter),
                "--tasks",
                "asr",
                "--split",
                "test-other",
                "--metrics-dir",
                str(rel_metrics_dir),
                "--no-interference-baselines",
            ),
            expected_path=expected,
        )


def discover_commands(families: Iterable[str], *, include_legacy_merged: bool = False) -> list[EvalCommand]:
    requested = {family.strip().lower() for family in families}
    commands: list[EvalCommand] = []
    if "single" in requested or "single_task" in requested:
        commands.extend(discover_single_task_commands())
    if "merged" in requested:
        if include_legacy_merged:
            commands.extend(discover_weighted_pairwise_commands())
        commands.extend(discover_merged_commands(include_legacy=include_legacy_merged))
    if "mtl" in requested:
        commands.extend(discover_mtl_commands())

    deduped: list[EvalCommand] = []
    seen: set[tuple[str, ...]] = set()
    for item in commands:
        if item.command in seen:
            continue
        seen.add(item.command)
        deduped.append(item)
    return deduped


def apply_batch_size(commands: Iterable[EvalCommand], batch_size: Optional[int]) -> list[EvalCommand]:
    if batch_size is None:
        return list(commands)
    if batch_size <= 0:
        raise ValueError("--batch-size must be a positive integer.")
    out: list[EvalCommand] = []
    for command in commands:
        if "--batch-size" in command.command:
            out.append(command)
            continue
        out.append(replace(command, command=(*command.command, "--batch-size", str(int(batch_size)))))
    return out


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--families",
        nargs="+",
        default=["single_task", "merged", "mtl"],
        choices=["single", "single_task", "merged", "mtl"],
        help="Experiment families to backfill.",
    )
    parser.add_argument("--execute", action="store_true", help="Run commands instead of printing them.")
    parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Append a per-device eval batch size override to every generated command.",
    )
    parser.add_argument(
        "--skip-existing",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Skip commands whose expected test_other output already exists (default: true).",
    )
    parser.add_argument(
        "--overwrite-existing",
        action="store_true",
        help="Allow overwriting existing test_other outputs. Equivalent to --no-skip-existing.",
    )
    parser.add_argument(
        "--include-legacy-merged",
        action="store_true",
        help="Also include legacy merged methods such as weighted_delta and uniform_delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    commands = discover_commands(args.families, include_legacy_merged=args.include_legacy_merged)
    commands = apply_batch_size(commands, args.batch_size)
    skip_existing = bool(args.skip_existing) and not bool(args.overwrite_existing)
    if skip_existing:
        commands = [cmd for cmd in commands if cmd.expected_path is None or not cmd.expected_path.exists()]

    if not commands:
        print("No ASR test-other backfill commands discovered.")
        return

    for idx, item in enumerate(commands, start=1):
        command_str = " ".join(item.command)
        print(f"[{idx}/{len(commands)}] {item.family}: {item.name}")
        print(f"  {command_str}")
        if item.expected_path is not None:
            print(f"  expected: {item.expected_path.relative_to(REPO_ROOT)}")
            if item.expected_path.exists() and not args.overwrite_existing:
                raise FileExistsError(
                    f"Refusing to overwrite existing test_other output: {item.expected_path.relative_to(REPO_ROOT)}. "
                    "Pass --overwrite-existing to allow this."
                )
        if args.execute:
            subprocess.run(item.command, cwd=REPO_ROOT, check=True)


if __name__ == "__main__":
    main()
