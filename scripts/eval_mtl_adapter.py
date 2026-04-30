"""Evaluate a saved MTL adapter on a list of tasks, saving results only to the MTL metrics dir."""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.data.io_utils import load_config
from core.evaluation.split_utils import (
    SUPPORTED_EVAL_SPLITS,
    apply_task_split_overrides,
    canonical_output_split,
    task_data_split,
)
from core.evaluation.eval_utils import (
    load_model_and_processor,
    prepare_task_for_evaluation,
    run_evaluation,
)
from core.evaluation.evaluate_task import prepare_dataset_cache, _resolve_batch_size, _get_generation_kwargs
from merging.evaluation.interference import maybe_add_interference_delta, maybe_compute_interference_baselines
from merging.runtime.utils import get_task_module


def _dedupe_preserve_order(values: list[str]) -> list[str]:
    ordered: list[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        ordered.append(value)
    return ordered


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved MTL LoRA adapter on one or more tasks."
    )
    parser.add_argument("--adapter", required=True, help="Path to the adapter directory.")
    parser.add_argument(
        "--tasks",
        nargs="+",
        required=True,
        help="Tasks to evaluate (e.g. emotion intent kws langid speaker_ver vocalsound asr speech_qa).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=SUPPORTED_EVAL_SPLITS,
        help="Dataset split to evaluate on (default: test). Use test-other for ASR LibriSpeech test.other.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device eval batch size override (defaults to task config value).")
    parser.add_argument(
        "--metrics-dir",
        default=None,
        help=(
            "Directory to save eval_results JSON. "
            "Defaults to <task_set_root>/metrics/<run_name>/ (three levels above the adapter run dir)."
        ),
    )
    parser.add_argument(
        "--no-save", action="store_true", help="Skip saving eval_results JSON."
    )
    parser.add_argument(
        "--no-interference-baselines",
        action="store_true",
        help="Skip auto-computing base_model/best_adapter baselines needed for interference_delta.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    adapter_path = Path(args.adapter).expanduser()
    if not adapter_path.is_absolute():
        adapter_path = Path.cwd() / adapter_path
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path not found: {adapter_path}")

    tasks = list(args.tasks)
    requested_split = args.split
    output_split = canonical_output_split(requested_split)
    batch_size = args.batch_size  # None means use per-task config value

    if not args.no_interference_baselines:
        maybe_compute_interference_baselines(
            tasks=tasks,
            split=requested_split,
            enable_cache=False,
            batch_size=batch_size,  # None lets each task use its own config default
            show_summary=True,
        )

    model_path = (PACKAGE_ROOT / "data/models/Qwen2.5-Omni-3B").resolve()
    print(f"\n🔧 Loading model + adapter from {adapter_path}...")
    model, processor = load_model_and_processor(model_path=model_path, adapter_path=adapter_path)

    results = {}
    for i, task in enumerate(tasks, 1):
        data_split = task_data_split(task, requested_split)
        print(f"\n[{i}/{len(tasks)}] Evaluating '{task}' on {output_split} split...")
        try:
            task_module = get_task_module(task)
            task_cfg = load_config(task_module.get_config_path(PACKAGE_ROOT, None))
            task_cfg, _split_metadata = apply_task_split_overrides(
                task=task,
                config=task_cfg,
                requested_split=requested_split,
            )
            task_cfg = prepare_dataset_cache(task_cfg, task_module.get_artifact_directories(PACKAGE_ROOT))
            setup = prepare_task_for_evaluation(task, processor, split=data_split, config=task_cfg)
            task_batch_size = _resolve_batch_size(args, task_cfg)
            task_generation_kwargs = _get_generation_kwargs(task_cfg)
            metrics = run_evaluation(
                model,
                setup,
                batch_size=task_batch_size,
                generation_kwargs=task_generation_kwargs,
                processor=processor,
            )
            if task == "asr":
                setup_meta = getattr(setup, "metadata", None) or {}
                metrics.update(
                    {
                        "dataset": "librispeech_asr",
                        "dataset_config": "clean",
                        "requested_split": requested_split,
                        "resolved_split": _split_metadata.get("resolved_split"),
                        "output_split": output_split,
                        "data_split": data_split,
                        "num_samples_before_filter": int(
                            setup_meta.get("num_samples_before_post_load_hook", len(setup.dataset))
                        ),
                        "num_samples_after_filter": int(
                            setup_meta.get("num_samples_after_post_load_hook", len(setup.dataset))
                        ),
                        "num_samples_evaluated": len(setup.dataset),
                    }
                )
            results[task] = metrics
            maybe_add_interference_delta(task, results[task], requested_split, show_summary=True, eval_tag=None)
            print(f"  ✅ {task}: {results[task]}")
        except Exception as exc:
            print(f"  ❌ Failed to evaluate '{task}': {exc}")
            results[task] = {"error": str(exc)}

    print("\n===== MTL Adapter Evaluation Summary =====")
    for task, metrics in results.items():
        if "error" in metrics:
            print(f"  {task}: ERROR — {metrics['error']}")
        else:
            delta = metrics.get("interference_delta")
            delta_str = f", interference_delta={delta:.4f}" if isinstance(delta, float) else ""
            primary = {k: v for k, v in metrics.items() if not k.startswith("_")}
            print(f"  {task}: {primary}{delta_str}")

    if not args.no_save:
        if args.metrics_dir is not None:
            metrics_dir = Path(args.metrics_dir).expanduser()
        else:
            # Layout: <task_set_root>/adapters/<subdir>/<run_name>/
            # Mirror _copy_mtl_metrics_to_subdirs: save under metrics/<run_name>/
            run_name = adapter_path.name  # e.g. "best", "latest", "run_20260309_120000"
            task_set_root = adapter_path.parent.parent.parent
            metrics_dir = task_set_root / "metrics" / run_name
        metrics_dir.mkdir(parents=True, exist_ok=True)

        out_path = metrics_dir / f"eval_results_{output_split}.json"
        existing_tasks: list[str] = []
        merged_results: dict = {}
        if out_path.exists():
            try:
                existing = json.loads(out_path.read_text(encoding="utf-8"))
                existing_results = existing.get("results", {})
                if isinstance(existing_results, dict):
                    merged_results.update(existing_results)
                if isinstance(existing.get("tasks"), list):
                    existing_tasks = [str(t) for t in existing["tasks"]]
            except Exception:
                pass

        merged_results.update(results)
        merged_tasks = _dedupe_preserve_order(existing_tasks + tasks)

        summary = {
            "split": output_split,
            "requested_split": requested_split,
            "output_split": output_split,
            "timestamp": datetime.now().isoformat(),
            "adapter_path": str(adapter_path),
            "tasks": merged_tasks,
            "last_requested_tasks": tasks,
            "results": merged_results,
        }
        with out_path.open("w", encoding="utf-8") as fh:
            json.dump(summary, fh, indent=2)
        print(f"\n💾 Aggregated results saved to {out_path}")

        for task, task_metrics in results.items():
            per_task_dir = metrics_dir / "per_task" / task
            per_task_dir.mkdir(parents=True, exist_ok=True)
            per_task_summary = {
                "split": output_split,
                "requested_split": requested_split,
                "output_split": output_split,
                "timestamp": summary["timestamp"],
                "adapter_path": str(adapter_path),
                "task": task,
                "metrics": task_metrics,
            }
            per_task_path = per_task_dir / f"eval_results_{output_split}.json"
            with per_task_path.open("w", encoding="utf-8") as fh:
                json.dump(per_task_summary, fh, indent=2)
            print(f"💾 Per-task results saved to {per_task_path}")


if __name__ == "__main__":
    main()
