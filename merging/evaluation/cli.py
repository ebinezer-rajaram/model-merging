"""CLI for evaluating merged adapters."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional

from merging.engine.registry import list_merge_methods
from merging.config.unified import load_merge_config
from merging.evaluation.evaluate import evaluate_merged_adapter
from core.evaluation.split_utils import SUPPORTED_EVAL_SPLITS


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    merge_methods = tuple(list_merge_methods())
    parser = argparse.ArgumentParser(description="Evaluate merged adapters on one or more tasks.")
    parser.add_argument("--adapter-path", default=None, help="Path to merged adapter run or base directory.")
    parser.add_argument(
        "--config",
        default=None,
        help="Optional merge config YAML used to source method/tasks/params for checkpoint replay.",
    )
    parser.add_argument(
        "--checkpoint",
        default=None,
        help="Path to a SuperMerge checkpoint file (.pt) to evaluate directly.",
    )
    parser.add_argument(
        "--optimizer-steps",
        type=int,
        default=None,
        help="Override optimizer steps for checkpoint replay (defaults to checkpoint's optimizer_step).",
    )
    parser.add_argument(
        "--method",
        default=None,
        choices=merge_methods,
        help="Merge method used (for resolving adapter path).",
    )
    parser.add_argument(
        "--tasks",
        nargs="+",
        default=None,
        help="Tasks merged into the adapter (e.g., asr intent).",
    )
    parser.add_argument(
        "--lambda",
        type=float,
        dest="lambda_weight",
        default=None,
        help="Lambda weight for weighted merges (optional).",
    )
    parser.add_argument(
        "--scale",
        type=float,
        default=None,
        help="Global scale for uniform_scalar_delta merges (optional).",
    )
    parser.add_argument(
        "--run-id",
        default=None,
        help="Run identifier to resolve (best, latest, or run_YYYYMMDD_HHMMSS).",
    )
    parser.add_argument(
        "--eval-tasks",
        nargs="+",
        default=None,
        help="Tasks to evaluate on (defaults to merged source tasks).",
    )
    parser.add_argument(
        "--split",
        default="test",
        choices=SUPPORTED_EVAL_SPLITS,
        help="Dataset split to evaluate. Use test-other for ASR LibriSpeech test.other.",
    )
    parser.add_argument("--batch-size", type=int, default=None, help="Per-device eval batch size.")
    parser.add_argument(
        "--confusion-matrix",
        action="store_true",
        dest="confusion_matrix",
        help="Generate confusion matrix for classification tasks.",
    )
    parser.add_argument(
        "--no-confusion-matrix",
        action="store_false",
        dest="confusion_matrix",
        help="Disable confusion matrix generation (default: True).",
    )
    parser.add_argument(
        "--no-save-results",
        action="store_false",
        dest="save_results",
        help="Disable saving merged evaluation summary.",
    )
    parser.add_argument(
        "--save",
        action="store_true",
        dest="save_merged",
        help="Save merged adapter before evaluation (default: in-memory only).",
    )
    parser.add_argument(
        "--no-compute-interference-baselines",
        action="store_false",
        dest="compute_missing_interference_baselines",
        help="Do not auto-compute base_model/best_adapter metrics needed for interference_delta.",
    )
    parser.set_defaults(compute_missing_interference_baselines=True)
    parser.set_defaults(confusion_matrix=True, save_results=True)
    return parser.parse_args()


def evaluate_from_args(args: argparse.Namespace) -> dict:
    """Execute merged adapter evaluation based on CLI args."""
    params: Dict[str, Any] = {}
    method = args.method
    task_names = _normalize_list(args.tasks)
    merge_mode = "common"

    if args.config is not None:
        config = load_merge_config(Path(args.config))
        if method is None:
            method = config.method
        if task_names is None:
            task_names = list(config.adapters)
        merge_mode = config.merge_mode
        params.update(dict(config.method_params))
        if config.lambda_policy is not None:
            params["lambda_policy"] = {
                "type": config.lambda_policy.type,
                "value": config.lambda_policy.value,
                "default": config.lambda_policy.default,
                "overrides": dict(config.lambda_policy.overrides),
            }
        if config.optimizer is not None:
            params["optimizer"] = {
                "type": config.optimizer.type,
                "params": dict(config.optimizer.params),
            }

    if args.scale is not None:
        params["scale"] = float(args.scale)
    if args.lambda_weight is not None:
        params["lambda"] = float(args.lambda_weight)

    if args.checkpoint is not None:
        if args.adapter_path is not None:
            raise ValueError("--checkpoint cannot be used together with --adapter-path.")
        if method is None:
            raise ValueError("--method is required when using --checkpoint (or provide --config).")
        if task_names is None:
            raise ValueError("--tasks are required when using --checkpoint (or provide --config).")

        checkpoint_path = Path(args.checkpoint).expanduser()
        if not checkpoint_path.is_absolute():
            checkpoint_path = Path.cwd() / checkpoint_path
        if not checkpoint_path.exists() or not checkpoint_path.is_file():
            raise ValueError(f"Checkpoint not found: {checkpoint_path}")

        optimizer_steps = args.optimizer_steps
        if optimizer_steps is None:
            optimizer_steps = _load_checkpoint_optimizer_step(checkpoint_path)
        if optimizer_steps is None:
            raise ValueError(
                f"Could not infer optimizer_step from checkpoint: {checkpoint_path}. "
                "Pass --optimizer-steps explicitly."
            )

        optimizer_cfg = params.get("optimizer")
        if isinstance(optimizer_cfg, Mapping):
            optimizer_cfg = {
                "type": str(optimizer_cfg.get("type", "supermerge")),
                "params": dict(optimizer_cfg.get("params", {})) if isinstance(optimizer_cfg.get("params"), Mapping) else {},
            }
        else:
            optimizer_cfg = {"type": "supermerge", "params": {}}

        if str(optimizer_cfg.get("type", "")).lower() not in {"", "supermerge"}:
            raise ValueError(
                f"--checkpoint currently supports optimizer.type=supermerge, got '{optimizer_cfg.get('type')}'."
            )
        optimizer_cfg["type"] = "supermerge"
        optimizer_params = dict(optimizer_cfg.get("params", {}))
        optimizer_params["resume_checkpoint"] = str(checkpoint_path)
        optimizer_params["steps"] = int(optimizer_steps)
        # Replay-only: avoid creating new heldout checkpoints during evaluation.
        optimizer_params["checkpoint_on_heldout_eval"] = False
        optimizer_cfg["params"] = optimizer_params
        params["optimizer"] = optimizer_cfg

        return evaluate_merged_adapter(
            adapter_path=None,
            method=method,
            task_names=task_names,
            params=params or None,
            run_id=None,
            eval_tasks=_normalize_list(args.eval_tasks),
            split=args.split,
            batch_size=args.batch_size,
            generate_confusion_matrix=args.confusion_matrix,
            save_merged=args.save_merged,
            save_results=args.save_results,
            show_summary=True,
            merge_mode=merge_mode,
            compute_missing_interference_baselines=args.compute_missing_interference_baselines,
        )

    return evaluate_merged_adapter(
        adapter_path=args.adapter_path,
        method=method,
        task_names=task_names,
        lambda_weight=args.lambda_weight,
        params=params or None,
        run_id=args.run_id,
        eval_tasks=_normalize_list(args.eval_tasks),
        split=args.split,
        batch_size=args.batch_size,
        generate_confusion_matrix=args.confusion_matrix,
        save_merged=args.save_merged,
        save_results=args.save_results,
        show_summary=True,
        merge_mode=merge_mode,
        compute_missing_interference_baselines=args.compute_missing_interference_baselines,
    )


def _normalize_list(value: Optional[List[str]]) -> Optional[List[str]]:
    if value:
        return list(value)
    return None


def _load_checkpoint_optimizer_step(checkpoint_path: Path) -> Optional[int]:
    try:
        import torch
    except Exception:
        return None
    try:
        payload = torch.load(checkpoint_path, map_location="cpu")
    except Exception:
        return None
    if not isinstance(payload, Mapping):
        return None
    raw = payload.get("optimizer_step")
    if isinstance(raw, int):
        return raw
    if isinstance(raw, float):
        return int(raw)
    return None


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    evaluate_from_args(args)


if __name__ == "__main__":
    main()
