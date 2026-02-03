"""Evaluation and metrics modules."""

from .eval_utils import (
    TaskEvalSetup,
    _compute_base_cache_path,
    get_registered_eval_tasks,
    load_model_and_processor,
    _print_metrics,
    prepare_task_for_evaluation,
    register_eval_task,
    _resolve_merged_eval_dir,
    run_evaluation,
)
from .metrics import compute_wer_from_texts, decode_tokens, sanitize_token_array
from .plotting import plot_confusion_matrix, plot_loss_and_wer

__all__ = [
    "TaskEvalSetup",
    "_compute_base_cache_path",
    "get_registered_eval_tasks",
    "load_model_and_processor",
    "_print_metrics",
    "prepare_task_for_evaluation",
    "register_eval_task",
    "_resolve_merged_eval_dir",
    "run_evaluation",
    "compute_wer_from_texts",
    "decode_tokens",
    "sanitize_token_array",
    "plot_loss_and_wer",
    "plot_confusion_matrix",
]
