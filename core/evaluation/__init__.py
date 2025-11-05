"""Evaluation and metrics modules."""

from .eval_utils import (
    TaskEvalSetup,
    get_registered_eval_tasks,
    load_model_and_processor,
    prepare_task_for_evaluation,
    register_eval_task,
    run_evaluation,
)
from .metrics import compute_wer_from_texts, decode_tokens, sanitize_token_array
from .plotting import plot_loss_and_wer

__all__ = [
    "TaskEvalSetup",
    "get_registered_eval_tasks",
    "load_model_and_processor",
    "prepare_task_for_evaluation",
    "register_eval_task",
    "run_evaluation",
    "compute_wer_from_texts",
    "decode_tokens",
    "sanitize_token_array",
    "plot_loss_and_wer",
]
