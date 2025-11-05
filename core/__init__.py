"""Core utilities for speech model experiments."""

from .dataset_utils import (
    add_duration,
    build_manifest,
    build_split_metadata,
    compute_split_hours,
    filter_dataset_columns,
    hours_key,
    hours_to_seconds,
    load_cached_split,
    normalize_split_metadata,
    num_proc_map_kwargs,
    resolve_num_proc,
    save_cached_split,
    select_indices_by_duration,
    select_random_indices,
    subset_dataset_by_metadata,
)
from .io_utils import dump_json, ensure_dir, load_config
from .eval_utils import (
    TaskEvalSetup,
    get_registered_eval_tasks,
    load_model_and_processor,
    prepare_task_for_evaluation,
    register_eval_task,
    run_evaluation,
)
from .logger import setup_logger
from .metrics import compute_wer_from_texts, decode_tokens, sanitize_token_array
from .models import load_qwen_asr_model, load_qwen_model
from .plotting import plot_loss_and_wer
from .seed_utils import set_global_seed
from .trainer import CustomTrainer, save_artifacts, save_history_to_csv
from .training_config import (
    TrainingConfig,
    build_early_stopping_kwargs,
    build_training_arguments,
    parse_training_config,
)
from .training_loop import build_history_record, run_training_with_evaluation

__all__ = [
    "dump_json",
    "ensure_dir",
    "load_config",
    "add_duration",
    "build_manifest",
    "build_split_metadata",
    "compute_split_hours",
    "filter_dataset_columns",
    "hours_key",
    "hours_to_seconds",
    "load_cached_split",
    "normalize_split_metadata",
    "num_proc_map_kwargs",
    "resolve_num_proc",
    "save_cached_split",
    "select_indices_by_duration",
    "select_random_indices",
    "subset_dataset_by_metadata",
    "setup_logger",
    "compute_wer_from_texts",
    "decode_tokens",
    "sanitize_token_array",
    "load_qwen_asr_model",
    "load_qwen_model",
    "load_model_and_processor",
    "prepare_task_for_evaluation",
    "register_eval_task",
    "get_registered_eval_tasks",
    "TaskEvalSetup",
    "run_evaluation",
    "plot_loss_and_wer",
    "set_global_seed",
    "CustomTrainer",
    "save_artifacts",
    "save_history_to_csv",
    "TrainingConfig",
    "parse_training_config",
    "build_training_arguments",
    "build_early_stopping_kwargs",
    "build_history_record",
    "run_training_with_evaluation",
]
