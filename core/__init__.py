"""Core utilities for speech model experiments.

The package root exposes commonly used helpers lazily so importing lightweight
subpackages such as ``core.results`` does not require optional data/model
dependencies.
"""

from __future__ import annotations

from importlib import import_module
from typing import Any

_EXPORT_MODULES = {
    # data
    "dump_json": "core.data",
    "ensure_dir": "core.data",
    "load_config": "core.data",
    "add_duration": "core.data",
    "build_manifest": "core.data",
    "build_split_metadata": "core.data",
    "compute_speaker_stats": "core.data",
    "compute_split_hours": "core.data",
    "filter_dataset_columns": "core.data",
    "hours_key": "core.data",
    "hours_to_seconds": "core.data",
    "load_cached_split": "core.data",
    "normalize_audio": "core.data",
    "normalize_split_metadata": "core.data",
    "num_proc_map_kwargs": "core.data",
    "resolve_num_proc": "core.data",
    "save_cached_split": "core.data",
    "select_indices_by_duration": "core.data",
    "select_random_indices": "core.data",
    "subset_dataset_by_metadata": "core.data",
    # evaluation
    "compute_wer_from_texts": "core.evaluation",
    "decode_tokens": "core.evaluation",
    "sanitize_token_array": "core.evaluation",
    "load_model_and_processor": "core.evaluation",
    "prepare_task_for_evaluation": "core.evaluation",
    "register_eval_task": "core.evaluation",
    "get_registered_eval_tasks": "core.evaluation",
    "TaskEvalSetup": "core.evaluation",
    "run_evaluation": "core.evaluation",
    "plot_loss_and_wer": "core.evaluation",
    "plot_confusion_matrix": "core.evaluation",
    "compute_base_cache_path": "core.evaluation",
    "compute_eval_subset_tag": "core.evaluation",
    "compute_task_eval_subset_tag": "core.evaluation",
    "resolve_merged_eval_dir": "core.evaluation",
    "print_metrics": "core.evaluation",
    # models
    "load_qwen_asr_model": "core.models",
    "load_qwen_model": "core.models",
    # tasks
    "BaseTaskConfig": "core.tasks",
    "create_simple_task_config": "core.tasks",
    "BaseAudioTextCollator": "core.tasks",
    "BaseClassificationCollator": "core.tasks",
    "BaseGenerationCollator": "core.tasks",
    # training
    "BalancedBatchSampler": "core.training",
    "CustomTrainer": "core.training",
    "WeightedClassSampler": "core.training",
    "save_artifacts": "core.training",
    "save_history_to_csv": "core.training",
    "TrainingConfig": "core.training",
    "parse_training_config": "core.training",
    "build_training_arguments": "core.training",
    "build_early_stopping_kwargs": "core.training",
    "build_history_record": "core.training",
    "create_classification_constraint": "core.training",
    "create_multi_token_constraint": "core.training",
    "run_training_with_evaluation": "core.training",
    # utils
    "setup_logger": "core.utils",
    "set_global_seed": "core.utils",
}

__all__ = list(_EXPORT_MODULES)


def __getattr__(name: str) -> Any:
    if name not in _EXPORT_MODULES:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module = import_module(_EXPORT_MODULES[name])
    value = getattr(module, name)
    globals()[name] = value
    return value
