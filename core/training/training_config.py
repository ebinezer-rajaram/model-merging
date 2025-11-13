"""Training configuration parser and builder for unified training workflows."""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Any, Dict, Optional

from transformers import TrainingArguments


@dataclass
class TrainingConfig:
    """Parsed training configuration with all hyperparameters."""

    # Batch sizes and accumulation
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int

    # Learning rate and schedule
    learning_rate: float
    lr_scheduler_type: str
    warmup_ratio: float
    warmup_steps: int

    # Training duration
    num_train_epochs: int

    # Logging and checkpointing
    logging_steps: int
    save_strategy: str
    save_steps: int
    save_total_limit: int

    # Evaluation
    eval_strategy: str
    eval_steps: int
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    eval_accumulation_steps: Optional[int]

    # Regularization
    max_grad_norm: float
    weight_decay: float

    # Precision
    bf16: bool
    fp16: bool

    # Dataloader
    dataloader_num_workers: Optional[int]
    dataloader_pin_memory: bool
    dataloader_prefetch_factor: Optional[int]
    group_by_length: bool
    length_column_name: Optional[str]

    # System
    report_to: list
    gradient_checkpointing: bool
    gradient_checkpointing_kwargs: Dict[str, Any]
    remove_unused_columns: bool

    # Generation
    generation_kwargs: Dict[str, Any]

    # Early stopping
    early_stopping_patience: int
    early_stopping_threshold: Optional[float]

    # Evaluation
    initial_eval: bool

    # Resume from checkpoint
    resume_from_checkpoint: Optional[str]


def parse_training_config(
    training_cfg: Dict[str, Any],
    *,
    num_train_examples: int,
    task_defaults: Optional[Dict[str, Any]] = None,
) -> TrainingConfig:
    """Parse training configuration from a dictionary.

    Args:
        training_cfg: Raw training configuration dictionary
        num_train_examples: Number of training examples (for warmup calculation)
        task_defaults: Optional task-specific default overrides

    Returns:
        Parsed TrainingConfig object
    """
    # Merge task defaults if provided
    defaults = task_defaults or {}

    # Parse learning rate with validation
    learning_rate_raw = training_cfg.get("learning_rate", defaults.get("learning_rate", 2e-5))
    try:
        learning_rate = float(learning_rate_raw)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid 'learning_rate' value: {learning_rate_raw!r}") from exc

    # Get basic hyperparameters
    per_device_train_batch_size = training_cfg.get(
        "per_device_train_batch_size", defaults.get("per_device_train_batch_size", 8)
    )
    per_device_eval_batch_size = training_cfg.get(
        "per_device_eval_batch_size", defaults.get("per_device_eval_batch_size", 8)
    )
    gradient_accumulation_steps = training_cfg.get(
        "gradient_accumulation_steps", defaults.get("gradient_accumulation_steps", 1)
    )
    num_train_epochs = training_cfg.get(
        "num_train_epochs", defaults.get("num_train_epochs", 5)
    )
    warmup_ratio = training_cfg.get(
        "warmup_ratio", defaults.get("warmup_ratio", 0.05)
    )

    # Calculate warmup steps
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    updates_per_epoch = math.ceil(
        max(1, num_train_examples)
        / (per_device_train_batch_size * gradient_accumulation_steps * max(1, world_size))
    )
    total_training_steps = max(1, updates_per_epoch * max(1, int(num_train_epochs)))
    warmup_steps = max(1, int(total_training_steps * float(warmup_ratio)))

    # Generation kwargs
    default_generation = {"max_new_tokens": 16, "do_sample": False, "temperature": 0.0, "num_beams": 1}
    generation_kwargs = training_cfg.get(
        "generation_kwargs",
        defaults.get("generation_kwargs", default_generation)
    )

    return TrainingConfig(
        per_device_train_batch_size=per_device_train_batch_size,
        per_device_eval_batch_size=per_device_eval_batch_size,
        gradient_accumulation_steps=gradient_accumulation_steps,
        learning_rate=learning_rate,
        lr_scheduler_type=training_cfg.get(
            "lr_scheduler_type", defaults.get("lr_scheduler_type", "cosine")
        ),
        warmup_ratio=warmup_ratio,
        warmup_steps=warmup_steps,
        num_train_epochs=num_train_epochs,
        logging_steps=training_cfg.get(
            "logging_steps", defaults.get("logging_steps", 50)
        ),
        save_strategy=training_cfg.get(
            "save_strategy", defaults.get("save_strategy", "steps")
        ),
        save_steps=training_cfg.get(
            "save_steps", defaults.get("save_steps", 250)
        ),
        save_total_limit=training_cfg.get(
            "save_total_limit", defaults.get("save_total_limit", 2)
        ),
        eval_strategy=training_cfg.get(
            "eval_strategy", defaults.get("eval_strategy", "steps")
        ),
        eval_steps=training_cfg.get(
            "eval_steps", defaults.get("eval_steps", 250)
        ),
        load_best_model_at_end=training_cfg.get(
            "load_best_model_at_end", defaults.get("load_best_model_at_end", True)
        ),
        metric_for_best_model=training_cfg.get(
            "metric_for_best_model", defaults.get("metric_for_best_model", "macro_f1")
        ),
        greater_is_better=training_cfg.get(
            "greater_is_better", defaults.get("greater_is_better", True)
        ),
        eval_accumulation_steps=training_cfg.get(
            "eval_accumulation_steps", defaults.get("eval_accumulation_steps")
        ),
        max_grad_norm=training_cfg.get(
            "max_grad_norm", defaults.get("max_grad_norm", 1.0)
        ),
        weight_decay=training_cfg.get(
            "weight_decay", defaults.get("weight_decay", 0.01)
        ),
        bf16=training_cfg.get(
            "bf16", defaults.get("bf16", True)
        ),
        fp16=training_cfg.get(
            "fp16", defaults.get("fp16", False)
        ),
        dataloader_num_workers=training_cfg.get(
            "dataloader_num_workers", defaults.get("dataloader_num_workers")
        ),
        dataloader_pin_memory=training_cfg.get(
            "dataloader_pin_memory", defaults.get("dataloader_pin_memory", True)
        ),
        dataloader_prefetch_factor=training_cfg.get(
            "dataloader_prefetch_factor", defaults.get("dataloader_prefetch_factor")
        ),
        group_by_length=training_cfg.get(
            "group_by_length", defaults.get("group_by_length", False)
        ),
        length_column_name=training_cfg.get(
            "length_column_name", defaults.get("length_column_name", "duration")
        ),
        report_to=training_cfg.get(
            "report_to", defaults.get("report_to", ["tensorboard", "wandb"])
        ),
        gradient_checkpointing=training_cfg.get(
            "gradient_checkpointing", defaults.get("gradient_checkpointing", True)
        ),
        gradient_checkpointing_kwargs=training_cfg.get(
            "gradient_checkpointing_kwargs",
            defaults.get("gradient_checkpointing_kwargs", {"use_reentrant": False})
        ),
        remove_unused_columns=training_cfg.get(
            "remove_unused_columns", defaults.get("remove_unused_columns", False)
        ),
        generation_kwargs=generation_kwargs,
        early_stopping_patience=training_cfg.get(
            "early_stopping_patience", defaults.get("early_stopping_patience", 3)
        ),
        early_stopping_threshold=training_cfg.get(
            "early_stopping_threshold", defaults.get("early_stopping_threshold")
        ),
        initial_eval=training_cfg.get(
            "initial_eval", defaults.get("initial_eval", False)
        ),
        resume_from_checkpoint=training_cfg.get(
            "resume_from_checkpoint", defaults.get("resume_from_checkpoint")
        ),
    )


def build_training_arguments(
    config: TrainingConfig,
    *,
    output_dir: str,
    run_name: Optional[str] = None,
) -> TrainingArguments:
    """Build HuggingFace TrainingArguments from parsed config.

    Args:
        config: Parsed training configuration
        output_dir: Output directory for checkpoints
        run_name: Optional run name for wandb logging

    Returns:
        TrainingArguments instance
    """
    return TrainingArguments(
        output_dir=output_dir,
        run_name=run_name,
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        lr_scheduler_type=config.lr_scheduler_type,
        warmup_steps=config.warmup_steps,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_strategy=config.save_strategy,
        save_steps=config.save_steps,
        save_total_limit=config.save_total_limit,
        eval_strategy=config.eval_strategy,
        eval_steps=config.eval_steps,
        load_best_model_at_end=config.load_best_model_at_end,
        metric_for_best_model=config.metric_for_best_model,
        greater_is_better=config.greater_is_better,
        max_grad_norm=config.max_grad_norm,
        weight_decay=config.weight_decay,
        bf16=config.bf16,
        fp16=config.fp16,
        dataloader_num_workers=config.dataloader_num_workers,
        dataloader_pin_memory=config.dataloader_pin_memory,
        dataloader_prefetch_factor=config.dataloader_prefetch_factor,
        group_by_length=config.group_by_length,
        eval_accumulation_steps=config.eval_accumulation_steps,
        length_column_name=config.length_column_name,
        report_to=config.report_to,
        gradient_checkpointing=config.gradient_checkpointing,
        gradient_checkpointing_kwargs=config.gradient_checkpointing_kwargs,
        remove_unused_columns=config.remove_unused_columns,
    )


def build_early_stopping_kwargs(config: TrainingConfig) -> Dict[str, Any]:
    """Build early stopping callback kwargs from parsed config.

    Args:
        config: Parsed training configuration

    Returns:
        Dictionary of early stopping parameters
    """
    early_stopping_kwargs = {"early_stopping_patience": config.early_stopping_patience}
    if config.early_stopping_threshold is not None:
        early_stopping_kwargs["early_stopping_threshold"] = config.early_stopping_threshold
    return early_stopping_kwargs


__all__ = [
    "TrainingConfig",
    "parse_training_config",
    "build_training_arguments",
    "build_early_stopping_kwargs",
]
