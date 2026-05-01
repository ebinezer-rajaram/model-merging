"""Unified experiment result schema — typed dataclasses."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

ExperimentType = Literal["single_task", "mtl", "merge"]

# Evaluation context:
#   full       — normal in-distribution evaluation on a trained task
#   subset     — eval_tag present; results cover a data subset only
#   cross_task — task not in source_tasks / training tasks (not a designated OOD task)
#   ood        — task in HELD_OUT_TASKS and not in the training task set
EvalContext = Literal["full", "subset", "cross_task", "ood"]


@dataclass
class TrainingHyperparameters:
    """Subset of training configuration relevant to comparing results."""

    learning_rate: Optional[float] = None
    lora_r: Optional[int] = None
    lora_alpha: Optional[int] = None
    num_train_epochs: Optional[int] = None
    per_device_train_batch_size: Optional[int] = None
    # MTL-only fields
    sampling_temperature: Optional[float] = None
    selection_criterion: Optional[str] = None  # e.g. "geometric_mean_interference_delta"
    num_tasks: Optional[int] = None


@dataclass
class InterferenceDeltaInfo:
    """Interference delta with full provenance for reproducibility.

    Formula: (oriented(merged) - oriented(base)) / (oriented(task_adapter) - oriented(base))
    where oriented() negates the value for lower-is-better metrics (e.g. WER).
    """

    value: float
    metric_name: str           # "macro_f1", "wer", "accuracy", …
    base_score: float          # base model performance on this metric
    task_adapter_score: float  # best single-task adapter performance
    merged_score: float        # raw metric value of the evaluated (merged/MTL) model


@dataclass
class TaskEvalResult:
    """Evaluation result for one task in one experiment."""

    task: str
    split: str                              # "test", "validation"
    eval_context: EvalContext = "full"

    # Primary task metrics — which are populated depends on the task.
    loss: Optional[float] = None
    accuracy: Optional[float] = None
    macro_f1: Optional[float] = None
    weighted_f1: Optional[float] = None
    wer: Optional[float] = None
    recognized_rate: Optional[float] = None

    # Interference delta (None for baseline single-task experiments).
    interference_delta: Optional[float] = None
    interference_delta_info: Optional[InterferenceDeltaInfo] = None

    # Runtime metadata.
    num_samples: Optional[int] = None
    runtime_seconds: Optional[float] = None

    # Eval subset tag (non-empty means this result covers a data subset).
    eval_tag: Optional[str] = None

    # Catch-all for any additional metrics not covered above.
    # These are flattened into extra long-format rows by the aggregator.
    extra_metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ExperimentMetadata:
    """Provenance for one experiment result."""

    experiment_type: ExperimentType
    experiment_id: str   # stable, lambda-free — see utils.make_experiment_id()
    method: str          # task name (single_task) | "mtl" | merge method name

    # Selection provenance — how this result was identified as the "best" run.
    selection_policy: Optional[str] = None          # "best_run_metric", "best_delta", "sweep_point", …
    selection_metric_name: Optional[str] = None     # metric name used to select this run/checkpoint
    selection_metric_value: Optional[float] = None  # value of that metric at the time of selection

    # Run identity.
    config_name: Optional[str] = None   # yaml filename or config hash
    seed: Optional[int] = None
    run_id: Optional[str] = None
    timestamp: Optional[str] = None     # ISO 8601

    # Task provenance.
    source_tasks: List[str] = field(default_factory=list)
    source_tasks_key: Optional[str] = None  # "+".join(sorted(source_tasks))

    # Merge-specific fields.
    merge_lambda: Optional[float] = None
    merge_tag: Optional[str] = None

    # Filesystem provenance.
    adapter_subdir: Optional[str] = None
    model_path: Optional[str] = None
    source_path: Optional[str] = None   # path of the result file that was read


@dataclass
class ExperimentResult:
    """Top-level container: one (experiment_run, task, split) triplet."""

    metadata: ExperimentMetadata
    task_result: TaskEvalResult
    hyperparameters: Optional[TrainingHyperparameters] = None

    # MTL aggregate metrics — duplicated across per-task rows for easy groupby.
    mtl_geometric_mean_interference_delta: Optional[float] = None
    mtl_arithmetic_mean_interference_delta: Optional[float] = None
    mtl_num_tasks_with_delta: Optional[int] = None
