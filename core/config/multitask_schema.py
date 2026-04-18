"""Pydantic schema for joint multi-task LoRA training configs."""

from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, field_validator

from core.config.registry import TASK_REGISTRY
from core.config.schemas import ModelConfig


class MultiTaskSpec(BaseModel):
    """One task entry in the joint training plan."""

    name: str = Field(description="Registered task key.")
    config: Optional[str] = Field(default=None, description="Optional task config filename override.")
    train_weight: float = Field(default=1.0, gt=0.0, description="Optional task-level weight.")

    model_config = {"extra": "forbid"}

    @field_validator("name")
    @classmethod
    def validate_name(cls, value: str) -> str:
        task = str(value).strip().lower()
        if task not in TASK_REGISTRY:
            valid = ", ".join(sorted(TASK_REGISTRY.keys()))
            raise ValueError(f"Unknown task '{value}'. Valid tasks: {valid}")
        return task


class MultiTaskSamplingConfig(BaseModel):
    """Task mixing configuration."""

    temperature: float = Field(default=1.0, gt=0.0, description="Temperature for size-based task sampling.")

    model_config = {"extra": "forbid"}


class MultiTaskTrainingConfig(BaseModel):
    """Training options for MTL runs."""

    sampling: MultiTaskSamplingConfig = Field(default_factory=MultiTaskSamplingConfig)

    selection_criterion: Literal[
        "geometric_mean_interference_delta",
        "arithmetic_mean_interference_delta",
    ] = Field(default="geometric_mean_interference_delta")
    selection_split: Literal["train", "validation", "test"] = Field(default="validation")
    selection_eval_subset: Optional[Dict[str, Any]] = Field(
        default=None,
        description="Optional eval subset config for faster model-selection evals.",
    )
    compute_missing_interference_baselines: bool = Field(default=True)
    final_eval_extra_tasks: List[str] = Field(
        default_factory=list,
        description="Optional extra tasks to include for final eval/test only (non-continual).",
    )
    final_eval_include_speech_qa: bool = Field(
        default=False,
        description="Include speech_qa in non-continual final eval/test task set.",
    )

    # Reuse existing training knobs by allowing extras here.
    model_config = {"extra": "allow"}

    @field_validator("final_eval_extra_tasks")
    @classmethod
    def validate_final_eval_extra_tasks(cls, tasks: List[str]) -> List[str]:
        normalized = []
        for raw in tasks:
            task = str(raw).strip().lower()
            if task not in TASK_REGISTRY:
                valid = ", ".join(sorted(TASK_REGISTRY.keys()))
                raise ValueError(f"Unknown task '{raw}'. Valid tasks: {valid}")
            normalized.append(task)
        return normalized


class MultiTaskLoggingConfig(BaseModel):
    """Optional logging-specific controls."""

    wandb_project: str = Field(default="speech-merging-mtl")

    model_config = {"extra": "allow"}


class MultiTaskArtifactsConfig(BaseModel):
    """Artifact storage configuration for MTL runs."""

    adapter_subdir: str = Field(description="Subdirectory name for LoRA adapters")
    root: str = Field(
        default="artifacts/mtl",
        description="Root directory for MTL artifacts.",
    )
    allow_mixed_output: bool = Field(
        default=False,
        description="Allow writing runs with a different mode into an existing task-set root.",
    )
    layout: Literal["task_set"] = Field(
        default="task_set",
        description="MTL artifact layout strategy.",
    )
    task_set_slug_mode: Literal["sorted_names", "base_then_added"] = Field(
        default="sorted_names",
        description="How to derive the task-set slug directory name.",
    )

    model_config = {"extra": "allow"}


class MultiTaskMetricsConfig(BaseModel):
    """Metrics output and plotting controls for MTL runs."""

    auto_plot: bool = Field(default=True, description="Auto-generate MTL metric trend plots.")

    model_config = {"extra": "allow"}


class MultiTaskContinualConfig(BaseModel):
    """Optional continual fine-tuning controls for MTL stage-2 updates."""

    enabled: bool = Field(default=False)
    base_adapter: Optional[str] = Field(
        default=None,
        description="Path to a base MTL adapter run/best/latest directory.",
    )
    base_adapter_run_id: Optional[str] = Field(
        default=None,
        description="Optional alias (best/latest/run_*) resolved under base_adapter when base_adapter points to adapter root.",
    )
    added_tasks: List[str] = Field(default_factory=list, description="Tasks to train in stage-2.")
    base_tasks_override: Optional[List[str]] = Field(
        default=None,
        description="Optional explicit base-task list; overrides metadata discovery.",
    )
    selection_mode: Literal["mtl_interference", "added_task_metric"] = Field(default="mtl_interference")
    selection_task_set: Literal["base_plus_added"] = Field(default="base_plus_added")
    final_eval_include_speech_qa: bool = Field(default=True)

    model_config = {"extra": "forbid"}

    @field_validator("added_tasks")
    @classmethod
    def validate_added_tasks(cls, tasks: List[str]) -> List[str]:
        normalized = []
        for raw in tasks:
            task = str(raw).strip().lower()
            if task not in TASK_REGISTRY:
                valid = ", ".join(sorted(TASK_REGISTRY.keys()))
                raise ValueError(f"Unknown task '{raw}'. Valid tasks: {valid}")
            normalized.append(task)
        return normalized

    @field_validator("base_tasks_override")
    @classmethod
    def validate_base_tasks_override(cls, tasks: Optional[List[str]]) -> Optional[List[str]]:
        if tasks is None:
            return None
        normalized = []
        for raw in tasks:
            task = str(raw).strip().lower()
            if task not in TASK_REGISTRY:
                valid = ", ".join(sorted(TASK_REGISTRY.keys()))
                raise ValueError(f"Unknown task '{raw}'. Valid tasks: {valid}")
            normalized.append(task)
        return normalized


class MultiTaskConfig(BaseModel):
    """Top-level config model for joint multi-task training."""

    seed: int = Field(default=0, ge=0)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: MultiTaskTrainingConfig = Field(default_factory=MultiTaskTrainingConfig)
    artifacts: MultiTaskArtifactsConfig
    metrics: Optional[MultiTaskMetricsConfig] = Field(default_factory=MultiTaskMetricsConfig)
    continual: Optional[MultiTaskContinualConfig] = Field(default_factory=MultiTaskContinualConfig)
    tasks: List[MultiTaskSpec] = Field(min_length=1)
    logging: Optional[MultiTaskLoggingConfig] = Field(default_factory=MultiTaskLoggingConfig)

    model_config = {"extra": "allow"}

    @field_validator("tasks")
    @classmethod
    def validate_unique_tasks(cls, tasks: List[MultiTaskSpec]) -> List[MultiTaskSpec]:
        names = [entry.name for entry in tasks]
        duplicates = sorted({name for name in names if names.count(name) > 1})
        if duplicates:
            raise ValueError(f"Duplicate task entries are not allowed: {duplicates}")
        return tasks


def parse_multitask_config(payload: Dict[str, Any]) -> MultiTaskConfig:
    """Validate and normalize MTL config payload."""
    return MultiTaskConfig(**payload)


__all__ = [
    "MultiTaskSpec",
    "MultiTaskSamplingConfig",
    "MultiTaskTrainingConfig",
    "MultiTaskLoggingConfig",
    "MultiTaskArtifactsConfig",
    "MultiTaskMetricsConfig",
    "MultiTaskContinualConfig",
    "MultiTaskConfig",
    "parse_multitask_config",
]
