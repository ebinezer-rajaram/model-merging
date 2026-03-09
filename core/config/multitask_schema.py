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

    # Reuse existing training knobs by allowing extras here.
    model_config = {"extra": "allow"}


class MultiTaskLoggingConfig(BaseModel):
    """Optional logging-specific controls."""

    wandb_project: str = Field(default="speech-merging-mtl")

    model_config = {"extra": "allow"}


class MultiTaskArtifactsConfig(BaseModel):
    """Artifact storage configuration for MTL runs."""

    adapter_subdir: str = Field(description="Subdirectory name for LoRA adapters")
    layout: Literal["task_set"] = Field(
        default="task_set",
        description="MTL artifact layout strategy.",
    )
    task_set_slug_mode: Literal["sorted_names"] = Field(
        default="sorted_names",
        description="How to derive the task-set slug directory name.",
    )

    model_config = {"extra": "allow"}


class MultiTaskMetricsConfig(BaseModel):
    """Metrics output and plotting controls for MTL runs."""

    auto_plot: bool = Field(default=True, description="Auto-generate MTL metric trend plots.")

    model_config = {"extra": "allow"}


class MultiTaskConfig(BaseModel):
    """Top-level config model for joint multi-task training."""

    seed: int = Field(default=0, ge=0)
    model: ModelConfig = Field(default_factory=ModelConfig)
    training: MultiTaskTrainingConfig = Field(default_factory=MultiTaskTrainingConfig)
    artifacts: MultiTaskArtifactsConfig
    metrics: Optional[MultiTaskMetricsConfig] = Field(default_factory=MultiTaskMetricsConfig)
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
    "MultiTaskConfig",
    "parse_multitask_config",
]
