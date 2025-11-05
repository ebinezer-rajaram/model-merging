"""Pydantic schemas for configuration validation."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Literal

from pydantic import BaseModel, Field, field_validator


class LoraConfig(BaseModel):
    """LoRA (Low-Rank Adaptation) configuration."""

    r: int = Field(default=32, ge=1, le=512, description="LoRA rank")
    alpha: int = Field(default=64, ge=1, description="LoRA alpha scaling factor")
    dropout: float = Field(default=0.1, ge=0.0, le=1.0, description="LoRA dropout probability")
    bias: Literal["none", "all", "lora_only"] = Field(
        default="none",
        description="Bias training configuration"
    )
    target_modules: List[str] = Field(
        default_factory=lambda: [
            "q_proj", "k_proj", "v_proj", "o_proj",
            "out_proj", "fc1", "fc2"
        ],
        description="List of module names to apply LoRA to"
    )
    task_type: str = Field(default="CAUSAL_LM", description="PEFT task type")

    model_config = {"extra": "forbid"}  # Reject unknown fields


class ModelConfig(BaseModel):
    """Model configuration."""

    path: str = Field(
        default="models/Qwen2.5-Omni-3B",
        description="Path to the base model"
    )
    lora: Optional[LoraConfig] = Field(
        default_factory=LoraConfig,
        description="LoRA configuration (None to disable LoRA)"
    )

    model_config = {"extra": "allow"}  # Allow additional model-specific fields


class DatasetConfig(BaseModel):
    """Dataset configuration."""

    # Common fields - specific tasks may have additional fields
    name: Optional[str] = None
    train_split: str = Field(default="train", description="Training split name")
    eval_split: str = Field(default="test", description="Evaluation split name")

    model_config = {"extra": "allow"}  # Allow task-specific dataset fields


class GenerationConfig(BaseModel):
    """Text generation configuration for evaluation."""

    max_new_tokens: int = Field(default=128, ge=1, description="Maximum tokens to generate")
    do_sample: bool = Field(default=False, description="Whether to use sampling")
    temperature: float = Field(default=0.0, ge=0.0, description="Sampling temperature")
    num_beams: int = Field(default=1, ge=1, description="Number of beams for beam search")

    model_config = {"extra": "allow"}  # Allow additional generation params


class TrainingConfig(BaseModel):
    """Training configuration."""

    # Batch sizes
    per_device_train_batch_size: int = Field(default=8, ge=1)
    per_device_eval_batch_size: int = Field(default=8, ge=1)
    gradient_accumulation_steps: int = Field(default=1, ge=1)

    # Optimization
    learning_rate: float = Field(default=5e-5, gt=0.0)
    weight_decay: float = Field(default=0.01, ge=0.0)
    max_grad_norm: float = Field(default=1.0, ge=0.0)
    warmup_ratio: float = Field(default=0.05, ge=0.0, le=1.0)
    warmup_steps: Optional[int] = Field(default=None, ge=0)

    # Training duration
    num_train_epochs: int = Field(default=5, ge=1)
    max_steps: int = Field(default=-1, description="-1 means use num_train_epochs")

    # Scheduler
    lr_scheduler_type: str = Field(default="cosine")

    # Logging and checkpointing
    logging_steps: int = Field(default=50, ge=1)
    save_steps: int = Field(default=250, ge=1)
    save_total_limit: int = Field(default=2, ge=1)
    eval_steps: int = Field(default=250, ge=1)

    # Early stopping
    early_stopping_patience: int = Field(default=3, ge=1)
    metric_for_best_model: str = Field(default="macro_f1")
    greater_is_better: bool = Field(default=True)

    # Precision
    bf16: bool = Field(default=True)
    fp16: bool = Field(default=False)

    # Other
    length_column_name: Optional[str] = Field(
        default=None,
        description="Column name for sequence length (for length-based sampling)"
    )

    model_config = {"extra": "allow"}  # Allow additional training args


class EvaluationConfig(BaseModel):
    """Evaluation-specific configuration."""

    generation: Optional[GenerationConfig] = Field(
        default_factory=GenerationConfig,
        description="Generation parameters for evaluation"
    )

    model_config = {"extra": "allow"}  # Allow task-specific eval fields


class ArtifactsConfig(BaseModel):
    """Artifact storage configuration."""

    adapter_subdir: str = Field(description="Subdirectory name for LoRA adapters")

    model_config = {"extra": "allow"}


class MetricsConfig(BaseModel):
    """Metrics logging configuration."""

    history_csv: Optional[str] = Field(default=None, description="CSV filename for metrics history")
    loss_plot: Optional[str] = Field(default=None, description="PNG filename for loss plot")

    model_config = {"extra": "allow"}


class TaskConfig(BaseModel):
    """Complete task configuration."""

    task: str = Field(description="Task name (asr, emotion, intent, speaker_id, speech_qa)")
    seed: int = Field(default=0, ge=0, description="Random seed for reproducibility")

    model: ModelConfig = Field(default_factory=ModelConfig)
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    training: TrainingConfig = Field(default_factory=TrainingConfig)
    evaluation: Optional[EvaluationConfig] = Field(default_factory=EvaluationConfig)
    artifacts: ArtifactsConfig
    metrics: Optional[MetricsConfig] = Field(default=None)

    model_config = {"extra": "forbid"}  # Top-level config should not have extra fields

    @field_validator("task")
    @classmethod
    def validate_task_name(cls, v: str) -> str:
        """Validate task name is one of the supported tasks."""
        valid_tasks = {"asr", "emotion", "intent", "speaker_id", "speech_qa"}
        if v not in valid_tasks:
            raise ValueError(f"Task must be one of {valid_tasks}, got: {v}")
        return v
