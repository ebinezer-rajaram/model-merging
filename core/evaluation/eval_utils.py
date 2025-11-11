"""Evaluation utilities with a pluggable task registry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Protocol, Sequence, Tuple

import torch
from peft import PeftModel
from transformers import (
    IntervalStrategy,
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
    TrainingArguments,
)

from core.data.io_utils import ensure_dir
from core.training.trainer import CustomTrainer

DEFAULT_GENERATION_KWARGS: Dict[str, Any] = {"max_new_tokens": 128, "do_sample": False}

CLASSIFICATION_EVAL_KEYS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_config",
    "max_train_samples",
    "max_validation_samples",
    "max_test_samples",
    "label_column",
    "text_column",
    "audio_column",
    "split_percentages",
    "train_split",
    "validation_split",
    "test_split",
    "stratify_by_column",
    "revision",
    "data_dir",
)

SPEAKER_EXTRA_EVAL_KEYS: Tuple[str, ...] = (
    "max_speakers",
    "max_samples_per_speaker",
)

SPEECH_QA_EVAL_KEYS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_config",
    "max_train_samples",
    "max_validation_samples",
    "max_test_samples",
    "audio_column",
    "question_column",
    "transcript_column",
    "context_column",
    "id_column",
    "answer_column",
    "split_percentages",
    "train_split",
    "validation_split",
    "test_split",
)

@dataclass
class TaskEvalSetup:
    """Container describing the dataset and helpers needed for evaluation."""

    dataset: Any  # HuggingFace Dataset
    data_collator: Callable[[Any], Dict[str, Any]]
    compute_metrics: Callable[[Any], Dict[str, float]]
    label_names: Optional[List[str]] = None  # For classification tasks (confusion matrix)


@dataclass
class TaskConfig:
    """Configuration for building task evaluation setup using template pattern.

    This dataclass encapsulates all task-specific differences, allowing a single
    generic builder function to handle all tasks instead of duplicating code.
    """

    # Dataset loading
    dataset_loader: Callable[..., Tuple[Any, ...]]  # Returns (train, val, test, [extra])
    loader_config_keys: Tuple[str, ...]  # Config keys to pass to loader

    # Data collation
    collator_class: type  # The collator class to instantiate

    # Metrics computation
    compute_metrics_fn: Callable[..., Dict[str, float]]  # The metrics function

    # Optional boolean flags
    has_label_names: bool = False  # Whether loader returns label_names as 4th element
    has_answers_map: bool = False  # Whether loader returns answers_map (Speech QA only)

    # Optional parameters
    collator_params: Dict[str, Any] = None  # Extra params beyond processor, sampling_rate
    metrics_params: Dict[str, Any] = None  # Extra params beyond processor

    # Column management
    required_columns: Tuple[str, ...] = ()  # Columns that must be kept
    optional_columns: Tuple[str, ...] = ()  # Columns to keep if they exist

    # Optional post-processing hooks
    post_load_hook: Optional[Callable[[Any, Dict[str, Any], str], Any]] = None  # (dataset, config, split) -> dataset

    def __post_init__(self):
        """Ensure mutable defaults are properly initialized."""
        if self.collator_params is None:
            self.collator_params = {}
        if self.metrics_params is None:
            self.metrics_params = {}


class EvalTaskBuilder(Protocol):
    """Protocol implemented by task-specific evaluation builders."""

    def __call__(
        self,
        *,
        processor: Qwen2_5OmniProcessor,
        split: str,
        config: Optional[Dict[str, Any]] = None,
    ) -> TaskEvalSetup:
        ...


_TASK_REGISTRY: Dict[str, Tuple[str, EvalTaskBuilder]] = {}


def register_eval_task(name: str, builder: EvalTaskBuilder, *, overwrite: bool = False) -> None:
    """Register a task-specific evaluation builder."""
    canonical = (name or "").strip()
    if not canonical:
        raise ValueError("Task name must be a non-empty string.")
    key = canonical.lower()
    if key in _TASK_REGISTRY and not overwrite:
        existing, _ = _TASK_REGISTRY[key]
        raise ValueError(f"Task '{existing}' is already registered.")
    _TASK_REGISTRY[key] = (canonical, builder)


def get_registered_eval_tasks() -> Sequence[str]:
    """Return the list of currently registered evaluation task names."""
    return [entry[0] for key, entry in sorted(_TASK_REGISTRY.items())]


def _configure_special_tokens(model, processor) -> None:
    """Align tokenizer special tokens with the model configuration."""
    tokenizer = processor.tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id


def load_model_and_processor(
    model_path: Path,
    adapter_path: Optional[Path] = None,
) -> tuple[Any, Qwen2_5OmniProcessor]:
    """Load the base Qwen Omni model and optionally attach a LoRA adapter."""
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), use_fast=False)
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    _configure_special_tokens(model, processor)

    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    model.eval()
    return model, processor


def prepare_task_for_evaluation(
    task_name: str,
    processor: Qwen2_5OmniProcessor,
    split: str = "validation",
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    """Return dataset, collator, and metric functions for the requested task."""
    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"
    key = (task_name or "").strip().lower()
    if key not in _TASK_REGISTRY:
        available = ", ".join(get_registered_eval_tasks()) or "none"
        raise NotImplementedError(
            f"Evaluation for task '{task_name}' is not registered. Available tasks: {available}."
        )
    _, builder = _TASK_REGISTRY[key]
    return builder(processor=processor, split=split, config=config or {})


def run_evaluation(
    model,
    setup: TaskEvalSetup,
    *,
    batch_size: int = 4,
    generation_kwargs: Optional[Dict[str, Any]] = None,
    output_dir: Path | None = None,
    store_predictions: bool = False,
) -> Dict[str, float]:
    """Execute evaluation on the provided dataset and return metric scores.

    Args:
        model: Model to evaluate
        setup: TaskEvalSetup with dataset, collator, and metrics
        batch_size: Per-device batch size
        generation_kwargs: Generation parameters
        output_dir: Output directory for evaluation artifacts
        store_predictions: If True, enable prediction storage in compute_metrics (for confusion matrix)

    Returns:
        Dictionary of evaluation metrics
    """
    generation_kwargs = {**DEFAULT_GENERATION_KWARGS, **(generation_kwargs or {})}
    output_dir = ensure_dir(Path(output_dir or "runs/eval"))

    training_args = TrainingArguments(
        output_dir=str(output_dir),
        per_device_eval_batch_size=batch_size,
        dataloader_drop_last=False,
        dataloader_num_workers=0,
        remove_unused_columns=False,
        report_to=[],
        eval_strategy=IntervalStrategy.NO,
        save_strategy=IntervalStrategy.NO,
        logging_strategy=IntervalStrategy.NO,
    )

    # Wrap compute_metrics to enable prediction storage if requested
    compute_metrics = setup.compute_metrics
    if store_predictions and compute_metrics is not None:
        import inspect
        sig = inspect.signature(compute_metrics)
        if 'store_predictions' in sig.parameters:
            from functools import partial
            compute_metrics = partial(
                compute_metrics.func if hasattr(compute_metrics, 'func') else compute_metrics,
                store_predictions=True
            )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=setup.dataset,
        data_collator=setup.data_collator,
        compute_metrics=compute_metrics,
        generation_kwargs=generation_kwargs,
    )

    metrics = trainer.evaluate()
    # The Trainer prefixes evaluation metrics with "eval_"; normalize keys for convenience.
    return {
        key.replace("eval_", "", 1) if key.startswith("eval_") else key: value
        for key, value in metrics.items()
    }


# ---------------------------------------------------------------------------
# Generic task builder using configuration pattern
# ---------------------------------------------------------------------------


def _build_generic_eval_setup(
    task_config: TaskConfig,
    *,
    processor: Qwen2_5OmniProcessor,
    split: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    """Generic evaluation setup builder that works for all tasks via configuration.

    This function implements the template pattern, eliminating ~400 lines of
    duplicated code across task-specific builders.

    Args:
        task_config: Task-specific configuration object
        processor: Qwen processor for tokenization and feature extraction
        split: Dataset split to load ("train", "validation", or "test")
        config: Optional runtime configuration overrides

    Returns:
        TaskEvalSetup with dataset, collator, and metrics configured for the task
    """
    cfg = config or {}
    dataset_cfg = cfg.get("dataset", {})

    # Build loader kwargs from config
    loader_kwargs: Dict[str, Any] = {
        "seed": dataset_cfg.get("seed", cfg.get("seed", 0)),
        "num_proc": dataset_cfg.get("num_proc"),
        "cache_dir": dataset_cfg.get("cache_dir"),
        "cache_splits": dataset_cfg.get("cache_splits", True),
        "force_rebuild": dataset_cfg.get("force_rebuild", False),
    }

    # Add task-specific config keys
    for key in task_config.loader_config_keys:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]

    # Load dataset
    dataset_bundle = task_config.dataset_loader(**loader_kwargs)
    train_ds = dataset_bundle[0]
    val_ds = dataset_bundle[1]
    test_ds = dataset_bundle[2] if len(dataset_bundle) > 2 else None

    # Extract label_names or answers_map if applicable
    label_names = None
    answers_map = None
    if task_config.has_label_names and len(dataset_bundle) > 3:
        label_names = dataset_bundle[3]
    elif task_config.has_answers_map and len(dataset_bundle) > 3:
        answers_map = dataset_bundle[3]

    # Select split
    normalized_split = split.strip().lower()
    split_map = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }

    if normalized_split not in split_map:
        raise ValueError(
            f"Unsupported split '{split}'. Use one of {list(split_map.keys())}."
        )

    dataset = split_map[normalized_split]
    if dataset is None:
        raise ValueError(f"Split '{split}' is unavailable for this task.")

    # Apply optional post-load hook (e.g., ASR duration filtering, Speech QA answers alignment)
    if task_config.post_load_hook is not None:
        dataset = task_config.post_load_hook(dataset, dataset_cfg, normalized_split)

    # Filter columns: keep required + optional (if they exist)
    keep_columns = set(task_config.required_columns)
    for optional_col in task_config.optional_columns:
        if optional_col in dataset.column_names:
            keep_columns.add(optional_col)

    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    if drop_columns:
        dataset = dataset.remove_columns(drop_columns)

    # Create collator
    target_sr = getattr(
        getattr(processor, "feature_extractor", None), "sampling_rate", 16000
    )
    collator_kwargs = {
        "processor": processor,
        "sampling_rate": target_sr,
        **task_config.collator_params,
    }
    # Inject label_names if needed
    if label_names is not None and "label_names" not in collator_kwargs:
        collator_kwargs["label_names"] = list(label_names or [])
    # Inject dataset_cfg overrides (e.g., include_transcript)
    for key in ("include_transcript", "include_context", "prepend_scenario"):
        if key in dataset_cfg:
            collator_kwargs[key] = dataset_cfg[key]

    collator = task_config.collator_class(**collator_kwargs)

    # Create metrics function
    metrics_kwargs = {"processor": processor, **task_config.metrics_params}
    if label_names is not None and "label_names" not in metrics_kwargs:
        metrics_kwargs["label_names"] = list(label_names or [])

    # Add config-specific metric parameters (e.g., wer_normalization for ASR)
    metrics_cfg = cfg.get("metrics", {})
    for key in ("wer_normalization",):
        if key in metrics_cfg:
            metrics_kwargs[key] = metrics_cfg[key]

    # Handle Speech QA special case with answers_map
    if answers_map is not None:
        answers_list = list(answers_map.get(normalized_split, []))
        # Align answers_list length with dataset
        if len(answers_list) != len(dataset):
            if len(answers_list) > len(dataset):
                answers_list = answers_list[: len(dataset)]
            else:
                answers_list.extend([[] for _ in range(len(dataset) - len(answers_list))])
        metrics_kwargs["reference_answers"] = answers_list

    # Use partial or closure for metrics
    if "reference_answers" in metrics_kwargs:
        # Speech QA case: use closure to capture reference_answers
        ref_answers = metrics_kwargs.pop("reference_answers")

        def metrics_fn(eval_pred: Any) -> Dict[str, float]:
            return task_config.compute_metrics_fn(
                eval_pred, **metrics_kwargs, reference_answers=ref_answers
            )

    else:
        # Standard case: use partial
        metrics_fn = partial(task_config.compute_metrics_fn, **metrics_kwargs)

    return TaskEvalSetup(
        dataset=dataset,
        data_collator=collator,
        compute_metrics=metrics_fn,
        label_names=label_names,
    )


# ---------------------------------------------------------------------------
# Task-specific post-load hooks
# ---------------------------------------------------------------------------


def _asr_post_load_hook(dataset: Any, dataset_cfg: Dict[str, Any], split: str) -> Any:
    """Filter ASR dataset by max duration if specified."""
    max_duration_seconds = dataset_cfg.get("max_duration_seconds")
    if max_duration_seconds is None:
        return dataset

    max_duration_seconds = float(max_duration_seconds)

    def _keep_duration(example: Dict[str, Any], *, _max: float = max_duration_seconds) -> bool:
        duration = example.get("duration") or 0.0
        return float(duration) <= _max

    before = len(dataset)
    dataset = dataset.filter(_keep_duration)
    if len(dataset) != before:
        print(
            f"⏱️ Filtered {before - len(dataset)} samples longer than"
            f" {max_duration_seconds:.1f}s from split '{split}'."
        )
    return dataset


# ---------------------------------------------------------------------------
# Task-specific configurations
# ---------------------------------------------------------------------------


def _get_asr_task_config() -> TaskConfig:
    """Create ASR task configuration."""
    from tasks.asr import OmniASRCollator, compute_asr_metrics, load_librispeech_subset

    # ASR has special loader args that need defaults
    def _asr_loader_wrapper(**kwargs):
        # Set ASR-specific defaults
        kwargs.setdefault("train_hours", 10.0)
        kwargs.setdefault("val_hours", 1.0)
        kwargs.setdefault("return_full_validation", True)
        kwargs.setdefault("return_test_split", True)
        kwargs.setdefault("test_split", "test.clean")
        return load_librispeech_subset(**kwargs)

    return TaskConfig(
        dataset_loader=_asr_loader_wrapper,
        loader_config_keys=("train_hours", "val_hours", "test_hours", "test_split"),
        has_label_names=False,
        collator_class=OmniASRCollator,
        collator_params={},
        compute_metrics_fn=compute_asr_metrics,
        metrics_params={},
        required_columns=("audio", "text"),
        optional_columns=("duration",),
        post_load_hook=_asr_post_load_hook,
    )


def _get_emotion_task_config() -> TaskConfig:
    """Create emotion classification task configuration."""
    from tasks.emotion import (
        EmotionDataCollator,
        compute_emotion_metrics,
        load_superb_emotion_dataset,
    )

    return TaskConfig(
        dataset_loader=load_superb_emotion_dataset,
        loader_config_keys=CLASSIFICATION_EVAL_KEYS,
        has_label_names=True,
        collator_class=EmotionDataCollator,
        collator_params={},
        compute_metrics_fn=compute_emotion_metrics,
        metrics_params={},
        required_columns=("audio", "label"),
        optional_columns=("duration", "text"),
    )


def _get_speaker_task_config() -> TaskConfig:
    """Create speaker identification task configuration."""
    from tasks.speaker_id import (
        SpeakerIdentificationCollator,
        compute_speaker_id_metrics,
        load_voxceleb_speaker_dataset,
    )

    return TaskConfig(
        dataset_loader=load_voxceleb_speaker_dataset,
        loader_config_keys=CLASSIFICATION_EVAL_KEYS + SPEAKER_EXTRA_EVAL_KEYS,
        has_label_names=True,
        collator_class=SpeakerIdentificationCollator,
        collator_params={},
        compute_metrics_fn=compute_speaker_id_metrics,
        metrics_params={},
        required_columns=("audio", "label"),
        optional_columns=("duration", "text"),
    )


def _get_intent_task_config() -> TaskConfig:
    """Create intent classification task configuration."""
    from tasks.intent import (
        IntentClassificationCollator,
        compute_intent_metrics,
        load_slurp_intent_dataset,
    )

    return TaskConfig(
        dataset_loader=load_slurp_intent_dataset,
        loader_config_keys=CLASSIFICATION_EVAL_KEYS,
        has_label_names=True,
        collator_class=IntentClassificationCollator,
        collator_params={},
        compute_metrics_fn=compute_intent_metrics,
        metrics_params={},
        required_columns=("audio", "label"),
        optional_columns=("duration", "text", "scenario", "action"),
    )


def _get_speech_qa_task_config() -> TaskConfig:
    """Create speech QA task configuration."""
    from tasks.speech_qa import (
        SpeechQACollator,
        compute_speech_qa_metrics,
        load_speech_qa_dataset,
    )

    return TaskConfig(
        dataset_loader=load_speech_qa_dataset,
        loader_config_keys=SPEECH_QA_EVAL_KEYS,
        has_label_names=False,
        has_answers_map=True,
        collator_class=SpeechQACollator,
        collator_params={},
        compute_metrics_fn=compute_speech_qa_metrics,
        metrics_params={},
        required_columns=("audio", "question", "answers", "label_text"),
        optional_columns=("transcript", "context", "duration", "id"),
    )


def _register_default_tasks() -> None:
    """Register built-in task builders.

    Note: Task names are hardcoded to avoid circular imports during module initialization.
    The actual task components are imported lazily when the evaluation setup is built.
    """
    # Register ASR
    try:
        register_eval_task(
            "asr",
            lambda **kwargs: _build_generic_eval_setup(_get_asr_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass  # Silently skip if ASR components are unavailable

    # Register Emotion
    try:
        register_eval_task(
            "emotion",
            lambda **kwargs: _build_generic_eval_setup(_get_emotion_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass

    # Register Speaker ID
    try:
        register_eval_task(
            "speaker_id",
            lambda **kwargs: _build_generic_eval_setup(_get_speaker_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass

    # Register Intent
    try:
        register_eval_task(
            "intent",
            lambda **kwargs: _build_generic_eval_setup(_get_intent_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass

    # Register Speech QA
    try:
        register_eval_task(
            "speech_qa",
            lambda **kwargs: _build_generic_eval_setup(_get_speech_qa_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass


_register_default_tasks()
