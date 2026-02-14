"""Evaluation utilities with a pluggable task registry."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial
import hashlib
import json
from pathlib import Path
from typing import Any, Callable, Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

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

def compute_eval_subset_tag(eval_subset: Mapping[str, Any]) -> str:
    """Return a stable short tag for evaluation-subset settings.

    This tag is used to namespace metrics so that subset-based evaluations do not
    overwrite full-split metrics (e.g., base_model.json).
    """
    payload = json.dumps(dict(eval_subset), sort_keys=True, default=str).encode("utf-8")
    digest = hashlib.md5(payload).hexdigest()[:10]
    return f"subset_{digest}"

CLASSIFICATION_EVAL_KEYS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_config",
    "max_train_samples",
    "max_validation_samples",
    "max_test_samples",
    "max_duration",
    "min_duration",
    "label_column",
    "text_column",
    "audio_column",
    "split_percentages",
    "train_split",
    "validation_split",
    "test_split",
    "stratify_by_column",
    "data_dir",
    "languages",
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

ST_EVAL_KEYS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_config",
    "max_train_samples",
    "max_validation_samples",
    "max_test_samples",
    "max_duration",
    "min_duration",
    "audio_column",
    "source_column",
    "translation_column",
    "train_split",
    "validation_split",
    "test_split",
)

SPEAKER_VER_EVAL_KEYS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_config",
    "max_train_samples",
    "max_validation_samples",
    "max_test_samples",
    "max_duration",
    "min_duration",
    "total_pairs",
    "split_by_speakers",
    "label_column",
    "text_column",
    "audio_column",
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
    delta_weights: Optional[Dict[str, torch.Tensor]] = None,
    device_map_override: Optional[Any] = None,
    torch_dtype_override: Optional[torch.dtype] = None,
) -> tuple[Any, Qwen2_5OmniProcessor]:
    """Load the base Qwen Omni model and optionally attach a LoRA adapter.

    For evaluation, we load the base model without LoRA, then attach a saved adapter.
    This is different from training which creates a new LoRA adapter.
    """
    from core.models.models import load_qwen_model

    # Load base model without creating new LoRA adapter (apply_lora=False)
    model, processor = load_qwen_model(
        model_path,
        apply_lora=False,
        torch_dtype=(
            torch_dtype_override
            if torch_dtype_override is not None
            else (torch.bfloat16 if torch.cuda.is_available() else torch.float32)
        ),
        device_map=("auto" if device_map_override is None else device_map_override),
        use_fast_tokenizer=False,
    )

    # Load saved adapter if provided
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path))

    if delta_weights is not None:
        if adapter_path is not None:
            raise ValueError("delta_weights cannot be used with adapter_path.")
        _apply_delta_weights(model, delta_weights)

    model.eval()
    return model, processor


def _apply_delta_weights(model, delta_weights: Dict[str, torch.Tensor]) -> None:
    """Apply in-memory delta weights to the base model parameters."""
    state_keys = list(model.state_dict().keys())
    state_key_set = set(state_keys)
    missing = []
    shape_mismatch = []
    applied = 0

    for base_key, delta in delta_weights.items():
        candidates = [f"{base_key}.weight"]
        suffix_bases = {base_key}

        def _add_candidate(prefix_from: str, prefix_to: str) -> None:
            if base_key.startswith(prefix_from):
                trimmed = base_key.replace(prefix_from, prefix_to, 1)
                candidates.append(f"{trimmed}.weight")
                suffix_bases.add(trimmed)

        # Strip base_model.* prefix (PEFT) to common model prefixes
        _add_candidate("base_model.model.model.", "model.")
        _add_candidate("base_model.model.", "model.")
        _add_candidate("base_model.model.", "")
        _add_candidate("base_model.", "")

        # Alternative nesting variants seen across model wrappers
        _add_candidate("model.model.model.", "model.")
        _add_candidate("model.model.", "model.")
        _add_candidate("model.", "model.model.")

        param = None
        matched_key = None
        for weight_key in candidates:
            if weight_key in state_key_set:
                param = model.get_parameter(weight_key)
                matched_key = weight_key
                break

        if param is None:
            # Fallback: suffix match for weight keys ending with base_key
            for suffix_base in suffix_bases:
                suffix = f".{suffix_base}.weight"
                for key in state_keys:
                    if key.endswith(suffix):
                        param = model.get_parameter(key)
                        matched_key = key
                        break
                if param is not None:
                    break

        if param is None:
            missing.append(candidates[0])
            continue

        if param.shape != delta.shape:
            shape_mismatch.append((matched_key or candidates[0], tuple(param.shape), tuple(delta.shape)))
            continue

        with torch.no_grad():
            param.add_(delta.to(device=param.device, dtype=param.dtype))
        applied += 1

    if missing:
        print(f"⚠️  Delta merge: {len(missing)} parameters not found in model.")
        for key in missing[:5]:
            print(f"   - missing: {key}")
    if shape_mismatch:
        print(f"⚠️  Delta merge: {len(shape_mismatch)} shape mismatches; skipped.")
        for key, model_shape, delta_shape in shape_mismatch[:5]:
            print(f"   - shape mismatch: {key} model={model_shape} delta={delta_shape}")
    print(f"✅ Applied delta weights to {applied} parameters.")


def _json_default(value: Any) -> str:
    """Fallback serializer for JSON encoding of pathlib and similar objects."""
    return str(value)


def compute_base_cache_path(
    eval_dir: Path,
    *,
    task: str,
    split: str,
    config_path: Path,
    batch_size: int,
    generation_kwargs: Dict[str, Any],
    dataset_cfg: Dict[str, Any],
) -> Path:
    """Generate a deterministic cache file path for base-model metrics."""
    fingerprint_payload = {
        "task": task,
        "split": split,
        "config_path": str(config_path.resolve()),
        "batch_size": int(batch_size),
        "generation_kwargs": generation_kwargs,
        "dataset_cfg": dataset_cfg,
    }
    fingerprint_raw = json.dumps(fingerprint_payload, sort_keys=True, default=_json_default).encode("utf-8")
    fingerprint = hashlib.md5(fingerprint_raw).hexdigest()[:12]
    filename = f"{task}_{split}_base_{fingerprint}.json"
    return eval_dir / filename


def resolve_merged_eval_dir(
    *,
    metrics_dir: Path,
    split: str,
    task: str,
    merged_tasks: list[str],
    adapter_label: Optional[str],
    merged_method: Optional[str],
) -> Path:
    """Resolve output directory for merged evaluation artifacts."""
    merged_base = ensure_dir(metrics_dir / "eval" / split / "merged")
    other_tasks = [t for t in merged_tasks if t != task]
    if not other_tasks:
        other_tag = "self"
    elif len(other_tasks) == 1:
        other_tag = other_tasks[0]
    else:
        other_tag = "multi_" + "_".join(sorted(other_tasks))

    method_tag = adapter_label or merged_method or "merged"
    return ensure_dir(merged_base / other_tag / method_tag)


def print_metrics(label: str, task: str, split: str, metrics: Dict[str, Any]) -> None:
    """Pretty-print evaluation metrics."""
    print(f"✅ Evaluation complete for {label} on {task}/{split}")
    for key, value in metrics.items():
        if isinstance(value, (int, float)):
            print(f"  {key}: {value:.4f}")
        else:
            print(f"  {key}: {value}")


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
    processor: Optional[Any] = None,
) -> Dict[str, float]:
    """Execute evaluation on the provided dataset and return metric scores.

    Args:
        model: Model to evaluate
        setup: TaskEvalSetup with dataset, collator, and metrics
        batch_size: Per-device batch size
        generation_kwargs: Generation parameters
        output_dir: Output directory for evaluation artifacts
        store_predictions: If True, enable prediction storage in compute_metrics (for confusion matrix)
        processor: Optional processor for the model (ensures correct tokenizer settings)

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
        # Get the underlying function signature (handles partial objects correctly)
        base_func = compute_metrics.func if hasattr(compute_metrics, 'func') else compute_metrics
        sig = inspect.signature(base_func)
        if 'store_predictions' in sig.parameters:
            from functools import partial
            # If it's already a partial, preserve existing kwargs and add store_predictions
            if hasattr(compute_metrics, 'func'):
                existing_kwargs = compute_metrics.keywords.copy()
                existing_kwargs['store_predictions'] = True
                compute_metrics = partial(base_func, **existing_kwargs)
            else:
                compute_metrics = partial(compute_metrics, store_predictions=True)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=setup.dataset,
        data_collator=setup.data_collator,
        processing_class=processor,
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

    # Special handling for ST task: map top-level 'language' to 'dataset_config'
    if "dataset_config" in task_config.loader_config_keys and "language" in cfg:
        loader_kwargs["dataset_config"] = cfg["language"]

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
    # Inject language from top-level config for ST task
    if "language" in cfg:
        collator_kwargs["language_pair"] = cfg["language"]

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
    """Filter ASR dataset by duration if specified."""
    max_duration = dataset_cfg.get("max_duration")
    min_duration = dataset_cfg.get("min_duration")

    if max_duration is None and min_duration is None:
        return dataset

    def _keep_duration(example: Dict[str, Any]) -> bool:
        duration = example.get("duration") or 0.0
        duration_float = float(duration)
        if max_duration is not None and duration_float > max_duration:
            return False
        if min_duration is not None and duration_float < min_duration:
            return False
        return True

    before = len(dataset)
    dataset = dataset.filter(_keep_duration)
    if len(dataset) != before:
        filtered_count = before - len(dataset)
        duration_info = []
        if max_duration is not None:
            duration_info.append(f">{max_duration:.1f}s")
        if min_duration is not None:
            duration_info.append(f"<{min_duration:.1f}s")
        duration_str = " or ".join(duration_info)
        print(f"⏱️ Filtered {filtered_count} samples ({duration_str}) from split '{split}'.")
    return dataset


def _classification_post_load_hook(dataset: Any, dataset_cfg: Dict[str, Any], split: str) -> Any:
    """Filter classification dataset by duration if specified.

    This ensures evaluation uses the same duration filtering as training.
    """
    # Classification dataset loaders already apply `filter_by_duration(...)` with
    # disk caching. Re-running `dataset.filter(...)` here duplicates work and
    # causes expensive progress bars during optimizer loops.
    #
    # Set `dataset.skip_post_load_duration_filter=false` to force this hook.
    if bool(dataset_cfg.get("skip_post_load_duration_filter", True)):
        return dataset

    max_duration = dataset_cfg.get("max_duration")
    min_duration = dataset_cfg.get("min_duration")

    if max_duration is None and min_duration is None:
        return dataset

    def _keep_duration(example: Dict[str, Any]) -> bool:
        duration = example.get("duration") or 0.0
        duration_float = float(duration)
        if max_duration is not None and duration_float > max_duration:
            return False
        if min_duration is not None and duration_float < min_duration:
            return False
        return True

    before = len(dataset)
    dataset = dataset.filter(_keep_duration)
    if len(dataset) != before:
        filtered_count = before - len(dataset)
        duration_info = []
        if max_duration is not None:
            duration_info.append(f">{max_duration:.1f}s")
        if min_duration is not None:
            duration_info.append(f"<{min_duration:.1f}s")
        duration_str = " or ".join(duration_info)
        print(f"⏱️ Filtered {filtered_count} samples ({duration_str}) from split '{split}'.")
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
        EmotionRecognitionCollator,
        compute_emotion_metrics,
        load_superb_emotion_dataset,
    )

    return TaskConfig(
        dataset_loader=load_superb_emotion_dataset,
        loader_config_keys=CLASSIFICATION_EVAL_KEYS,
        has_label_names=True,
        collator_class=EmotionRecognitionCollator,
        collator_params={},
        compute_metrics_fn=compute_emotion_metrics,
        metrics_params={},
        required_columns=("audio", "label"),
        optional_columns=("duration", "text"),
        post_load_hook=_classification_post_load_hook,
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
        post_load_hook=_classification_post_load_hook,
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
        post_load_hook=_classification_post_load_hook,
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


def _get_st_task_config() -> TaskConfig:
    """Create speech translation task configuration."""
    from tasks.st import (
        STCollator,
        compute_st_metrics,
        load_covost2_dataset,
    )

    return TaskConfig(
        dataset_loader=load_covost2_dataset,
        loader_config_keys=ST_EVAL_KEYS,
        has_label_names=False,
        collator_class=STCollator,
        collator_params={},
        compute_metrics_fn=compute_st_metrics,
        metrics_params={},
        required_columns=("audio", "text", "translation"),
        optional_columns=("duration",),
        post_load_hook=_asr_post_load_hook,  # Reuse ASR duration filtering
    )


def _get_kws_task_config() -> TaskConfig:
    """Create keyword spotting task configuration."""
    from tasks.kws import (
        KeywordSpottingCollator,
        compute_kws_metrics,
        load_speech_commands_kws_dataset,
    )

    return TaskConfig(
        dataset_loader=load_speech_commands_kws_dataset,
        loader_config_keys=CLASSIFICATION_EVAL_KEYS,
        has_label_names=True,
        collator_class=KeywordSpottingCollator,
        collator_params={},
        compute_metrics_fn=compute_kws_metrics,
        metrics_params={},
        required_columns=("audio", "text", "label"),
        optional_columns=("duration",),
        post_load_hook=_classification_post_load_hook,
    )


def _get_langid_task_config() -> TaskConfig:
    """Create language identification task configuration."""
    from tasks.langid import (
        LanguageIdentificationCollator,
        compute_langid_metrics,
        load_fleurs_langid_dataset,
    )

    return TaskConfig(
        dataset_loader=load_fleurs_langid_dataset,
        loader_config_keys=CLASSIFICATION_EVAL_KEYS,
        has_label_names=True,
        collator_class=LanguageIdentificationCollator,
        collator_params={},
        compute_metrics_fn=compute_langid_metrics,
        metrics_params={},
        required_columns=("audio", "text", "label"),
        optional_columns=("duration",),
        post_load_hook=_classification_post_load_hook,
    )


def _get_speaker_ver_task_config() -> TaskConfig:
    """Create speaker verification task configuration."""
    from tasks.speaker_ver import (
        SpeakerVerCollator,
        compute_speaker_ver_metrics,
        load_speaker_ver_dataset,
    )

    return TaskConfig(
        dataset_loader=load_speaker_ver_dataset,
        loader_config_keys=SPEAKER_VER_EVAL_KEYS,
        has_label_names=True,
        collator_class=SpeakerVerCollator,
        collator_params={},
        compute_metrics_fn=compute_speaker_ver_metrics,
        metrics_params={},
        required_columns=("audio_a", "audio_b", "label"),
        optional_columns=(),
        post_load_hook=None,  # Speaker verification has its own filtering logic
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

    # Register Speech Translation
    try:
        register_eval_task(
            "st",
            lambda **kwargs: _build_generic_eval_setup(_get_st_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass

    # Register Keyword Spotting
    try:
        register_eval_task(
            "kws",
            lambda **kwargs: _build_generic_eval_setup(_get_kws_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass

    # Register Language Identification
    try:
        register_eval_task(
            "langid",
            lambda **kwargs: _build_generic_eval_setup(_get_langid_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass

    # Register Speaker Verification
    try:
        register_eval_task(
            "speaker_ver",
            lambda **kwargs: _build_generic_eval_setup(_get_speaker_ver_task_config(), **kwargs),
            overwrite=True,
        )
    except Exception:
        pass


_register_default_tasks()
