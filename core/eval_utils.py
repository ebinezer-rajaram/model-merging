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

from core.io_utils import ensure_dir
from core.trainer import CustomTrainer

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
) -> Dict[str, float]:
    """Execute evaluation on the provided dataset and return metric scores."""
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

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        eval_dataset=setup.dataset,
        data_collator=setup.data_collator,
        compute_metrics=setup.compute_metrics,
        generation_kwargs=generation_kwargs,
    )

    metrics = trainer.evaluate()
    # The Trainer prefixes evaluation metrics with "eval_"; normalize keys for convenience.
    return {
        key.replace("eval_", "", 1) if key.startswith("eval_") else key: value
        for key, value in metrics.items()
    }


# ---------------------------------------------------------------------------
# Default task builders
# ---------------------------------------------------------------------------


def _build_asr_eval_setup(
    *,
    processor: Qwen2_5OmniProcessor,
    split: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    from tasks.asr import (
        OmniASRCollator,
        compute_asr_metrics,
        load_librispeech_subset,
    )

    cfg = config or {}
    dataset_cfg = cfg.get("dataset", {})
    loader_kwargs: Dict[str, Any] = {
        "train_hours": dataset_cfg.get("train_hours", 10.0),
        "val_hours": dataset_cfg.get("val_hours", 1.0),
        "seed": dataset_cfg.get("seed", cfg.get("seed", 0)),
        "num_proc": dataset_cfg.get("num_proc"),
        "cache_dir": dataset_cfg.get("cache_dir"),
        "cache_splits": dataset_cfg.get("cache_splits", True),
        "force_rebuild": dataset_cfg.get("force_rebuild", False),
        "return_full_validation": True,
        "return_test_split": True,
        "test_split": dataset_cfg.get("test_split", "test.clean"),
    }
    if dataset_cfg.get("test_hours") is not None:
        loader_kwargs["test_hours"] = dataset_cfg["test_hours"]

    dataset_bundle = load_librispeech_subset(**loader_kwargs)
    train_ds = dataset_bundle[0]
    val_ds = dataset_bundle[1]
    test_ds = dataset_bundle[-1] if loader_kwargs["return_test_split"] else None

    normalized_split = split.strip().lower()
    split_map = {
        "train": train_ds,
        "validation": val_ds,
    }
    if loader_kwargs["return_test_split"] and test_ds is not None:
        split_map["test"] = test_ds

    if normalized_split not in split_map:
        raise ValueError("Unsupported split '%s'. Use one of %s." % (split, list(split_map.keys())))

    dataset = split_map[normalized_split]
    if dataset is None:
        raise ValueError(f"Split '{split}' is unavailable for the ASR dataset.")

    max_duration_seconds = dataset_cfg.get("max_duration_seconds")
    if max_duration_seconds is not None:
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

    keep_columns = {"audio", "text"}
    if "duration" in dataset.column_names:
        keep_columns.add("duration")
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns) if drop_columns else dataset

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = OmniASRCollator(processor=processor, sampling_rate=target_sr)
    metrics_fn = partial(compute_asr_metrics, processor=processor)

    return TaskEvalSetup(dataset=dataset, data_collator=collator, compute_metrics=metrics_fn)



def _build_speaker_eval_setup(
    *,
    processor: Qwen2_5OmniProcessor,
    split: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    from tasks.speaker_id import (
        SpeakerIdentificationCollator,
        compute_speaker_id_metrics,
        load_voxceleb_speaker_dataset,
    )

    cfg = config or {}
    dataset_cfg = cfg.get('dataset', {})
    loader_kwargs: Dict[str, Any] = {
        'seed': dataset_cfg.get('seed', cfg.get('seed', 0)),
        'num_proc': dataset_cfg.get('num_proc'),
        'cache_dir': dataset_cfg.get('cache_dir'),
        'cache_splits': dataset_cfg.get('cache_splits', True),
        'force_rebuild': dataset_cfg.get('force_rebuild', False),
    }
    for key in CLASSIFICATION_EVAL_KEYS + SPEAKER_EXTRA_EVAL_KEYS:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]

    train_ds, val_ds, test_ds, label_names = load_voxceleb_speaker_dataset(**loader_kwargs)
    normalized_split = split.strip().lower()
    split_map = {
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds,
    }
    if normalized_split not in split_map:
        raise ValueError("Unsupported split '%s'. Use one of %s." % (split, list(split_map.keys())))

    dataset = split_map[normalized_split]
    if dataset is None:
        raise ValueError(f"Split '{split}' is unavailable for the speaker identification dataset.")

    keep_columns = {'audio', 'label'}
    for optional in ('duration', 'text'):
        if optional in dataset.column_names:
            keep_columns.add(optional)
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns) if drop_columns else dataset

    target_sr = getattr(getattr(processor, 'feature_extractor', None), 'sampling_rate', 16000)
    collator = SpeakerIdentificationCollator(
        processor=processor,
        sampling_rate=target_sr,
        label_names=list(label_names or []),
        include_transcript=dataset_cfg.get('include_transcript', False),
    )
    metrics_fn = partial(
        compute_speaker_id_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )
    return TaskEvalSetup(dataset=dataset, data_collator=collator, compute_metrics=metrics_fn)



def _build_intent_eval_setup(
    *,
    processor: Qwen2_5OmniProcessor,
    split: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    from tasks.intent import (
        IntentClassificationCollator,
        compute_intent_metrics,
        load_slurp_intent_dataset,
    )

    cfg = config or {}
    dataset_cfg = cfg.get('dataset', {})
    loader_kwargs: Dict[str, Any] = {
        'seed': dataset_cfg.get('seed', cfg.get('seed', 0)),
        'num_proc': dataset_cfg.get('num_proc'),
        'cache_dir': dataset_cfg.get('cache_dir'),
        'cache_splits': dataset_cfg.get('cache_splits', True),
        'force_rebuild': dataset_cfg.get('force_rebuild', False),
    }
    for key in CLASSIFICATION_EVAL_KEYS:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]

    train_ds, val_ds, test_ds, label_names = load_slurp_intent_dataset(**loader_kwargs)
    normalized_split = split.strip().lower()
    split_map = {
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds,
    }
    if normalized_split not in split_map:
        raise ValueError("Unsupported split '%s'. Use one of %s." % (split, list(split_map.keys())))

    dataset = split_map[normalized_split]
    if dataset is None:
        raise ValueError(f"Split '{split}' is unavailable for the intent dataset.")

    keep_columns = {'audio', 'label'}
    for optional in ('duration', 'text', 'scenario', 'action'):
        if optional in dataset.column_names:
            keep_columns.add(optional)
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns) if drop_columns else dataset

    target_sr = getattr(getattr(processor, 'feature_extractor', None), 'sampling_rate', 16000)
    collator = IntentClassificationCollator(
        processor=processor,
        sampling_rate=target_sr,
        label_names=list(label_names or []),
        include_transcript=dataset_cfg.get('include_transcript', True),
        prepend_scenario=dataset_cfg.get('prepend_scenario', False),
    )
    metrics_fn = partial(
        compute_intent_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )
    return TaskEvalSetup(dataset=dataset, data_collator=collator, compute_metrics=metrics_fn)



def _build_speech_qa_eval_setup(
    *,
    processor: Qwen2_5OmniProcessor,
    split: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    from tasks.speech_qa import (
        SpeechQACollator,
        compute_speech_qa_metrics,
        load_speech_qa_dataset,
    )

    cfg = config or {}
    dataset_cfg = cfg.get('dataset', {})
    loader_kwargs: Dict[str, Any] = {
        'seed': dataset_cfg.get('seed', cfg.get('seed', 0)),
        'num_proc': dataset_cfg.get('num_proc'),
        'cache_dir': dataset_cfg.get('cache_dir'),
        'cache_splits': dataset_cfg.get('cache_splits', True),
        'force_rebuild': dataset_cfg.get('force_rebuild', False),
    }
    for key in SPEECH_QA_EVAL_KEYS:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]

    train_ds, val_ds, test_ds, answers_map = load_speech_qa_dataset(**loader_kwargs)
    normalized_split = split.strip().lower()
    split_map = {
        'train': train_ds,
        'validation': val_ds,
        'test': test_ds,
    }
    if normalized_split not in split_map:
        raise ValueError("Unsupported split '%s'. Use one of %s." % (split, list(split_map.keys())))

    dataset = split_map[normalized_split]
    if dataset is None:
        raise ValueError(f"Split '{split}' is unavailable for the speech QA dataset.")

    keep_columns = {'audio', 'question', 'answers', 'label_text'}
    for optional in ('transcript', 'context', 'duration', 'id'):
        if optional in dataset.column_names:
            keep_columns.add(optional)
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns) if drop_columns else dataset

    answers_list = list(answers_map.get(normalized_split, []))
    if len(answers_list) != len(dataset):
        if len(answers_list) > len(dataset):
            answers_list = answers_list[: len(dataset)]
        else:
            answers_list.extend([[] for _ in range(len(dataset) - len(answers_list))])
    target_sr = getattr(getattr(processor, 'feature_extractor', None), 'sampling_rate', 16000)
    collator = SpeechQACollator(
        processor=processor,
        sampling_rate=target_sr,
        include_transcript=dataset_cfg.get('include_transcript', True),
        include_context=dataset_cfg.get('include_context', False),
    )

    def metrics_fn(eval_pred: Any) -> Dict[str, float]:
        return compute_speech_qa_metrics(
            eval_pred,
            processor=processor,
            reference_answers=answers_list,
        )

    return TaskEvalSetup(dataset=dataset, data_collator=collator, compute_metrics=metrics_fn)


def _build_emotion_eval_setup(
    *,
    processor: Qwen2_5OmniProcessor,
    split: str,
    config: Optional[Dict[str, Any]] = None,
) -> TaskEvalSetup:
    from tasks.emotion import (
        EmotionDataCollator,
        compute_emotion_metrics,
        load_superb_emotion_dataset,
    )

    cfg = config or {}
    dataset_cfg = cfg.get("dataset", {})
    loader_kwargs: Dict[str, Any] = {
        "seed": dataset_cfg.get("seed", cfg.get("seed", 0)),
        "num_proc": dataset_cfg.get("num_proc"),
        "cache_dir": dataset_cfg.get("cache_dir"),
        "cache_splits": dataset_cfg.get("cache_splits", True),
        "force_rebuild": dataset_cfg.get("force_rebuild", False),
    }

    optional_args = (
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
    )
    for key in optional_args:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]

    train_ds, val_ds, test_ds, label_names = load_superb_emotion_dataset(**loader_kwargs)
    normalized_split = split.strip().lower()
    split_map = {
        "train": train_ds,
        "validation": val_ds,
        "test": test_ds,
    }
    if normalized_split not in split_map:
        raise ValueError("Unsupported split '%s'. Use one of %s." % (split, list(split_map.keys())))

    dataset = split_map[normalized_split]
    if dataset is None:
        raise ValueError(f"Split '{split}' is unavailable for the emotion dataset.")

    keep_columns = {"audio", "label"}
    if "duration" in dataset.column_names:
        keep_columns.add("duration")
    if "text" in dataset.column_names:
        keep_columns.add("text")
    drop_columns = [col for col in dataset.column_names if col not in keep_columns]
    dataset = dataset.remove_columns(drop_columns) if drop_columns else dataset

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = EmotionDataCollator(
        processor=processor,
        sampling_rate=target_sr,
        label_names=list(label_names or []),
        include_transcript=dataset_cfg.get("include_transcript", True),
    )
    metrics_fn = partial(
        compute_emotion_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )

    return TaskEvalSetup(dataset=dataset, data_collator=collator, compute_metrics=metrics_fn)



def _register_default_tasks() -> None:
    """Register built-in task builders."""
    try:
        from tasks.asr import TASK_NAME as asr_task_name  # type: ignore
    except ImportError:
        asr_task_name = None
    if asr_task_name:
        register_eval_task(asr_task_name, _build_asr_eval_setup, overwrite=True)

    try:
        from tasks.emotion import TASK_NAME as emotion_task_name  # type: ignore
    except ImportError:
        emotion_task_name = None
    if emotion_task_name:
        register_eval_task(emotion_task_name, _build_emotion_eval_setup, overwrite=True)

    try:
        from tasks.speaker_id import TASK_NAME as speaker_task_name  # type: ignore
    except ImportError:
        speaker_task_name = None
    if speaker_task_name:
        register_eval_task(speaker_task_name, _build_speaker_eval_setup, overwrite=True)

    try:
        from tasks.intent import TASK_NAME as intent_task_name  # type: ignore
    except ImportError:
        intent_task_name = None
    if intent_task_name:
        register_eval_task(intent_task_name, _build_intent_eval_setup, overwrite=True)

    try:
        from tasks.speech_qa import TASK_NAME as speech_qa_task_name  # type: ignore
    except ImportError:
        speech_qa_task_name = None
    if speech_qa_task_name:
        register_eval_task(speech_qa_task_name, _build_speech_qa_eval_setup, overwrite=True)


_register_default_tasks()
