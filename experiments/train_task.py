"""Generic task training entry point."""

import argparse
import sys
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Sequence, Tuple

import torch
from transformers import EarlyStoppingCallback

torch.set_float32_matmul_precision("high")

CURRENT_DIR = Path(__file__).resolve().parent
PACKAGE_ROOT = CURRENT_DIR.parent
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core import (
    build_early_stopping_kwargs,
    build_training_arguments,
    create_classification_constraint,
    CustomTrainer,
    ensure_dir,
    filter_dataset_columns,
    load_config,
    load_qwen_model,
    parse_training_config,
    run_training_with_evaluation,
    set_global_seed,
)
from core.models.models import create_lora_config_from_dict
from tasks.asr import (
    compute_asr_metrics,
    get_artifact_directories as get_asr_artifact_directories,
    get_config_path as get_asr_config_path,
    load_librispeech_subset,
    OmniASRCollator,
    TASK_NAME as ASR_TASK_NAME,
)
from tasks.emotion import (
    EmotionDataCollator,
    compute_emotion_metrics,
    get_artifact_directories as get_emotion_artifact_directories,
    get_config_path as get_emotion_config_path,
    load_superb_emotion_dataset,
    TASK_NAME as EMOTION_TASK_NAME,
)
from tasks.intent import (
    IntentClassificationCollator,
    compute_intent_metrics,
    get_artifact_directories as get_intent_artifact_directories,
    get_config_path as get_intent_config_path,
    load_slurp_intent_dataset,
    TASK_NAME as INTENT_TASK_NAME,
)
from tasks.speaker_id import (
    SpeakerIdentificationCollator,
    compute_speaker_id_metrics,
    get_artifact_directories as get_speaker_artifact_directories,
    get_config_path as get_speaker_config_path,
    load_voxceleb_speaker_dataset,
    TASK_NAME as SPEAKER_TASK_NAME,
)
from tasks.speech_qa import (
    SpeechQACollator,
    compute_speech_qa_metrics,
    get_artifact_directories as get_speech_qa_artifact_directories,
    get_config_path as get_speech_qa_config_path,
    load_speech_qa_dataset,
    TASK_NAME as SPEECH_QA_TASK_NAME,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train task adapters.")
    parser.add_argument("--task", default=ASR_TASK_NAME, help="Task name to run.")
    parser.add_argument("--config", default=None, help="Override config filename.")
    return parser.parse_args()


def _resolve_dataset_cache_path(dataset_cfg: Dict[str, Any], artifact_dirs: Dict[str, Path]) -> Path:
    """Normalize dataset cache paths relative to the task artifact directory."""
    dataset_cache_dir = dataset_cfg.get("cache_dir")
    if dataset_cache_dir is not None:
        dataset_cache_path = Path(dataset_cache_dir)
        if not dataset_cache_path.is_absolute():
            dataset_cache_path = artifact_dirs["base"] / dataset_cache_path
    else:
        dataset_cache_path = ensure_dir(artifact_dirs["datasets"])
    dataset_cfg["cache_dir"] = dataset_cache_path
    return dataset_cache_path


@dataclass
class ClassificationTaskSpec:
    """Configuration for an audio classification task."""

    task_name: str
    get_config_path: Callable[[Path, Optional[str]], Path]
    get_artifact_directories: Callable[[Path], Dict[str, Path]]
    dataset_loader: Callable[..., Tuple[Optional[Any], Optional[Any], Optional[Any], Sequence[str]]]
    loader_extra_keys: Sequence[str]
    collator_builder: Callable[[Any, Dict[str, Any], Sequence[str], int], Any]
    metrics_builder: Callable[[Any, Sequence[str]], Callable[[Any], Dict[str, float]]]
    keep_columns: Sequence[str]
    default_adapter_subdir: str


CLASSIFICATION_BASE_KEYS: Tuple[str, ...] = (
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

SPEECH_QA_LOADER_KEYS: Tuple[str, ...] = (
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


def _build_classification_loader_kwargs(
    dataset_cfg: Dict[str, Any],
    *,
    seed: int,
    cache_path: Path,
    extra_keys: Sequence[str],
) -> Dict[str, Any]:
    """Assemble keyword arguments for classification dataset loaders."""
    loader_kwargs: Dict[str, Any] = {
        "seed": dataset_cfg.get("seed", seed),
        "num_proc": dataset_cfg.get("num_proc"),
        "cache_dir": cache_path,
        "cache_splits": dataset_cfg.get("cache_splits", True),
        "force_rebuild": dataset_cfg.get("force_rebuild", False),
    }
    for key in CLASSIFICATION_BASE_KEYS:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]
    for key in extra_keys:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]
    return loader_kwargs


def _build_speech_qa_loader_kwargs(
    dataset_cfg: Dict[str, Any],
    *,
    seed: int,
    cache_path: Path,
) -> Dict[str, Any]:
    """Assemble kwargs for speech QA dataset loaders."""
    loader_kwargs: Dict[str, Any] = {
        "seed": dataset_cfg.get("seed", seed),
        "num_proc": dataset_cfg.get("num_proc"),
        "cache_dir": cache_path,
        "cache_splits": dataset_cfg.get("cache_splits", True),
        "force_rebuild": dataset_cfg.get("force_rebuild", False),
    }
    for key in SPEECH_QA_LOADER_KEYS:
        if key in dataset_cfg and dataset_cfg[key] is not None:
            loader_kwargs[key] = dataset_cfg[key]
    return loader_kwargs


def _build_emotion_collator(
    processor,
    dataset_cfg: Dict[str, Any],
    label_names: Sequence[str],
    sampling_rate: int,
):
    return EmotionDataCollator(
        processor=processor,
        sampling_rate=sampling_rate,
        label_names=list(label_names or []),
        include_transcript=dataset_cfg.get("include_transcript", True),
    )


def _build_speaker_collator(
    processor,
    dataset_cfg: Dict[str, Any],
    label_names: Sequence[str],
    sampling_rate: int,
):
    return SpeakerIdentificationCollator(
        processor=processor,
        sampling_rate=sampling_rate,
        label_names=list(label_names or []),
        include_transcript=dataset_cfg.get("include_transcript", False),
    )


def _build_intent_collator(
    processor,
    dataset_cfg: Dict[str, Any],
    label_names: Sequence[str],
    sampling_rate: int,
):
    return IntentClassificationCollator(
        processor=processor,
        sampling_rate=sampling_rate,
        label_names=list(label_names or []),
        include_transcript=dataset_cfg.get("include_transcript", True),
        prepend_scenario=dataset_cfg.get("prepend_scenario", False),
    )


def _build_emotion_metrics(processor, label_names: Sequence[str]):
    return partial(
        compute_emotion_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )


def _build_speaker_metrics(processor, label_names: Sequence[str]):
    return partial(
        compute_speaker_id_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )


def _build_intent_metrics(processor, label_names: Sequence[str]):
    return partial(
        compute_intent_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )


CLASSIFICATION_TASK_SPECS: Dict[str, ClassificationTaskSpec] = {
    EMOTION_TASK_NAME: ClassificationTaskSpec(
        task_name=EMOTION_TASK_NAME,
        get_config_path=get_emotion_config_path,
        get_artifact_directories=get_emotion_artifact_directories,
        dataset_loader=load_superb_emotion_dataset,
        loader_extra_keys=(),
        collator_builder=_build_emotion_collator,
        metrics_builder=_build_emotion_metrics,
        keep_columns=("audio", "label", "duration", "text"),
        default_adapter_subdir="qwen2_5_omni_lora_emotion",
    ),
    SPEAKER_TASK_NAME: ClassificationTaskSpec(
        task_name=SPEAKER_TASK_NAME,
        get_config_path=get_speaker_config_path,
        get_artifact_directories=get_speaker_artifact_directories,
        dataset_loader=load_voxceleb_speaker_dataset,
        loader_extra_keys=("max_speakers", "max_samples_per_speaker"),
        collator_builder=_build_speaker_collator,
        metrics_builder=_build_speaker_metrics,
        keep_columns=("audio", "label", "duration", "text"),
        default_adapter_subdir="qwen2_5_omni_lora_speaker_id",
    ),
    INTENT_TASK_NAME: ClassificationTaskSpec(
        task_name=INTENT_TASK_NAME,
        get_config_path=get_intent_config_path,
        get_artifact_directories=get_intent_artifact_directories,
        dataset_loader=load_slurp_intent_dataset,
        loader_extra_keys=(),
        collator_builder=_build_intent_collator,
        metrics_builder=_build_intent_metrics,
        keep_columns=("audio", "label", "duration", "text", "scenario", "action"),
        default_adapter_subdir="qwen2_5_omni_lora_intent",
    ),
}


def run_audio_classification_task(config_path: Path, spec: ClassificationTaskSpec) -> None:
    """Generic runner for audio-based classification tasks."""
    config = load_config(config_path)
    seed = config.get("seed")
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    artifacts_cfg = config.get("artifacts", {})
    metrics_cfg = config.get("metrics", {})
    model_cfg = config.get("model", {})

    model_path = PACKAGE_ROOT / model_cfg.get("path", "models/Qwen2.5-Omni-3B")
    artifact_dirs = spec.get_artifact_directories(PACKAGE_ROOT)

    output_dir = ensure_dir(
        artifact_dirs["adapters"] / artifacts_cfg.get("adapter_subdir", spec.default_adapter_subdir)
    )
    metrics_dir = ensure_dir(artifact_dirs["metrics"])

    history_csv_path = metrics_dir / metrics_cfg.get(
        "history_csv", f"{spec.task_name}_training_history.csv"
    )
    loss_plot_path = metrics_dir / metrics_cfg.get("loss_plot", f"{spec.task_name}_loss_plot.png")
    final_adapter_dir = output_dir / "final"

    print("🔧 Loading model and processor...")
    lora_config = None
    if "lora" in model_cfg:
        lora_config = create_lora_config_from_dict(model_cfg["lora"])
        print(f"  Using LoRA config from YAML: r={model_cfg['lora'].get('r')}, alpha={model_cfg['lora'].get('alpha')}")
    model, processor = load_qwen_model(model_path, lora_config=lora_config)

    dataset_seed = dataset_cfg.get("seed", seed)
    cache_path = _resolve_dataset_cache_path(dataset_cfg, artifact_dirs)
    loader_kwargs = _build_classification_loader_kwargs(
        dataset_cfg,
        seed=dataset_seed,
        cache_path=cache_path,
        extra_keys=spec.loader_extra_keys,
    )
    train_ds, val_ds, test_ds, label_names = spec.dataset_loader(**loader_kwargs)

    if train_ds is None:
        raise RuntimeError(f"{spec.task_name} dataset did not provide a training split.")
    eval_ds = val_ds or test_ds
    if eval_ds is None:
        raise RuntimeError(f"{spec.task_name} dataset requires a validation or test split.")

    full_eval_ds = eval_ds

    # Filter datasets to keep only necessary columns
    train_ds = filter_dataset_columns(train_ds, spec.keep_columns, always_keep=["duration"])
    full_eval_ds = filter_dataset_columns(full_eval_ds, spec.keep_columns, always_keep=["duration"])

    eval_ds_for_trainer = full_eval_ds
    max_eval_samples = training_cfg.get("max_eval_samples")
    if max_eval_samples is not None:
        max_eval_samples = int(max(0, max_eval_samples))
        if 0 < max_eval_samples < len(full_eval_ds):
            eval_ds_for_trainer = full_eval_ds.select(range(max_eval_samples))
            print(f"🔍 Validation truncated to {len(eval_ds_for_trainer)} samples for faster eval.")

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = spec.collator_builder(processor, dataset_cfg, label_names, target_sr)

    # Parse training configuration with task-specific defaults
    task_defaults = {
        "metric_for_best_model": "macro_f1",
        "greater_is_better": True,
        "early_stopping_patience": 3,
        "length_column_name": "duration",
    }
    train_config = parse_training_config(
        training_cfg,
        num_train_examples=len(train_ds),
        task_defaults=task_defaults,
    )

    # Build training arguments
    training_args = build_training_arguments(train_config, output_dir=str(output_dir))
    early_stopping_kwargs = build_early_stopping_kwargs(train_config)
    compute_metrics_fn = spec.metrics_builder(processor, label_names or [])

    # Create constrained decoding function to ensure only valid labels are generated
    print(f"🔒 Creating constrained decoding with {len(label_names)} valid labels...")
    constraint_fn = create_classification_constraint(
        processor=processor,
        label_names=label_names,
        allow_eos=True,
    )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds_for_trainer,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=compute_metrics_fn,
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
        constrained_decoding_fn=constraint_fn,
    )

    # Run training with evaluation
    run_training_with_evaluation(
        trainer,
        processor=processor,
        full_eval_dataset=full_eval_ds,
        initial_eval=train_config.initial_eval,
        history_csv_path=history_csv_path,
        loss_plot_path=loss_plot_path,
        final_adapter_dir=final_adapter_dir,
    )


def run_asr_task(config_path: Path) -> None:
    """Run the ASR fine-tuning workflow."""
    config = load_config(config_path)
    seed = config.get("seed")
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    artifacts_cfg = config.get("artifacts", {})
    metrics_cfg = config.get("metrics", {})
    model_cfg = config.get("model", {})

    model_path = PACKAGE_ROOT / model_cfg.get("path", "models/Qwen2.5-Omni-3B")
    artifact_dirs = get_asr_artifact_directories(PACKAGE_ROOT)

    output_dir = ensure_dir(
        artifact_dirs["adapters"] / artifacts_cfg.get("adapter_subdir", "qwen2_5_omni_lora_asr_10h")
    )
    metrics_dir = ensure_dir(artifact_dirs["metrics"])

    history_csv_path = metrics_dir / metrics_cfg.get("history_csv", "training_history.csv")
    loss_plot_path = metrics_dir / metrics_cfg.get("loss_plot", "loss_wer_plot.png")
    final_adapter_dir = output_dir / "final"

    print("🔧 Loading model and processor...")
    lora_config = None
    if "lora" in model_cfg:
        lora_config = create_lora_config_from_dict(model_cfg["lora"])
        print(f"  Using LoRA config from YAML: r={model_cfg['lora'].get('r')}, alpha={model_cfg['lora'].get('alpha')}")
    model, processor = load_qwen_model(model_path, lora_config=lora_config)

    dataset_seed = dataset_cfg.get("seed", seed)
    dataset_cache_dir = dataset_cfg.get("cache_dir")
    if dataset_cache_dir is not None:
        dataset_cache_path = Path(dataset_cache_dir)
        if not dataset_cache_path.is_absolute():
            dataset_cache_path = artifact_dirs["base"] / dataset_cache_path
    else:
        dataset_cache_path = ensure_dir(artifact_dirs["datasets"])

    loader_kwargs = dict(
        train_hours=dataset_cfg.get("train_hours", 10.0),
        val_hours=dataset_cfg.get("val_hours", 1.0),
        seed=dataset_seed,
        num_proc=dataset_cfg.get("num_proc"),
        cache_dir=dataset_cache_path,
        cache_splits=dataset_cfg.get("cache_splits", True),
        force_rebuild=dataset_cfg.get("force_rebuild", False),
        return_full_validation=dataset_cfg.get("return_full_validation", False),
    )
    train_val = load_librispeech_subset(**loader_kwargs)
    if dataset_cfg.get("return_full_validation", False):
        train_ds, val_ds, raw_full_val_ds = train_val
        print(f"📏 Full validation set retained with {len(raw_full_val_ds)} samples.")
    else:
        train_ds, val_ds = train_val
        raw_full_val_ds = val_ds

    max_duration_seconds = dataset_cfg.get("max_duration_seconds")
    if max_duration_seconds is not None:
        max_duration_seconds = float(max_duration_seconds)

        def _keep_duration(example: dict, *, max_duration: float = max_duration_seconds) -> bool:
            duration = example.get("duration") or 0.0
            return float(duration) <= max_duration

        train_before = len(train_ds)
        train_ds = train_ds.filter(_keep_duration)
        if len(train_ds) != train_before:
            print(
                f"⏱️ Filtered {train_before - len(train_ds)} training samples longer than"
                f" {max_duration_seconds:.1f}s."
            )

        val_before = len(val_ds)
        val_ds = val_ds.filter(_keep_duration)
        if len(val_ds) != val_before:
            print(
                f"⏱️ Filtered {val_before - len(val_ds)} validation samples longer than"
                f" {max_duration_seconds:.1f}s."
            )

        full_before = len(raw_full_val_ds)
        raw_full_val_ds = raw_full_val_ds.filter(_keep_duration)
        if len(raw_full_val_ds) != full_before:
            print(
                f"⏱️ Filtered {full_before - len(raw_full_val_ds)} full-eval samples longer than"
                f" {max_duration_seconds:.1f}s."
            )

    # Filter datasets to keep only necessary columns
    keep_columns = ["audio", "text", "duration"]
    train_ds = filter_dataset_columns(train_ds, keep_columns)
    val_ds = filter_dataset_columns(val_ds, keep_columns)
    full_val_ds = filter_dataset_columns(raw_full_val_ds, keep_columns)

    eval_val_ds = val_ds

    max_eval_samples = training_cfg.get("max_eval_samples")
    if max_eval_samples is not None:
        max_eval_samples = int(max(0, max_eval_samples))
        if 0 < max_eval_samples < len(val_ds):
            eval_val_ds = val_ds.select(range(max_eval_samples))
            print(f"🔍 Validation truncated to {len(eval_val_ds)} samples for faster eval.")

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = OmniASRCollator(processor=processor, sampling_rate=target_sr)

    # Parse training configuration with ASR-specific defaults
    task_defaults = {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 4,
        "learning_rate": 5e-5,
        "num_train_epochs": 2,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "early_stopping_patience": 1,
        "length_column_name": None,
        "generation_kwargs": {
            "max_new_tokens": 196,
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
        },
    }
    train_config = parse_training_config(
        training_cfg,
        num_train_examples=len(train_ds),
        task_defaults=task_defaults,
    )

    # Build training arguments
    training_args = build_training_arguments(train_config, output_dir=str(output_dir))
    early_stopping_kwargs = build_early_stopping_kwargs(train_config)

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=partial(compute_asr_metrics, processor=processor),
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
    )

    # Run training with evaluation
    run_training_with_evaluation(
        trainer,
        processor=processor,
        full_eval_dataset=full_val_ds,
        initial_eval=train_config.initial_eval,
        history_csv_path=history_csv_path,
        loss_plot_path=loss_plot_path,
        final_adapter_dir=final_adapter_dir,
    )



def run_emotion_task(config_path: Path) -> None:
    """Run the emotion recognition fine-tuning workflow."""
    run_audio_classification_task(config_path, CLASSIFICATION_TASK_SPECS[EMOTION_TASK_NAME])


def run_speaker_id_task(config_path: Path) -> None:
    """Run the speaker identification fine-tuning workflow."""
    run_audio_classification_task(config_path, CLASSIFICATION_TASK_SPECS[SPEAKER_TASK_NAME])


def run_intent_task(config_path: Path) -> None:
    """Run the intent classification fine-tuning workflow."""
    run_audio_classification_task(config_path, CLASSIFICATION_TASK_SPECS[INTENT_TASK_NAME])


def run_speech_qa_task(config_path: Path) -> None:
    """Run the speech question answering fine-tuning workflow."""
    config = load_config(config_path)
    seed = config.get('seed')
    set_global_seed(seed)

    training_cfg = config.get('training', {})
    dataset_cfg = config.get('dataset', {})
    artifacts_cfg = config.get('artifacts', {})
    metrics_cfg = config.get('metrics', {})
    model_cfg = config.get('model', {})

    model_path = PACKAGE_ROOT / model_cfg.get('path', 'models/Qwen2.5-Omni-3B')
    artifact_dirs = get_speech_qa_artifact_directories(PACKAGE_ROOT)

    output_dir = ensure_dir(
        artifact_dirs['adapters'] / artifacts_cfg.get('adapter_subdir', 'qwen2_5_omni_lora_speech_qa')
    )
    metrics_dir = ensure_dir(artifact_dirs['metrics'])

    history_csv_path = metrics_dir / metrics_cfg.get('history_csv', 'speech_qa_training_history.csv')
    loss_plot_path = metrics_dir / metrics_cfg.get('loss_plot', 'speech_qa_loss_metrics.png')
    final_adapter_dir = output_dir / 'final'

    print('🔧 Loading model and processor...')
    lora_config = None
    if "lora" in model_cfg:
        lora_config = create_lora_config_from_dict(model_cfg["lora"])
        print(f"  Using LoRA config from YAML: r={model_cfg['lora'].get('r')}, alpha={model_cfg['lora'].get('alpha')}")
    model, processor = load_qwen_model(model_path, lora_config=lora_config)

    dataset_seed = dataset_cfg.get('seed', seed)
    cache_path = _resolve_dataset_cache_path(dataset_cfg, artifact_dirs)
    loader_kwargs = _build_speech_qa_loader_kwargs(
        dataset_cfg,
        seed=dataset_seed,
        cache_path=cache_path,
    )
    train_ds, val_ds, test_ds, answers_map = load_speech_qa_dataset(**loader_kwargs)

    if train_ds is None:
        raise RuntimeError('Speech QA dataset did not provide a training split.')

    # Filter datasets to keep only necessary columns
    keep_columns = ['audio', 'question', 'answers', 'label_text', 'transcript', 'context', 'duration', 'id']
    train_ds = filter_dataset_columns(train_ds, keep_columns)
    if val_ds is not None:
        val_ds = filter_dataset_columns(val_ds, keep_columns)
    if test_ds is not None:
        test_ds = filter_dataset_columns(test_ds, keep_columns)

    if val_ds is not None:
        eval_ds_full = val_ds
        eval_answers_full = answers_map.get('validation', [])
    elif test_ds is not None:
        eval_ds_full = test_ds
        eval_answers_full = answers_map.get('test', [])
    else:
        raise RuntimeError('Speech QA dataset requires a validation or test split.')

    eval_ds_for_trainer = eval_ds_full
    eval_answers_for_trainer = eval_answers_full

    max_eval_samples = training_cfg.get('max_eval_samples')
    if max_eval_samples is not None:
        max_eval_samples = int(max(0, max_eval_samples))
        if 0 < max_eval_samples < len(eval_ds_full):
            indices = list(range(max_eval_samples))
            eval_ds_for_trainer = eval_ds_full.select(indices)
            eval_answers_for_trainer = [eval_answers_full[idx] for idx in indices]
            print(f'🔍 Validation truncated to {len(eval_ds_for_trainer)} samples for faster eval.')

    target_sr = getattr(getattr(processor, 'feature_extractor', None), 'sampling_rate', 16000)
    collator = SpeechQACollator(
        processor=processor,
        sampling_rate=target_sr,
        include_transcript=dataset_cfg.get('include_transcript', True),
        include_context=dataset_cfg.get('include_context', False),
    )

    # Parse training configuration with Speech QA-specific defaults
    task_defaults = {
        "gradient_accumulation_steps": 2,
        "learning_rate": 3e-5,
        "num_train_epochs": 6,
        "logging_steps": 100,
        "save_steps": 200,
        "eval_steps": 200,
        "metric_for_best_model": "f1",
        "greater_is_better": True,
        "early_stopping_patience": 4,
        "length_column_name": "duration",
        "generation_kwargs": {
            "max_new_tokens": 48,
            "do_sample": False,
            "temperature": 0.0,
            "num_beams": 1,
        },
    }
    train_config = parse_training_config(
        training_cfg,
        num_train_examples=len(train_ds),
        task_defaults=task_defaults,
    )

    # Build training arguments
    training_args = build_training_arguments(train_config, output_dir=str(output_dir))
    early_stopping_kwargs = build_early_stopping_kwargs(train_config)

    reference_store = {'values': eval_answers_for_trainer}

    def _qa_metrics(eval_pred: Any) -> Dict[str, float]:
        return compute_speech_qa_metrics(
            eval_pred,
            processor=processor,
            reference_answers=reference_store['values'],
        )

    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_ds_for_trainer,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=_qa_metrics,
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
    )

    # Run training with evaluation (using full eval dataset)
    reference_store['values'] = eval_answers_full
    run_training_with_evaluation(
        trainer,
        processor=processor,
        full_eval_dataset=eval_ds_full,
        initial_eval=train_config.initial_eval,
        history_csv_path=history_csv_path,
        loss_plot_path=loss_plot_path,
        final_adapter_dir=final_adapter_dir,
    )

    # Optional: additional test set evaluation
    if test_ds is not None and test_ds is not eval_ds_full:
        test_answers = answers_map.get('test', [])
        if test_answers:
            reference_store['values'] = test_answers
            print(f'🧪 Running evaluation on test split ({len(test_ds)} samples)...')
            test_metrics = trainer.evaluate(eval_dataset=test_ds)
            if isinstance(test_metrics, dict) and test_metrics:
                scalar_test_metrics = {
                    key: value for key, value in test_metrics.items() if isinstance(value, (int, float))
                }
                if scalar_test_metrics:
                    formatted = ', '.join(f"{k}={v:.4f}" for k, v in scalar_test_metrics.items())
                    print(f'📦 Test metrics: {formatted}')


def main() -> None:
    """CLI entry point."""
    args = parse_args()
    if args.task == ASR_TASK_NAME:
        config_path = get_asr_config_path(PACKAGE_ROOT, args.config)
        run_asr_task(config_path)
    elif args.task == EMOTION_TASK_NAME:
        config_path = get_emotion_config_path(PACKAGE_ROOT, args.config)
        run_emotion_task(config_path)
    elif args.task == SPEAKER_TASK_NAME:
        config_path = get_speaker_config_path(PACKAGE_ROOT, args.config)
        run_speaker_id_task(config_path)
    elif args.task == INTENT_TASK_NAME:
        config_path = get_intent_config_path(PACKAGE_ROOT, args.config)
        run_intent_task(config_path)
    elif args.task == SPEECH_QA_TASK_NAME:
        config_path = get_speech_qa_config_path(PACKAGE_ROOT, args.config)
        run_speech_qa_task(config_path)
    else:
        raise NotImplementedError(f"Task '{args.task}' is not supported yet.")


if __name__ == "__main__":
    main()
