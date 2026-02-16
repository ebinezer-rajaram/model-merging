"""Generic task training entry point."""

import argparse
import os
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
    BalancedBatchSampler,
    build_early_stopping_kwargs,
    build_training_arguments,
    CustomTrainer,
    ensure_dir,
    filter_dataset_columns,
    load_config,
    load_qwen_model,
    parse_training_config,
    run_training_with_evaluation,
    set_global_seed,
    WeightedClassSampler,
)
from core.training.losses import (
    FocalLoss,
    WeightedCrossEntropyLoss,
    compute_class_weights,
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
    EmotionRecognitionCollator,
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
from tasks.st import (
    STCollator,
    compute_st_metrics,
    get_artifact_directories as get_st_artifact_directories,
    get_config_path as get_st_config_path,
    load_covost2_dataset,
    TASK_NAME as SPEECH_TRANSLATION_TASK_NAME,
)
from tasks.langid import (
    LanguageIdentificationCollator,
    compute_langid_metrics,
    get_artifact_directories as get_langid_artifact_directories,
    get_config_path as get_langid_config_path,
    load_fleurs_langid_dataset,
    TASK_NAME as LANGID_TASK_NAME,
)
from tasks.kws import (
    KeywordSpottingCollator,
    compute_kws_metrics,
    get_artifact_directories as get_kws_artifact_directories,
    get_config_path as get_kws_config_path,
    load_speech_commands_kws_dataset,
    TASK_NAME as KWS_TASK_NAME,
)
from tasks.speaker_ver import (
    SpeakerVerCollator,
    compute_speaker_ver_metrics,
    get_artifact_directories as get_speaker_ver_artifact_directories,
    get_config_path as get_speaker_ver_config_path,
    load_speaker_ver_dataset,
    TASK_NAME as SPEAKER_VER_TASK_NAME,
)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train task adapters.")
    parser.add_argument("--task", default=ASR_TASK_NAME, help="Task name to run.")
    parser.add_argument("--config", default=None, help="Override config filename.")
    return parser.parse_args()


def setup_wandb_for_task(task_name: str, training_cfg: Dict[str, Any]) -> None:
    """Configure wandb with task-specific project name and logging directories.

    Args:
        task_name: Name of the task (e.g., 'asr', 'emotion', 'intent')
        training_cfg: Training configuration dictionary
    """
    # Configure wandb directory to be under logs/
    os.environ["WANDB_DIR"] = str(PACKAGE_ROOT / "logs" / "wandb")

    # Configure tensorboard directory to be under logs/runs
    # HuggingFace Trainer will use output_dir for tensorboard logs,
    # but this ensures any other tensorboard usage goes to logs/
    os.environ["TENSORBOARD_LOG_DIR"] = str(PACKAGE_ROOT / "logs" / "runs")

    report_to = training_cfg.get("report_to", [])
    if "wandb" in report_to:
        try:
            import wandb
            # Set wandb project based on task name
            project_name = f"speech-merging-{task_name}"
            os.environ["WANDB_PROJECT"] = project_name
            print(f"📊 Configured wandb project: {project_name}")
        except ImportError:
            print("⚠️ wandb not available, skipping wandb configuration")
            pass


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


def _setup_common_components(
    config: Dict[str, Any],
    task_name: str,
    artifact_dirs: Dict[str, Path],
    default_adapter_subdir: str,
    default_history_csv: str,
    default_loss_plot: str,
):
    """Setup components common to all tasks (model, directories, paths)."""
    training_cfg = config.get("training", {})
    artifacts_cfg = config.get("artifacts", {})
    metrics_cfg = config.get("metrics", {})
    model_cfg = config.get("model", {})

    setup_wandb_for_task(task_name, training_cfg)

    model_path = PACKAGE_ROOT / model_cfg.get("path", "data/models/Qwen2.5-Omni-3B")
    output_dir = ensure_dir(
        artifact_dirs["adapters"] / artifacts_cfg.get("adapter_subdir", default_adapter_subdir)
    )
    metrics_dir = ensure_dir(artifact_dirs["metrics"])

    history_csv_path = metrics_dir / metrics_cfg.get("history_csv", default_history_csv)
    loss_plot_path = metrics_dir / metrics_cfg.get("loss_plot", default_loss_plot)
    final_adapter_dir = output_dir / "final"

    print("🔧 Loading model and processor...")
    lora_config = None
    if "lora" in model_cfg:
        lora_config = create_lora_config_from_dict(model_cfg["lora"])
        print(f"  Using LoRA config from YAML: r={model_cfg['lora'].get('r')}, alpha={model_cfg['lora'].get('alpha')}")
    model, processor = load_qwen_model(model_path, lora_config=lora_config)

    return {
        "model": model,
        "processor": processor,
        "output_dir": output_dir,
        "metrics_dir": metrics_dir,
        "history_csv_path": history_csv_path,
        "loss_plot_path": loss_plot_path,
        "final_adapter_dir": final_adapter_dir,
    }


def _truncate_eval_dataset(
    full_eval_ds,
    max_eval_samples: Optional[int],
    seed: Optional[int] = None,
    shuffle: bool = False,
    return_indices: bool = False,
):
    """Truncate evaluation dataset if max_eval_samples is specified.

    Args:
        full_eval_ds: Full evaluation dataset
        max_eval_samples: Maximum number of samples to use (None = use all)
        seed: Random seed for shuffling (only used if shuffle=True)
        shuffle: If True, randomly sample instead of taking first N samples
    """
    if max_eval_samples is None:
        if return_indices:
            return full_eval_ds, None
        return full_eval_ds

    max_eval_samples = int(max(0, max_eval_samples))
    if 0 < max_eval_samples < len(full_eval_ds):
        if shuffle:
            # Random sampling with optional seed
            import random
            indices = list(range(len(full_eval_ds)))
            if seed is not None:
                random.Random(seed).shuffle(indices)
            else:
                random.shuffle(indices)
            selected_indices = sorted(indices[:max_eval_samples])
            truncated = full_eval_ds.select(selected_indices)
            print(f"🔍 Validation randomly sampled to {len(truncated)} samples (seed={seed}).")
        else:
            # Sequential sampling (first N samples)
            selected_indices = list(range(max_eval_samples))
            truncated = full_eval_ds.select(range(max_eval_samples))
            print(f"🔍 Validation truncated to {len(truncated)} samples for faster eval.")
        if return_indices:
            return truncated, selected_indices
        return truncated
    if return_indices:
        return full_eval_ds, None
    return full_eval_ds


def _reindex_reference_answers(
    reference_answers: Sequence[Sequence[str]],
    selected_indices: Optional[Sequence[int]],
    expected_length: int,
) -> list[list[str]]:
    """Align reference answers with selected evaluation indices."""
    base = list(reference_answers or [])
    if selected_indices is None:
        aligned = [list(answer) for answer in base[:expected_length]]
    else:
        aligned = []
        for idx in selected_indices:
            idx_int = int(idx)
            if 0 <= idx_int < len(base):
                aligned.append(list(base[idx_int]))
            else:
                aligned.append([])
    if len(aligned) < expected_length:
        aligned.extend([[] for _ in range(expected_length - len(aligned))])
    elif len(aligned) > expected_length:
        aligned = aligned[:expected_length]
    return aligned


def _filter_by_duration(
    dataset,
    max_duration_seconds: Optional[float],
    split_name: str = "dataset",
    min_duration_seconds: Optional[float] = None,
):
    """Filter dataset to keep only samples within duration range."""
    if max_duration_seconds is None and min_duration_seconds is None:
        return dataset

    def _keep_duration(example: dict) -> bool:
        duration = example.get("duration") or 0.0
        duration_float = float(duration)
        if max_duration_seconds is not None and duration_float > max_duration_seconds:
            return False
        if min_duration_seconds is not None and duration_float < min_duration_seconds:
            return False
        return True

    before = len(dataset)
    filtered = dataset.filter(_keep_duration)
    if len(filtered) != before:
        filtered_count = before - len(filtered)
        duration_info = []
        if max_duration_seconds is not None:
            duration_info.append(f">{max_duration_seconds:.1f}s")
        if min_duration_seconds is not None:
            duration_info.append(f"<{min_duration_seconds:.1f}s")
        duration_str = " or ".join(duration_info)
        print(f"⏱️ Filtered {filtered_count} {split_name} samples ({duration_str}).")
    return filtered


class _DatasetWithReferenceStore:
    """Wrapper to update reference store before dataset access (for Speech QA)."""

    def __init__(self, dataset, reference_store: Dict[str, Any], reference_values: Any):
        self._dataset = dataset
        self._reference_store = reference_store
        self._reference_values = reference_values
        # Copy dataset attributes for compatibility
        for attr in dir(dataset):
            if not attr.startswith('_') and attr not in ('_dataset', '_reference_store', '_reference_values'):
                try:
                    setattr(self, attr, getattr(dataset, attr))
                except (AttributeError, TypeError):
                    pass

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, idx):
        self._reference_store['values'] = self._reference_values
        return self._dataset[idx]

    def __iter__(self):
        self._reference_store['values'] = self._reference_values
        return iter(self._dataset)


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
    "revision",
    "data_dir",
)

SPEECH_QA_LOADER_KEYS: Tuple[str, ...] = (
    "dataset_name",
    "dataset_config",
    "max_train_samples",
    "max_validation_samples",
    "max_test_samples",
    "max_duration",
    "min_duration",
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
    return EmotionRecognitionCollator(
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
        max_audio_length=dataset_cfg.get("max_audio_length", None),  # Trim audio to this duration
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


def _build_langid_collator(
    processor,
    dataset_cfg: Dict[str, Any],
    label_names: Sequence[str],
    sampling_rate: int,
):
    return LanguageIdentificationCollator(
        processor=processor,
        sampling_rate=sampling_rate,
        label_names=list(label_names or []),
    )


def _build_kws_collator(
    processor,
    dataset_cfg: Dict[str, Any],
    label_names: Sequence[str],
    sampling_rate: int,
):
    return KeywordSpottingCollator(
        processor=processor,
        sampling_rate=sampling_rate,
        label_names=list(label_names or []),
    )


def _build_speaker_ver_collator(
    processor,
    dataset_cfg: Dict[str, Any],
    label_names: Sequence[str],
    sampling_rate: int,
):
    return SpeakerVerCollator(
        processor=processor,
        sampling_rate=sampling_rate,
        label_names=list(label_names or []),
        max_audio_length=dataset_cfg.get("max_audio_length", None),
        audio_gap_seconds=dataset_cfg.get("audio_gap_seconds", 0.5),
    )


def _build_emotion_metrics(processor, label_names: Sequence[str], store_predictions: bool = False):
    return partial(
        compute_emotion_metrics,
        processor=processor,
        label_names=list(label_names or []),
        store_predictions=store_predictions,
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


def _build_langid_metrics(processor, label_names: Sequence[str]):
    return partial(
        compute_langid_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )


def _build_kws_metrics(processor, label_names: Sequence[str]):
    return partial(
        compute_kws_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )


def _build_speaker_ver_metrics(processor, label_names: Sequence[str]):
    return partial(
        compute_speaker_ver_metrics,
        processor=processor,
        label_names=list(label_names or []),
    )


def _create_loss_function(
    loss_config: Optional[Dict[str, Any]],
    train_dataset,
    num_classes: int,
    label_column: str = "label",
):
    """Create custom loss function based on configuration.

    Args:
        loss_config: Loss configuration from YAML
        train_dataset: Training dataset (for computing class weights)
        num_classes: Number of classes
        label_column: Name of the label column in dataset

    Returns:
        Custom loss function or None (to use default)
    """
    if loss_config is None:
        return None

    loss_type = loss_config.get("type", "default")

    if loss_type == "default":
        return None

    elif loss_type == "focal":
        focal_cfg = loss_config.get("focal", {})
        gamma = focal_cfg.get("gamma", 2.0)
        alpha = focal_cfg.get("alpha")

        print(f"🎯 Using Focal Loss (gamma={gamma}, alpha={alpha})")
        return FocalLoss(gamma=gamma, alpha=alpha, ignore_index=-100)

    elif loss_type == "weighted":
        weighted_cfg = loss_config.get("weighted", {})
        weights = weighted_cfg.get("weights")
        auto_compute = weighted_cfg.get("auto_compute", False)

        if weights is not None:
            # Use explicitly provided weights
            print(f"⚖️  Using Weighted Cross-Entropy with manual weights: {weights}")
            return WeightedCrossEntropyLoss(weights=weights, ignore_index=-100)

        elif auto_compute:
            # Auto-compute weights from training data
            method = weighted_cfg.get("method", "inverse")
            print(f"⚖️  Computing class weights using method: {method}")

            # Extract labels from training dataset
            labels = [example[label_column] for example in train_dataset]
            weights = compute_class_weights(labels, num_classes, method=method)

            print(f"   Computed weights: {weights.tolist()}")
            return WeightedCrossEntropyLoss(weights=weights, ignore_index=-100)

        else:
            print("⚠️  Weighted loss requested but no weights provided. Using default loss.")
            return None

    else:
        print(f"⚠️  Unknown loss type '{loss_type}'. Using default loss.")
        return None


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
    LANGID_TASK_NAME: ClassificationTaskSpec(
        task_name=LANGID_TASK_NAME,
        get_config_path=get_langid_config_path,
        get_artifact_directories=get_langid_artifact_directories,
        dataset_loader=load_fleurs_langid_dataset,
        loader_extra_keys=("languages",),
        collator_builder=_build_langid_collator,
        metrics_builder=_build_langid_metrics,
        keep_columns=("audio", "label", "duration"),
        default_adapter_subdir="qwen2_5_omni_lora_langid",
    ),
    KWS_TASK_NAME: ClassificationTaskSpec(
        task_name=KWS_TASK_NAME,
        get_config_path=get_kws_config_path,
        get_artifact_directories=get_kws_artifact_directories,
        dataset_loader=load_speech_commands_kws_dataset,
        loader_extra_keys=("revision",),
        collator_builder=_build_kws_collator,
        metrics_builder=_build_kws_metrics,
        keep_columns=("audio", "label", "duration"),
        default_adapter_subdir="qwen2_5_omni_lora_kws",
    ),
    SPEAKER_VER_TASK_NAME: ClassificationTaskSpec(
        task_name=SPEAKER_VER_TASK_NAME,
        get_config_path=get_speaker_ver_config_path,
        get_artifact_directories=get_speaker_ver_artifact_directories,
        dataset_loader=load_speaker_ver_dataset,
        loader_extra_keys=("total_pairs", "audio_gap_seconds", "split_by_speakers"),
        collator_builder=_build_speaker_ver_collator,
        metrics_builder=_build_speaker_ver_metrics,
        keep_columns=("audio_a", "audio_b", "label", "duration_a", "duration_b"),
        default_adapter_subdir="qwen2_5_omni_lora_speaker_ver",
    ),
}


def run_audio_classification_task(config_path: Path, spec: ClassificationTaskSpec) -> None:
    """Generic runner for audio-based classification tasks."""
    config = load_config(config_path)
    seed = config.get("seed")
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    artifact_dirs = spec.get_artifact_directories(PACKAGE_ROOT)

    # Setup common components
    components = _setup_common_components(
        config,
        spec.task_name,
        artifact_dirs,
        spec.default_adapter_subdir,
        f"{spec.task_name}_training_history.csv",
        f"{spec.task_name}_loss_plot.png",
    )
    model = components["model"]
    processor = components["processor"]
    output_dir = components["output_dir"]
    metrics_dir = components["metrics_dir"]
    history_csv_path = components["history_csv_path"]
    loss_plot_path = components["loss_plot_path"]
    final_adapter_dir = components["final_adapter_dir"]

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

    # Use validation for early stopping during training
    eval_ds = val_ds or test_ds
    if eval_ds is None:
        raise RuntimeError(f"{spec.task_name} dataset requires a validation or test split.")

    full_eval_ds = eval_ds

    # Filter datasets to keep only necessary columns
    train_ds = filter_dataset_columns(train_ds, spec.keep_columns, always_keep=["duration"])
    full_eval_ds = filter_dataset_columns(full_eval_ds, spec.keep_columns, always_keep=["duration"])

    # Prepare test dataset if available and different from validation
    test_ds_for_eval = None
    if test_ds is not None and val_ds is not None:
        # We have both validation and test, so use test for final evaluation
        test_ds_for_eval = filter_dataset_columns(test_ds, spec.keep_columns, always_keep=["duration"])

    # Get eval sampling configuration
    shuffle_eval = training_cfg.get("shuffle_eval_subset", False)
    eval_ds_for_trainer = _truncate_eval_dataset(
        full_eval_ds,
        training_cfg.get("max_eval_samples"),
        seed=seed,
        shuffle=shuffle_eval
    )

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

    # Create custom loss function if configured
    loss_config = config.get("loss")
    custom_loss_fn = _create_loss_function(
        loss_config,
        train_ds,
        num_classes=len(label_names),
        label_column="label",
    )

    # Create custom sampler for balanced training if configured
    custom_sampler = None
    balanced_sampling = training_cfg.get("balanced_sampling")
    if balanced_sampling:
        if balanced_sampling == "balanced_batch":
            # Create balanced batch sampler (each batch has equal class representation)
            custom_sampler = BalancedBatchSampler(
                dataset=train_ds,
                batch_size=training_args.per_device_train_batch_size,
                num_classes=len(label_names),
                drop_last=training_args.dataloader_drop_last,
                shuffle=True,
            )
            # When using batch sampler, disable group_by_length
            if training_args.group_by_length:
                print("⚠️  Disabling group_by_length since balanced_batch sampler is used")
                training_args.group_by_length = False
        elif balanced_sampling == "weighted":
            # Create weighted sampler (oversample minority classes)
            # Get weighting method from training config
            weighting_method = training_cfg.get("sampling_method", "inverse")

            # Create a temporary dataset with integer labels for the sampler
            # WeightedClassSampler expects integer class indices, not strings
            label_to_idx = {name: idx for idx, name in enumerate(label_names)}

            # Create a wrapper dataset that maps string labels to integers
            class LabelMappedDataset:
                def __init__(self, dataset, label_to_idx):
                    self.dataset = dataset
                    self.label_to_idx = label_to_idx

                def __len__(self):
                    return len(self.dataset)

                def __getitem__(self, idx):
                    item = self.dataset[idx]
                    # Convert string label to integer index
                    label_str = item['label']
                    item_copy = dict(item)
                    item_copy['label'] = self.label_to_idx.get(label_str, 0)
                    return item_copy

            mapped_dataset = LabelMappedDataset(train_ds, label_to_idx)

            custom_sampler = WeightedClassSampler(
                dataset=mapped_dataset,
                num_samples=len(train_ds),
                replacement=True,
                method=weighting_method,
            )
        else:
            print(f"⚠️  Unknown balanced_sampling option: {balanced_sampling}")

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
        constrained_decoding_fn=None,
        custom_loss_fn=custom_loss_fn,
        custom_sampler=custom_sampler,
    )

    # Extract confusion matrix configuration
    metrics_cfg = config.get("metrics", {})
    confusion_matrix_filename = metrics_cfg.get("confusion_matrix")
    confusion_matrix_normalize = metrics_cfg.get("normalize_confusion_matrix", True)
    confusion_matrix_path = None
    if confusion_matrix_filename:
        confusion_matrix_path = metrics_dir / confusion_matrix_filename

    # Resolve checkpoint path for resuming
    from core.training.training_loop import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(
        train_config.resume_from_checkpoint,
        output_dir,
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
        config=config,
        config_path=config_path,
        metrics_dir=metrics_dir,
        test_dataset=test_ds_for_eval,
        test_split_name="test",
        confusion_matrix_path=confusion_matrix_path,
        confusion_matrix_normalize=confusion_matrix_normalize,
        label_names=label_names,
        resume_from_checkpoint=checkpoint_path,
    )


def run_asr_task(config_path: Path) -> None:
    """Run the ASR fine-tuning workflow."""
    config = load_config(config_path)
    seed = config.get("seed")
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})
    metrics_cfg = config.get("metrics", {})
    artifact_dirs = get_asr_artifact_directories(PACKAGE_ROOT)

    # Setup common components
    components = _setup_common_components(
        config,
        ASR_TASK_NAME,
        artifact_dirs,
        "qwen2_5_omni_lora_asr_10h",
        "training_history.csv",
        "loss_wer_plot.png",
    )
    model = components["model"]
    processor = components["processor"]
    output_dir = components["output_dir"]
    metrics_dir = components["metrics_dir"]
    history_csv_path = components["history_csv_path"]
    loss_plot_path = components["loss_plot_path"]
    final_adapter_dir = components["final_adapter_dir"]

    dataset_seed = dataset_cfg.get("seed", seed)
    dataset_cache_path = _resolve_dataset_cache_path(dataset_cfg, artifact_dirs)

    # Load train, val, and optionally test splits
    return_test = dataset_cfg.get("return_test_split", True)
    loader_kwargs = dict(
        train_hours=dataset_cfg.get("train_hours", 10.0),
        val_hours=dataset_cfg.get("val_hours", 1.0),
        seed=dataset_seed,
        num_proc=dataset_cfg.get("num_proc"),
        cache_dir=dataset_cache_path,
        cache_splits=dataset_cfg.get("cache_splits", True),
        force_rebuild=dataset_cfg.get("force_rebuild", False),
        return_full_validation=dataset_cfg.get("return_full_validation", False),
        return_test_split=return_test,
        test_split=dataset_cfg.get("test_split", "test.clean"),
        test_hours=dataset_cfg.get("test_hours"),
    )
    train_val = load_librispeech_subset(**loader_kwargs)

    # Unpack based on what was returned
    test_ds = None
    if dataset_cfg.get("return_full_validation", False) and return_test:
        train_ds, val_ds, raw_full_val_ds, test_ds = train_val
        print(f"📏 Full validation set retained with {len(raw_full_val_ds)} samples.")
    elif dataset_cfg.get("return_full_validation", False):
        train_ds, val_ds, raw_full_val_ds = train_val
        print(f"📏 Full validation set retained with {len(raw_full_val_ds)} samples.")
    elif return_test:
        train_ds, val_ds, test_ds = train_val
        raw_full_val_ds = val_ds
    else:
        train_ds, val_ds = train_val
        raw_full_val_ds = val_ds

    max_duration = dataset_cfg.get("max_duration")
    min_duration = dataset_cfg.get("min_duration")
    if max_duration is not None or min_duration is not None:
        if max_duration is not None:
            max_duration = float(max_duration)
        if min_duration is not None:
            min_duration = float(min_duration)
        train_ds = _filter_by_duration(train_ds, max_duration, "training", min_duration)
        val_ds = _filter_by_duration(val_ds, max_duration, "validation", min_duration)
        raw_full_val_ds = _filter_by_duration(raw_full_val_ds, max_duration, "full-eval", min_duration)
        if test_ds is not None:
            test_ds = _filter_by_duration(test_ds, max_duration, "test", min_duration)

    # Filter datasets to keep only necessary columns
    keep_columns = ["audio", "text", "duration"]
    train_ds = filter_dataset_columns(train_ds, keep_columns)
    val_ds = filter_dataset_columns(val_ds, keep_columns)
    full_val_ds = filter_dataset_columns(raw_full_val_ds, keep_columns)
    if test_ds is not None:
        test_ds = filter_dataset_columns(test_ds, keep_columns)

    # Get eval sampling configuration
    shuffle_eval = training_cfg.get("shuffle_eval_subset", False)
    eval_val_ds = _truncate_eval_dataset(
        val_ds,
        training_cfg.get("max_eval_samples"),
        seed=seed,
        shuffle=shuffle_eval
    )

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

    wer_normalization = metrics_cfg.get("wer_normalization", "default")
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=eval_val_ds,
        data_collator=collator,
        processing_class=processor,
        compute_metrics=partial(compute_asr_metrics, processor=processor, wer_normalization=wer_normalization),
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
    )

    # Resolve checkpoint path for resuming
    from core.training.training_loop import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(
        train_config.resume_from_checkpoint,
        output_dir,
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
        config=config,
        config_path=config_path,
        metrics_dir=metrics_dir,
        test_dataset=test_ds,
        test_split_name=dataset_cfg.get("test_split", "test.clean"),
        resume_from_checkpoint=checkpoint_path,
    )


def run_st_task(config_path: Path) -> None:
    """Run the Speech Translation fine-tuning workflow."""
    config = load_config(config_path)
    seed = config.get("seed")
    set_global_seed(seed)

    training_cfg = config.get("training", {})
    dataset_cfg = config.get("dataset", {})

    # Get language pair for language-specific artifacts and prompts
    language = config.get("language", "en_de")
    artifact_dirs = get_st_artifact_directories(PACKAGE_ROOT, language=language)

    # Setup common components
    components = _setup_common_components(
        config,
        SPEECH_TRANSLATION_TASK_NAME,
        artifact_dirs,
        "qwen2_5_omni_lora_st",
        "st_training_history.csv",
        "st_loss_bleu_plot.png",
    )
    model = components["model"]
    processor = components["processor"]
    output_dir = components["output_dir"]
    metrics_dir = components["metrics_dir"]
    history_csv_path = components["history_csv_path"]
    loss_plot_path = components["loss_plot_path"]
    final_adapter_dir = components["final_adapter_dir"]

    dataset_seed = dataset_cfg.get("seed", seed)
    dataset_cache_path = _resolve_dataset_cache_path(dataset_cfg, artifact_dirs)

    # Load CoVoST2 dataset
    # Use 'language' parameter as dataset_config (single source of truth)
    loader_kwargs = dict(
        dataset_name=dataset_cfg.get("dataset_name", "fixie-ai/covost2"),
        dataset_config=language,  # Use language parameter instead of separate dataset_config
        max_train_samples=dataset_cfg.get("max_train_samples"),
        max_validation_samples=dataset_cfg.get("max_validation_samples"),
        max_test_samples=dataset_cfg.get("max_test_samples"),
        max_duration=dataset_cfg.get("max_duration"),
        min_duration=dataset_cfg.get("min_duration"),
        seed=dataset_seed,
        num_proc=dataset_cfg.get("num_proc"),
        cache_dir=dataset_cache_path,
        cache_splits=dataset_cfg.get("cache_splits", True),
        force_rebuild=dataset_cfg.get("force_rebuild", False),
        audio_column=dataset_cfg.get("audio_column"),
        source_column=dataset_cfg.get("source_column"),
        translation_column=dataset_cfg.get("translation_column"),
        train_split=dataset_cfg.get("train_split", "train"),
        validation_split=dataset_cfg.get("validation_split", "validation"),
        test_split=dataset_cfg.get("test_split", "test"),
    )
    train_ds, val_ds, test_ds = load_covost2_dataset(**loader_kwargs)

    if train_ds is None:
        raise RuntimeError("ST dataset did not provide a training split.")
    if val_ds is None:
        raise RuntimeError("ST dataset did not provide a validation split.")

    # Filter datasets to keep only necessary columns
    keep_columns = ["audio", "text", "translation", "duration"]
    train_ds = filter_dataset_columns(train_ds, keep_columns)
    val_ds = filter_dataset_columns(val_ds, keep_columns)
    full_val_ds = filter_dataset_columns(val_ds, keep_columns)
    if test_ds is not None:
        test_ds = filter_dataset_columns(test_ds, keep_columns)

    # Get eval sampling configuration
    shuffle_eval = training_cfg.get("shuffle_eval_subset", False)
    eval_val_ds = _truncate_eval_dataset(
        val_ds,
        training_cfg.get("max_eval_samples"),
        seed=seed,
        shuffle=shuffle_eval
    )

    target_sr = getattr(getattr(processor, "feature_extractor", None), "sampling_rate", 16000)
    collator = STCollator(processor=processor, sampling_rate=target_sr, language_pair=language)

    # Parse training configuration with ST-specific defaults
    task_defaults = {
        "per_device_train_batch_size": 16,
        "per_device_eval_batch_size": 16,
        "learning_rate": 3e-5,
        "num_train_epochs": 3,
        "metric_for_best_model": "bleu",
        "greater_is_better": True,
        "early_stopping_patience": 5,
        "length_column_name": "duration",
        "generation_kwargs": {
            "max_new_tokens": 128,
            "do_sample": False,
            "num_beams": 4,
            "length_penalty": 1.0,
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
        compute_metrics=partial(compute_st_metrics, processor=processor, target_lang=language),
        callbacks=[EarlyStoppingCallback(**early_stopping_kwargs)],
        generation_kwargs=train_config.generation_kwargs,
    )

    # Resolve checkpoint path for resuming
    from core.training.training_loop import resolve_checkpoint_path
    checkpoint_path = resolve_checkpoint_path(
        train_config.resume_from_checkpoint,
        output_dir,
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
        config=config,
        config_path=config_path,
        metrics_dir=metrics_dir,
        test_dataset=test_ds,
        test_split_name=dataset_cfg.get("test_split", "test"),
        resume_from_checkpoint=checkpoint_path,
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


def run_langid_task(config_path: Path) -> None:
    """Run the language identification fine-tuning workflow."""
    run_audio_classification_task(config_path, CLASSIFICATION_TASK_SPECS[LANGID_TASK_NAME])


def run_kws_task(config_path: Path) -> None:
    """Run the keyword spotting fine-tuning workflow."""
    run_audio_classification_task(config_path, CLASSIFICATION_TASK_SPECS[KWS_TASK_NAME])


def run_speaker_ver_task(config_path: Path) -> None:
    """Run the speaker verification fine-tuning workflow."""
    run_audio_classification_task(config_path, CLASSIFICATION_TASK_SPECS[SPEAKER_VER_TASK_NAME])


def run_speech_qa_task(config_path: Path) -> None:
    """Run the speech question answering fine-tuning workflow."""
    config = load_config(config_path)
    seed = config.get('seed')
    set_global_seed(seed)

    training_cfg = config.get('training', {})
    dataset_cfg = config.get('dataset', {})
    artifact_dirs = get_speech_qa_artifact_directories(PACKAGE_ROOT)

    # Setup common components
    components = _setup_common_components(
        config,
        SPEECH_QA_TASK_NAME,
        artifact_dirs,
        'qwen2_5_omni_lora_speech_qa',
        'speech_qa_training_history.csv',
        'speech_qa_loss_metrics.png',
    )
    model = components["model"]
    processor = components["processor"]
    output_dir = components["output_dir"]
    metrics_dir = components["metrics_dir"]
    history_csv_path = components["history_csv_path"]
    loss_plot_path = components["loss_plot_path"]
    final_adapter_dir = components["final_adapter_dir"]

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

    max_eval_samples = training_cfg.get('max_eval_samples')
    shuffle_eval = training_cfg.get("shuffle_eval_subset", False)
    eval_ds_for_trainer, eval_selected_indices = _truncate_eval_dataset(
        eval_ds_full,
        max_eval_samples,
        seed=seed,
        shuffle=shuffle_eval,
        return_indices=True,
    )
    eval_answers_for_trainer = _reindex_reference_answers(
        eval_answers_full,
        eval_selected_indices,
        len(eval_ds_for_trainer),
    )

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

    # Prepare test dataset and answers if available
    test_ds_for_eval = None
    test_answers = None
    if test_ds is not None and val_ds is not None:
        # We have both validation and test, so use test for final evaluation
        test_ds_for_eval = test_ds
        test_answers = answers_map.get('test', [])

    # Run training with evaluation (using full eval dataset)
    reference_store['values'] = eval_answers_full

    # Wrap test dataset to update reference answers before evaluation
    reference_store['values'] = eval_answers_full
    test_ds_wrapped = None
    if test_ds_for_eval is not None and test_answers:
        test_ds_wrapped = _DatasetWithReferenceStore(test_ds_for_eval, reference_store, test_answers)

    run_training_with_evaluation(
        trainer,
        processor=processor,
        full_eval_dataset=eval_ds_full,
        initial_eval=train_config.initial_eval,
        history_csv_path=history_csv_path,
        loss_plot_path=loss_plot_path,
        final_adapter_dir=final_adapter_dir,
        config=config,
        config_path=config_path,
        metrics_dir=metrics_dir,
        test_dataset=test_ds_wrapped,
        test_split_name="test",
    )


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
    elif args.task == LANGID_TASK_NAME:
        config_path = get_langid_config_path(PACKAGE_ROOT, args.config)
        run_langid_task(config_path)
    elif args.task == KWS_TASK_NAME:
        config_path = get_kws_config_path(PACKAGE_ROOT, args.config)
        run_kws_task(config_path)
    elif args.task == SPEAKER_VER_TASK_NAME:
        config_path = get_speaker_ver_config_path(PACKAGE_ROOT, args.config)
        run_speaker_ver_task(config_path)
    elif args.task == SPEECH_QA_TASK_NAME:
        config_path = get_speech_qa_config_path(PACKAGE_ROOT, args.config)
        run_speech_qa_task(config_path)
    elif args.task == SPEECH_TRANSLATION_TASK_NAME:
        config_path = get_st_config_path(PACKAGE_ROOT, args.config)
        run_st_task(config_path)
    else:
        raise NotImplementedError(f"Task '{args.task}' is not supported yet.")


if __name__ == "__main__":
    main()
