from __future__ import annotations

from dataclasses import dataclass, field

import pytest
from datasets import Dataset

from core.config.loader import load_task_config
from core.config.registry import TASK_REGISTRY
from core.config.schemas import TaskConfig as TaskSchemaConfig
from core.evaluation.eval_utils import (
    SPEECH_QA_EVAL_KEYS,
    _TASK_CONFIG_GETTERS,
    get_registered_eval_tasks,
    prepare_task_for_evaluation,
)
from core.training.train_task import CLASSIFICATION_TASK_SPECS, SPEECH_QA_LOADER_KEYS
from merging.config.unified import load_merge_config
from merging.continual.suite import VALID_TASKS
from merging.evaluation.interference import TASK_METRICS
from merging.runtime.utils import TASK_REGISTRY as MERGE_TASK_REGISTRY


ALL_TASKS = set(TASK_REGISTRY)
CLASSIFICATION_TASKS = {
    "emotion",
    "intent",
    "kws",
    "langid",
    "speaker_id",
    "speaker_ver",
    "speech_qa",
    "vocalsound",
}
GENERATION_TASKS = {"asr", "st"}


def test_all_task_registries_have_the_same_task_surface() -> None:
    assert ALL_TASKS == set(MERGE_TASK_REGISTRY)
    assert ALL_TASKS == set(get_registered_eval_tasks())
    assert ALL_TASKS == set(_TASK_CONFIG_GETTERS)
    assert ALL_TASKS == set(VALID_TASKS)
    assert CLASSIFICATION_TASKS | GENERATION_TASKS == ALL_TASKS
    assert CLASSIFICATION_TASKS & GENERATION_TASKS == set()


def test_classification_training_specs_cover_trainable_classification_tasks() -> None:
    assert set(CLASSIFICATION_TASK_SPECS) == (CLASSIFICATION_TASKS - {"speech_qa"})


def test_interference_metric_registry_covers_all_primary_scored_tasks() -> None:
    # ST reports BLEU/chrF rather than a single primary scalar used by the
    # interference/continual helpers, so it is intentionally absent here.
    assert set(TASK_METRICS) == (ALL_TASKS - {"st"})
    assert TASK_METRICS["asr"] == ("wer", False)
    assert TASK_METRICS["speech_qa"] == ("accuracy", True)


@pytest.mark.parametrize("task_name", sorted(ALL_TASKS))
def test_default_task_configs_parse_for_every_registered_task(task_name: str) -> None:
    info = TASK_REGISTRY[task_name]
    task_cfg = load_task_config(task_name, info.default_config_file, validate=False)

    assert task_cfg["task"] == task_name
    schema_cfg = TaskSchemaConfig(task=task_name, artifacts={"adapter_subdir": "tmp"})
    assert schema_cfg.task == task_name


def test_speech_qa_allowlists_include_local_dataset_keys() -> None:
    required = {
        "data_dir",
        "train_json",
        "test_json",
        "audio_root",
        "noisy_test_variant",
        "min_wavs_per_split",
        "max_missing_audio_rate",
        "allow_train_only_fallback",
        "audio_merge_policy",
        "max_total_samples",
        "include_choices_in_prompt",
        "subtask_column",
    }
    assert required.issubset(set(SPEECH_QA_LOADER_KEYS))
    assert required.issubset(set(SPEECH_QA_EVAL_KEYS))


def test_merge_presets_cover_vocalsound_when_expected() -> None:
    supermerge_cfg = load_merge_config(
        "configs/merge/supermerge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml"
    )
    assert "vocalsound" in supermerge_cfg.adapters
    assert supermerge_cfg.eval_tasks is not None
    assert "vocalsound" in supermerge_cfg.eval_tasks

    scalar_cfg = load_merge_config(
        "configs/merge/uniform_scalar_delta/merge_uniform_scalar_delta_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml"
    )
    assert "vocalsound" in scalar_cfg.adapters
    assert scalar_cfg.eval_tasks is not None
    assert "vocalsound" in scalar_cfg.eval_tasks


@dataclass
class _DummyTokenizer:
    padding_side: str = "right"
    pad_token_id: int = 0


@dataclass
class _DummyFeatureExtractor:
    sampling_rate: int = 16000


@dataclass
class _DummyProcessor:
    tokenizer: _DummyTokenizer = field(default_factory=_DummyTokenizer)
    feature_extractor: _DummyFeatureExtractor = field(default_factory=_DummyFeatureExtractor)


def _audio() -> dict:
    return {"array": [0.0, 0.0], "sampling_rate": 16000}


def _dataset_for_task(task_name: str) -> Dataset:
    if task_name == "asr":
        return Dataset.from_list([{"audio": _audio(), "text": "hello", "duration": 0.1}])
    if task_name == "st":
        return Dataset.from_list([{"audio": _audio(), "text": "hello", "translation": "hallo", "duration": 0.1}])
    if task_name == "speech_qa":
        return Dataset.from_list(
            [
                {
                    "audio": _audio(),
                    "question": "Which option?",
                    "answers": ["alpha"],
                    "label_text": "A",
                    "id": "q1",
                    "task_name": "demo",
                    "choice_a": "alpha",
                    "choice_b": "beta",
                    "choice_c": "gamma",
                    "choice_d": "delta",
                    "duration": 0.1,
                }
            ]
        )
    if task_name == "speaker_ver":
        return Dataset.from_list(
            [{"audio_a": _audio(), "audio_b": _audio(), "label": 1, "duration_a": 0.1, "duration_b": 0.1}]
        )
    return Dataset.from_list([{"audio": _audio(), "label": 0, "text": "hello", "duration": 0.1}])


def _bundle_for_task(task_name: str) -> tuple:
    dataset = _dataset_for_task(task_name)
    if task_name == "speech_qa":
        return dataset, dataset, dataset, {"train": [["alpha"]], "validation": [["alpha"]], "test": [["alpha"]]}
    if task_name in CLASSIFICATION_TASKS:
        labels = ["no", "yes"] if task_name == "speaker_ver" else ["class_0", "class_1"]
        return dataset, dataset, dataset, labels
    return dataset, dataset, dataset


@pytest.mark.parametrize("task_name", sorted(ALL_TASKS))
def test_prepare_task_for_evaluation_builds_setup_for_every_registered_task(task_name: str) -> None:
    setup = prepare_task_for_evaluation(
        task_name,
        _DummyProcessor(),
        split="validation",
        config={"dataset": {"cache_splits": False}},
        bundle=_bundle_for_task(task_name),
    )

    assert len(setup.dataset) == 1
    assert setup.data_collator is not None
    assert setup.compute_metrics is not None
    if task_name in CLASSIFICATION_TASKS - {"speech_qa"}:
        assert setup.label_names is not None
    if task_name == "speech_qa":
        assert setup.apply_subset_indices is not None
