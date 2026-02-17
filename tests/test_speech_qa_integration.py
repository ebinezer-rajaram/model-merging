from __future__ import annotations

from core.evaluation.eval_utils import SPEECH_QA_EVAL_KEYS
from core.training.train_task import SPEECH_QA_LOADER_KEYS
from merging.evaluation.interference import TASK_METRICS


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
    }
    assert required.issubset(set(SPEECH_QA_LOADER_KEYS))
    assert required.issubset(set(SPEECH_QA_EVAL_KEYS))


def test_speech_qa_interference_metric_mapping_uses_f1() -> None:
    assert TASK_METRICS["speech_qa"] == ("f1", True)
