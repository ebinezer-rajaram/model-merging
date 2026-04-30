from __future__ import annotations

from core.evaluation.split_utils import (
    apply_task_split_overrides,
    asr_resolved_librispeech_split,
    canonical_output_split,
    task_data_split,
)


def test_test_other_uses_separate_output_namespace() -> None:
    assert canonical_output_split("test-other") == "test_other"
    assert canonical_output_split("test_other") == "test_other"
    assert canonical_output_split("test.other") == "test_other"


def test_test_other_uses_generic_test_dataset_slot() -> None:
    assert task_data_split("asr", "test-other") == "test"
    assert task_data_split("emotion", "test-other") == "test"


def test_asr_test_other_overrides_librispeech_split() -> None:
    config = {"dataset": {"test_split": "test"}}
    updated, metadata = apply_task_split_overrides(
        task="asr",
        config=config,
        requested_split="test-other",
    )

    assert updated["dataset"]["test_split"] == "test-other"
    assert config["dataset"]["test_split"] == "test"
    assert metadata["dataset"] == "librispeech_asr"
    assert metadata["dataset_config"] == "clean"
    assert metadata["requested_split"] == "test-other"
    assert metadata["resolved_split"] == "test.other"
    assert metadata["output_split"] == "test_other"


def test_asr_clean_aliases_resolve_to_clean_test() -> None:
    assert asr_resolved_librispeech_split("test") == "test"
    assert asr_resolved_librispeech_split("test-clean") == "test"
