"""Helpers for evaluation split aliases and output namespaces."""

from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Tuple

ASR_TASK_NAME = "asr"

TEST_CLEAN_ALIASES = {"test", "test.clean", "test-clean", "test_clean"}
TEST_OTHER_ALIASES = {"test.other", "test-other", "test_other"}
SUPPORTED_EVAL_SPLITS = ("train", "validation", "test", "test-clean", "test_clean", "test-other", "test_other")


def normalize_split_name(split: str) -> str:
    """Return a normalized split token for comparisons."""
    return str(split or "").strip().lower()


def canonical_output_split(split: str) -> str:
    """Return the split namespace used for artifact paths and cache files."""
    normalized = normalize_split_name(split)
    if normalized in TEST_OTHER_ALIASES:
        return "test_other"
    if normalized in TEST_CLEAN_ALIASES:
        return "test"
    return normalized.replace(".", "_").replace("-", "_")


def task_data_split(task: str, split: str) -> str:
    """Return the generic split name used to select train/val/test datasets."""
    normalized = normalize_split_name(split)
    if normalized in TEST_CLEAN_ALIASES or normalized in TEST_OTHER_ALIASES:
        return "test"
    return normalized


def asr_resolved_librispeech_split(split: str) -> str:
    """Return the underlying LibriSpeech ASR split name."""
    normalized = normalize_split_name(split)
    if normalized in TEST_OTHER_ALIASES:
        return "test.other"
    if normalized in TEST_CLEAN_ALIASES:
        return "test"
    return normalized


def apply_task_split_overrides(
    *,
    task: str,
    config: Dict[str, Any],
    requested_split: str,
) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Apply task-specific dataset split overrides without mutating the caller config.

    For ASR, the generic evaluation split remains ``test`` while
    ``dataset.test_split`` chooses LibriSpeech ``test`` vs ``test.other``.
    """
    cfg = deepcopy(config)
    metadata: Dict[str, Any] = {
        "requested_split": requested_split,
        "output_split": canonical_output_split(requested_split),
        "data_split": task_data_split(task, requested_split),
    }
    if normalize_split_name(task) == ASR_TASK_NAME:
        dataset_cfg = cfg.setdefault("dataset", {})
        if normalize_split_name(requested_split) in TEST_OTHER_ALIASES:
            dataset_cfg["test_split"] = "test-other"
        elif normalize_split_name(requested_split) in TEST_CLEAN_ALIASES:
            dataset_cfg["test_split"] = dataset_cfg.get("test_split") or "test"
        metadata.update(
            {
                "dataset": "librispeech_asr",
                "dataset_config": "clean",
                "resolved_split": asr_resolved_librispeech_split(dataset_cfg.get("test_split", requested_split)),
            }
        )
    return cfg, metadata
