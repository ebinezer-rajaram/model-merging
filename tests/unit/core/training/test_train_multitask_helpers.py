from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.training.train_multitask import (
    _assert_mode_compatible_task_set_root,
    _build_task_set_slug,
    _compute_config_hash_short,
    _compute_continual_headline_metrics,
    _dedupe_preserve_order,
    _discover_base_tasks_from_adapter,
    _extract_base_adapter_lora_config,
    _override_lora_config_from_base_adapter,
    _resolve_base_adapter_path,
    _resolve_mtl_paths,
    _write_mode_marker,
)


def test_task_set_slug_modes_and_path_resolution(tmp_path: Path) -> None:
    first = _build_task_set_slug(["asr", "emotion"], mode="sorted_names")
    second = _build_task_set_slug(["emotion", "asr"], mode="sorted_names")
    assert first == second == "asr_emotion"
    assert _build_task_set_slug(["asr"], mode="base_then_added", base_task_names=[], added_task_names=["asr"]) == "base_none__added_asr"
    with pytest.raises(ValueError, match="Unsupported task_set_slug_mode"):
        _build_task_set_slug(["asr"], mode="bad")

    paths = _resolve_mtl_paths(adapter_subdir="adapter", task_names=["asr"], artifacts_root=tmp_path)
    assert paths["output_dir"].exists()
    with pytest.raises(ValueError, match="Unsupported MTL artifacts.layout"):
        _resolve_mtl_paths(adapter_subdir="adapter", task_names=["asr"], layout="flat", artifacts_root=tmp_path)


def test_continual_adapter_resolution_and_lora_override(tmp_path: Path) -> None:
    root = tmp_path / "adapter_root"
    run = root / "runs" / "run_20260101_010101"
    run.mkdir(parents=True)
    (run / "adapter_config.json").write_text(
        json.dumps({"r": 64, "lora_alpha": 128, "lora_dropout": 0.05, "bias": "none", "target_modules": ["q_proj"], "task_type": "CAUSAL_LM"}),
        encoding="utf-8",
    )
    assert _resolve_base_adapter_path(str(root), "run_20260101_010101") == run.resolve()
    with pytest.raises(FileNotFoundError, match="base_adapter_run_id"):
        _resolve_base_adapter_path(str(root), "missing")

    raw_cfg = {"model": {"lora": {"r": 8, "alpha": 16, "dropout": 0.1, "bias": "none", "target_modules": ["k_proj"]}}}
    base_lora = _override_lora_config_from_base_adapter(raw_cfg, run)
    assert base_lora["r"] == 64
    assert raw_cfg["model"]["lora"]["alpha"] == 128

    bad = tmp_path / "bad"
    bad.mkdir()
    (bad / "adapter_config.json").write_text(json.dumps({"target_modules": []}), encoding="utf-8")
    with pytest.raises(ValueError, match="missing r/lora_alpha"):
        _extract_base_adapter_lora_config(bad)


def test_mtl_mode_markers_hashes_and_headline_metrics(tmp_path: Path) -> None:
    assert _dedupe_preserve_order(["ASR", "asr", "", "Emotion"]) == ["asr", "emotion"]
    assert len(_compute_config_hash_short({"b": 2, "a": 1})) == 12

    _write_mode_marker(task_set_root=tmp_path, mode="joint", config_hash="abc", allow_mixed_output=False)
    with pytest.raises(ValueError, match="Output collision"):
        _assert_mode_compatible_task_set_root(task_set_root=tmp_path, mode="continual", allow_mixed_output=False)
    _assert_mode_compatible_task_set_root(task_set_root=tmp_path, mode="continual", allow_mixed_output=True)

    report = _compute_continual_headline_metrics(
        validation_metrics={},
        test_metrics={"eval_asr_interference_delta": 0.5, "eval_asr_wer": 0.2, "eval_speech_qa_accuracy": 0.8},
        constituent_tasks=["asr"],
        added_tasks=["asr"],
    )
    assert report["constituent_interference_delta_mean"] == 0.5
    assert report["added_tasks_primary_metrics"]["asr"]["oriented_value"] == -0.2
    assert report["speech_qa_transfer_accuracy"] == 0.8


def test_discover_base_tasks_from_adapter_metadata(tmp_path: Path) -> None:
    adapter = tmp_path / "adapter"
    adapter.mkdir()
    (adapter / "mtl_config_resolved.json").write_text(json.dumps({"tasks": [{"name": "ASR"}, {"name": "emotion"}]}), encoding="utf-8")
    assert _discover_base_tasks_from_adapter(adapter) == ["asr", "emotion"]
    with pytest.raises(ValueError, match="Could not discover"):
        _discover_base_tasks_from_adapter(tmp_path / "missing")
