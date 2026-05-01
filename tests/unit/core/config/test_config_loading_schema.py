from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.loader import _deep_merge, _load_yaml_config, get_artifact_directories, get_config_path, load_task_config
from core.config.multitask_schema import parse_multitask_config
from core.config.registry import TASK_REGISTRY, get_task_info, list_tasks
from core.config.schemas import TaskConfig
from core.data.io_utils import dump_json, ensure_dir, load_config


def _minimal_task_payload(task: str = "asr") -> dict:
    return {
        "task": task,
        "artifacts": {"adapter_subdir": "adapter"},
        "model": {"lora": {"r": 8, "alpha": 16, "dropout": 0.1, "bias": "none", "target_modules": ["q_proj"]}},
    }


def _minimal_mtl_payload() -> dict:
    return {
        "seed": 0,
        "model": {
            "path": "data/models/Qwen2.5-Omni-3B",
            "lora": {"r": 8, "alpha": 16, "dropout": 0.1, "bias": "none", "target_modules": ["q_proj"]},
        },
        "training": {"selection_criterion": "geometric_mean_interference_delta"},
        "artifacts": {"adapter_subdir": "tmp"},
        "tasks": [{"name": "emotion"}],
    }


def test_config_registry_lists_expected_tasks_and_artifact_dirs(tmp_path: Path) -> None:
    assert set(list_tasks()) == set(TASK_REGISTRY)
    info = get_task_info("asr")
    assert info.default_config_file == "asr.yaml"
    assert info.get_artifact_dirs(tmp_path)["adapters"] == tmp_path / "adapters"
    assert get_config_path("asr").name == "asr.yaml"
    assert get_artifact_directories("asr", tmp_path)["metrics"] == tmp_path / "metrics"
    with pytest.raises(ValueError, match="Unknown task"):
        get_task_info("missing")


def test_yaml_config_loading_merges_base_and_override(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "task.yaml"
    base.write_text("a: 1\nnested:\n  x: 2\n  y: 3\n", encoding="utf-8")
    override.write_text("nested:\n  y: 9\nb: 4\n", encoding="utf-8")

    assert _deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}
    assert _load_yaml_config(override, base)["nested"] == {"x": 2, "y": 9}
    assert load_config(override, base)["b"] == 4

    out_dir = ensure_dir(tmp_path / "created")
    payload_path = out_dir / "payload.json"
    dump_json({"z": 1}, payload_path)
    assert json.loads(payload_path.read_text(encoding="utf-8")) == {"z": 1}


def test_task_schema_accepts_known_tasks_and_rejects_unknowns() -> None:
    cfg = TaskConfig(**_minimal_task_payload("emotion"))
    assert cfg.task == "emotion"
    assert cfg.model.lora.r == 8
    with pytest.raises(ValueError, match="Task must be one of"):
        TaskConfig(**_minimal_task_payload("unknown"))
    with pytest.raises(ValueError):
        TaskConfig(**{**_minimal_task_payload("asr"), "unexpected": True})


def test_load_task_config_validates_and_reports_missing_or_invalid(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    good = tmp_path / "good.yaml"
    good.write_text("task: asr\nartifacts:\n  adapter_subdir: adapter\n", encoding="utf-8")
    monkeypatch.setattr("core.config.loader.get_config_path", lambda task_name, config_filename=None: good)
    assert isinstance(load_task_config("asr"), TaskConfig)
    assert load_task_config("asr", validate=False)["task"] == "asr"

    missing = tmp_path / "missing.yaml"
    monkeypatch.setattr("core.config.loader.get_config_path", lambda task_name, config_filename=None: missing)
    with pytest.raises(FileNotFoundError):
        load_task_config("asr")

    bad = tmp_path / "bad.yaml"
    bad.write_text("task: not_a_task\nartifacts:\n  adapter_subdir: adapter\n", encoding="utf-8")
    monkeypatch.setattr("core.config.loader.get_config_path", lambda task_name, config_filename=None: bad)
    with pytest.raises(ValueError, match="Invalid configuration"):
        load_task_config("asr")


def test_multitask_schema_validates_tasks_continual_and_duplicates() -> None:
    cfg = parse_multitask_config(_minimal_mtl_payload())
    assert cfg.continual is not None
    assert cfg.continual.enabled is False
    assert cfg.artifacts.task_set_slug_mode == "sorted_names"

    payload = _minimal_mtl_payload()
    payload["tasks"] = [{"name": "emotion"}, {"name": "emotion"}]
    with pytest.raises(ValueError, match="Duplicate task"):
        parse_multitask_config(payload)

    payload = _minimal_mtl_payload()
    payload["continual"] = {"enabled": True, "base_adapter": "adapter", "added_tasks": ["not_a_task"]}
    with pytest.raises(ValueError, match="Unknown task"):
        parse_multitask_config(payload)

    payload = _minimal_mtl_payload()
    payload["training"]["final_eval_extra_tasks"] = ["ASR"]
    assert parse_multitask_config(payload).training.final_eval_extra_tasks == ["asr"]
