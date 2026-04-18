from __future__ import annotations

import json
from pathlib import Path

import pytest

from core.config.multitask_schema import parse_multitask_config
from core.training.train_multitask import (
    _override_lora_config_from_base_adapter,
    _resolve_base_adapter_path,
)


def _minimal_mtl_payload() -> dict:
    return {
        "seed": 0,
        "model": {
            "path": "data/models/Qwen2.5-Omni-3B",
            "lora": {
                "r": 8,
                "alpha": 16,
                "dropout": 0.1,
                "bias": "none",
                "target_modules": ["q_proj"],
                "task_type": "CAUSAL_LM",
            },
        },
        "training": {"selection_criterion": "geometric_mean_interference_delta"},
        "artifacts": {"adapter_subdir": "tmp"},
        "tasks": [{"name": "emotion"}],
    }


def test_multitask_continual_defaults_parse() -> None:
    cfg = parse_multitask_config(_minimal_mtl_payload())
    assert cfg.continual is not None
    assert cfg.continual.enabled is False
    assert cfg.continual.selection_mode == "mtl_interference"
    assert cfg.continual.final_eval_include_speech_qa is True
    assert cfg.artifacts.task_set_slug_mode == "sorted_names"


def test_multitask_continual_validates_added_tasks() -> None:
    payload = _minimal_mtl_payload()
    payload["continual"] = {
        "enabled": True,
        "base_adapter": "artifacts/mtl/some_adapter/best",
        "added_tasks": ["not_a_task"],
    }
    with pytest.raises(ValueError):
        parse_multitask_config(payload)


def test_multitask_artifacts_accepts_base_then_added_slug_mode() -> None:
    payload = _minimal_mtl_payload()
    payload["artifacts"]["task_set_slug_mode"] = "base_then_added"
    cfg = parse_multitask_config(payload)
    assert cfg.artifacts.task_set_slug_mode == "base_then_added"


def test_resolve_base_adapter_path_supports_run_alias(tmp_path: Path) -> None:
    root = tmp_path / "adapter_root"
    run = root / "runs" / "run_20260101_010101"
    run.mkdir(parents=True)
    (run / "adapter_config.json").write_text("{}")

    resolved = _resolve_base_adapter_path(str(root), "run_20260101_010101")
    assert resolved == run.resolve()


def test_override_lora_config_from_base_adapter(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True)
    (adapter_dir / "adapter_config.json").write_text(
        json.dumps(
            {
                "r": 64,
                "lora_alpha": 128,
                "lora_dropout": 0.05,
                "bias": "none",
                "target_modules": ["q_proj", "k_proj"],
                "task_type": "CAUSAL_LM",
            }
        )
    )

    raw_cfg = _minimal_mtl_payload()
    _override_lora_config_from_base_adapter(raw_cfg, adapter_dir)
    lora_cfg = raw_cfg["model"]["lora"]
    assert lora_cfg["r"] == 64
    assert lora_cfg["alpha"] == 128
    assert lora_cfg["dropout"] == 0.05
    assert lora_cfg["target_modules"] == ["q_proj", "k_proj"]
