from __future__ import annotations

import pytest

from core.training.training_config import build_early_stopping_kwargs, parse_training_config


def test_parse_training_config_merges_defaults_and_computes_warmup(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("WORLD_SIZE", "2")
    cfg = parse_training_config(
        {"learning_rate": "0.001", "num_train_epochs": 3, "warmup_ratio": 0.2, "early_stopping_threshold": 0.01},
        num_train_examples=80,
        task_defaults={"per_device_train_batch_size": 4, "gradient_accumulation_steps": 2, "report_to": []},
    )
    assert cfg.learning_rate == 0.001
    assert cfg.warmup_steps == 3
    assert cfg.report_to == []
    assert build_early_stopping_kwargs(cfg) == {"early_stopping_patience": 3, "early_stopping_threshold": 0.01}


def test_parse_training_config_supports_max_steps_and_bad_learning_rate() -> None:
    cfg = parse_training_config({"max_steps": 10, "warmup_ratio": 0.2}, num_train_examples=100)
    assert cfg.warmup_steps == 2
    with pytest.raises(ValueError, match="learning_rate"):
        parse_training_config({"learning_rate": "bad"}, num_train_examples=1)
