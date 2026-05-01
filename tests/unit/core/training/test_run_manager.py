from __future__ import annotations

import json
from pathlib import Path

from core.training.run_manager import RunManager


def test_run_manager_registers_ranks_latest_and_best(tmp_path: Path) -> None:
    manager = RunManager(tmp_path / "adapter", metric_for_ranking="accuracy", greater_is_better=True)
    first = manager.create_run_directory()
    manager.register_run(first, {"accuracy": 0.7}, {"training": {"learning_rate": 1e-4}, "model": {"lora": {"r": 4}}})
    second = manager.create_run_directory()
    manager.register_run(second, {"eval_accuracy": 0.9}, {"training": {}, "model": {"lora": {}}})

    registry = json.loads((tmp_path / "adapter" / "runs_registry.json").read_text(encoding="utf-8"))
    assert registry["latest_run_id"] == second.name
    assert manager.get_latest_run_path() == second.resolve()
    assert manager.get_best_run_path() == second.resolve()
    assert manager.get_run_path(second.name) == second
    assert manager.get_run_path("missing") is None


def test_run_manager_serializes_path_values_in_config_hash(tmp_path: Path) -> None:
    manager = RunManager(tmp_path / "adapter", metric_for_ranking="loss", greater_is_better=False)
    digest = manager._compute_config_hash({"path": tmp_path / "x", "nested": {"items": [tmp_path / "y"]}})
    assert len(digest) == 12
