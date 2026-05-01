from __future__ import annotations

from pathlib import Path

import pytest

from core.training.training_loop import (
    _is_checkpoint_complete,
    build_history_record,
    find_latest_checkpoint,
    resolve_checkpoint_path,
)
from tests.helpers.core import complete_checkpoint


def test_checkpoint_helpers_find_latest_complete_checkpoint(tmp_path: Path) -> None:
    incomplete = tmp_path / "checkpoint-3"
    incomplete.mkdir()
    complete = complete_checkpoint(tmp_path, step=2)
    latest = complete_checkpoint(tmp_path, step=5)

    assert _is_checkpoint_complete(complete) is True
    assert _is_checkpoint_complete(incomplete) is False
    assert find_latest_checkpoint(tmp_path) == str(latest)
    assert find_latest_checkpoint(tmp_path / "missing") is None


def test_resolve_checkpoint_path_supports_auto_explicit_and_errors(tmp_path: Path) -> None:
    complete = complete_checkpoint(tmp_path, step=1)
    assert resolve_checkpoint_path(None, tmp_path) is None
    assert resolve_checkpoint_path("auto", tmp_path) == str(complete)
    assert resolve_checkpoint_path(str(complete), tmp_path) == str(complete)
    with pytest.raises(ValueError, match="does not exist"):
        resolve_checkpoint_path(str(tmp_path / "missing"), tmp_path)
    file_path = tmp_path / "file"
    file_path.write_text("x", encoding="utf-8")
    with pytest.raises(ValueError, match="not a directory"):
        resolve_checkpoint_path(str(file_path), tmp_path)


def test_build_history_record_filters_non_metrics_and_runtime_values() -> None:
    row = build_history_record(
        {
            "epoch": "1.5",
            "eval_accuracy": "0.8",
            "eval_runtime": 3.0,
            "loss": 0.2,
            "note": "skip",
        },
        step=4,
    )
    assert row == {"step": 4, "epoch": 1.5, "eval_accuracy": 0.8, "loss": 0.2}
    assert build_history_record({"epoch": 1.0, "note": "skip"}, step=1) is None
