from __future__ import annotations

import logging

import pytest
import torch

from core.tasks.config import BaseTaskConfig, create_simple_task_config
from core.utils.logger import setup_logger
from core.utils.seed_utils import set_global_seed


def test_task_config_factory_and_base_validation(tmp_path) -> None:
    with pytest.raises(NotImplementedError):
        BaseTaskConfig.get_config_path(tmp_path)
    with pytest.raises(NotImplementedError):
        BaseTaskConfig.get_artifact_directories(tmp_path)

    task_name, default_file, get_path, get_dirs = create_simple_task_config("demo", "demo.yaml")
    assert task_name == "demo"
    assert default_file == "demo.yaml"
    assert get_path(tmp_path) == tmp_path / "configs" / "demo.yaml"
    assert get_dirs(tmp_path)["base"] == tmp_path / "artifacts" / "demo"


def test_seed_helper_and_logger_are_idempotent(tmp_path) -> None:
    set_global_seed(None)
    set_global_seed(123)
    first = torch.rand(1).item()
    set_global_seed(123)
    assert torch.rand(1).item() == pytest.approx(first)

    logger = setup_logger("unit-test-logger-core", tmp_path / "test.log")
    logger.info("hello")
    assert logger.level == logging.INFO
    assert logger is setup_logger("unit-test-logger-core", tmp_path / "ignored.log")


def test_color_formatter_leaves_plain_output_when_not_tty() -> None:
    record = logging.LogRecord("x", logging.INFO, __file__, 1, "message", (), None)
    handler = setup_logger("unit-test-color-logger").handlers[0]
    assert handler.format(record) == "message"
