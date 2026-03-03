from __future__ import annotations

import importlib.util
from pathlib import Path


def _load_script_module():
    script_path = Path(__file__).resolve().parents[1] / "analysis" / "scripts" / "build_mmsu_subtask_interference.py"
    spec = importlib.util.spec_from_file_location("build_mmsu_subtask_interference", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError("Failed to load build_mmsu_subtask_interference module")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_build_delta_table_computes_variant_deltas() -> None:
    module = _load_script_module()
    table = module.build_delta_table(
        base_subtask_accuracy={"accent": 0.6, "age": 0.5},
        variant_subtask_accuracy={
            "adapter_a": {"accent": 0.7, "age": 0.4},
            "adapter_b": {"accent": 0.55, "age": 0.65},
        },
    )

    assert table["accent"]["base_accuracy"] == 0.6
    assert table["accent"]["adapter_a"] == 0.1
    assert table["accent"]["adapter_b"] == -0.05
    assert table["age"]["adapter_a"] == -0.1
    assert table["age"]["adapter_b"] == 0.15


def test_rank_sensitive_subtasks_orders_by_impact_then_variance() -> None:
    module = _load_script_module()
    ranked = module.rank_sensitive_subtasks(
        {
            "accent": {"base_accuracy": 0.6, "a": 0.2, "b": -0.1},
            "age": {"base_accuracy": 0.5, "a": 0.05, "b": 0.03},
            "scene": {"base_accuracy": 0.4, "a": -0.25, "b": 0.25},
        }
    )

    assert ranked[0]["task_name"] == "scene"
    assert ranked[0]["max_abs_delta"] == 0.25
    assert ranked[1]["task_name"] == "accent"
    assert ranked[2]["task_name"] == "age"
