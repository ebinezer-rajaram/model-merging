from __future__ import annotations

from pathlib import Path

import pytest

from merging.config.specs import MergeSpec, OptimizerSpec
from merging.optimizers.registry import OptimizerContext, get_optimizer, optimize_lambda_policy
from tests.helpers.merging import write_sweep_json


def _context() -> OptimizerContext:
    return OptimizerContext(
        method="weighted",
        adapter_specs=["a", "b"],
        adapter_paths=[Path("a"), Path("b")],
        source_metadata=[{"task": "task_a"}, {"task": "task_b"}],
        merge_mode="common",
        output_dir=None,
        method_params={},
        lambda_policy=None,
    )


def test_bayes_optimizer_reuses_compatible_sweep_files(tmp_path: Path) -> None:
    write_sweep_json(
        tmp_path / "sweep_001.json",
        method="weighted",
        adapters=["task_b", "task_a"],
        runs=[
            {"params": {"lambda": 0.2}, "score": 1.0},
            {"params": {"lambda": 0.8}, "score": 2.0},
        ],
    )
    spec = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"sweep_paths": str(tmp_path)}),
    )
    result = optimize_lambda_policy(spec, _context())
    assert result.method_params_overrides == {"lambda": 0.8}
    assert result.provenance["status"] == "applied_from_sweep"


def test_bayes_optimizer_reports_no_compatible_sweep_and_bad_inputs(tmp_path: Path) -> None:
    write_sweep_json(
        tmp_path / "sweep_001.json",
        method="uniform",
        adapters=["task_a", "task_b"],
        runs=[{"params": {"lambda": 0.2}, "score": 1.0}],
    )
    spec = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"sweep_paths": str(tmp_path)}),
    )
    result = optimize_lambda_policy(spec, _context())
    assert result.method_params_overrides == {}
    assert result.provenance["status"] == "no_compatible_sweep_found"

    bad_spec = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"sweep_paths": [123]}),
    )
    with pytest.raises(ValueError, match="sweep_paths"):
        optimize_lambda_policy(bad_spec, _context())

    explicit_bad = MergeSpec(
        adapters=["a", "b"],
        method="weighted",
        optimizer=OptimizerSpec(type="bayes", params={"best_params": 1}),
    )
    with pytest.raises(ValueError, match="best_params"):
        optimize_lambda_policy(explicit_bad, _context())

    with pytest.raises(ValueError, match="Unknown optimizer"):
        get_optimizer("missing")
