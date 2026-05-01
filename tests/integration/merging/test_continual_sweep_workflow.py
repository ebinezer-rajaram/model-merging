from __future__ import annotations

from pathlib import Path

import pytest
import torch

from merging.config.unified import normalize_merge_config
from merging.continual.supermerge_optimizer import _coefficients, _ordered_seen_tasks
from merging.evaluation import sweep as sweep_module
from merging.evaluation.continual_sweep import (
    ContinualSweepContext,
    is_bayes_continual_method,
    is_continual_supermerge_method,
)


class _Source:
    source_id = "source"


def _context(seen_tasks: list[str], eval_tasks: list[str] | None = None) -> ContinualSweepContext:
    return ContinualSweepContext(
        x_source=_Source(),  # type: ignore[arg-type]
        y_source=_Source(),  # type: ignore[arg-type]
        seen_tasks=seen_tasks,
        eval_tasks=eval_tasks or seen_tasks,
        merge_mode="common",
        energy_threshold=1.0,
        store_dtype="float32",
        alpha_default=1.0,
        lambda_default=0.5,
        batch_size=None,
        use_cache=False,
        continual_source_mode="compressed_recursive",
    )


def test_continual_method_routing_helpers_distinguish_bayes_and_supermerge() -> None:
    assert is_bayes_continual_method("continual")
    assert not is_bayes_continual_method("continual_supermerge")
    assert is_continual_supermerge_method("continual_supermerge")
    assert not is_continual_supermerge_method("continual")


def test_run_sweep_routes_continual_supermerge_to_optimizer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    config = normalize_merge_config(
        {
            "adapters": ["old", "new"],
            "method": "continual_supermerge",
            "output_dir": str(tmp_path / "sweeps"),
            "optimizer": {"type": "continual_supermerge", "params": {"steps": 1}},
        }
    )
    ctx = _context(["emotion", "asr"])
    called = {}

    monkeypatch.setattr(sweep_module, "prepare_continual_context", lambda cfg: ctx)

    from merging.continual import supermerge_optimizer as opt_module

    def fake_run(*, config, context, summary_dir):
        called["config"] = config
        called["context"] = context
        called["summary_dir"] = summary_dir
        return {"method": "continual_supermerge"}

    monkeypatch.setattr(opt_module, "run_continual_supermerge_optimizer", fake_run)

    result = sweep_module.run_sweep(config)

    assert result == {"method": "continual_supermerge"}
    assert called["config"] is config
    assert called["context"] is ctx
    assert called["summary_dir"] == tmp_path / "sweeps"


def test_continual_supermerge_scalar_constraints_and_seen_task_resolution() -> None:
    coeffs = _coefficients(torch.tensor(-20.0), torch.tensor(20.0))
    assert coeffs.shape == (2,)
    assert torch.all(coeffs >= 0)
    alpha = float(coeffs.sum().item())
    lambda_weight = float((coeffs[0] / coeffs.sum()).item())
    assert alpha >= 0.0
    assert 0.0 <= lambda_weight <= 1.0

    ctx = _context(["emotion", "asr"], eval_tasks=["emotion", "asr", "speech_qa"])
    assert _ordered_seen_tasks(ctx, {}) == ["emotion", "asr"]
    with pytest.raises(ValueError, match="seen tasks only"):
        _ordered_seen_tasks(ctx, {"tasks": ["emotion", "speech_qa"]})
