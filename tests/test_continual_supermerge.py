from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from merging.config.unified import normalize_merge_config
from merging.continual.supermerge_optimizer import _coefficients, _ordered_seen_tasks
from merging.delta_sources.resolver import resolve_delta_source
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


def test_run_sweep_routes_continual_supermerge_to_optimizer(tmp_path: Path, monkeypatch) -> None:
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


def test_continual_supermerge_scalar_constraints() -> None:
    coeffs = _coefficients(torch.tensor(-20.0), torch.tensor(20.0))
    assert coeffs.shape == (2,)
    assert torch.all(coeffs >= 0)
    alpha = float(coeffs.sum().item())
    lambda_weight = float((coeffs[0] / coeffs.sum()).item())
    assert alpha >= 0.0
    assert 0.0 <= lambda_weight <= 1.0


def test_continual_supermerge_seen_task_resolution_excludes_eval_only() -> None:
    ctx = _context(["emotion", "asr"], eval_tasks=["emotion", "asr", "speech_qa"])
    assert _ordered_seen_tasks(ctx, {}) == ["emotion", "asr"]

    try:
        _ordered_seen_tasks(ctx, {"tasks": ["emotion", "speech_qa"]})
        raise AssertionError("Expected eval-only task to be rejected")
    except ValueError as exc:
        assert "seen tasks only" in str(exc)


def _write_lora_adapter(path: Path, *, a: torch.Tensor, b: torch.Tensor) -> None:
    path.mkdir(parents=True, exist_ok=True)
    (path / "adapter_config.json").write_text(
        json.dumps({"r": int(a.shape[0]), "lora_alpha": int(a.shape[0])}),
        encoding="utf-8",
    )
    save_file(
        {
            "base_model.model.layer.lora_A.weight": a,
            "base_model.model.layer.lora_B.weight": b,
        },
        str(path / "adapter_model.safetensors"),
    )


def test_path_lora_source_infers_task_from_artifacts_layout(tmp_path: Path) -> None:
    adapter = tmp_path / "artifacts" / "kws" / "adapters" / "run"
    _write_lora_adapter(
        adapter,
        a=torch.ones(1, 2),
        b=torch.ones(3, 1),
    )

    source = resolve_delta_source(adapter)

    assert source.constituent_tasks_flat() == ["kws"]


def test_fused_lora_leaf_source_concatenates_weighted_prior_leaves(tmp_path: Path) -> None:
    speaker = tmp_path / "artifacts" / "speaker_ver" / "adapters" / "run"
    langid = tmp_path / "artifacts" / "langid" / "adapters" / "run"
    _write_lora_adapter(
        speaker,
        a=torch.tensor([[1.0, 2.0]]),
        b=torch.tensor([[3.0], [4.0]]),
    )
    _write_lora_adapter(
        langid,
        a=torch.tensor([[5.0, 6.0]]),
        b=torch.tensor([[7.0], [8.0]]),
    )

    artifact = tmp_path / "merged"
    factors = artifact / "factors"
    factors.mkdir(parents=True)
    save_file({"dummy": torch.zeros(1)}, str(factors / "shard_00000.safetensors"))
    (artifact / "continual_artifact_manifest.json").write_text(
        json.dumps(
            {
                "schema_version": "1",
                "artifact_type": "continual_compressed_delta",
                "constituent_tasks_flat": ["speaker_ver", "langid"],
                "provenance_tree": {
                    "kind": "continual_merge",
                    "label": "continual_alpha_lambda",
                    "params": {"alpha": 2.0, "lambda": 0.25},
                    "children": [
                        {
                            "kind": "lora_adapter",
                            "label": "speaker",
                            "params": {"path": str(speaker), "task": "speaker_ver"},
                            "children": [],
                        },
                        {
                            "kind": "lora_adapter",
                            "label": "langid",
                            "params": {"path": str(langid), "task": "langid"},
                            "children": [],
                        },
                    ],
                },
                "target_params": [
                    {
                        "source_key": "base_model.model.layer",
                        "original_shape": [2, 2],
                        "matrix_shape": [2, 2],
                        "rank": 1,
                        "retained_energy": 1.0,
                        "relative_error": 0.0,
                        "frobenius_norm": 1.0,
                        "scale": 1.0,
                        "shard_file": "factors/shard_00000.safetensors",
                        "tensor_key_a": "dummy",
                        "tensor_key_b": "dummy",
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    source = resolve_delta_source(artifact, continual_source_mode="fused_lora_leaves")
    a, b, scale = source.get_factor_tensors("base_model.model.layer")
    dense = b @ a * scale
    expected = (
        2.0 * 0.25 * torch.tensor([[3.0], [4.0]]) @ torch.tensor([[1.0, 2.0]])
        + 2.0 * 0.75 * torch.tensor([[7.0], [8.0]]) @ torch.tensor([[5.0, 6.0]])
    )

    assert source.constituent_tasks_flat() == ["speaker_ver", "langid"]
    assert a.shape == (2, 2)
    assert b.shape == (2, 2)
    assert torch.allclose(dense, expected)
