"""Streaming continual merge engine with dense semantics and SVD-compressed storage."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch

from merging.artifacts.continual_format import ContinualArtifactWriter
from merging.compression.svd import CompressedParam, compress_dense_delta_to_svd
from merging.continual.policy import ContinualMergePolicy
from merging.delta_sources.base import DeltaSource, ParamDeltaSpec, ProvenanceNode
from merging.delta_sources.lora_source import LoRADeltaSource
from merging.policies.lambda_policy import extract_layer_index
from merging.runtime.utils import infer_task_from_path


@dataclass(frozen=True)
class ContinualMergeResult:
    artifact_dir: Path
    manifest_path: Path
    num_merged_params: int
    num_skipped_params: int
    skipped_reasons: Dict[str, int]
    aggregate_stats: Dict[str, float]


def _ordered_union(values: Iterable[str]) -> List[str]:
    out: List[str] = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def _resolve_keys_for_merge(
    source_specs: Sequence[List[ParamDeltaSpec]],
    *,
    merge_mode: str,
) -> List[str]:
    if merge_mode not in {"common", "strict"}:
        raise ValueError(f"Unsupported merge_mode='{merge_mode}'")

    key_sets = [set(spec.source_key for spec in specs) for specs in source_specs]
    if not key_sets:
        return []

    if merge_mode == "strict":
        ref = key_sets[0]
        for idx, other in enumerate(key_sets[1:], start=1):
            if other != ref:
                missing = sorted(ref - other)[:10]
                extra = sorted(other - ref)[:10]
                raise ValueError(
                    "Strict merge requires identical key sets across sources. "
                    f"Mismatch at source index {idx}; missing={missing}, extra={extra}"
                )
        return sorted(ref)

    return sorted(set.intersection(*key_sets))


def _shape_for_key(specs: Sequence[ParamDeltaSpec], key: str) -> Optional[Tuple[int, ...]]:
    for spec in specs:
        if spec.source_key == key:
            return tuple(int(x) for x in spec.shape)
    return None


def _resolve_merge_coefficients_for_existing_key(
    *,
    merge_method: str,
    merge_params: Mapping[str, Any],
    coefficient_policy: Optional[Mapping[str, Any]],
    source_key: str,
    num_sources: int,
) -> List[float]:
    method = str(merge_method).strip().lower()

    def _validate(coeffs: Sequence[float]) -> List[float]:
        if len(coeffs) != num_sources:
            raise ValueError(
                f"Coefficient length mismatch for key '{source_key}': expected {num_sources}, got {len(coeffs)}"
            )
        out = [float(x) for x in coeffs]
        for idx, value in enumerate(out):
            if not math.isfinite(value):
                raise ValueError(f"Non-finite coefficient at index {idx} for key '{source_key}': {value}")
        return out

    if method in {"uniform_delta", "task_vector"}:
        return [1.0 / float(num_sources)] * num_sources

    if method == "uniform_scalar_delta":
        scale = float(merge_params.get("scale", 1.0))
        return [scale] * num_sources

    if method == "weighted_delta":
        if num_sources != 2:
            raise ValueError("weighted_delta coefficient recovery requires exactly 2 sources.")
        lam = float(merge_params.get("lambda"))
        if not 0.0 <= lam <= 1.0:
            raise ValueError(f"weighted_delta lambda must be in [0,1], got {lam}")
        return [lam, 1.0 - lam]

    if method == "weighted_delta_n":
        policy = dict(coefficient_policy or {})
        normalize = bool(policy.get("normalize_coefficients", False))

        default_coeffs = policy.get("default_task_coefficients")
        task_coeffs = policy.get("task_coefficients")
        layer_map = policy.get("layer_task_coefficients")

        coeffs: Optional[List[float]] = None
        layer_idx = extract_layer_index(source_key)
        if isinstance(layer_map, Mapping) and layer_idx is not None:
            if str(layer_idx) in layer_map:
                coeffs = _validate(layer_map[str(layer_idx)])
            elif layer_idx in layer_map:
                coeffs = _validate(layer_map[layer_idx])
        if coeffs is None and isinstance(default_coeffs, Sequence):
            coeffs = _validate(default_coeffs)
        if coeffs is None and isinstance(task_coeffs, Sequence):
            coeffs = _validate(task_coeffs)
        if coeffs is None:
            coeffs = [1.0 / float(num_sources)] * num_sources

        if normalize:
            denom = float(sum(coeffs))
            if denom <= 0.0:
                raise ValueError(
                    f"Coefficient sum must be > 0 for normalized weighted_delta_n key '{source_key}'."
                )
            coeffs = [x / denom for x in coeffs]
        return coeffs

    raise ValueError(
        f"Materialization from merge metadata does not support merge_method='{merge_method}'. "
        "Supported methods: uniform_delta, uniform_scalar_delta, weighted_delta, weighted_delta_n, task_vector."
    )


def _aggregate_compression_stats(params: Sequence[CompressedParam]) -> Dict[str, float]:
    if not params:
        return {
            "avg_rank": 0.0,
            "avg_retained_energy": 0.0,
            "avg_relative_error": 0.0,
            "max_relative_error": 0.0,
        }
    ranks = [float(p.stats.rank) for p in params]
    retained = [float(p.stats.retained_energy) for p in params]
    rel_err = [float(p.stats.relative_error) for p in params]
    return {
        "avg_rank": float(sum(ranks) / len(ranks)),
        "avg_retained_energy": float(sum(retained) / len(retained)),
        "avg_relative_error": float(sum(rel_err) / len(rel_err)),
        "max_relative_error": float(max(rel_err)),
    }


def _validate_energy_threshold(energy_threshold: float) -> None:
    if not math.isfinite(float(energy_threshold)):
        raise ValueError(f"energy_threshold must be finite, got {energy_threshold}")
    if not 0.0 < float(energy_threshold) <= 1.0:
        raise ValueError(f"energy_threshold must be in (0,1], got {energy_threshold}")


def _merge_param_with_coefficients(
    deltas: Sequence[torch.Tensor],
    coeffs: Sequence[float],
    *,
    out_dtype: torch.dtype,
) -> torch.Tensor:
    if len(deltas) != len(coeffs):
        raise ValueError("deltas and coeffs length mismatch")
    merged = torch.zeros_like(deltas[0], dtype=out_dtype)
    for delta, coeff in zip(deltas, coeffs):
        merged = merged + (float(coeff) * delta.to(dtype=out_dtype))
    return merged


def _source_metadata(source: DeltaSource) -> Dict[str, Any]:
    return source.metadata().to_dict()


def _build_constituent_tasks(sources: Sequence[DeltaSource], fallback_from_paths: bool = True) -> List[str]:
    tasks: List[str] = []
    for source in sources:
        tasks.extend(source.constituent_tasks_flat())
        if fallback_from_paths:
            meta = source.metadata()
            if meta.task:
                tasks.append(meta.task)
            elif meta.path:
                inferred = infer_task_from_path(meta.path)
                if inferred:
                    tasks.append(inferred)
    return _ordered_union([t for t in tasks if t])


def _materialize_params_to_artifact(
    *,
    output_dir: Path,
    sources: Sequence[DeltaSource],
    key_resolver_specs: Sequence[List[ParamDeltaSpec]],
    merge_mode: str,
    energy_threshold: float,
    merge_semantics: Dict[str, Any],
    provenance_tree: ProvenanceNode,
    stored_representation_extra: Optional[Dict[str, Any]] = None,
    coeff_provider: Any,
    compute_dtype: torch.dtype,
    store_dtype: str,
) -> ContinualMergeResult:
    keys = _resolve_keys_for_merge(key_resolver_specs, merge_mode=merge_mode)
    if not keys:
        raise ValueError("No mergeable parameter keys were resolved.")

    stored_representation = {
        "type": "factorized_svd",
        "energy_threshold": float(energy_threshold),
        "compute_dtype": str(compute_dtype).replace("torch.", ""),
        "store_dtype": str(store_dtype),
    }
    if stored_representation_extra:
        stored_representation.update(dict(stored_representation_extra))

    writer = ContinualArtifactWriter(
        output_dir=output_dir,
        dense_merge_semantics=merge_semantics,
        stored_representation=stored_representation,
        provenance_tree=provenance_tree.to_dict(),
        constituent_tasks_flat=_build_constituent_tasks(list(sources)),
        source_metadata=[_source_metadata(source) for source in sources],
    )

    compressed_entries: List[CompressedParam] = []
    skipped = 0
    skipped_reasons: Dict[str, int] = {}

    # deterministic ordering by source key
    for key in keys:
        shapes = []
        for specs in key_resolver_specs:
            shape = _shape_for_key(specs, key)
            if shape is None:
                shape = ()
            shapes.append(shape)
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"Shape mismatch in strict mode for key '{key}': {shapes}")
            skipped += 1
            skipped_reasons["shape_mismatch"] = skipped_reasons.get("shape_mismatch", 0) + 1
            continue

        deltas = [source.materialize_dense_param_delta(key, dtype=compute_dtype) for source in sources]
        coeffs = coeff_provider(key)
        merged = _merge_param_with_coefficients(deltas, coeffs, out_dtype=compute_dtype)
        if not torch.isfinite(merged).all():
            raise ValueError(f"Merged dense delta for key '{key}' contains non-finite values.")

        compressed = compress_dense_delta_to_svd(
            merged,
            energy_threshold=energy_threshold,
            min_rank=0,
            max_rank=None,
            compute_dtype=compute_dtype,
            store_dtype=store_dtype,
        )
        compressed_entries.append(compressed)
        writer.add_param(
            source_key=key,
            a_factor=compressed.a_factor,
            b_factor=compressed.b_factor,
            scale=compressed.scale,
            original_shape=compressed.stats.original_shape,
            matrix_shape=compressed.stats.matrix_shape,
            rank=compressed.stats.rank,
            retained_energy=compressed.stats.retained_energy,
            relative_error=compressed.stats.relative_error,
            frobenius_norm=compressed.stats.frobenius_norm,
        )

    stats = _aggregate_compression_stats(compressed_entries)
    diagnostics = {
        "num_merged_params": int(len(compressed_entries)),
        "num_skipped_params": int(skipped),
        "skipped_reasons": dict(skipped_reasons),
        "compression": stats,
    }
    manifest_path = writer.finalize(diagnostics=diagnostics)

    return ContinualMergeResult(
        artifact_dir=output_dir,
        manifest_path=manifest_path,
        num_merged_params=len(compressed_entries),
        num_skipped_params=skipped,
        skipped_reasons=skipped_reasons,
        aggregate_stats=stats,
    )


def materialize_existing_merge_to_artifact(
    *,
    merged_run_path: Path,
    output_dir: Path,
    energy_threshold: float,
    merge_mode: Optional[str] = None,
    compute_dtype: torch.dtype = torch.float32,
    store_dtype: str = "float16",
) -> ContinualMergeResult:
    """Materialize an existing merged run into a reusable compressed artifact."""
    _validate_energy_threshold(energy_threshold)

    run_path = Path(merged_run_path).resolve()
    meta_path = run_path / "merge_metadata.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"merge_metadata.json not found under {run_path}")

    with meta_path.open("r") as handle:
        metadata = json.load(handle)

    merge_method = str(metadata.get("merge_method", ""))
    if not merge_method:
        raise ValueError("merge_metadata.json is missing merge_method")

    source_entries = metadata.get("source_adapters")
    if not isinstance(source_entries, list) or not source_entries:
        raise ValueError("merge_metadata.json must contain non-empty source_adapters list")

    sources: List[DeltaSource] = []
    for idx, entry in enumerate(source_entries):
        if not isinstance(entry, Mapping):
            raise ValueError(f"source_adapters[{idx}] must be a mapping")
        path_value = entry.get("path")
        if not isinstance(path_value, str) or not path_value:
            raise ValueError(f"source_adapters[{idx}] missing valid 'path'")
        task_name = entry.get("task") if isinstance(entry.get("task"), str) else None
        adapter_path = Path(path_value).expanduser()
        if not adapter_path.is_absolute():
            adapter_path = (Path.cwd() / adapter_path).resolve()
        sources.append(LoRADeltaSource(adapter_path, task=task_name))

    effective_merge_mode = str(merge_mode or metadata.get("merge_mode") or "common")
    params = metadata.get("params")
    params_map: Dict[str, Any] = dict(params) if isinstance(params, Mapping) else {}
    coeff_policy = metadata.get("coefficient_policy")
    coeff_policy_map = dict(coeff_policy) if isinstance(coeff_policy, Mapping) else None

    def _coeff_provider(key: str) -> List[float]:
        return _resolve_merge_coefficients_for_existing_key(
            merge_method=merge_method,
            merge_params=params_map,
            coefficient_policy=coeff_policy_map,
            source_key=key,
            num_sources=len(sources),
        )

    provenance = ProvenanceNode(
        kind="materialized_existing_merge",
        label=f"{merge_method}_materialized",
        params={
            "merge_method": merge_method,
            "merge_mode": effective_merge_mode,
            "source_run": str(run_path),
            "materialized_at": datetime.now(timezone.utc).isoformat(),
        },
        children=[source.provenance() for source in sources],
    )

    semantics = {
        "semantic_type": "existing_merge_materialization",
        "merge_method": merge_method,
        "merge_mode": effective_merge_mode,
        "params": params_map,
        "coefficient_policy": coeff_policy_map,
        "source_run": str(run_path),
    }

    key_specs = [source.list_target_params() for source in sources]
    return _materialize_params_to_artifact(
        output_dir=Path(output_dir).resolve(),
        sources=sources,
        key_resolver_specs=key_specs,
        merge_mode=effective_merge_mode,
        energy_threshold=energy_threshold,
        merge_semantics=semantics,
        provenance_tree=provenance,
        stored_representation_extra={
            "source_artifact_type": "merged_run_metadata",
        },
        coeff_provider=_coeff_provider,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
    )


def continual_merge_sources_to_artifact(
    *,
    x_source: DeltaSource,
    y_source: DeltaSource,
    policy: ContinualMergePolicy,
    output_dir: Path,
    energy_threshold: float,
    merge_mode: str = "common",
    compute_dtype: torch.dtype = torch.float32,
    store_dtype: str = "float16",
) -> ContinualMergeResult:
    """Merge two delta sources with global alpha/lambda and save compressed artifact."""
    _validate_energy_threshold(energy_threshold)
    policy.validate()
    x_coeff, y_coeff = policy.source_coefficients()

    def _coeff_provider(_: str) -> List[float]:
        return [x_coeff, y_coeff]

    provenance = ProvenanceNode(
        kind="continual_merge",
        label="continual_alpha_lambda",
        params={
            "alpha": float(policy.alpha),
            "lambda": float(policy.lambda_weight),
            "formula": "alpha * (lambda * x + (1-lambda) * y)",
            "merge_mode": merge_mode,
        },
        children=[x_source.provenance(), y_source.provenance()],
    )

    semantics = {
        "semantic_type": "continual_two_source",
        "formula": "alpha * (lambda * x + (1-lambda) * y)",
        "alpha": float(policy.alpha),
        "lambda": float(policy.lambda_weight),
        "merge_mode": merge_mode,
        "source_order": ["x", "y"],
        "source_ids": [x_source.source_id, y_source.source_id],
    }

    return _materialize_params_to_artifact(
        output_dir=Path(output_dir).resolve(),
        sources=[x_source, y_source],
        key_resolver_specs=[x_source.list_target_params(), y_source.list_target_params()],
        merge_mode=merge_mode,
        energy_threshold=energy_threshold,
        merge_semantics=semantics,
        provenance_tree=provenance,
        stored_representation_extra={
            "source_artifact_type": "continual_merge",
        },
        coeff_provider=_coeff_provider,
        compute_dtype=compute_dtype,
        store_dtype=store_dtype,
    )


__all__ = [
    "ContinualMergeResult",
    "continual_merge_sources_to_artifact",
    "materialize_existing_merge_to_artifact",
]
