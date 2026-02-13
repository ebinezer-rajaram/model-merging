"""Register built-in merge methods."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

from experiments.extract_vector import extract_task_vector_from_lora
from merging.policies.lambda_policy import build_lambda_policy
from merging.methods.task_vector import merge_uniform_via_task_vectors
from merging.engine.registry import MergeMethod, MergeOutput, build_merge_metadata, register_merge_method
from merging.config.specs import merge_spec_from_legacy_args
from merging.plugins.transforms import apply_transforms
from merging.methods.dare import merge_dare, _validate_dare_params
from merging.methods.ties import merge_ties, _validate_ties_params
from merging.methods.uniform import merge_adapters_uniform
from merging.runtime.utils import compute_delta_from_lora_weights, load_adapter_weights
from merging.methods.weighted import merge_adapters_weighted, merge_task_vectors_weighted
from merging.methods.weighted_delta_n import merge_task_vectors_weighted_n
from merging.methods.uniform_scalar_delta import merge_task_vectors_uniform_scalar


def _uniform_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="uniform",
        merge_mode=merge_mode,
        lambda_weight=None,
        params=params,
    )
    all_weights = [apply_transforms(load_adapter_weights(p), spec.transforms) for p in adapter_paths]
    merged_weights = merge_adapters_uniform(all_weights, merge_mode=merge_mode)
    merged_delta = compute_delta_from_lora_weights(merged_weights, adapter_paths[0])
    metadata = build_merge_metadata(
        method="uniform",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params=params or {},
        method_params=spec.method_params,
        lambda_policy=spec.method_params.get("lambda_policy"),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=merged_weights, metadata=metadata)


def _weighted_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="weighted",
        merge_mode=merge_mode,
        lambda_weight=None if params is None else params.get("lambda"),
        params=params,
    )

    lambda_weight = spec.method_params.get("lambda")
    if lambda_weight is None:
        raise ValueError("weighted requires lambda_weight.")
    lambda_policy = build_lambda_policy(spec.lambda_policy, fallback_lambda=float(lambda_weight))

    weights1 = apply_transforms(load_adapter_weights(adapter_paths[0]), spec.transforms)
    weights2 = apply_transforms(load_adapter_weights(adapter_paths[1]), spec.transforms)
    merged_weights = merge_adapters_weighted(
        weights1,
        weights2,
        lambda_weight=float(lambda_weight),
        merge_mode=merge_mode,
        lambda_resolver=lambda_policy.lambda_for_key,
    )
    merged_delta = compute_delta_from_lora_weights(merged_weights, adapter_paths[0])
    weighted_metadata = [dict(source_metadata[0]), dict(source_metadata[1])]
    weighted_metadata[0]["weight"] = float(lambda_weight)
    weighted_metadata[1]["weight"] = 1.0 - float(lambda_weight)
    metadata = build_merge_metadata(
        method="weighted",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=weighted_metadata,
        num_parameters=len(merged_delta),
        params={"lambda": float(lambda_weight)},
        lambda_weight=float(lambda_weight),
        method_params=spec.method_params,
        lambda_policy=lambda_policy.describe(),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=merged_weights, metadata=metadata)


def _task_vector_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="task_vector",
        merge_mode=merge_mode,
        lambda_weight=None,
        params=params,
    )
    task_vectors = [extract_task_vector_from_lora(p) for p in adapter_paths]
    merged_delta = merge_adapters_uniform(task_vectors, merge_mode=merge_mode)
    metadata = build_merge_metadata(
        method="task_vector",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params=params or {},
        method_params=spec.method_params,
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _uniform_delta_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="uniform_delta",
        merge_mode=merge_mode,
        lambda_weight=None,
        params=params,
    )
    task_vectors = []
    for adapter_path in adapter_paths:
        transformed = apply_transforms(load_adapter_weights(adapter_path), spec.transforms)
        task_vectors.append(compute_delta_from_lora_weights(transformed, adapter_path))

    merged_delta = merge_adapters_uniform(task_vectors, merge_mode=merge_mode)
    metadata = build_merge_metadata(
        method="uniform_delta",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params=params or {},
        method_params=spec.method_params,
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _weighted_delta_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="weighted_delta",
        merge_mode=merge_mode,
        lambda_weight=None if params is None else params.get("lambda"),
        params=params,
    )
    lambda_weight = spec.method_params.get("lambda")
    if lambda_weight is None:
        raise ValueError("weighted_delta requires lambda_weight.")
    tv1 = extract_task_vector_from_lora(adapter_paths[0])
    tv2 = extract_task_vector_from_lora(adapter_paths[1])
    merged_delta = merge_task_vectors_weighted(
        tv1,
        tv2,
        lambda_weight=float(lambda_weight),
        merge_mode=merge_mode,
    )
    weighted_metadata = [dict(source_metadata[0]), dict(source_metadata[1])]
    weighted_metadata[0]["weight"] = float(lambda_weight)
    weighted_metadata[1]["weight"] = 1.0 - float(lambda_weight)
    metadata = build_merge_metadata(
        method="weighted_delta",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=weighted_metadata,
        num_parameters=len(merged_delta),
        params={"lambda": float(lambda_weight)},
        lambda_weight=float(lambda_weight),
        method_params=spec.method_params,
        lambda_policy=spec.method_params.get("lambda_policy"),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _weighted_delta_n_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="weighted_delta_n",
        merge_mode=merge_mode,
        lambda_weight=None,
        params=params,
    )
    task_vectors = [extract_task_vector_from_lora(p) for p in adapter_paths]
    task_coefficients = spec.method_params.get("task_coefficients")
    normalize_coefficients = bool(spec.method_params.get("normalize_coefficients", True))
    layer_task_coefficients = spec.method_params.get("layer_task_coefficients")
    default_task_coefficients = spec.method_params.get("default_task_coefficients")
    allow_negative_coefficients = bool(spec.method_params.get("allow_negative_coefficients", False))
    merged_delta = merge_task_vectors_weighted_n(
        task_vectors,
        merge_mode=merge_mode,
        task_coefficients=task_coefficients,
        normalize_coefficients=normalize_coefficients,
        layer_task_coefficients=layer_task_coefficients,
        default_task_coefficients=default_task_coefficients,
        allow_negative_coefficients=allow_negative_coefficients,
    )
    policy_type = "layer_task_coefficients" if layer_task_coefficients else "task_coefficients"
    metadata = build_merge_metadata(
        method="weighted_delta_n",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params=params or {},
        method_params=spec.method_params,
        lambda_policy=spec.method_params.get("lambda_policy"),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    metadata["coefficient_policy"] = {
        "type": policy_type,
        "normalize_coefficients": normalize_coefficients,
        "allow_negative_coefficients": allow_negative_coefficients,
        "task_coefficients": task_coefficients,
        "default_task_coefficients": default_task_coefficients,
        "layer_task_coefficients": layer_task_coefficients,
    }
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _uniform_scalar_delta_in_memory(
    *,
    adapter_paths: List[Path],
    source_metadata: List[Dict],
    merge_mode: str,
    params: Optional[Dict[str, object]],
) -> MergeOutput:
    spec = merge_spec_from_legacy_args(
        adapters=[str(p) for p in adapter_paths],
        method="uniform_scalar_delta",
        merge_mode=merge_mode,
        lambda_weight=None,
        params=params,
    )
    task_vectors = []
    for adapter_path in adapter_paths:
        transformed = apply_transforms(load_adapter_weights(adapter_path), spec.transforms)
        task_vectors.append(compute_delta_from_lora_weights(transformed, adapter_path))
    scale = float(spec.method_params.get("scale", 1.0))
    merged_delta = merge_task_vectors_uniform_scalar(
        task_vectors,
        scale=scale,
        merge_mode=merge_mode,
    )
    metadata = build_merge_metadata(
        method="uniform_scalar_delta",
        merge_mode=merge_mode,
        num_adapters=len(adapter_paths),
        source_metadata=source_metadata,
        num_parameters=len(merged_delta),
        params={"scale": scale},
        method_params=spec.method_params,
        lambda_policy=spec.method_params.get("lambda_policy"),
        transforms=[{"name": t.name, "params": dict(t.params)} for t in spec.transforms],
        optimizer=spec.method_params.get("optimizer"),
    )
    return MergeOutput(merged_delta=merged_delta, merged_weights=None, metadata=metadata)


def _task_vector_save(
    *,
    adapter_paths: List[Path],
    output_path: Path,
    merge_mode: str,
    show_progress: bool,
) -> Path:
    merge_uniform_via_task_vectors(
        adapter_paths=adapter_paths,
        output_path=output_path,
        merge_mode=merge_mode,
        show_progress=show_progress,
    )
    return output_path


def register_builtin_methods() -> None:
    register_merge_method(
        MergeMethod(
            name="uniform",
            required_params=(),
            params_defaults={},
            params_validator=None,
            min_adapters=2,
            max_adapters=None,
            saveable=True,
            merge_in_memory=_uniform_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="weighted",
            required_params=("lambda",),
            params_defaults={},
            params_validator=lambda p: build_lambda_policy(
                merge_spec_from_legacy_args(
                    adapters=[],
                    method="weighted",
                    merge_mode="common",
                    lambda_weight=p.get("lambda"),
                    params=p,
                ).lambda_policy,
                fallback_lambda=float(p.get("lambda")),
            ),
            min_adapters=2,
            max_adapters=2,
            saveable=True,
            merge_in_memory=_weighted_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="task_vector",
            required_params=(),
            params_defaults={},
            params_validator=None,
            min_adapters=2,
            max_adapters=None,
            saveable=True,
            merge_in_memory=_task_vector_in_memory,
            save_fn=_task_vector_save,
        )
    )
    register_merge_method(
        MergeMethod(
            name="uniform_delta",
            required_params=(),
            params_defaults={},
            params_validator=None,
            min_adapters=2,
            max_adapters=None,
            saveable=False,
            merge_in_memory=_uniform_delta_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="weighted_delta",
            required_params=("lambda",),
            params_defaults={},
            params_validator=None,
            min_adapters=2,
            max_adapters=2,
            saveable=False,
            merge_in_memory=_weighted_delta_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="weighted_delta_n",
            required_params=(),
            params_defaults={"normalize_coefficients": True},
            params_validator=None,
            min_adapters=2,
            max_adapters=None,
            saveable=False,
            merge_in_memory=_weighted_delta_n_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="uniform_scalar_delta",
            required_params=(),
            params_defaults={"scale": 1.0},
            params_validator=None,
            min_adapters=2,
            max_adapters=None,
            saveable=False,
            merge_in_memory=_uniform_scalar_delta_in_memory,
        )
    )
    register_merge_method(
        MergeMethod(
            name="dare",
            required_params=(),
            params_defaults={"drop_rate": 0.9, "seed": 42},
            params_validator=_validate_dare_params,
            min_adapters=2,
            max_adapters=None,
            saveable=False,
            merge_in_memory=merge_dare,
        )
    )
    register_merge_method(
        MergeMethod(
            name="ties",
            required_params=(),
            params_defaults={"k": 20.0, "lambda": 1.0},
            params_validator=_validate_ties_params,
            min_adapters=2,
            max_adapters=None,
            saveable=False,
            merge_in_memory=merge_ties,
        )
    )


# Register on import
register_builtin_methods()
