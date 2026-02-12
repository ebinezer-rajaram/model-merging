"""Shared optimizer helpers for AdaMerging-style coefficient optimization."""

from __future__ import annotations

import math
import random
from dataclasses import dataclass
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from core import load_config, prepare_task_for_evaluation
from experiments.evaluate_task import _prepare_dataset_cache
from merging.plugins.adamerging_streaming import StreamingDeltaEntry
from merging.plugins.optimizers import OptimizerContext
from merging.policies.lambda_policy import extract_layer_index
from merging.runtime.utils import PACKAGE_ROOT, get_task_module

_CLASSIFICATION_TASKS = {
    "emotion",
    "intent",
    "speaker_id",
    "kws",
    "langid",
    "speaker_ver",
}

_DTYPE_ALIASES = {
    "bf16": torch.bfloat16,
    "bfloat16": torch.bfloat16,
    "fp16": torch.float16,
    "float16": torch.float16,
    "fp32": torch.float32,
    "float32": torch.float32,
}

_SAMPLING_STRATEGIES = {"uniform", "none", "inverse", "sqrt_inverse", "balanced"}


@dataclass(frozen=True)
class _DeltaEntry:
    task_key: str
    param_key: str
    layer_idx: Optional[int]
    deltas: List[torch.Tensor]


def _logit(p: float) -> float:
    p = float(min(1.0 - 1e-6, max(1e-6, p)))
    return math.log(p / (1.0 - p))


def _resolve_dtype(name: str) -> Optional[torch.dtype]:
    if name == "auto":
        return None
    mapped = _DTYPE_ALIASES.get(name)
    if mapped is None:
        raise ValueError(f"Unsupported dtype '{name}'.")
    return mapped


def _to_device(batch: Any, device: torch.device, *, non_blocking: bool = False) -> Any:
    if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, torch.Tensor):
        try:
            return batch.to(device, non_blocking=non_blocking)
        except Exception:
            pass
    if isinstance(batch, torch.Tensor):
        return batch.to(device, non_blocking=non_blocking)
    if isinstance(batch, dict):
        return {k: _to_device(v, device, non_blocking=non_blocking) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_to_device(x, device, non_blocking=non_blocking) for x in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(x, device, non_blocking=non_blocking) for x in batch)
    return batch


def _maybe_subset_dataset(
    dataset: Any,
    *,
    task: str,
    eval_subset: Optional[Mapping[str, Any]],
) -> Any:
    if not eval_subset or not bool(eval_subset.get("enabled", True)):
        return dataset
    max_samples = eval_subset.get("max_samples")
    per_task = eval_subset.get("per_task")
    if isinstance(per_task, Mapping) and task in per_task:
        task_cfg = per_task[task]
        if isinstance(task_cfg, Mapping):
            if task_cfg.get("max_samples") is not None:
                max_samples = task_cfg.get("max_samples")
        elif isinstance(task_cfg, (int, float)):
            max_samples = task_cfg
    if max_samples is None:
        return dataset
    max_samples = int(max_samples)
    if max_samples <= 0:
        raise ValueError("optimizer.params.eval_subset.max_samples must be > 0.")
    size = len(dataset)
    if max_samples >= size:
        return dataset
    shuffle = bool(eval_subset.get("shuffle", False))
    seed = int(eval_subset.get("seed", 0) or 0)
    indices = list(range(size))
    if shuffle:
        random.Random(seed).shuffle(indices)
    return dataset.select(indices[:max_samples])


def _resolve_eval_tasks(context: OptimizerContext, params: Mapping[str, Any]) -> List[str]:
    explicit = params.get("tasks")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("optimizer.params.tasks must be a list when provided.")
        tasks = [str(t).strip().lower() for t in explicit if str(t).strip()]
    else:
        tasks = []
        for meta in context.source_metadata:
            task_name = meta.get("task")
            if task_name:
                tasks.append(str(task_name).strip().lower())
    tasks = [t for t in tasks if t in _CLASSIFICATION_TASKS]
    deduped: List[str] = []
    seen = set()
    for task_name in tasks:
        if task_name in seen:
            continue
        seen.add(task_name)
        deduped.append(task_name)
    tasks = deduped
    if not tasks:
        raise ValueError(
            "AdaMerging-like optimizers require at least one classification task in "
            "optimizer.params.tasks or source adapter metadata."
        )
    return tasks


def _resolve_delta_param_name(base_key: str, parameter_keys: List[str]) -> Optional[str]:
    parameter_key_set = set(parameter_keys)
    candidates = [f"{base_key}.weight"]
    suffix_bases = {base_key}

    def _add_candidate(prefix_from: str, prefix_to: str) -> None:
        if base_key.startswith(prefix_from):
            trimmed = base_key.replace(prefix_from, prefix_to, 1)
            candidates.append(f"{trimmed}.weight")
            suffix_bases.add(trimmed)

    _add_candidate("base_model.model.model.", "model.")
    _add_candidate("base_model.model.", "model.")
    _add_candidate("base_model.model.", "")
    _add_candidate("base_model.", "")
    _add_candidate("model.model.model.", "model.")
    _add_candidate("model.model.", "model.")
    _add_candidate("model.", "model.model.")

    for candidate in candidates:
        if candidate in parameter_key_set:
            return candidate

    for suffix_base in suffix_bases:
        suffix = f".{suffix_base}.weight"
        for key in parameter_keys:
            if key.endswith(suffix):
                return key
    return None


def _build_delta_entries(
    task_vectors: List[Dict[str, torch.Tensor]],
    *,
    parameter_keys: List[str],
    merge_mode: str,
) -> List[_DeltaEntry]:
    key_sets = [set(tv.keys()) for tv in task_vectors]
    if merge_mode == "strict":
        reference = key_sets[0]
        for i, keys in enumerate(key_sets[1:], start=1):
            if keys != reference:
                raise ValueError(
                    "AdaMerging strict mode requires identical task-vector keys; "
                    f"mismatch at adapter {i}."
                )
        keys_to_merge = reference
    else:
        keys_to_merge = set.intersection(*key_sets)

    entries: List[_DeltaEntry] = []
    for key in sorted(keys_to_merge):
        param_key = _resolve_delta_param_name(key, parameter_keys)
        if param_key is None:
            continue
        deltas = [tv[key] for tv in task_vectors]
        shapes = [tuple(d.shape) for d in deltas]
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"AdaMerging strict mode shape mismatch for key '{key}': {shapes}")
            continue
        entries.append(
            _DeltaEntry(
                task_key=key,
                param_key=param_key,
                layer_idx=extract_layer_index(key),
                deltas=deltas,
            )
        )
    if not entries:
        raise ValueError("AdaMerging could not map any task-vector deltas to model parameters.")
    return entries


def _build_task_loader(
    task: str,
    *,
    processor: Any,
    split: str,
    batch_size: int,
    eval_subset: Optional[Mapping[str, Any]],
    sampling: Optional[Mapping[str, Any]],
    num_workers: int,
    pin_memory: bool,
) -> DataLoader:
    task_module = get_task_module(task)
    config_path = task_module.get_config_path(PACKAGE_ROOT, None)
    config = load_config(config_path)
    artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
    config = _prepare_dataset_cache(config, artifact_dirs)
    setup = prepare_task_for_evaluation(task, processor, split=split, config=config)
    setup.dataset = _maybe_subset_dataset(setup.dataset, task=task, eval_subset=eval_subset)
    resolved_sampling = _resolve_sampling_for_task(task=task, sampling=sampling)
    loader_kwargs: Dict[str, Any] = {
        "dataset": setup.dataset,
        "batch_size": batch_size,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": setup.data_collator,
    }
    if resolved_sampling["enabled"]:
        from core.training.samplers import WeightedClassSampler

        loader_kwargs["sampler"] = WeightedClassSampler(
            setup.dataset,
            num_samples=len(setup.dataset),
            replacement=True,
            method=resolved_sampling["strategy"],
        )
        loader_kwargs["shuffle"] = False
    else:
        loader_kwargs["shuffle"] = True
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)
    return loader


def _resolve_sampling_for_task(
    *,
    task: str,
    sampling: Optional[Mapping[str, Any]],
) -> Dict[str, Any]:
    """Resolve sampling policy for a task using default + per-task overrides.

    Schema:
      sampling:
        default:
          enabled: bool
          strategy: uniform|inverse|sqrt_inverse|balanced|none
        per_task:
          <task_name>:
            enabled: bool
            strategy: ...
    """
    default_enabled = False
    default_strategy = "uniform"
    per_task_cfg: Optional[Mapping[str, Any]] = None
    if isinstance(sampling, Mapping):
        default_cfg = sampling.get("default")
        if isinstance(default_cfg, Mapping):
            if "enabled" in default_cfg:
                default_enabled = bool(default_cfg.get("enabled"))
            if "strategy" in default_cfg and default_cfg.get("strategy") is not None:
                default_strategy = str(default_cfg.get("strategy")).strip().lower()
        per_task_raw = sampling.get("per_task")
        if isinstance(per_task_raw, Mapping):
            cfg = per_task_raw.get(task)
            if isinstance(cfg, Mapping):
                per_task_cfg = cfg
    enabled = default_enabled
    strategy = default_strategy
    if per_task_cfg is not None:
        if "enabled" in per_task_cfg:
            enabled = bool(per_task_cfg.get("enabled"))
        if "strategy" in per_task_cfg and per_task_cfg.get("strategy") is not None:
            strategy = str(per_task_cfg.get("strategy")).strip().lower()
    if strategy not in _SAMPLING_STRATEGIES:
        raise ValueError(
            "optimizer.params.sampling strategy must be one of: "
            + "|".join(sorted(_SAMPLING_STRATEGIES))
            + f". Got '{strategy}' for task '{task}'."
        )
    if strategy in {"uniform", "none"}:
        enabled = False
    return {"enabled": bool(enabled), "strategy": strategy}


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("Model output does not contain logits required for optimization.")
    return logits


def _merge_coeffs(
    *,
    alpha_task: torch.Tensor,
    alpha_default: Optional[torch.Tensor],
    alpha_layer: Optional[torch.Tensor],
    layer_indices: List[int],
    coefficient_parameterization: str,
    normalize_coefficients: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[int, torch.Tensor]]:
    if coefficient_parameterization == "sigmoid_alpha":
        task = torch.sigmoid(alpha_task)
        default = torch.sigmoid(alpha_default) if alpha_default is not None else None
    elif coefficient_parameterization == "tanh_alpha":
        task = torch.tanh(alpha_task)
        default = torch.tanh(alpha_default) if alpha_default is not None else None
    elif coefficient_parameterization == "projected_lambda":
        task = alpha_task
        default = alpha_default
    else:
        raise ValueError(
            "coefficient_parameterization must be one of: "
            "sigmoid_alpha|tanh_alpha|projected_lambda."
        )
    layer_map: Dict[int, torch.Tensor] = {}
    if alpha_layer is not None:
        for i, layer in enumerate(layer_indices):
            if coefficient_parameterization == "sigmoid_alpha":
                layer_map[int(layer)] = torch.sigmoid(alpha_layer[i])
            elif coefficient_parameterization == "tanh_alpha":
                layer_map[int(layer)] = torch.tanh(alpha_layer[i])
            else:
                layer_map[int(layer)] = alpha_layer[i]

    def _norm(x: torch.Tensor) -> torch.Tensor:
        if not normalize_coefficients:
            return x
        denom = x.sum().clamp_min(1e-8)
        return x / denom

    task = _norm(task)
    if default is not None:
        default = _norm(default)
    layer_map = {k: _norm(v) for k, v in layer_map.items()}
    return task, default, layer_map


def _project_coefficients_inplace(
    *,
    alpha_task: torch.Tensor,
    alpha_default: Optional[torch.Tensor],
    alpha_layer: Optional[torch.Tensor],
    coefficient_parameterization: str,
    coeff_min: float,
    coeff_max: float,
) -> None:
    if coefficient_parameterization != "projected_lambda":
        return
    with torch.no_grad():
        alpha_task.clamp_(min=coeff_min, max=coeff_max)
        if alpha_default is not None:
            alpha_default.clamp_(min=coeff_min, max=coeff_max)
        if alpha_layer is not None:
            alpha_layer.clamp_(min=coeff_min, max=coeff_max)


def _build_functional_params(
    *,
    base_params: Dict[str, torch.Tensor],
    delta_entries: List[_DeltaEntry],
    task_coeffs: torch.Tensor,
    default_coeffs: Optional[torch.Tensor],
    layer_coeffs: Dict[int, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    params = dict(base_params)
    for entry in delta_entries:
        coeffs = task_coeffs
        if entry.layer_idx is not None and entry.layer_idx in layer_coeffs:
            coeffs = layer_coeffs[entry.layer_idx]
        elif default_coeffs is not None:
            coeffs = default_coeffs

        base = base_params[entry.param_key]
        merged_param = base.clone()
        for i, delta in enumerate(entry.deltas):
            if delta.device != base.device or delta.dtype != base.dtype:
                delta_cast = delta.to(device=base.device, dtype=base.dtype)
            else:
                delta_cast = delta
            merged_param.add_(delta_cast * coeffs[i])
        params[entry.param_key] = merged_param
    return params


def _functional_forward(model, params: Dict[str, torch.Tensor], batch: Dict[str, Any]):
    try:
        from torch.func import functional_call

        buffers = dict(model.named_buffers())
        return functional_call(model, (params, buffers), args=(), kwargs=batch)
    except ImportError:
        from torch.nn.utils.stateless import functional_call  # type: ignore

        return functional_call(model, params, args=(), kwargs=batch)


def _infer_input_device(model) -> torch.device:
    try:
        emb = model.get_input_embeddings()
        if emb is not None and hasattr(emb, "weight"):
            return emb.weight.device
    except Exception:
        pass
    return next(model.parameters()).device


def _next_batch(loader: DataLoader, iterator: Optional[Iterator[Any]]) -> Tuple[Any, Iterator[Any]]:
    if iterator is None:
        iterator = iter(loader)
    try:
        batch = next(iterator)
        return batch, iterator
    except StopIteration:
        iterator = iter(loader)
        batch = next(iterator)
        return batch, iterator


def _to_streaming_entries(delta_entries: List[_DeltaEntry]) -> List[StreamingDeltaEntry]:
    return [
        StreamingDeltaEntry(
            param_key=entry.param_key,
            layer_idx=entry.layer_idx,
            deltas=entry.deltas,
        )
        for entry in delta_entries
    ]


def _global_topk_threshold(abs_values: torch.Tensor, k_percent: float) -> torch.Tensor:
    if abs_values.numel() == 0 or k_percent <= 0.0:
        return torch.tensor(float("inf"), dtype=torch.float32, device=abs_values.device)
    if k_percent >= 100.0:
        return torch.tensor(0.0, dtype=torch.float32, device=abs_values.device)

    keep_count = int(math.ceil(abs_values.numel() * (k_percent / 100.0)))
    keep_count = max(1, min(keep_count, abs_values.numel()))
    topk_values = torch.topk(abs_values, k=keep_count, largest=True, sorted=False).values
    return torch.min(topk_values)


def _trim_task_vector_global(task_vector: Dict[str, torch.Tensor], k_percent: float) -> Dict[str, torch.Tensor]:
    if not task_vector:
        return {}
    if k_percent <= 0.0:
        return {key: torch.zeros_like(tensor, dtype=torch.float32) for key, tensor in task_vector.items()}
    if k_percent >= 100.0:
        return {key: tensor.detach().to(dtype=torch.float32).clone() for key, tensor in task_vector.items()}

    flat = [tensor.detach().to(dtype=torch.float32).abs().reshape(-1) for tensor in task_vector.values()]
    abs_values = torch.cat(flat) if flat else torch.empty(0, dtype=torch.float32)
    threshold = _global_topk_threshold(abs_values, k_percent)

    trimmed: Dict[str, torch.Tensor] = {}
    for key, tensor in task_vector.items():
        tensor_f32 = tensor.detach().to(dtype=torch.float32)
        keep_mask = tensor_f32.abs() >= threshold
        trimmed[key] = torch.where(keep_mask, tensor_f32, torch.zeros_like(tensor_f32))
    return trimmed


def _resolve_keys_to_merge(task_vectors: List[Dict[str, torch.Tensor]], merge_mode: str) -> Tuple[List[str], int]:
    if merge_mode not in {"common", "strict"}:
        raise ValueError(f"Unsupported merge_mode='{merge_mode}'.")
    key_sets = [set(tv.keys()) for tv in task_vectors]
    if merge_mode == "strict":
        reference = key_sets[0]
        for i, keys in enumerate(key_sets[1:], start=1):
            if keys != reference:
                raise ValueError(
                    f"Task vectors have different parameters in strict mode. "
                    f"Mismatch at adapter index {i}."
                )
        return sorted(reference), 0
    common_keys = set.intersection(*key_sets) if key_sets else set()
    all_keys = set.union(*key_sets) if key_sets else set()
    missing_count = len(all_keys - common_keys)
    return sorted(common_keys), missing_count


def _apply_ties_consensus_preprocess(
    task_vectors: List[Dict[str, torch.Tensor]],
    *,
    merge_mode: str,
    k_percent: float,
    lambda_scale: float,
) -> Tuple[List[Dict[str, torch.Tensor]], Dict[str, Any]]:
    if not task_vectors:
        return [], {
            "enabled": True,
            "k": float(k_percent),
            "lambda": float(lambda_scale),
            "trim_density": 0.0,
            "sign_conflict_rate": 0.0,
            "merged_nonzero_entries": 0,
            "skipped_missing_key_count": 0,
            "skipped_shape_mismatch_count": 0,
            "keys_to_merge": 0,
        }
    keys_to_merge, missing_key_count = _resolve_keys_to_merge(task_vectors, merge_mode)
    shape_mismatch_count = 0
    for key in list(keys_to_merge):
        shapes = [tuple(tv[key].shape) for tv in task_vectors]
        if len(set(shapes)) != 1:
            if merge_mode == "strict":
                raise ValueError(f"Shape mismatch for key '{key}' in strict mode: {shapes}")
            keys_to_merge.remove(key)
            shape_mismatch_count += 1
    mergeable_vectors = [{key: tv[key] for key in keys_to_merge} for tv in task_vectors]
    trimmed_vectors = [_trim_task_vector_global(tv, k_percent) for tv in mergeable_vectors]

    elected_sign_map: Dict[str, torch.Tensor] = {}
    for key in keys_to_merge:
        stacked = torch.stack([tv[key] for tv in trimmed_vectors], dim=0)
        elected_sign_map[key] = torch.sign(stacked.sum(dim=0))

    processed_vectors: List[Dict[str, torch.Tensor]] = []
    merged_nonzero_entries = 0
    active_sign_positions = 0
    conflict_positions = 0
    total_entries = 0
    kept_entries = 0
    for i, tv in enumerate(task_vectors):
        out_i: Dict[str, torch.Tensor] = {}
        for key in keys_to_merge:
            original = tv[key]
            trimmed = trimmed_vectors[i][key]
            elected_sign = elected_sign_map[key]
            aligned_mask = (
                (torch.sign(trimmed) == elected_sign)
                & (elected_sign != 0.0)
                & (trimmed != 0.0)
            )
            filtered = torch.where(aligned_mask, trimmed * float(lambda_scale), torch.zeros_like(trimmed))
            out_i[key] = filtered.to(dtype=original.dtype)
            merged_nonzero_entries += int((filtered != 0.0).sum().item())
            total_entries += int(filtered.numel())
            kept_entries += int((filtered != 0.0).sum().item())
        processed_vectors.append(out_i)

    for key in keys_to_merge:
        stacked = torch.stack([tv[key] for tv in trimmed_vectors], dim=0)
        active = (stacked != 0.0).any(dim=0)
        pos_present = (stacked > 0.0).any(dim=0)
        neg_present = (stacked < 0.0).any(dim=0)
        conflict = pos_present & neg_present
        active_sign_positions += int(active.sum().item())
        conflict_positions += int((conflict & active).sum().item())

    trim_density = float(kept_entries) / float(total_entries) if total_entries > 0 else 0.0
    sign_conflict_rate = (
        float(conflict_positions) / float(active_sign_positions) if active_sign_positions > 0 else 0.0
    )
    stats: Dict[str, Any] = {
        "enabled": True,
        "k": float(k_percent),
        "lambda": float(lambda_scale),
        "trim_density": trim_density,
        "sign_conflict_rate": sign_conflict_rate,
        "merged_nonzero_entries": int(merged_nonzero_entries),
        "skipped_missing_key_count": int(missing_key_count),
        "skipped_shape_mismatch_count": int(shape_mismatch_count),
        "keys_to_merge": int(len(keys_to_merge)),
    }
    return processed_vectors, stats


__all__ = [
    "_CLASSIFICATION_TASKS",
    "_DTYPE_ALIASES",
    "_DeltaEntry",
    "_build_delta_entries",
    "_build_functional_params",
    "_build_task_loader",
    "_extract_logits",
    "_functional_forward",
    "_infer_input_device",
    "_logit",
    "_maybe_subset_dataset",
    "_merge_coeffs",
    "_next_batch",
    "_project_coefficients_inplace",
    "_resolve_delta_param_name",
    "_resolve_dtype",
    "_resolve_eval_tasks",
    "_resolve_sampling_for_task",
    "_apply_ties_consensus_preprocess",
    "_to_device",
    "_to_streaming_entries",
]
