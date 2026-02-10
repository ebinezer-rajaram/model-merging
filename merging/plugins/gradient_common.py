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
    loader_kwargs: Dict[str, Any] = {
        "dataset": setup.dataset,
        "batch_size": batch_size,
        "shuffle": True,
        "drop_last": False,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "collate_fn": setup.data_collator,
    }
    if num_workers > 0:
        loader_kwargs["persistent_workers"] = True
    loader = DataLoader(**loader_kwargs)
    return loader


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
    elif coefficient_parameterization == "projected_lambda":
        task = alpha_task
        default = alpha_default
    else:
        raise ValueError(
            "coefficient_parameterization must be one of: sigmoid_alpha|projected_lambda."
        )
    layer_map: Dict[int, torch.Tensor] = {}
    if alpha_layer is not None:
        for i, layer in enumerate(layer_indices):
            if coefficient_parameterization == "sigmoid_alpha":
                layer_map[int(layer)] = torch.sigmoid(alpha_layer[i])
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
    "_to_device",
    "_to_streaming_entries",
]
