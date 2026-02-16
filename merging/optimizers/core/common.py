"""Shared optimizer helpers for AdaMerging-style coefficient optimization."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional, Tuple

import numpy as np
import torch
from safetensors import safe_open
from torch import nn
from torch.utils.data import DataLoader

from core import load_config, prepare_task_for_evaluation
from core.evaluation.evaluate_task import prepare_dataset_cache
from merging.optimizers.core.streaming import StreamingDeltaEntry, StreamingLoraEntry
from merging.optimizers.registry import OptimizerContext
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


def _seed_dataloader_worker(worker_id: int) -> None:
    """Seed Python/NumPy RNGs from the PyTorch worker seed for reproducible loading."""
    del worker_id
    worker_seed = int(torch.initial_seed()) % (2 ** 32)
    random.seed(worker_seed)
    np.random.seed(worker_seed)


@dataclass(frozen=True)
class _DeltaEntry:
    task_key: str
    param_key: str
    layer_idx: Optional[int]
    deltas: List[torch.Tensor]


@dataclass(frozen=True)
class _LoraEntry:
    task_key: str
    param_key: str
    layer_idx: Optional[int]
    a_factors: List[torch.Tensor]
    b_factors: List[torch.Tensor]
    scales: List[float]


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


def _force_zero_dropout(model: nn.Module) -> Dict[str, int]:
    """Set dropout probabilities to zero to remove stochasticity and save activation memory."""
    dropout_modules = (
        nn.Dropout,
        nn.Dropout1d,
        nn.Dropout2d,
        nn.Dropout3d,
        nn.AlphaDropout,
        nn.FeatureAlphaDropout,
    )
    module_updates = 0
    module_seen = 0
    for module in model.modules():
        if isinstance(module, dropout_modules):
            module_seen += 1
            if float(getattr(module, "p", 0.0)) != 0.0:
                module.p = 0.0
                module_updates += 1

    config_updates = 0
    cfg = getattr(model, "config", None)
    if cfg is not None:
        for name in (
            "dropout",
            "attention_dropout",
            "activation_dropout",
            "classifier_dropout",
            "hidden_dropout",
            "hidden_dropout_prob",
            "attention_probs_dropout_prob",
            "embd_pdrop",
            "resid_pdrop",
            "attn_pdrop",
            "summary_first_dropout",
        ):
            value = getattr(cfg, name, None)
            if isinstance(value, (float, int)) and float(value) != 0.0:
                setattr(cfg, name, 0.0)
                config_updates += 1

    return {
        "dropout_modules_seen": int(module_seen),
        "dropout_modules_updated": int(module_updates),
        "dropout_config_fields_updated": int(config_updates),
    }


def _maybe_subset_dataset(
    dataset: Any,
    *,
    task: str,
    eval_subset: Optional[Mapping[str, Any]],
) -> Any:
    if not eval_subset or not bool(eval_subset.get("enabled", True)):
        return dataset
    max_samples = eval_subset.get("max_samples")
    stratified = bool(eval_subset.get("stratified", False))
    stratify_by = eval_subset.get("stratify_by", "label")
    per_task = eval_subset.get("per_task")
    if isinstance(per_task, Mapping) and task in per_task:
        task_cfg = per_task[task]
        if isinstance(task_cfg, Mapping):
            if task_cfg.get("max_samples") is not None:
                max_samples = task_cfg.get("max_samples")
            if "stratified" in task_cfg:
                stratified = bool(task_cfg.get("stratified"))
            if task_cfg.get("stratify_by") is not None:
                stratify_by = task_cfg.get("stratify_by")
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
    if stratified:
        column = str(stratify_by or "label")
        columns = list(getattr(dataset, "column_names", []) or [])
        if column in columns:
            groups: Dict[Any, List[int]] = {}
            values = dataset[column]
            for idx, value in enumerate(values):
                groups.setdefault(value, []).append(idx)
            if groups:
                rng = random.Random(seed)
                if shuffle:
                    for idxs in groups.values():
                        rng.shuffle(idxs)
                total = float(size)
                keys = list(groups.keys())
                exact_alloc = {
                    k: (float(len(groups[k])) / total) * float(max_samples)
                    for k in keys
                }
                alloc = {k: int(exact_alloc[k]) for k in keys}
                for k in keys:
                    if len(groups[k]) > 0 and alloc[k] == 0 and max_samples >= len(keys):
                        alloc[k] = 1
                used = sum(alloc.values())
                if used > max_samples:
                    ranked = sorted(
                        keys,
                        key=lambda k: (exact_alloc[k] - float(alloc[k]), float(alloc[k])),
                    )
                    for k in ranked:
                        if used <= max_samples:
                            break
                        if alloc[k] > 0:
                            alloc[k] -= 1
                            used -= 1
                if used < max_samples:
                    ranked = sorted(
                        keys,
                        key=lambda k: (exact_alloc[k] - float(alloc[k])),
                        reverse=True,
                    )
                    ptr = 0
                    while used < max_samples and ranked:
                        k = ranked[ptr % len(ranked)]
                        if alloc[k] < len(groups[k]):
                            alloc[k] += 1
                            used += 1
                        ptr += 1
                        if ptr > (10 * max_samples + 10):
                            break
                selected: List[int] = []
                for k in keys:
                    take = min(int(alloc[k]), len(groups[k]))
                    if take > 0:
                        selected.extend(groups[k][:take])
                if shuffle:
                    rng.shuffle(selected)
                return dataset.select(selected[:max_samples])
            print(f"[subset] stratified requested for task='{task}' but no groups found; falling back to random.")
        else:
            print(
                f"[subset] stratified requested for task='{task}' but column '{column}' is missing; "
                "falling back to random."
            )
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


def _resolve_lora_pattern_value(pattern: Mapping[str, Any], key: str, default: float) -> float:
    if not pattern:
        return default
    if key in pattern:
        return float(pattern[key])
    matches = [k for k in pattern.keys() if key.endswith(str(k))]
    if not matches:
        return default
    best = max(matches, key=len)
    return float(pattern[best])


def _extract_lora_factors_from_adapter(adapter_path: Path) -> Dict[str, Tuple[torch.Tensor, torch.Tensor, float]]:
    config_path = adapter_path / "adapter_config.json"
    with open(config_path, "r") as f:
        config = json.load(f)

    default_rank = float(config["r"])
    default_alpha = float(config["lora_alpha"])
    rank_pattern = config.get("rank_pattern")
    alpha_pattern = config.get("alpha_pattern")
    if rank_pattern is not None and not isinstance(rank_pattern, Mapping):
        raise ValueError(f"Adapter {adapter_path}: rank_pattern must be a mapping when provided.")
    if alpha_pattern is not None and not isinstance(alpha_pattern, Mapping):
        raise ValueError(f"Adapter {adapter_path}: alpha_pattern must be a mapping when provided.")
    rank_pattern = rank_pattern if isinstance(rank_pattern, Mapping) else {}
    alpha_pattern = alpha_pattern if isinstance(alpha_pattern, Mapping) else {}

    safetensors_path = adapter_path / "adapter_model.safetensors"
    lora_pairs: Dict[str, Dict[str, torch.Tensor]] = {}
    with safe_open(safetensors_path, framework="pt", device="cpu") as f:
        for key in f.keys():
            if ".lora_A." in key:
                base_key = key.replace(".lora_A.weight", "")
                lora_pairs.setdefault(base_key, {})["A"] = f.get_tensor(key).clone()
            elif ".lora_B." in key:
                base_key = key.replace(".lora_B.weight", "")
                lora_pairs.setdefault(base_key, {})["B"] = f.get_tensor(key).clone()

    incomplete = [k for k, pair in lora_pairs.items() if "A" not in pair or "B" not in pair]
    if incomplete:
        sample = ", ".join(sorted(incomplete)[:8])
        extra = " ..." if len(incomplete) > 8 else ""
        raise ValueError(
            f"Adapter {adapter_path}: missing A/B LoRA pair(s) for {len(incomplete)} module(s): {sample}{extra}"
        )

    out: Dict[str, Tuple[torch.Tensor, torch.Tensor, float]] = {}
    for base_key, pair in lora_pairs.items():
        rank = _resolve_lora_pattern_value(rank_pattern, base_key, default_rank)
        alpha = _resolve_lora_pattern_value(alpha_pattern, base_key, default_alpha)
        if rank <= 0:
            raise ValueError(f"Adapter {adapter_path}: non-positive rank for '{base_key}' (rank={rank}).")
        out[base_key] = (pair["A"], pair["B"], float(alpha / rank))
    return out


def _build_lora_entries(
    adapter_paths: List[Path],
    *,
    parameter_keys: List[str],
    merge_mode: str,
) -> List[_LoraEntry]:
    per_adapter = [_extract_lora_factors_from_adapter(path) for path in adapter_paths]
    key_sets = [set(x.keys()) for x in per_adapter]
    if merge_mode == "strict":
        reference = key_sets[0]
        for i, keys in enumerate(key_sets[1:], start=1):
            if keys != reference:
                raise ValueError(
                    "fused_lora_linear strict mode requires identical LoRA module keys across adapters; "
                    f"mismatch at adapter {i}."
                )
        keys_to_merge = reference
    else:
        keys_to_merge = set.intersection(*key_sets)

    if not keys_to_merge:
        raise ValueError("fused_lora_linear found no common LoRA modules across adapters.")

    entries: List[_LoraEntry] = []
    unmapped: List[str] = []
    shape_mismatch: List[str] = []
    for key in sorted(keys_to_merge):
        param_key = _resolve_delta_param_name(key, parameter_keys)
        if param_key is None:
            unmapped.append(key)
            continue
        factors = [adapter[key] for adapter in per_adapter]
        a_shapes = [tuple(item[0].shape) for item in factors]
        b_shapes = [tuple(item[1].shape) for item in factors]
        if len(set(a_shapes)) != 1 or len(set(b_shapes)) != 1:
            shape_mismatch.append(f"{key}: A={a_shapes} B={b_shapes}")
            continue
        entries.append(
            _LoraEntry(
                task_key=key,
                param_key=param_key,
                layer_idx=extract_layer_index(key),
                a_factors=[item[0] for item in factors],
                b_factors=[item[1] for item in factors],
                scales=[float(item[2]) for item in factors],
            )
        )

    if shape_mismatch:
        sample = "; ".join(shape_mismatch[:8])
        extra = " ..." if len(shape_mismatch) > 8 else ""
        raise ValueError(
            "fused_lora_linear detected LoRA shape/rank mismatches across adapters: "
            f"{sample}{extra}"
        )
    if unmapped:
        sample = ", ".join(unmapped[:8])
        extra = " ..." if len(unmapped) > 8 else ""
        raise ValueError(
            "fused_lora_linear could not map LoRA modules to model parameters "
            f"for {len(unmapped)} key(s): {sample}{extra}"
        )
    if not entries:
        raise ValueError("fused_lora_linear could not build any LoRA entries for model parameters.")
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
    seed: Optional[int] = None,
) -> DataLoader:
    task_module = get_task_module(task)
    config_path = task_module.get_config_path(PACKAGE_ROOT, None)
    config = load_config(config_path)
    artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
    config = prepare_dataset_cache(config, artifact_dirs)
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
    if seed is not None:
        loader_generator = torch.Generator()
        loader_generator.manual_seed(int(seed))
        loader_kwargs["generator"] = loader_generator
        loader_kwargs["worker_init_fn"] = _seed_dataloader_worker
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
    magnitude_task: Optional[torch.Tensor] = None,
    magnitude_default: Optional[torch.Tensor] = None,
    magnitude_layer: Optional[torch.Tensor] = None,
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
    elif coefficient_parameterization == "simplex_softmax_scaled":
        task = torch.softmax(alpha_task, dim=-1)
        default = torch.softmax(alpha_default, dim=-1) if alpha_default is not None else None
    else:
        raise ValueError(
            "coefficient_parameterization must be one of: "
            "sigmoid_alpha|tanh_alpha|projected_lambda|simplex_softmax_scaled."
        )
    layer_map: Dict[int, torch.Tensor] = {}
    if alpha_layer is not None:
        for i, layer in enumerate(layer_indices):
            if coefficient_parameterization == "sigmoid_alpha":
                layer_map[int(layer)] = torch.sigmoid(alpha_layer[i])
            elif coefficient_parameterization == "tanh_alpha":
                layer_map[int(layer)] = torch.tanh(alpha_layer[i])
            elif coefficient_parameterization == "simplex_softmax_scaled":
                layer_map[int(layer)] = torch.softmax(alpha_layer[i], dim=-1)
            else:
                layer_map[int(layer)] = alpha_layer[i]

    def _scale(x: torch.Tensor, magnitude: Optional[torch.Tensor]) -> torch.Tensor:
        if coefficient_parameterization != "simplex_softmax_scaled":
            return x
        if magnitude is None:
            return x
        return x * magnitude

    task = _scale(task, magnitude_task)
    if default is not None:
        default = _scale(default, magnitude_default)
    if layer_map:
        scaled_layers: Dict[int, torch.Tensor] = {}
        for i, layer in enumerate(layer_indices):
            coeffs = layer_map.get(int(layer))
            if coeffs is None:
                continue
            layer_mag: Optional[torch.Tensor] = None
            if magnitude_layer is not None:
                layer_mag = magnitude_layer[i]
            elif magnitude_default is not None:
                layer_mag = magnitude_default
            elif magnitude_task is not None:
                layer_mag = magnitude_task
            scaled_layers[int(layer)] = _scale(coeffs, layer_mag)
        layer_map = scaled_layers

    def _norm(x: torch.Tensor) -> torch.Tensor:
        if not normalize_coefficients:
            return x
        if coefficient_parameterization == "simplex_softmax_scaled":
            # Coefficients are already simplex-normalized pre-scale; keep learned magnitude intact.
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


def _to_streaming_lora_entries(lora_entries: List[_LoraEntry]) -> List[StreamingLoraEntry]:
    return [
        StreamingLoraEntry(
            param_key=entry.param_key,
            layer_idx=entry.layer_idx,
            a_factors=entry.a_factors,
            b_factors=entry.b_factors,
            scales=entry.scales,
        )
        for entry in lora_entries
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
    "_LoraEntry",
    "_build_delta_entries",
    "_build_lora_entries",
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
    "_force_zero_dropout",
    "_to_streaming_lora_entries",
    "_to_streaming_entries",
]
