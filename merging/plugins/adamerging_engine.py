"""AdaMerging optimizer engine."""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch
from torch.utils.data import DataLoader

from core import load_config, prepare_task_for_evaluation
from core.evaluation.eval_utils import load_model_and_processor
from experiments.evaluate_task import _get_model_path, _prepare_dataset_cache
from experiments.extract_vector import extract_task_vector_from_lora
from merging.config.specs import MergeSpec
from merging.plugins.adamerging_streaming import (
    StreamingDeltaEntry,
    TaskCoefficientProvider,
    register_fused_linear_modules,
    register_streaming_parametrizations,
    unregister_fused_linear_modules,
    unregister_streaming_parametrizations,
)
from merging.plugins.optimizers import OptimizerContext, OptimizerResult
from merging.policies.lambda_policy import extract_layer_index
from merging.runtime.utils import get_task_module, PACKAGE_ROOT

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


_CLASSIFICATION_TASKS = {
    "emotion",
    "intent",
    "speaker_id",
    "kws",
    "langid",
    "speaker_ver",
}

_PLUS_PLUS_VARIANTS = {"task_wise_plus_plus", "layer_wise_plus_plus", "plusplus", "plus_plus"}
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
    tasks = sorted(set(tasks))
    if not tasks:
        raise ValueError(
            "AdaMerging v1 requires at least one classification task in optimizer.params.tasks "
            "or source adapter metadata."
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
                    f"AdaMerging strict mode requires identical task-vector keys; mismatch at adapter {i}."
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
        raise ValueError("Model output does not contain logits required for entropy optimization.")
    return logits


def _compute_entropy_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    *,
    entropy_temperature: float = 1.0,
    entropy_chunk_size: int = 4096,
    require_valid_label_mask: bool = False,
    return_stats: bool = False,
):
    scaled_logits = logits.float() / float(entropy_temperature)
    lse = torch.logsumexp(scaled_logits, dim=-1, keepdim=True)
    max_logit = scaled_logits.max(dim=-1).values if return_stats else None
    weighted_sum = torch.zeros_like(lse.squeeze(-1))
    chunk = int(entropy_chunk_size)
    use_chunking = (chunk > 0) and (scaled_logits.shape[-1] > chunk)
    if use_chunking:
        for start in range(0, scaled_logits.shape[-1], chunk):
            stop = min(start + chunk, scaled_logits.shape[-1])
            z_chunk = scaled_logits[..., start:stop]
            probs_chunk = torch.exp(z_chunk - lse)
            weighted_sum = weighted_sum + (probs_chunk * z_chunk).sum(dim=-1)
    else:
        probs = torch.exp(scaled_logits - lse)
        weighted_sum = (probs * scaled_logits).sum(dim=-1)
    entropy = lse.squeeze(-1) - weighted_sum
    pmax: Optional[torch.Tensor] = None
    if return_stats and max_logit is not None:
        pmax = torch.exp(max_logit - lse.squeeze(-1))
    valid_label_tokens: Optional[int] = None
    if labels is not None and logits.ndim >= 3 and labels.shape[:2] == logits.shape[:2]:
        mask = labels != -100
        valid_label_tokens = int(mask.sum().detach().item())
        if mask.any():
            entropy = entropy[mask]
            if pmax is not None and pmax.shape == mask.shape:
                pmax = pmax[mask]
        elif require_valid_label_mask:
            raise ValueError("No valid label tokens found for entropy masking (labels are fully masked).")
    elif labels is not None and require_valid_label_mask:
        raise ValueError("No valid label mask available: label shape is incompatible with logits.")

    if entropy.numel() == 0:
        if require_valid_label_mask:
            raise ValueError("Entropy tensor is empty after label masking.")
        loss = logits.sum() * 0.0
        if not return_stats:
            return loss
        stats = {
            "entropy_min": float("nan"),
            "entropy_max": float("nan"),
            "pmax_mean": float("nan"),
            "valid_label_tokens": (0 if valid_label_tokens is None else int(valid_label_tokens)),
            "num_entropy_positions": 0,
        }
        return loss, stats

    loss = entropy.mean()
    if not return_stats:
        return loss
    stats = {
        "entropy_min": float(entropy.min().detach().item()),
        "entropy_max": float(entropy.max().detach().item()),
        "pmax_mean": float(pmax.mean().detach().item()) if pmax is not None else float("nan"),
        "valid_label_tokens": (
            int(valid_label_tokens) if valid_label_tokens is not None else int(entropy.numel())
        ),
        "num_entropy_positions": int(entropy.numel()),
    }
    return loss, stats


def _merge_coeffs(
    *,
    alpha_task: torch.Tensor,
    alpha_default: Optional[torch.Tensor],
    alpha_layer: Optional[torch.Tensor],
    layer_indices: List[int],
    normalize_coefficients: bool,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Dict[int, torch.Tensor]]:
    task = torch.sigmoid(alpha_task)
    default = torch.sigmoid(alpha_default) if alpha_default is not None else None
    layer_map: Dict[int, torch.Tensor] = {}
    if alpha_layer is not None:
        for i, layer in enumerate(layer_indices):
            layer_map[int(layer)] = torch.sigmoid(alpha_layer[i])

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
    """Infer best device for model inputs.

    With `device_map="auto"` the first parameter can live on CPU while embeddings
    are on CUDA. Inputs must match embedding device for initial lookup.
    """
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


def run_adamerging_optimizer(spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
    params = dict(spec.optimizer.params if spec.optimizer is not None else {})
    variant = str(params.get("variant", "task_wise")).strip().lower()
    plus_plus = bool(params.get("plus_plus", False))
    if plus_plus or variant in _PLUS_PLUS_VARIANTS:
        raise NotImplementedError("AdaMerging++ requires TIES implementation; currently scaffold-only.")
    if variant not in {"task_wise", "layer_wise"}:
        raise ValueError(f"Unsupported optimizer.params.variant='{variant}'.")
    if context.method != "weighted_delta_n":
        raise ValueError("AdaMerging v1 only supports method='weighted_delta_n'.")

    task_scope = str(params.get("task_scope", "classification")).strip().lower()
    if task_scope != "classification":
        raise ValueError("AdaMerging v1 supports only optimizer.params.task_scope='classification'.")

    steps = int(params.get("steps", 500))
    lr = float(params.get("lr", 1e-3))
    betas = params.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.params.betas must be a 2-item list/tuple.")
    betas = (float(betas[0]), float(betas[1]))
    batch_size = int(params.get("batch_size", 2))
    init_lambda = float(params.get("init_lambda", 0.3))
    split = str(params.get("split", "validation"))
    merge_mode = context.merge_mode
    normalize_coefficients = bool(params.get("normalize_coefficients", False))
    if normalize_coefficients:
        print(
            "[AdaMerging] normalize_coefficients=true applies cross-task sum-to-1 normalization. "
            "This is not part of the AdaMerging paper objective and may force near-uniform coefficients."
        )
    use_autocast = bool(params.get("use_autocast", True))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", True))
    allow_tf32 = bool(params.get("allow_tf32", True))
    force_cpu = bool(params.get("force_cpu", False))
    merge_impl = str(params.get("merge_impl", "streaming_parametrize")).strip().lower()
    delta_residency = str(params.get("delta_residency", "cpu_stream")).strip().lower()
    dtype_compute = str(params.get("dtype_compute", "auto")).strip().lower()
    task_vector_dtype = str(params.get("task_vector_dtype", "auto")).strip().lower()
    model_dtype = str(params.get("model_dtype", "auto")).strip().lower()
    empty_cache_interval = int(params.get("empty_cache_interval", 0))
    logging_steps = int(params.get("logging_steps", params.get("log_every", 0)))
    entropy_rolling_window = int(params.get("entropy_rolling_window", 10))
    entropy_temperature = float(params.get("entropy_temperature", 1.0))
    entropy_chunk_size = int(params.get("entropy_chunk_size", 4096))
    entropy_debug_stats = bool(params.get("entropy_debug_stats", False))
    require_valid_label_mask = bool(params.get("require_valid_label_mask", True))
    log_alpha_grad_stats = bool(params.get("log_alpha_grad_stats", True))
    log_label_token_stats = bool(params.get("log_label_token_stats", True))
    entropy_debug_steps = int(
        params.get("entropy_debug_steps", logging_steps if logging_steps > 0 else 0)
    )
    progress_bar = bool(params.get("progress_bar", True))
    dataloader_num_workers = int(params.get("dataloader_num_workers", 0))
    dataloader_pin_memory = bool(params.get("dataloader_pin_memory", True))
    non_blocking_transfer = bool(params.get("non_blocking_transfer", True))
    gradient_accumulation_steps = int(
        params.get("gradient_accumulation_steps", params.get("grad_accum_steps", 1))
    )
    early_stopping_patience = int(params.get("early_stopping_patience", 0))
    early_stopping_threshold = float(
        params.get("early_stopping_threshold", params.get("early_stopping_min_delta", 0.0))
    )
    eval_subset = params.get("eval_subset")
    if eval_subset is not None and not isinstance(eval_subset, Mapping):
        raise ValueError("optimizer.params.eval_subset must be a mapping when provided.")

    if steps <= 0:
        raise ValueError("optimizer.params.steps must be > 0.")
    if batch_size <= 0:
        raise ValueError("optimizer.params.batch_size must be > 0.")
    if empty_cache_interval < 0:
        raise ValueError("optimizer.params.empty_cache_interval must be >= 0.")
    if logging_steps < 0:
        raise ValueError("optimizer.params.logging_steps must be >= 0.")
    if entropy_rolling_window <= 0:
        raise ValueError("optimizer.params.entropy_rolling_window must be > 0.")
    if entropy_temperature <= 0.0:
        raise ValueError("optimizer.params.entropy_temperature must be > 0.")
    if entropy_chunk_size < 0:
        raise ValueError("optimizer.params.entropy_chunk_size must be >= 0.")
    if entropy_debug_steps < 0:
        raise ValueError("optimizer.params.entropy_debug_steps must be >= 0.")
    if dataloader_num_workers < 0:
        raise ValueError("optimizer.params.dataloader_num_workers must be >= 0.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("optimizer.params.gradient_accumulation_steps must be > 0.")
    if early_stopping_patience < 0:
        raise ValueError("optimizer.params.early_stopping_patience must be >= 0.")
    if early_stopping_threshold < 0.0:
        raise ValueError("optimizer.params.early_stopping_threshold must be >= 0.")
    if merge_impl not in {"streaming_parametrize", "functional_clone_legacy", "fused_linear"}:
        raise ValueError(
            "optimizer.params.merge_impl must be one of: "
            "streaming_parametrize|functional_clone_legacy|fused_linear."
        )
    if delta_residency not in {"cpu_stream", "gpu_cache"}:
        raise ValueError("optimizer.params.delta_residency must be one of: cpu_stream|gpu_cache.")
    if dtype_compute not in {"auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        raise ValueError(
            "optimizer.params.dtype_compute must be one of: auto|bf16|bfloat16|fp16|float16|fp32|float32."
        )
    if task_vector_dtype not in {"auto", "none", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        raise ValueError(
            "optimizer.params.task_vector_dtype must be one of: "
            "auto|none|bf16|bfloat16|fp16|float16|fp32|float32."
        )
    if model_dtype not in {"auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        raise ValueError(
            "optimizer.params.model_dtype must be one of: auto|bf16|bfloat16|fp16|float16|fp32|float32."
        )

    tasks = _resolve_eval_tasks(context, params)
    task_vectors = [extract_task_vector_from_lora(path) for path in context.adapter_paths]
    num_adapters = len(task_vectors)

    t0 = time.time()
    first_task_module = get_task_module(tasks[0])
    first_config_path = first_task_module.get_config_path(PACKAGE_ROOT, None)
    first_config = load_config(first_config_path)
    first_artifacts = first_task_module.get_artifact_directories(PACKAGE_ROOT)
    first_config = _prepare_dataset_cache(first_config, first_artifacts)
    model_path = _get_model_path(first_config, tasks[0])
    if force_cpu:
        model, processor = load_model_and_processor(
            model_path,
            adapter_path=None,
            delta_weights=None,
            device_map_override={"": "cpu"},
            torch_dtype_override=torch.float32,
        )
    else:
        if torch.cuda.is_available():
            try:
                torch.backends.cuda.matmul.allow_tf32 = allow_tf32
                torch.backends.cudnn.allow_tf32 = allow_tf32
            except Exception:
                pass
        model_torch_dtype = _resolve_dtype(model_dtype)
        model, processor = load_model_and_processor(
            model_path,
            adapter_path=None,
            delta_weights=None,
            torch_dtype_override=model_torch_dtype,
        )
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable()
        except Exception:
            pass

    loaders = {
        task: _build_task_loader(
            task,
            processor=processor,
            split=split,
            batch_size=batch_size,
            eval_subset=eval_subset if isinstance(eval_subset, Mapping) else None,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
        for task in tasks
    }

    # Load only one base model (source tasks share the same base checkpoint in this pipeline).
    for p in model.parameters():
        p.requires_grad_(False)
    model.eval()

    device = _infer_input_device(model)
    parameter_keys = [name for name, _ in model.named_parameters()]
    base_params = {name: p for name, p in model.named_parameters()}
    delta_entries = _build_delta_entries(task_vectors, parameter_keys=parameter_keys, merge_mode=merge_mode)

    if task_vector_dtype in {"auto", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        cast_dtype: Optional[torch.dtype] = None
        if task_vector_dtype == "auto":
            cast_dtype = torch.bfloat16 if device.type == "cuda" else None
        elif task_vector_dtype in {"bf16", "bfloat16"}:
            cast_dtype = torch.bfloat16
        elif task_vector_dtype in {"fp16", "float16"}:
            cast_dtype = torch.float16
        elif task_vector_dtype in {"fp32", "float32"}:
            cast_dtype = torch.float32
        if cast_dtype is not None:
            for entry in delta_entries:
                for i, delta in enumerate(entry.deltas):
                    entry.deltas[i] = delta.to(dtype=cast_dtype, copy=False)

    layer_indices = sorted({entry.layer_idx for entry in delta_entries if entry.layer_idx is not None})
    if variant == "layer_wise" and not layer_indices:
        raise ValueError("Layer-wise AdaMerging requested but no `.layers.<idx>.` keys were detected.")

    init_logit = _logit(init_lambda)
    alpha_task = torch.nn.Parameter(torch.full((num_adapters,), init_logit, device=device))
    alpha_default: Optional[torch.nn.Parameter] = None
    alpha_layer: Optional[torch.nn.Parameter] = None
    optim_params: List[torch.nn.Parameter] = [alpha_task]
    if variant == "layer_wise":
        alpha_default = torch.nn.Parameter(torch.full((num_adapters,), init_logit, device=device))
        alpha_layer = torch.nn.Parameter(
            torch.full((len(layer_indices), num_adapters), init_logit, device=device)
        )
        optim_params.extend([alpha_default, alpha_layer])

    optimizer = torch.optim.Adam(optim_params, lr=lr, betas=betas)
    task_iters: Dict[str, Optional[Iterator[Any]]] = {task: None for task in tasks}
    initial_entropy: Optional[float] = None
    final_entropy: Optional[float] = None
    best_entropy: Optional[float] = None
    no_improve_steps = 0
    steps_completed = 0
    optimizer_steps_completed = 0
    update_entropy_sum = 0.0
    update_entropy_count = 0
    update_entropy_min_sum = 0.0
    update_entropy_max_sum = 0.0
    update_pmax_sum = 0.0
    update_stats_count = 0
    update_label_token_sum = 0.0
    update_label_token_count = 0
    update_label_token_by_task_sum: Dict[str, float] = {task: 0.0 for task in tasks}
    update_label_token_by_task_count: Dict[str, int] = {task: 0 for task in tasks}
    global_label_token_by_task_sum: Dict[str, float] = {task: 0.0 for task in tasks}
    global_label_token_by_task_count: Dict[str, int] = {task: 0 for task in tasks}
    alpha_task_grad_norm_sum = 0.0
    alpha_task_grad_norm_count = 0
    alpha_task_grad_norm_max: Optional[float] = None
    alpha_task_grad_norm_last: Optional[float] = None
    alpha_task_grad_dim_abs_sum = torch.zeros((num_adapters,), dtype=torch.float64)
    alpha_task_grad_dim_abs_count = 0
    alpha_task_grad_dim_last: Optional[List[float]] = None
    update_entropy_rolling: List[float] = []
    rolling_entropy: Optional[float] = None
    early_stopping_monitor = (
        f"rolling_entropy_{entropy_rolling_window}"
        if entropy_rolling_window > 1
        else "avg_entropy_update"
    )
    streaming_handles = []
    fused_linear_handles = []
    coefficient_provider = TaskCoefficientProvider()
    # Parametrization registration performs an immediate forward pass; seed
    # coefficients upfront so streaming mode is valid before the first step.
    init_task_coeffs, init_default_coeffs, init_layer_coeffs = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        layer_indices=layer_indices,
        normalize_coefficients=normalize_coefficients,
    )
    coefficient_provider.set_coefficients(
        task_coeffs=init_task_coeffs,
        default_coeffs=init_default_coeffs,
        layer_coeffs=init_layer_coeffs,
    )
    if merge_impl == "streaming_parametrize":
        streaming_handles = register_streaming_parametrizations(
            model=model,
            entries=_to_streaming_entries(delta_entries),
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
    elif merge_impl == "fused_linear":
        fused_linear_handles = register_fused_linear_modules(
            model=model,
            entries=_to_streaming_entries(delta_entries),
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
    try:
        step_iter: Iterable[int]
        if progress_bar and tqdm is not None:
            step_iter = tqdm(range(steps), desc="AdaMerging", total=steps)
        else:
            step_iter = range(steps)
        optimizer.zero_grad(set_to_none=True)
        for step in step_iter:

            num_tasks = len(tasks)
            if num_tasks == 0:
                raise RuntimeError("AdaMerging optimization has no tasks to optimize.")

            entropy_sum = 0.0
            step_entropy_min_sum = 0.0
            step_entropy_max_sum = 0.0
            step_pmax_sum = 0.0
            step_stats_count = 0
            active_task = "<none>"
            try:
                # Keep per-task backward to release each task graph immediately.
                for task in tasks:
                    active_task = task
                    task_coeffs, default_coeffs, layer_coeffs = _merge_coeffs(
                        alpha_task=alpha_task,
                        alpha_default=alpha_default,
                        alpha_layer=alpha_layer,
                        layer_indices=layer_indices,
                        normalize_coefficients=normalize_coefficients,
                    )
                    if merge_impl == "functional_clone_legacy":
                        func_params = _build_functional_params(
                            base_params=base_params,
                            delta_entries=delta_entries,
                            task_coeffs=task_coeffs,
                            default_coeffs=default_coeffs,
                            layer_coeffs=layer_coeffs,
                        )
                    else:
                        coefficient_provider.set_coefficients(
                            task_coeffs=task_coeffs,
                            default_coeffs=default_coeffs,
                            layer_coeffs=layer_coeffs,
                        )
                        func_params = None
                    batch, task_iters[task] = _next_batch(loaders[task], task_iters[task])
                    batch = _to_device(batch, device, non_blocking=(non_blocking_transfer and device.type == "cuda"))
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=(use_autocast and device.type == "cuda"),
                    ):
                        outputs = (
                            _functional_forward(model, func_params, batch)
                            if func_params is not None
                            else model(**batch)
                        )
                        logits = _extract_logits(outputs)
                        labels = batch.get("labels") if isinstance(batch, Mapping) else None
                        collect_entropy_stats = bool(entropy_debug_stats or log_label_token_stats)
                        if collect_entropy_stats:
                            task_loss, task_stats = _compute_entropy_loss(
                                logits,
                                labels,
                                entropy_temperature=entropy_temperature,
                                entropy_chunk_size=entropy_chunk_size,
                                require_valid_label_mask=require_valid_label_mask,
                                return_stats=True,
                            )
                            if entropy_debug_stats:
                                step_entropy_min_sum += float(task_stats["entropy_min"])
                                step_entropy_max_sum += float(task_stats["entropy_max"])
                                step_pmax_sum += float(task_stats["pmax_mean"])
                                step_stats_count += 1
                            if log_label_token_stats:
                                valid_tokens = int(task_stats.get("valid_label_tokens", 0))
                                update_label_token_sum += float(valid_tokens)
                                update_label_token_count += 1
                                update_label_token_by_task_sum[task] += float(valid_tokens)
                                update_label_token_by_task_count[task] += 1
                                global_label_token_by_task_sum[task] += float(valid_tokens)
                                global_label_token_by_task_count[task] += 1
                        else:
                            task_loss = _compute_entropy_loss(
                                logits,
                                labels,
                                entropy_temperature=entropy_temperature,
                                entropy_chunk_size=entropy_chunk_size,
                                require_valid_label_mask=require_valid_label_mask,
                            )
                    entropy_sum += float(task_loss.detach().item())
                    (task_loss / float(num_tasks * gradient_accumulation_steps)).backward()
                    del func_params, batch, outputs, logits, labels, task_loss
            except ValueError as exc:
                raise ValueError(
                    f"AdaMerging entropy objective failed at step={step}, task='{active_task}': {exc}"
                ) from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "AdaMerging OOM during entropy optimization "
                        f"(step={step}, task='{active_task}', merge_impl='{merge_impl}', "
                        f"delta_residency='{delta_residency}', batch_size={batch_size}, force_cpu={force_cpu}). "
                        "Try smaller optimizer.params.batch_size (e.g., 1-2), "
                        "fewer optimizer.params.tasks, and/or optimizer.params.eval_subset.max_samples. "
                        "Set optimizer.params.merge_impl='streaming_parametrize' and "
                        "optimizer.params.delta_residency='cpu_stream' for lower VRAM."
                    ) from exc
                raise

            avg_entropy = entropy_sum / float(num_tasks)
            if initial_entropy is None:
                initial_entropy = avg_entropy
            update_entropy_sum += avg_entropy
            update_entropy_count += 1
            if entropy_debug_stats and step_stats_count > 0:
                update_entropy_min_sum += step_entropy_min_sum / float(step_stats_count)
                update_entropy_max_sum += step_entropy_max_sum / float(step_stats_count)
                update_pmax_sum += step_pmax_sum / float(step_stats_count)
                update_stats_count += 1
            should_step = (((step + 1) % gradient_accumulation_steps) == 0) or ((step + 1) == steps)
            if should_step:
                grad_norm_value: Optional[float] = None
                grad_dim_values: Optional[List[float]] = None
                if log_alpha_grad_stats and alpha_task.grad is not None:
                    grad = alpha_task.grad.detach().float()
                    grad_norm_value = float(grad.norm(p=2).item())
                    grad_dim_values = [float(x) for x in grad.tolist()]
                    alpha_task_grad_norm_sum += grad_norm_value
                    alpha_task_grad_norm_count += 1
                    alpha_task_grad_norm_last = grad_norm_value
                    if alpha_task_grad_norm_max is None or grad_norm_value > alpha_task_grad_norm_max:
                        alpha_task_grad_norm_max = grad_norm_value
                    alpha_task_grad_dim_abs_sum += grad.abs().to(dtype=torch.float64).cpu()
                    alpha_task_grad_dim_abs_count += 1
                    alpha_task_grad_dim_last = grad_dim_values

                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps_completed += 1
                update_entropy = update_entropy_sum / float(max(1, update_entropy_count))
                update_entropy_rolling.append(update_entropy)
                if len(update_entropy_rolling) > entropy_rolling_window:
                    update_entropy_rolling.pop(0)
                rolling_entropy = sum(update_entropy_rolling) / float(len(update_entropy_rolling))
                debug_entropy_min = (
                    update_entropy_min_sum / float(update_stats_count)
                    if (entropy_debug_stats and update_stats_count > 0)
                    else None
                )
                debug_entropy_max = (
                    update_entropy_max_sum / float(update_stats_count)
                    if (entropy_debug_stats and update_stats_count > 0)
                    else None
                )
                debug_pmax = (
                    update_pmax_sum / float(update_stats_count)
                    if (entropy_debug_stats and update_stats_count > 0)
                    else None
                )
                update_entropy_sum = 0.0
                update_entropy_count = 0
                update_entropy_min_sum = 0.0
                update_entropy_max_sum = 0.0
                update_pmax_sum = 0.0
                update_stats_count = 0
                label_tokens_avg = (
                    update_label_token_sum / float(max(1, update_label_token_count))
                    if log_label_token_stats
                    else None
                )
                label_tokens_by_task_avg = {
                    task: (
                        update_label_token_by_task_sum[task] / float(max(1, update_label_token_by_task_count[task]))
                        if update_label_token_by_task_count[task] > 0
                        else 0.0
                    )
                    for task in tasks
                }
                update_label_token_sum = 0.0
                update_label_token_count = 0
                update_label_token_by_task_sum = {task: 0.0 for task in tasks}
                update_label_token_by_task_count = {task: 0 for task in tasks}
                monitor_entropy = rolling_entropy if rolling_entropy is not None else update_entropy
                if best_entropy is None or ((best_entropy - monitor_entropy) > early_stopping_threshold):
                    best_entropy = monitor_entropy
                    no_improve_steps = 0
                else:
                    no_improve_steps += 1
                if logging_steps > 0 and (
                    (optimizer_steps_completed % logging_steps) == 0 or (step + 1) == steps
                ):
                    print(
                        f"[AdaMerging] step {step + 1}/{steps} "
                        f"update_step={optimizer_steps_completed} "
                        f"avg_entropy={update_entropy:.6f} "
                        f"rolling_entropy_{entropy_rolling_window}={rolling_entropy:.6f}"
                    )
                    if log_alpha_grad_stats and grad_norm_value is not None:
                        print(
                            "[AdaMerging][grad] "
                            f"alpha_task_grad_norm={grad_norm_value:.6e} "
                            f"alpha_task_grad={grad_dim_values}"
                        )
                    if log_label_token_stats and label_tokens_avg is not None:
                        print(
                            "[AdaMerging][label_tokens] "
                            f"avg_valid_tokens={label_tokens_avg:.3f} "
                            f"per_task={{{', '.join(f'{k}: {v:.3f}' for k, v in label_tokens_by_task_avg.items())}}}"
                        )
                if (
                    entropy_debug_stats
                    and entropy_debug_steps > 0
                    and debug_entropy_min is not None
                    and debug_entropy_max is not None
                    and debug_pmax is not None
                    and (
                        (optimizer_steps_completed % entropy_debug_steps) == 0
                        or (step + 1) == steps
                    )
                ):
                    print(
                        "[AdaMerging][entropy_debug] "
                        f"entropy_min={debug_entropy_min:.6f} "
                        f"entropy_max={debug_entropy_max:.6f} "
                        f"pmax_mean={debug_pmax:.6f}"
                    )
                if early_stopping_patience > 0 and no_improve_steps >= early_stopping_patience:
                    print(
                        f"[AdaMerging] early stopping at step {step + 1}/{steps} "
                        f"(update_step={optimizer_steps_completed}, monitor={early_stopping_monitor}, "
                        f"best_entropy={best_entropy:.6f}, "
                        f"patience={early_stopping_patience}, threshold={early_stopping_threshold:.2e})"
                    )
                    final_entropy = update_entropy
                    steps_completed = step + 1
                    break
            if empty_cache_interval > 0 and device.type == "cuda" and ((step + 1) % empty_cache_interval == 0):
                torch.cuda.empty_cache()
            final_entropy = avg_entropy
            steps_completed = step + 1
            if tqdm is not None and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix({"entropy": f"{avg_entropy:.4f}"})
    finally:
        if streaming_handles:
            unregister_streaming_parametrizations(streaming_handles)
        if fused_linear_handles:
            unregister_fused_linear_modules(fused_linear_handles)

    task_out, default_out, layer_out = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        layer_indices=layer_indices,
        normalize_coefficients=normalize_coefficients,
    )
    task_coefficients = [float(x) for x in task_out.detach().cpu().tolist()]
    alpha_task_grad_norm_mean = (
        alpha_task_grad_norm_sum / float(alpha_task_grad_norm_count)
        if alpha_task_grad_norm_count > 0
        else None
    )
    alpha_task_grad_dim_abs_mean = (
        (alpha_task_grad_dim_abs_sum / float(alpha_task_grad_dim_abs_count)).tolist()
        if alpha_task_grad_dim_abs_count > 0
        else None
    )
    avg_valid_label_tokens_by_task = {
        task: (
            global_label_token_by_task_sum[task] / float(max(1, global_label_token_by_task_count[task]))
            if global_label_token_by_task_count[task] > 0
            else 0.0
        )
        for task in tasks
    }

    method_params_overrides: Dict[str, Any] = {
        "normalize_coefficients": normalize_coefficients,
    }
    if variant == "task_wise":
        method_params_overrides["task_coefficients"] = task_coefficients
    else:
        method_params_overrides["task_coefficients"] = task_coefficients
        method_params_overrides["default_task_coefficients"] = [
            float(x) for x in default_out.detach().cpu().tolist()
        ] if default_out is not None else task_coefficients
        method_params_overrides["layer_task_coefficients"] = {
            int(layer): [float(x) for x in coeff.detach().cpu().tolist()]
            for layer, coeff in layer_out.items()
        }

    provenance = {
        "optimizer": "adamerging",
        "status": "optimized",
        "variant": variant,
        "task_scope": task_scope,
        "tasks": tasks,
        "split": split,
        "steps": steps,
        "batch_size": batch_size,
        "lr": lr,
        "betas": [betas[0], betas[1]],
        "init_lambda": init_lambda,
        "normalize_coefficients": normalize_coefficients,
        "gradient_checkpointing": gradient_checkpointing,
        "allow_tf32": allow_tf32,
        "force_cpu": force_cpu,
        "merge_impl": merge_impl,
        "delta_residency": delta_residency,
        "dtype_compute": dtype_compute,
        "actual_device_mode": "cpu" if device.type == "cpu" else "gpu",
        "task_vector_dtype": task_vector_dtype,
        "model_dtype": model_dtype,
        "empty_cache_interval": empty_cache_interval,
        "logging_steps": logging_steps,
        "entropy_rolling_window": entropy_rolling_window,
        "early_stopping_monitor": early_stopping_monitor,
        "entropy_temperature": entropy_temperature,
        "entropy_chunk_size": entropy_chunk_size,
        "entropy_debug_stats": entropy_debug_stats,
        "entropy_debug_steps": entropy_debug_steps,
        "require_valid_label_mask": require_valid_label_mask,
        "log_alpha_grad_stats": log_alpha_grad_stats,
        "log_label_token_stats": log_label_token_stats,
        "progress_bar": progress_bar,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "non_blocking_transfer": non_blocking_transfer,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_threshold": early_stopping_threshold,
        "objective": "entropy_minimization",
        "initial_entropy": initial_entropy,
        "final_entropy": final_entropy,
        "best_entropy": best_entropy,
        "final_rolling_entropy": rolling_entropy,
        "steps_completed": steps_completed,
        "optimizer_steps_completed": optimizer_steps_completed,
        "elapsed_sec": time.time() - t0,
        "num_adapters": num_adapters,
        "num_delta_entries": len(delta_entries),
        "initial_task_coefficients": [float(init_lambda)] * num_adapters,
        "final_task_coefficients": task_coefficients,
        "alpha_task_grad_norm_mean": alpha_task_grad_norm_mean,
        "alpha_task_grad_norm_max": alpha_task_grad_norm_max,
        "alpha_task_grad_norm_last": alpha_task_grad_norm_last,
        "alpha_task_grad_dim_abs_mean": alpha_task_grad_dim_abs_mean,
        "alpha_task_grad_dim_last": alpha_task_grad_dim_last,
        "avg_valid_label_tokens_by_task": avg_valid_label_tokens_by_task,
        "label_token_batches_by_task": dict(global_label_token_by_task_count),
        "eval_subset": dict(eval_subset) if isinstance(eval_subset, Mapping) else None,
    }
    if variant == "layer_wise":
        provenance["num_layers"] = len(layer_indices)
        provenance["final_default_task_coefficients"] = method_params_overrides.get("default_task_coefficients")
        provenance["final_layer_task_coefficients"] = method_params_overrides.get("layer_task_coefficients")

    return OptimizerResult(
        lambda_policy=spec.lambda_policy,
        provenance=provenance,
        method_params_overrides=method_params_overrides,
    )
