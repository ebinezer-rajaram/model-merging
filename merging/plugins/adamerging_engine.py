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
from merging.plugins.optimizers import OptimizerContext, OptimizerResult
from merging.policies.lambda_policy import extract_layer_index
from merging.runtime.utils import get_task_module, PACKAGE_ROOT


_CLASSIFICATION_TASKS = {
    "emotion",
    "intent",
    "speaker_id",
    "kws",
    "langid",
    "speaker_ver",
}

_PLUS_PLUS_VARIANTS = {"task_wise_plus_plus", "layer_wise_plus_plus", "plusplus", "plus_plus"}


@dataclass(frozen=True)
class _DeltaEntry:
    task_key: str
    param_key: str
    layer_idx: Optional[int]
    deltas: List[torch.Tensor]


def _logit(p: float) -> float:
    p = float(min(1.0 - 1e-6, max(1e-6, p)))
    return math.log(p / (1.0 - p))


def _to_device(batch: Any, device: torch.device) -> Any:
    if hasattr(batch, "to") and callable(getattr(batch, "to")) and not isinstance(batch, torch.Tensor):
        try:
            return batch.to(device)
        except Exception:
            pass
    if isinstance(batch, torch.Tensor):
        return batch.to(device)
    if isinstance(batch, dict):
        return {k: _to_device(v, device) for k, v in batch.items()}
    if isinstance(batch, list):
        return [_to_device(x, device) for x in batch]
    if isinstance(batch, tuple):
        return tuple(_to_device(x, device) for x in batch)
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
) -> DataLoader:
    task_module = get_task_module(task)
    config_path = task_module.get_config_path(PACKAGE_ROOT, None)
    config = load_config(config_path)
    artifact_dirs = task_module.get_artifact_directories(PACKAGE_ROOT)
    config = _prepare_dataset_cache(config, artifact_dirs)
    setup = prepare_task_for_evaluation(task, processor, split=split, config=config)
    setup.dataset = _maybe_subset_dataset(setup.dataset, task=task, eval_subset=eval_subset)
    loader = DataLoader(
        setup.dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=0,
        collate_fn=setup.data_collator,
    )
    return loader


def _extract_logits(outputs: Any) -> torch.Tensor:
    if isinstance(outputs, Mapping):
        logits = outputs.get("logits")
    else:
        logits = getattr(outputs, "logits", None)
    if logits is None:
        raise ValueError("Model output does not contain logits required for entropy optimization.")
    return logits


def _compute_entropy_loss(logits: torch.Tensor, labels: Optional[torch.Tensor]) -> torch.Tensor:
    probs = torch.softmax(logits.float(), dim=-1)
    entropy = -(probs * probs.clamp_min(1e-8).log()).sum(dim=-1)
    if labels is not None and logits.ndim >= 3 and labels.shape[:2] == logits.shape[:2]:
        mask = labels != -100
        if mask.any():
            entropy = entropy[mask]
    return entropy.mean()


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
    normalize_coefficients = bool(params.get("normalize_coefficients", True))
    use_autocast = bool(params.get("use_autocast", True))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", True))
    force_cpu = bool(params.get("force_cpu", False))
    task_vector_dtype = str(params.get("task_vector_dtype", "auto")).strip().lower()
    empty_cache_interval = int(params.get("empty_cache_interval", 0))
    eval_subset = params.get("eval_subset")
    if eval_subset is not None and not isinstance(eval_subset, Mapping):
        raise ValueError("optimizer.params.eval_subset must be a mapping when provided.")

    if steps <= 0:
        raise ValueError("optimizer.params.steps must be > 0.")
    if batch_size <= 0:
        raise ValueError("optimizer.params.batch_size must be > 0.")
    if empty_cache_interval < 0:
        raise ValueError("optimizer.params.empty_cache_interval must be >= 0.")
    if task_vector_dtype not in {"auto", "none", "bf16", "bfloat16", "fp16", "float16", "fp32", "float32"}:
        raise ValueError(
            "optimizer.params.task_vector_dtype must be one of: "
            "auto|none|bf16|bfloat16|fp16|float16|fp32|float32."
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
    model, processor = load_model_and_processor(model_path, adapter_path=None, delta_weights=None)
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if force_cpu:
        try:
            model = model.to("cpu")
        except Exception:
            # If accelerate/device_map wrappers block `.to`, continue with inferred device.
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

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        num_tasks = len(tasks)
        if num_tasks == 0:
            raise RuntimeError("AdaMerging optimization has no tasks to optimize.")

        entropy_sum = 0.0
        try:
            # Rebuild merged functional params per task so each backward pass can
            # free its graph immediately (prevents VRAM growth with many tasks).
            for task in tasks:
                task_coeffs, default_coeffs, layer_coeffs = _merge_coeffs(
                    alpha_task=alpha_task,
                    alpha_default=alpha_default,
                    alpha_layer=alpha_layer,
                    layer_indices=layer_indices,
                    normalize_coefficients=normalize_coefficients,
                )
                func_params = _build_functional_params(
                    base_params=base_params,
                    delta_entries=delta_entries,
                    task_coeffs=task_coeffs,
                    default_coeffs=default_coeffs,
                    layer_coeffs=layer_coeffs,
                )
                batch, task_iters[task] = _next_batch(loaders[task], task_iters[task])
                batch = _to_device(batch, device)
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(use_autocast and device.type == "cuda"),
                ):
                    outputs = _functional_forward(model, func_params, batch)
                    logits = _extract_logits(outputs)
                    labels = batch.get("labels") if isinstance(batch, Mapping) else None
                    task_loss = _compute_entropy_loss(logits, labels)
                entropy_sum += float(task_loss.detach().item())
                (task_loss / float(num_tasks)).backward()
                del func_params, batch, outputs, logits, labels, task_loss
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower():
                raise RuntimeError(
                    "AdaMerging OOM during entropy optimization. "
                    "Try smaller optimizer.params.batch_size (e.g., 1-2), "
                    "fewer optimizer.params.tasks, and/or optimizer.params.eval_subset.max_samples."
                ) from exc
            raise

        avg_entropy = entropy_sum / float(num_tasks)
        if initial_entropy is None:
            initial_entropy = avg_entropy
        optimizer.step()
        if empty_cache_interval > 0 and device.type == "cuda" and ((step + 1) % empty_cache_interval == 0):
            torch.cuda.empty_cache()
        final_entropy = avg_entropy

    task_out, default_out, layer_out = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        layer_indices=layer_indices,
        normalize_coefficients=normalize_coefficients,
    )
    task_coefficients = [float(x) for x in task_out.detach().cpu().tolist()]
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
        "force_cpu": force_cpu,
        "task_vector_dtype": task_vector_dtype,
        "empty_cache_interval": empty_cache_interval,
        "objective": "entropy_minimization",
        "initial_entropy": initial_entropy,
        "final_entropy": final_entropy,
        "elapsed_sec": time.time() - t0,
        "num_adapters": num_adapters,
        "num_delta_entries": len(delta_entries),
        "initial_task_coefficients": [float(init_lambda)] * num_adapters,
        "final_task_coefficients": task_coefficients,
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
