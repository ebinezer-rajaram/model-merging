"""Exact SuperMerge optimizer engine (supervised, tanh-signed coefficients)."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional

import torch

from core import load_config
from core.evaluation.eval_utils import load_model_and_processor
from experiments.evaluate_task import _get_model_path, _prepare_dataset_cache
from experiments.extract_vector import extract_task_vector_from_lora
from merging.config.specs import MergeSpec
from merging.optimizers.core.streaming import (
    TaskCoefficientProvider,
    register_fused_linear_modules,
    register_fused_lora_linear_modules,
    register_streaming_parametrizations,
    unregister_fused_linear_modules,
    unregister_fused_lora_linear_modules,
    unregister_streaming_parametrizations,
)
from merging.optimizers.core.common import (
    _apply_ties_consensus_preprocess,
    _build_delta_entries,
    _build_lora_entries,
    _build_functional_params,
    _build_task_loader,
    _extract_logits,
    _force_zero_dropout,
    _functional_forward,
    _infer_input_device,
    _merge_coeffs,
    _next_batch,
    _project_coefficients_inplace,
    _resolve_dtype,
    _to_device,
    _to_streaming_lora_entries,
    _to_streaming_entries,
)
from merging.optimizers.registry import OptimizerContext, OptimizerResult
from merging.optimizers.core.shared_training import (
    _compute_single_task_ce_baselines,
    _compute_supervised_ce_loss,
    _compute_task_ce_multipliers,
    _extract_model_loss,
    _validate_selection_split,
)
from merging.runtime.utils import PACKAGE_ROOT, get_task_module

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


_PLUS_PLUS_VARIANTS = {"task_wise_plus_plus", "layer_wise_plus_plus", "plusplus", "plus_plus"}


def _atanh_stable(x: float) -> float:
    clipped = float(min(1.0 - 1e-6, max(-1.0 + 1e-6, x)))
    return 0.5 * math.log((1.0 + clipped) / (1.0 - clipped))


def _resolve_supermerge_tasks(context: OptimizerContext, params: Mapping[str, Any]) -> List[str]:
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
    deduped: List[str] = []
    seen = set()
    for task_name in tasks:
        if task_name in seen:
            continue
        seen.add(task_name)
        deduped.append(task_name)
    if not deduped:
        raise ValueError(
            "SuperMerge requires at least one task in optimizer.params.tasks "
            "or source adapter metadata."
        )
    return deduped


def _validate_hierarchical_stub(params: Mapping[str, Any]) -> None:
    hierarchical = params.get("hierarchical")
    if hierarchical is None:
        return
    if not isinstance(hierarchical, Mapping):
        raise ValueError("optimizer.params.hierarchical must be a mapping when provided.")
    enabled = bool(hierarchical.get("enabled", False))
    if enabled:
        raise ValueError(
            "optimizer.params.hierarchical.enabled=true is not implemented yet. "
            "Hierarchy config keys are accepted as a stub for future support."
        )


def run_supermerge_optimizer(spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
    params = dict(spec.optimizer.params if spec.optimizer is not None else {})
    _validate_hierarchical_stub(params)
    variant = str(params.get("variant", "task_wise")).strip().lower()
    plus_plus = bool(params.get("plus_plus", False) or (variant in _PLUS_PLUS_VARIANTS))
    if variant in {"task_wise_plus_plus", "plusplus", "plus_plus"}:
        variant = "task_wise"
    elif variant == "layer_wise_plus_plus":
        variant = "layer_wise"
    if variant not in {"task_wise", "layer_wise"}:
        raise ValueError(f"Unsupported optimizer.params.variant='{variant}'.")
    if context.method != "weighted_delta_n":
        raise ValueError("supermerge optimizer only supports method='weighted_delta_n'.")
    task_scope = str(params.get("task_scope", "all")).strip().lower()

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
            "[supermerge] normalize_coefficients=true applies cross-task sum-to-1 normalization. "
            "This may force near-uniform coefficients."
        )
    use_autocast = bool(params.get("use_autocast", True))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", True))
    zero_dropout = bool(params.get("zero_dropout", True))
    allow_tf32 = bool(params.get("allow_tf32", True))
    force_cpu = bool(params.get("force_cpu", False))
    merge_impl = str(params.get("merge_impl", "streaming_parametrize")).strip().lower()
    delta_residency = str(params.get("delta_residency", "cpu_stream")).strip().lower()
    dtype_compute = str(params.get("dtype_compute", "auto")).strip().lower()
    task_vector_dtype = str(params.get("task_vector_dtype", "auto")).strip().lower()
    model_dtype = str(params.get("model_dtype", "auto")).strip().lower()
    empty_cache_interval = int(params.get("empty_cache_interval", 0))
    logging_steps = int(params.get("logging_steps", params.get("log_every", 0)))
    rolling_window = int(params.get("entropy_rolling_window", 10))
    log_alpha_grad_stats = bool(params.get("log_alpha_grad_stats", True))
    log_label_token_stats = bool(params.get("log_label_token_stats", True))
    coefficient_parameterization = str(
        params.get("coefficient_parameterization", "tanh_alpha")
    ).strip().lower()
    magnitude_scope = str(params.get("magnitude_scope", "per_layer")).strip().lower()
    magnitude_parameterization = str(params.get("magnitude_parameterization", "linear")).strip().lower()
    magnitude_init = float(params.get("magnitude_init", 1.0))
    if normalize_coefficients and coefficient_parameterization == "simplex_softmax_scaled":
        print(
            "[supermerge] coefficient_parameterization='simplex_softmax_scaled' ignores "
            "normalize_coefficients to preserve learned magnitudes."
        )
    project_coefficients = bool(params.get("project_coefficients", True))
    coefficient_min = float(params.get("coefficient_min", -1.0))
    coefficient_max = float(params.get("coefficient_max", 1.0))
    progress_bar = bool(params.get("progress_bar", True))
    log_coefficients_during_training = bool(params.get("log_coefficients_during_training", True))
    log_layer_coefficients_full = bool(params.get("log_layer_coefficients_full", False))
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
    restore_best_checkpoint = bool(params.get("restore_best_checkpoint", False))
    min_optimizer_steps_before_early_stop = int(params.get("min_optimizer_steps_before_early_stop", 20))
    early_stopping_warmup_steps = int(params.get("early_stopping_warmup_steps", 20))
    eval_subset = params.get("eval_subset")
    if eval_subset is not None and not isinstance(eval_subset, Mapping):
        raise ValueError("optimizer.params.eval_subset must be a mapping when provided.")
    sampling = params.get("sampling")
    if sampling is not None and not isinstance(sampling, Mapping):
        raise ValueError("optimizer.params.sampling must be a mapping when provided.")

    ce_ignore_index = int(params.get("ce_ignore_index", -100))
    ce_label_smoothing = float(params.get("ce_label_smoothing", 0.0))
    ce_reduction = str(params.get("ce_reduction", "mean")).strip().lower()
    ce_use_model_loss = bool(params.get("ce_use_model_loss", True))
    ce_task_weighting = str(params.get("ce_task_weighting", "equal")).strip().lower()
    ce_baseline_source = str(params.get("ce_baseline_source", "single_task_eval")).strip().lower()
    ce_baseline_floor = float(params.get("ce_baseline_floor", 1e-6))
    ce_baseline_batches = int(params.get("ce_baseline_batches", 32))
    ce_manual_task_weights = params.get("ce_manual_task_weights")
    ce_multiplier_cap_raw = params.get("ce_multiplier_cap")
    ce_multiplier_cap: Optional[float]
    if ce_multiplier_cap_raw is None:
        ce_multiplier_cap = None
    else:
        ce_multiplier_cap = float(ce_multiplier_cap_raw)
    enforce_validation_only_selection = bool(params.get("enforce_validation_only_selection", True))
    if ce_label_smoothing < 0.0 or ce_label_smoothing >= 1.0:
        raise ValueError("optimizer.params.ce_label_smoothing must be in [0.0, 1.0).")
    if ce_reduction not in {"mean", "sum"}:
        raise ValueError("optimizer.params.ce_reduction must be one of: mean|sum.")
    if ce_task_weighting not in {"baseline_normalized", "equal", "manual"}:
        raise ValueError(
            "optimizer.params.ce_task_weighting must be one of: baseline_normalized|equal|manual."
        )
    if ce_baseline_source not in {"single_task_eval"}:
        raise ValueError("optimizer.params.ce_baseline_source must be one of: single_task_eval.")
    if ce_baseline_floor <= 0.0:
        raise ValueError("optimizer.params.ce_baseline_floor must be > 0.")
    if ce_baseline_batches <= 0:
        raise ValueError("optimizer.params.ce_baseline_batches must be > 0.")
    if ce_multiplier_cap is not None and ce_multiplier_cap <= 0.0:
        raise ValueError("optimizer.params.ce_multiplier_cap must be > 0 when provided.")

    ignored_entropy_keys = [
        key
        for key in params.keys()
        if key.startswith("entropy_") and key != "entropy_rolling_window"
    ]
    if ignored_entropy_keys:
        print(
            "[supermerge] ignoring entropy-only optimizer params: "
            + ", ".join(sorted(ignored_entropy_keys))
        )

    if steps <= 0:
        raise ValueError("optimizer.params.steps must be > 0.")
    if batch_size <= 0:
        raise ValueError("optimizer.params.batch_size must be > 0.")
    if empty_cache_interval < 0:
        raise ValueError("optimizer.params.empty_cache_interval must be >= 0.")
    if logging_steps < 0:
        raise ValueError("optimizer.params.logging_steps must be >= 0.")
    if rolling_window <= 0:
        raise ValueError("optimizer.params.entropy_rolling_window must be > 0.")
    if dataloader_num_workers < 0:
        raise ValueError("optimizer.params.dataloader_num_workers must be >= 0.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("optimizer.params.gradient_accumulation_steps must be > 0.")
    if early_stopping_patience < 0:
        raise ValueError("optimizer.params.early_stopping_patience must be >= 0.")
    if early_stopping_threshold < 0.0:
        raise ValueError("optimizer.params.early_stopping_threshold must be >= 0.")
    if min_optimizer_steps_before_early_stop < 0:
        raise ValueError("optimizer.params.min_optimizer_steps_before_early_stop must be >= 0.")
    if early_stopping_warmup_steps < 0:
        raise ValueError("optimizer.params.early_stopping_warmup_steps must be >= 0.")
    if coefficient_parameterization not in {"tanh_alpha", "simplex_softmax_scaled"}:
        raise ValueError(
            "supermerge exact mode requires "
            "optimizer.params.coefficient_parameterization in "
            "{'tanh_alpha', 'simplex_softmax_scaled'}."
        )
    if coefficient_parameterization == "simplex_softmax_scaled":
        if magnitude_parameterization != "linear":
            raise ValueError(
                "optimizer.params.magnitude_parameterization must be 'linear' "
                "when optimizer.params.coefficient_parameterization='simplex_softmax_scaled'."
            )
        if magnitude_scope not in {"per_layer", "global", "per_task"}:
            raise ValueError(
                "optimizer.params.magnitude_scope must be one of: per_layer|global|per_task "
                "when optimizer.params.coefficient_parameterization='simplex_softmax_scaled'."
            )
    if coefficient_min > coefficient_max:
        raise ValueError("optimizer.params.coefficient_min must be <= optimizer.params.coefficient_max.")
    if merge_impl not in {"streaming_parametrize", "functional_clone_legacy", "fused_linear", "fused_lora_linear"}:
        raise ValueError(
            "optimizer.params.merge_impl must be one of: "
            "streaming_parametrize|functional_clone_legacy|fused_linear|fused_lora_linear."
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

    tasks = _resolve_supermerge_tasks(context, params)
    _validate_selection_split(split=split, enforce_validation_only_selection=enforce_validation_only_selection)
    from merging.transforms.registry import apply_transforms
    if merge_impl == "fused_lora_linear":
        if plus_plus:
            raise ValueError("optimizer.params.plus_plus is not supported with merge_impl='fused_lora_linear'.")
        task_vectors = []
    else:
        task_vectors = [
            apply_transforms(extract_task_vector_from_lora(path), spec.transforms)
            for path in context.adapter_paths
        ]
    plus_plus_k = float(params.get("plus_plus_k", params.get("ties_k", 20.0)))
    plus_plus_lambda = float(params.get("plus_plus_lambda", params.get("ties_lambda", 1.0)))
    plus_plus_stats: Optional[Dict[str, Any]] = None
    if plus_plus:
        if not 0.0 <= plus_plus_k <= 100.0:
            raise ValueError("optimizer.params.plus_plus_k must be in [0, 100].")
        task_vectors, plus_plus_stats = _apply_ties_consensus_preprocess(
            task_vectors,
            merge_mode=merge_mode,
            k_percent=plus_plus_k,
            lambda_scale=plus_plus_lambda,
        )
    num_adapters = len(context.adapter_paths)

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
            sampling=sampling if isinstance(sampling, Mapping) else None,
            num_workers=dataloader_num_workers,
            pin_memory=dataloader_pin_memory,
        )
        for task in tasks
    }

    for p in model.parameters():
        p.requires_grad_(False)
    if gradient_checkpointing:
        model.train()
    else:
        model.eval()
    dropout_stats = _force_zero_dropout(model) if zero_dropout else None
    if dropout_stats is not None:
        print(
            "[supermerge] zero_dropout enabled: "
            f"modules_updated={dropout_stats['dropout_modules_updated']}/"
            f"{dropout_stats['dropout_modules_seen']}, "
            f"config_fields_updated={dropout_stats['dropout_config_fields_updated']}"
        )

    device = _infer_input_device(model)
    parameter_keys = [name for name, _ in model.named_parameters()]
    base_params = {name: p for name, p in model.named_parameters()}
    delta_entries = []
    lora_entries = []
    if merge_impl == "fused_lora_linear":
        lora_entries = _build_lora_entries(context.adapter_paths, parameter_keys=parameter_keys, merge_mode=merge_mode)
    else:
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

    active_entries = lora_entries if merge_impl == "fused_lora_linear" else delta_entries
    layer_indices = sorted({entry.layer_idx for entry in active_entries if entry.layer_idx is not None})
    if variant == "layer_wise" and not layer_indices:
        raise ValueError("Layer-wise supermerge optimizer requested but no `.layers.<idx>.` keys were detected.")

    if coefficient_parameterization == "tanh_alpha":
        init_value = _atanh_stable(init_lambda)
    elif coefficient_parameterization == "simplex_softmax_scaled":
        # Start from uniform simplex logits.
        init_value = 0.0
    else:
        init_value = init_lambda
    alpha_default: Optional[torch.nn.Parameter] = None
    alpha_layer: Optional[torch.nn.Parameter] = None
    magnitude_task: Optional[torch.nn.Parameter] = None
    magnitude_default: Optional[torch.nn.Parameter] = None
    magnitude_layer: Optional[torch.nn.Parameter] = None
    if variant == "layer_wise":
        # In layer-wise mode, task coefficients act as the global default anchor.
        # Share storage so task/default remain consistent and gradients are observable.
        alpha_default = torch.nn.Parameter(torch.full((num_adapters,), init_value, device=device))
        alpha_task = alpha_default
        alpha_layer = torch.nn.Parameter(
            torch.full((len(layer_indices), num_adapters), init_value, device=device)
        )
        optim_params: List[torch.nn.Parameter] = [alpha_default, alpha_layer]
    else:
        alpha_task = torch.nn.Parameter(torch.full((num_adapters,), init_value, device=device))
        optim_params = [alpha_task]

    if coefficient_parameterization == "simplex_softmax_scaled":
        if variant == "layer_wise":
            if magnitude_scope == "per_task":
                magnitude_default = torch.nn.Parameter(
                    torch.full((num_adapters,), magnitude_init, device=device)
                )
                magnitude_layer = torch.nn.Parameter(
                    torch.full((len(layer_indices), num_adapters), magnitude_init, device=device)
                )
            elif magnitude_scope == "global":
                magnitude_default = torch.nn.Parameter(torch.full((1,), magnitude_init, device=device))
                magnitude_layer = None
            else:
                # per_layer: one scalar per layer plus a scalar default fallback.
                magnitude_default = torch.nn.Parameter(torch.full((1,), magnitude_init, device=device))
                magnitude_layer = torch.nn.Parameter(
                    torch.full((len(layer_indices), 1), magnitude_init, device=device)
                )
            magnitude_task = magnitude_default
            if magnitude_default is not None:
                optim_params.append(magnitude_default)
            if magnitude_layer is not None:
                optim_params.append(magnitude_layer)
        else:
            if magnitude_scope == "per_task":
                magnitude_task = torch.nn.Parameter(torch.full((num_adapters,), magnitude_init, device=device))
            else:
                magnitude_task = torch.nn.Parameter(torch.full((1,), magnitude_init, device=device))
            optim_params.append(magnitude_task)

    optimizer = torch.optim.Adam(optim_params, lr=lr, betas=betas)
    task_iters: Dict[str, Optional[Iterator[Any]]] = {task: None for task in tasks}
    initial_ce: Optional[float] = None
    initial_normalized_ce: Optional[float] = None
    final_ce: Optional[float] = None
    final_normalized_ce: Optional[float] = None
    best_selected_ce: Optional[float] = None
    best_selected_update_step: Optional[int] = None
    best_ce: Optional[float] = None
    no_improve_steps = 0
    steps_completed = 0
    optimizer_steps_completed = 0
    update_raw_ce_sum = 0.0
    update_raw_ce_count = 0
    update_normalized_ce_sum = 0.0
    update_normalized_ce_count = 0
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
    update_ce_rolling: List[float] = []
    rolling_ce: Optional[float] = None
    best_alpha_task: Optional[torch.Tensor] = None
    best_alpha_default: Optional[torch.Tensor] = None
    best_alpha_layer: Optional[torch.Tensor] = None
    best_magnitude_task: Optional[torch.Tensor] = None
    best_magnitude_default: Optional[torch.Tensor] = None
    best_magnitude_layer: Optional[torch.Tensor] = None
    early_stopping_monitor = f"rolling_ce_{rolling_window}" if rolling_window > 1 else "avg_ce_update"
    streaming_handles = []
    fused_linear_handles = []
    fused_lora_handles = []
    coefficient_provider = TaskCoefficientProvider()
    init_task_coeffs, init_default_coeffs, init_layer_coeffs = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        magnitude_task=magnitude_task,
        magnitude_default=magnitude_default,
        magnitude_layer=magnitude_layer,
        layer_indices=layer_indices,
        coefficient_parameterization=coefficient_parameterization,
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
    elif merge_impl == "fused_lora_linear":
        fused_lora_handles = register_fused_lora_linear_modules(
            model=model,
            entries=_to_streaming_lora_entries(lora_entries),
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
    try:
        ce_baselines: Optional[Dict[str, float]] = None
        if ce_task_weighting == "baseline_normalized":
            if ce_baseline_source != "single_task_eval":
                raise ValueError(
                    "optimizer.params.ce_baseline_source must be single_task_eval "
                    "when optimizer.params.ce_task_weighting=baseline_normalized."
                )
            ce_baselines = _compute_single_task_ce_baselines(
                tasks=tasks,
                num_adapters=num_adapters,
                alpha_task=alpha_task,
                alpha_default=alpha_default,
                alpha_layer=alpha_layer,
                magnitude_task=magnitude_task,
                magnitude_default=magnitude_default,
                magnitude_layer=magnitude_layer,
                layer_indices=layer_indices,
                coefficient_parameterization=coefficient_parameterization,
                normalize_coefficients=normalize_coefficients,
                coefficient_provider=coefficient_provider,
                merge_impl=merge_impl,
                model=model,
                loaders=loaders,
                task_iters=task_iters,
                device=device,
                use_autocast=use_autocast,
                non_blocking_transfer=non_blocking_transfer,
                ce_use_model_loss=ce_use_model_loss,
                ce_ignore_index=ce_ignore_index,
                ce_label_smoothing=ce_label_smoothing,
                ce_reduction=ce_reduction,
                ce_baseline_batches=ce_baseline_batches,
            )
            print(
                "[supermerge] CE baselines (single_task_eval): "
                + ", ".join(f"{k}={v:.6f}" for k, v in ce_baselines.items())
            )
        ce_multipliers = _compute_task_ce_multipliers(
            tasks=tasks,
            ce_task_weighting=ce_task_weighting,
            ce_baselines=ce_baselines,
            ce_baseline_floor=ce_baseline_floor,
            ce_manual_task_weights=ce_manual_task_weights if isinstance(ce_manual_task_weights, Mapping) else None,
            ce_multiplier_cap=ce_multiplier_cap,
        )
        step_iter: Iterable[int]
        if progress_bar and tqdm is not None:
            step_iter = tqdm(range(steps), desc="supermerge", total=steps)
        else:
            step_iter = range(steps)
        optimizer.zero_grad(set_to_none=True)
        for step in step_iter:
            num_tasks = len(tasks)
            if num_tasks == 0:
                raise RuntimeError("supermerge optimization has no tasks to optimize.")

            ce_sum = 0.0
            normalized_ce_sum = 0.0
            raw_ce_by_task_sum: Dict[str, float] = {task: 0.0 for task in tasks}
            norm_ce_by_task_sum: Dict[str, float] = {task: 0.0 for task in tasks}
            active_task = "<none>"
            try:
                for task in tasks:
                    active_task = task
                    task_coeffs, default_coeffs, layer_coeffs = _merge_coeffs(
                        alpha_task=alpha_task,
                        alpha_default=alpha_default,
                        alpha_layer=alpha_layer,
                        magnitude_task=magnitude_task,
                        magnitude_default=magnitude_default,
                        magnitude_layer=magnitude_layer,
                        layer_indices=layer_indices,
                        coefficient_parameterization=coefficient_parameterization,
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
                        model_loss = _extract_model_loss(outputs) if ce_use_model_loss else None
                        collect_stats = bool(log_label_token_stats)
                        if model_loss is not None:
                            task_loss = model_loss.float()
                            valid_tokens = 0
                            if labels is not None:
                                valid_tokens = int((labels != ce_ignore_index).sum().detach().item())
                            if collect_stats:
                                task_stats = {
                                    "valid_label_tokens": valid_tokens,
                                    "num_ce_positions": valid_tokens,
                                }
                            else:
                                task_stats = None
                        elif collect_stats:
                            task_loss, task_stats = _compute_supervised_ce_loss(
                                logits,
                                labels,
                                ignore_index=ce_ignore_index,
                                label_smoothing=ce_label_smoothing,
                                reduction=ce_reduction,
                                return_stats=True,
                            )
                            valid_tokens = int(task_stats.get("valid_label_tokens", 0))
                            update_label_token_sum += float(valid_tokens)
                            update_label_token_count += 1
                            update_label_token_by_task_sum[task] += float(valid_tokens)
                            update_label_token_by_task_count[task] += 1
                            global_label_token_by_task_sum[task] += float(valid_tokens)
                            global_label_token_by_task_count[task] += 1
                        else:
                            task_loss = _compute_supervised_ce_loss(
                                logits,
                                labels,
                                ignore_index=ce_ignore_index,
                                label_smoothing=ce_label_smoothing,
                                reduction=ce_reduction,
                            )
                        if collect_stats and model_loss is not None:
                            update_label_token_sum += float(valid_tokens)
                            update_label_token_count += 1
                            update_label_token_by_task_sum[task] += float(valid_tokens)
                            update_label_token_by_task_count[task] += 1
                            global_label_token_by_task_sum[task] += float(valid_tokens)
                            global_label_token_by_task_count[task] += 1
                    ce_sum += float(task_loss.detach().item())
                    raw_ce_by_task_sum[task] += float(task_loss.detach().item())
                    task_multiplier = float(ce_multipliers[task])
                    weighted_task_loss = task_loss * task_multiplier
                    normalized_ce_sum += float(weighted_task_loss.detach().item()) * float(num_tasks)
                    norm_ce_by_task_sum[task] += float(weighted_task_loss.detach().item()) * float(num_tasks)
                    (weighted_task_loss / float(gradient_accumulation_steps)).backward()
                    del func_params, batch, outputs, logits, labels, task_loss
            except ValueError as exc:
                raise ValueError(
                    f"supermerge supervised objective failed at step={step}, task='{active_task}': {exc}"
                ) from exc
            except RuntimeError as exc:
                if "out of memory" in str(exc).lower():
                    raise RuntimeError(
                        "supermerge optimizer OOM during CE optimization "
                        f"(step={step}, task='{active_task}', merge_impl='{merge_impl}', "
                        f"delta_residency='{delta_residency}', batch_size={batch_size}, force_cpu={force_cpu}). "
                        "Try smaller optimizer.params.batch_size (e.g., 1-2), "
                        "fewer optimizer.params.tasks, and/or optimizer.params.eval_subset.max_samples. "
                        "Set optimizer.params.merge_impl='streaming_parametrize' and "
                        "optimizer.params.delta_residency='cpu_stream' for lower VRAM."
                    ) from exc
                raise

            avg_ce = ce_sum / float(num_tasks)
            avg_normalized_ce = normalized_ce_sum / float(num_tasks)
            if initial_ce is None:
                initial_ce = avg_ce
            if initial_normalized_ce is None:
                initial_normalized_ce = avg_normalized_ce
            if best_selected_ce is None:
                best_selected_ce = avg_normalized_ce
            update_raw_ce_sum += avg_ce
            update_raw_ce_count += 1
            update_normalized_ce_sum += avg_normalized_ce
            update_normalized_ce_count += 1
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
                if project_coefficients:
                    _project_coefficients_inplace(
                        alpha_task=alpha_task,
                        alpha_default=alpha_default,
                        alpha_layer=alpha_layer,
                        coefficient_parameterization=coefficient_parameterization,
                        coeff_min=coefficient_min,
                        coeff_max=coefficient_max,
                    )
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps_completed += 1
                update_raw_ce = update_raw_ce_sum / float(max(1, update_raw_ce_count))
                update_normalized_ce = update_normalized_ce_sum / float(max(1, update_normalized_ce_count))
                update_ce_rolling.append(update_normalized_ce)
                if len(update_ce_rolling) > rolling_window:
                    update_ce_rolling.pop(0)
                rolling_ce = sum(update_ce_rolling) / float(len(update_ce_rolling))
                monitor_ce = rolling_ce if rolling_ce is not None else update_normalized_ce
                in_warmup = optimizer_steps_completed < early_stopping_warmup_steps
                if best_ce is None or ((best_ce - monitor_ce) > early_stopping_threshold):
                    best_ce = monitor_ce
                    best_selected_ce = monitor_ce
                    best_selected_update_step = optimizer_steps_completed
                    if restore_best_checkpoint:
                        best_alpha_task = alpha_task.detach().clone()
                        if alpha_default is not None:
                            best_alpha_default = alpha_default.detach().clone()
                        if alpha_layer is not None:
                            best_alpha_layer = alpha_layer.detach().clone()
                        if magnitude_task is not None:
                            best_magnitude_task = magnitude_task.detach().clone()
                        if magnitude_default is not None:
                            best_magnitude_default = magnitude_default.detach().clone()
                        if magnitude_layer is not None:
                            best_magnitude_layer = magnitude_layer.detach().clone()
                    if not in_warmup:
                        no_improve_steps = 0
                elif not in_warmup:
                    no_improve_steps += 1

                update_raw_ce_sum = 0.0
                update_raw_ce_count = 0
                update_normalized_ce_sum = 0.0
                update_normalized_ce_count = 0
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
                if logging_steps > 0 and (
                    (optimizer_steps_completed % logging_steps) == 0 or (step + 1) == steps
                ):
                    raw_ce_by_task = {
                        task: raw_ce_by_task_sum[task]
                        for task in tasks
                    }
                    normalized_ce_by_task = {
                        task: norm_ce_by_task_sum[task]
                        for task in tasks
                    }
                    print(
                        f"[supermerge] step {step + 1}/{steps} "
                        f"update_step={optimizer_steps_completed} "
                        f"avg_ce={update_raw_ce:.6f} "
                        f"avg_normalized_ce={update_normalized_ce:.6f} "
                        f"rolling_ce_{rolling_window}={rolling_ce:.6f}"
                    )
                    print(
                        "[supermerge][ce] "
                        f"raw_by_task={raw_ce_by_task} normalized_by_task={normalized_ce_by_task}"
                    )
                    if log_alpha_grad_stats and grad_norm_value is not None:
                        print(
                            "[supermerge][grad] "
                            f"alpha_task_grad_norm={grad_norm_value:.6e} "
                            f"alpha_task_grad={grad_dim_values}"
                        )
                    if log_label_token_stats and label_tokens_avg is not None:
                        print(
                            "[supermerge][label_tokens] "
                            f"avg_valid_tokens={label_tokens_avg:.3f} "
                            f"per_task={{{', '.join(f'{k}: {v:.3f}' for k, v in label_tokens_by_task_avg.items())}}}"
                        )
                    if log_coefficients_during_training:
                        coeff_task, coeff_default, coeff_layers = _merge_coeffs(
                            alpha_task=alpha_task,
                            alpha_default=alpha_default,
                            alpha_layer=alpha_layer,
                            magnitude_task=magnitude_task,
                            magnitude_default=magnitude_default,
                            magnitude_layer=magnitude_layer,
                            layer_indices=layer_indices,
                            coefficient_parameterization=coefficient_parameterization,
                            normalize_coefficients=normalize_coefficients,
                        )
                        print(
                            "[supermerge][lambda] "
                            f"task={ [float(x) for x in coeff_task.detach().cpu().tolist()] }"
                        )
                        if coeff_default is not None:
                            print(
                                "[supermerge][lambda] "
                                f"default={ [float(x) for x in coeff_default.detach().cpu().tolist()] }"
                            )
                        if coeff_layers:
                            if log_layer_coefficients_full:
                                layer_payload = {
                                    int(layer): [float(x) for x in vec.detach().cpu().tolist()]
                                    for layer, vec in coeff_layers.items()
                                }
                                print(f"[supermerge][lambda] layer={layer_payload}")
                            else:
                                print(
                                    "[supermerge][lambda] "
                                    f"num_layer_overrides={len(coeff_layers)} "
                                    "(set optimizer.params.log_layer_coefficients_full=true to print all)"
                                )
                if early_stopping_patience > 0 and no_improve_steps >= early_stopping_patience:
                    if optimizer_steps_completed >= min_optimizer_steps_before_early_stop:
                        print(
                            f"[supermerge] early stopping at step {step + 1}/{steps} "
                            f"(update_step={optimizer_steps_completed}, monitor={early_stopping_monitor}, "
                            f"best_ce={best_ce:.6f}, "
                            f"patience={early_stopping_patience}, threshold={early_stopping_threshold:.2e}, "
                            f"warmup={early_stopping_warmup_steps}, "
                            f"min_optimizer_steps_before_early_stop={min_optimizer_steps_before_early_stop})"
                        )
                        final_ce = update_raw_ce
                        final_normalized_ce = update_normalized_ce
                        steps_completed = step + 1
                        break
            if empty_cache_interval > 0 and device.type == "cuda" and ((step + 1) % empty_cache_interval == 0):
                torch.cuda.empty_cache()
            final_ce = avg_ce
            final_normalized_ce = avg_normalized_ce
            steps_completed = step + 1
            if tqdm is not None and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix({"ce": f"{avg_ce:.4f}"})
    finally:
        if streaming_handles:
            unregister_streaming_parametrizations(streaming_handles)
        if fused_linear_handles:
            unregister_fused_linear_modules(fused_linear_handles)
        if fused_lora_handles:
            unregister_fused_lora_linear_modules(fused_lora_handles)

    last_task, last_default, last_layer = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        magnitude_task=magnitude_task,
        magnitude_default=magnitude_default,
        magnitude_layer=magnitude_layer,
        layer_indices=layer_indices,
        coefficient_parameterization=coefficient_parameterization,
        normalize_coefficients=normalize_coefficients,
    )
    last_task_coefficients = [float(x) for x in last_task.detach().cpu().tolist()]
    restored_best_checkpoint = False
    if restore_best_checkpoint and best_alpha_task is not None:
        with torch.no_grad():
            alpha_task.copy_(best_alpha_task)
            if alpha_default is not None and best_alpha_default is not None:
                alpha_default.copy_(best_alpha_default)
            if alpha_layer is not None and best_alpha_layer is not None:
                alpha_layer.copy_(best_alpha_layer)
            if magnitude_task is not None and best_magnitude_task is not None:
                magnitude_task.copy_(best_magnitude_task)
            if magnitude_default is not None and best_magnitude_default is not None:
                magnitude_default.copy_(best_magnitude_default)
            if magnitude_layer is not None and best_magnitude_layer is not None:
                magnitude_layer.copy_(best_magnitude_layer)
        restored_best_checkpoint = True

    task_out, default_out, layer_out = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        magnitude_task=magnitude_task,
        magnitude_default=magnitude_default,
        magnitude_layer=magnitude_layer,
        layer_indices=layer_indices,
        coefficient_parameterization=coefficient_parameterization,
        normalize_coefficients=normalize_coefficients,
    )
    def _to_floats_checked(name: str, tensor: torch.Tensor) -> List[float]:
        if not bool(torch.isfinite(tensor).all().item()):
            raise ValueError(f"supermerge produced non-finite coefficients in '{name}'.")
        return [float(x) for x in tensor.detach().cpu().tolist()]

    task_coefficients = _to_floats_checked("task_coefficients", task_out)
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
        "allow_negative_coefficients": True,
    }
    if coefficient_parameterization == "simplex_softmax_scaled" and magnitude_parameterization == "linear":
        method_params_overrides["allow_unbounded_coefficients"] = True
    if variant == "task_wise":
        method_params_overrides["task_coefficients"] = task_coefficients
    else:
        method_params_overrides["task_coefficients"] = task_coefficients
        method_params_overrides["default_task_coefficients"] = [
            float(x) for x in _to_floats_checked("default_task_coefficients", default_out)
        ] if default_out is not None else task_coefficients
        method_params_overrides["layer_task_coefficients"] = {
            int(layer): _to_floats_checked(f"layer_task_coefficients[{int(layer)}]", coeff)
            for layer, coeff in layer_out.items()
        }

    provenance = {
        "optimizer": "supermerge",
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
        "zero_dropout": zero_dropout,
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
        "entropy_rolling_window": rolling_window,
        "early_stopping_monitor": early_stopping_monitor,
        "log_alpha_grad_stats": log_alpha_grad_stats,
        "log_label_token_stats": log_label_token_stats,
        "coefficient_parameterization": coefficient_parameterization,
        "magnitude_scope": magnitude_scope,
        "magnitude_parameterization": magnitude_parameterization,
        "magnitude_init": magnitude_init,
        "project_coefficients": project_coefficients,
        "coefficient_min": coefficient_min,
        "coefficient_max": coefficient_max,
        "progress_bar": progress_bar,
        "log_coefficients_during_training": log_coefficients_during_training,
        "log_layer_coefficients_full": log_layer_coefficients_full,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "non_blocking_transfer": non_blocking_transfer,
        "gradient_accumulation_steps": gradient_accumulation_steps,
        "early_stopping_patience": early_stopping_patience,
        "early_stopping_threshold": early_stopping_threshold,
        "min_optimizer_steps_before_early_stop": min_optimizer_steps_before_early_stop,
        "early_stopping_warmup_steps": early_stopping_warmup_steps,
        "restore_best_checkpoint": restore_best_checkpoint,
        "restored_best_checkpoint": restored_best_checkpoint,
        "objective": "supervised_cross_entropy",
        "exact_mode": True,
        "plus_plus": plus_plus,
        "plus_plus_k": plus_plus_k,
        "plus_plus_lambda": plus_plus_lambda,
        "plus_plus_stats": plus_plus_stats,
        "ce_ignore_index": ce_ignore_index,
        "ce_label_smoothing": ce_label_smoothing,
        "ce_reduction": ce_reduction,
        "ce_use_model_loss": ce_use_model_loss,
        "ce_task_weighting": ce_task_weighting,
        "ce_baseline_source": ce_baseline_source,
        "ce_baseline_floor": ce_baseline_floor,
        "ce_baseline_batches": ce_baseline_batches,
        "ce_multiplier_cap": ce_multiplier_cap,
        "ce_manual_task_weights": dict(ce_manual_task_weights) if isinstance(ce_manual_task_weights, Mapping) else None,
        "enforce_validation_only_selection": enforce_validation_only_selection,
        "ce_baselines": ce_baselines if 'ce_baselines' in locals() else None,
        "ce_task_multipliers": ce_multipliers if 'ce_multipliers' in locals() else None,
        "ignored_entropy_params": sorted(ignored_entropy_keys),
        "initial_ce": initial_ce,
        "initial_normalized_ce": initial_normalized_ce,
        "final_ce": final_ce,
        "final_normalized_ce": final_normalized_ce,
        "best_ce": best_ce,
        "best_selected_ce": best_selected_ce,
        "best_selected_update_step": best_selected_update_step,
        "final_rolling_ce": rolling_ce,
        "steps_completed": steps_completed,
        "optimizer_steps_completed": optimizer_steps_completed,
        "elapsed_sec": time.time() - t0,
        "num_adapters": num_adapters,
        "num_delta_entries": len(delta_entries),
        "num_lora_entries": len(lora_entries),
        "initial_task_coefficients": [float(x) for x in init_task_coeffs.detach().cpu().tolist()],
        "final_task_coefficients": task_coefficients,
        "last_step_task_coefficients": last_task_coefficients,
        "alpha_task_grad_norm_mean": alpha_task_grad_norm_mean,
        "alpha_task_grad_norm_max": alpha_task_grad_norm_max,
        "alpha_task_grad_norm_last": alpha_task_grad_norm_last,
        "alpha_task_grad_dim_abs_mean": alpha_task_grad_dim_abs_mean,
        "alpha_task_grad_dim_last": alpha_task_grad_dim_last,
        "avg_valid_label_tokens_by_task": avg_valid_label_tokens_by_task,
        "label_token_batches_by_task": dict(global_label_token_by_task_count),
        "eval_subset": dict(eval_subset) if isinstance(eval_subset, Mapping) else None,
    }
    if magnitude_task is not None:
        provenance["final_task_magnitude"] = _to_floats_checked("final_task_magnitude", magnitude_task)
    if magnitude_default is not None:
        provenance["final_default_magnitude"] = _to_floats_checked("final_default_magnitude", magnitude_default)
    if magnitude_layer is not None:
        provenance["final_layer_magnitude"] = {
            int(layer): _to_floats_checked(f"final_layer_magnitude[{int(layer)}]", magnitude_layer[i])
            for i, layer in enumerate(layer_indices)
        }
    if variant == "layer_wise":
        provenance["num_layers"] = len(layer_indices)
        provenance["final_default_task_coefficients"] = method_params_overrides.get("default_task_coefficients")
        provenance["final_layer_task_coefficients"] = method_params_overrides.get("layer_task_coefficients")

    return OptimizerResult(
        lambda_policy=context.lambda_policy,
        provenance=provenance,
        method_params_overrides=method_params_overrides,
    )


__all__ = [
    "run_supermerge_optimizer",
    "_compute_supervised_ce_loss",
    "_compute_task_ce_multipliers",
    "_validate_selection_split",
    "_validate_hierarchical_stub",
]
