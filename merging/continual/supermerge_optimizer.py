"""Differentiable two-source optimizer for continual alpha/lambda merges."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch

from core import load_config, set_global_seed
from core.evaluation.eval_utils import load_model_and_processor
from core.evaluation.evaluate_task import get_model_path, prepare_dataset_cache
from merging.config.unified import MergeConfig
from merging.continual.engine import continual_merge_sources_to_artifact
from merging.continual.evaluate import build_continual_tag, evaluate_continual_artifact
from merging.continual.policy import ContinualMergePolicy
from merging.evaluation.continual_sweep import (
    CONTINUAL_SUPERMERGE_METHOD,
    ContinualSweepContext,
)
from merging.optimizers.core.common import (
    _DeltaEntry,
    _LoraEntry,
    _build_task_loader,
    _extract_logits,
    _force_zero_dropout,
    _infer_input_device,
    _next_batch,
    _resolve_delta_param_name,
    _resolve_dtype,
    _to_device,
    _to_streaming_entries,
    _to_streaming_lora_entries,
)
from merging.optimizers.core.heldout_selection import (
    PeriodicHeldoutEvaluator,
    resolve_heldout_eval_config,
)
from merging.optimizers.core.shared_training import (
    _compute_supervised_ce_loss,
    _compute_task_ce_multipliers,
    _extract_model_loss,
    _validate_selection_split,
)
from merging.optimizers.core.streaming import (
    TaskCoefficientProvider,
    register_fused_lora_linear_modules,
    register_streaming_parametrizations,
    unregister_fused_lora_linear_modules,
    unregister_streaming_parametrizations,
)
from merging.policies.lambda_policy import extract_layer_index
from merging.runtime.utils import PACKAGE_ROOT, get_task_module

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


@dataclass(frozen=True)
class ContinualScalarParameters:
    alpha: float
    lambda_weight: float


def _logit(p: float) -> float:
    p = min(1.0 - 1e-6, max(1e-6, float(p)))
    return math.log(p / (1.0 - p))


def _ordered_seen_tasks(context: ContinualSweepContext, params: Mapping[str, Any]) -> List[str]:
    explicit = params.get("tasks")
    if explicit is not None:
        if not isinstance(explicit, list):
            raise ValueError("optimizer.params.tasks must be a list when provided.")
        tasks = [str(task).strip().lower() for task in explicit if str(task).strip()]
    else:
        tasks = [str(task).strip().lower() for task in context.seen_tasks if str(task).strip()]
    if not tasks:
        raise ValueError(
            "continual_supermerge could not infer optimization tasks from seen continual sources. "
            "Set optimizer.params.tasks explicitly."
        )
    seen = set(str(task).strip().lower() for task in context.seen_tasks)
    extras = [task for task in tasks if task not in seen]
    if seen and extras and bool(params.get("enforce_seen_tasks_only", True)):
        raise ValueError(
            "continual_supermerge optimizer trains on seen tasks only; unexpected task(s): "
            + ", ".join(sorted(extras))
        )
    return tasks


def _build_source_delta_entries(
    *,
    context: ContinualSweepContext,
    parameter_keys: List[str],
    merge_mode: str,
    dtype: torch.dtype,
) -> List[_DeltaEntry]:
    x_specs = {spec.source_key: spec for spec in context.x_source.list_target_params()}
    y_specs = {spec.source_key: spec for spec in context.y_source.list_target_params()}
    if merge_mode == "strict" and set(x_specs) != set(y_specs):
        raise ValueError("continual_supermerge strict mode requires identical source parameter keys.")
    keys_to_merge = set(x_specs).intersection(y_specs)

    entries: List[_DeltaEntry] = []
    for key in sorted(keys_to_merge):
        param_key = _resolve_delta_param_name(key, parameter_keys)
        if param_key is None:
            continue
        x_delta = context.x_source.materialize_dense_param_delta(key, dtype=dtype, device=torch.device("cpu"))
        y_delta = context.y_source.materialize_dense_param_delta(key, dtype=dtype, device=torch.device("cpu"))
        if tuple(x_delta.shape) != tuple(y_delta.shape):
            if merge_mode == "strict":
                raise ValueError(
                    f"continual_supermerge strict mode shape mismatch for key '{key}': "
                    f"{tuple(x_delta.shape)} vs {tuple(y_delta.shape)}"
                )
            continue
        entries.append(
            _DeltaEntry(
                task_key=key,
                param_key=param_key,
                layer_idx=None,
                deltas=[x_delta, y_delta],
            )
        )
    if not entries:
        raise ValueError("continual_supermerge could not map any continual deltas to model parameters.")
    return entries


def _build_source_lora_entries(
    *,
    context: ContinualSweepContext,
    parameter_keys: List[str],
    merge_mode: str,
) -> List[_LoraEntry]:
    x_specs = {spec.source_key: spec for spec in context.x_source.list_target_params()}
    y_specs = {spec.source_key: spec for spec in context.y_source.list_target_params()}
    if merge_mode == "strict" and set(x_specs) != set(y_specs):
        raise ValueError("continual_supermerge strict fused_lora_linear mode requires identical source keys.")
    keys_to_merge = set(x_specs).intersection(y_specs)

    missing_factor_api: List[str] = []
    entries: List[_LoraEntry] = []
    for key in sorted(keys_to_merge):
        param_key = _resolve_delta_param_name(key, parameter_keys)
        if param_key is None:
            continue
        if not hasattr(context.x_source, "get_factor_tensors") or not hasattr(context.y_source, "get_factor_tensors"):
            missing_factor_api.append(key)
            continue
        x_a, x_b, x_scale = context.x_source.get_factor_tensors(key)  # type: ignore[attr-defined]
        y_a, y_b, y_scale = context.y_source.get_factor_tensors(key)  # type: ignore[attr-defined]
        if x_a.ndim != 2 or x_b.ndim != 2 or y_a.ndim != 2 or y_b.ndim != 2:
            raise ValueError(f"continual_supermerge fused_lora_linear factors must be 2D for key '{key}'.")
        if x_b.shape[0] != y_b.shape[0] or x_a.shape[1] != y_a.shape[1]:
            if merge_mode == "strict":
                raise ValueError(
                    f"continual_supermerge fused_lora_linear shape mismatch for key '{key}': "
                    f"x A={tuple(x_a.shape)} B={tuple(x_b.shape)}; "
                    f"y A={tuple(y_a.shape)} B={tuple(y_b.shape)}"
                )
            continue
        entries.append(
            _LoraEntry(
                task_key=key,
                param_key=param_key,
                layer_idx=extract_layer_index(key),
                a_factors=[x_a, y_a],
                b_factors=[x_b, y_b],
                scales=[float(x_scale), float(y_scale)],
            )
        )
    if missing_factor_api:
        raise ValueError(
            "continual_supermerge fused_lora_linear requires factorized sources; "
            f"{len(missing_factor_api)} key(s) lacked get_factor_tensors."
        )
    if not entries:
        raise ValueError("continual_supermerge fused_lora_linear could not build any low-rank entries.")
    return entries


def _scalar_values(alpha_raw: torch.Tensor, lambda_raw: torch.Tensor) -> ContinualScalarParameters:
    return ContinualScalarParameters(
        alpha=float(torch.nn.functional.softplus(alpha_raw).detach().item()),
        lambda_weight=float(torch.sigmoid(lambda_raw).detach().item()),
    )


def _coefficients(alpha_raw: torch.Tensor, lambda_raw: torch.Tensor) -> torch.Tensor:
    alpha = torch.nn.functional.softplus(alpha_raw)
    lambda_weight = torch.sigmoid(lambda_raw)
    return torch.stack([alpha * lambda_weight, alpha * (1.0 - lambda_weight)])


def _write_optimizer_metadata(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(dict(payload), handle, indent=2)


def _atomic_write_json(path: Path, payload: Mapping[str, Any]) -> None:
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w") as handle:
        json.dump(dict(payload), handle, indent=2)
    tmp_path.replace(path)


def _score_min_interference(results: Mapping[str, Mapping[str, Any]], constraint_nonnegative: bool) -> tuple[float, Dict[str, Any]]:
    deltas: Dict[str, float] = {}
    for task, metrics in results.items():
        value = metrics.get("interference_delta")
        if isinstance(value, (int, float)):
            deltas[str(task)] = float(value)
    if not deltas:
        return float("-inf"), {"reason": "missing_interference_delta"}
    min_delta = min(deltas.values())
    mean_delta = sum(deltas.values()) / float(len(deltas))
    if constraint_nonnegative and min_delta < 0:
        return float("-inf"), {"min_interference_delta": min_delta, "mean_interference_delta": mean_delta}
    return min_delta, {"min_interference_delta": min_delta, "mean_interference_delta": mean_delta}


def run_continual_supermerge_optimizer(
    *,
    config: MergeConfig,
    context: ContinualSweepContext,
    summary_dir: Path,
) -> Dict[str, Any]:
    optimizer_spec = config.optimizer
    params = dict(optimizer_spec.params if optimizer_spec is not None else {})
    opt_type = str(optimizer_spec.type if optimizer_spec is not None else "supermerge").strip().lower()
    if opt_type not in {"supermerge", "continual_supermerge", "gradient"}:
        raise ValueError(
            "method='continual_supermerge' expects optimizer.type to be "
            "supermerge|continual_supermerge|gradient."
        )

    steps = int(params.get("steps", 500))
    batch_size = int(params.get("batch_size", 2))
    lr = float(params.get("lr", 1e-3))
    betas_raw = params.get("betas", [0.9, 0.999])
    if not isinstance(betas_raw, (list, tuple)) or len(betas_raw) != 2:
        raise ValueError("optimizer.params.betas must be a 2-item list/tuple.")
    betas = (float(betas_raw[0]), float(betas_raw[1]))
    split = str(params.get("split", "validation")).strip().lower()
    gradient_accumulation_steps = int(params.get("gradient_accumulation_steps", params.get("grad_accum_steps", 1)))
    early_stopping_patience = int(params.get("early_stopping_patience", 0))
    early_stopping_threshold = float(params.get("early_stopping_threshold", params.get("early_stopping_min_delta", 0.0)))
    early_stopping_warmup_steps = int(params.get("early_stopping_warmup_steps", 20))
    min_optimizer_steps_before_early_stop = int(params.get("min_optimizer_steps_before_early_stop", 20))
    early_stopping_rolling_window = int(params.get("early_stopping_rolling_window", 10))
    restore_best_checkpoint = bool(params.get("restore_best_checkpoint", True))
    init_alpha = float(params.get("init_alpha", context.alpha_default))
    init_lambda = float(params.get("init_lambda", context.lambda_default))
    use_autocast = bool(params.get("use_autocast", True))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", False))
    gradient_checkpointing_use_reentrant = bool(params.get("gradient_checkpointing_use_reentrant", False))
    zero_dropout = bool(params.get("zero_dropout", True))
    allow_tf32 = bool(params.get("allow_tf32", True))
    force_cpu = bool(params.get("force_cpu", False))
    dtype_compute = str(params.get("dtype_compute", "auto")).strip().lower()
    task_vector_dtype = str(params.get("task_vector_dtype", "auto")).strip().lower()
    model_dtype = str(params.get("model_dtype", "auto")).strip().lower()
    delta_residency = str(params.get("delta_residency", "cpu_stream")).strip().lower()
    merge_impl = str(params.get("merge_impl", "fused_lora_linear")).strip().lower()
    logging_steps = int(params.get("logging_steps", params.get("log_every", 0)))
    progress_bar = bool(params.get("progress_bar", True))
    dataloader_num_workers = int(params.get("dataloader_num_workers", 0))
    dataloader_pin_memory = bool(params.get("dataloader_pin_memory", True))
    non_blocking_transfer = bool(params.get("non_blocking_transfer", True))
    seed_raw = params.get("seed")
    seed: Optional[int] = None if seed_raw is None else int(seed_raw)
    eval_subset = params.get("eval_subset")
    sampling = params.get("sampling")
    enforce_validation_only_selection = bool(params.get("enforce_validation_only_selection", True))

    ce_ignore_index = int(params.get("ce_ignore_index", -100))
    ce_label_smoothing = float(params.get("ce_label_smoothing", 0.0))
    ce_reduction = str(params.get("ce_reduction", "mean")).strip().lower()
    ce_use_model_loss = bool(params.get("ce_use_model_loss", True))
    ce_task_weighting = str(params.get("ce_task_weighting", params.get("task_weighting", "equal"))).strip().lower()
    ce_baseline_floor = float(params.get("ce_baseline_floor", 1e-6))
    ce_baseline_batches = int(params.get("ce_baseline_batches", 32))
    ce_manual_task_weights = params.get("ce_manual_task_weights", params.get("manual_task_weights"))
    ce_multiplier_cap_raw = params.get("ce_multiplier_cap")
    ce_multiplier_cap = None if ce_multiplier_cap_raw is None else float(ce_multiplier_cap_raw)

    if steps <= 0:
        raise ValueError("optimizer.params.steps must be > 0.")
    if batch_size <= 0:
        raise ValueError("optimizer.params.batch_size must be > 0.")
    if gradient_accumulation_steps <= 0:
        raise ValueError("optimizer.params.gradient_accumulation_steps must be > 0.")
    if early_stopping_rolling_window <= 0:
        raise ValueError("optimizer.params.early_stopping_rolling_window must be > 0.")
    if init_alpha < 0.0:
        raise ValueError("optimizer.params.init_alpha must be >= 0.")
    if not 0.0 <= init_lambda <= 1.0:
        raise ValueError("optimizer.params.init_lambda must be in [0, 1].")
    if ce_task_weighting not in {"baseline_normalized", "equal", "manual"}:
        raise ValueError("continual_supermerge supports ce_task_weighting baseline_normalized|equal|manual.")
    if ce_baseline_floor <= 0.0:
        raise ValueError("optimizer.params.ce_baseline_floor must be > 0.")
    if ce_baseline_batches <= 0:
        raise ValueError("optimizer.params.ce_baseline_batches must be > 0.")
    if ce_reduction not in {"mean", "sum"}:
        raise ValueError("optimizer.params.ce_reduction must be one of: mean|sum.")
    if eval_subset is not None and not isinstance(eval_subset, Mapping):
        raise ValueError("optimizer.params.eval_subset must be a mapping when provided.")
    if sampling is not None and not isinstance(sampling, Mapping):
        raise ValueError("optimizer.params.sampling must be a mapping when provided.")
    if delta_residency not in {"cpu_stream", "gpu_cache"}:
        raise ValueError("optimizer.params.delta_residency must be one of: cpu_stream|gpu_cache.")
    if merge_impl not in {"fused_lora_linear", "streaming_parametrize"}:
        raise ValueError(
            "optimizer.params.merge_impl must be one of: fused_lora_linear|streaming_parametrize."
        )

    tasks = _ordered_seen_tasks(context, params)
    set_global_seed(seed)
    _validate_selection_split(split=split, enforce_validation_only_selection=enforce_validation_only_selection)
    heldout_eval_cfg = resolve_heldout_eval_config(
        params=params,
        tasks=tasks,
        optimization_split=split,
        default_batch_size=batch_size,
        default_patience=early_stopping_patience,
        default_threshold=early_stopping_threshold,
        default_restore_best_checkpoint=restore_best_checkpoint,
    )
    heldout_enabled = bool(heldout_eval_cfg is not None and heldout_eval_cfg.enabled)

    first_task_module = get_task_module(tasks[0])
    first_config_path = first_task_module.get_config_path(PACKAGE_ROOT, None)
    first_config = load_config(first_config_path)
    first_artifacts = first_task_module.get_artifact_directories(PACKAGE_ROOT)
    first_config = prepare_dataset_cache(first_config, first_artifacts)
    model_path = get_model_path(first_config, tasks[0])
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
        model, processor = load_model_and_processor(
            model_path,
            adapter_path=None,
            delta_weights=None,
            torch_dtype_override=_resolve_dtype(model_dtype),
        )
    if hasattr(model, "config"):
        try:
            model.config.use_cache = False
        except Exception:
            pass
    if gradient_checkpointing and hasattr(model, "gradient_checkpointing_enable"):
        try:
            model.gradient_checkpointing_enable(
                gradient_checkpointing_kwargs={"use_reentrant": gradient_checkpointing_use_reentrant}
            )
        except TypeError:
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
            seed=(None if seed is None else int(seed) + idx),
        )
        for idx, task in enumerate(tasks)
    }

    for p in model.parameters():
        p.requires_grad_(False)
    model.train() if gradient_checkpointing else model.eval()
    if zero_dropout:
        stats = _force_zero_dropout(model)
        print(
            "[continual_supermerge] zero_dropout enabled: "
            f"modules_updated={stats['dropout_modules_updated']}/"
            f"{stats['dropout_modules_seen']}, "
            f"config_fields_updated={stats['dropout_config_fields_updated']}"
        )

    device = _infer_input_device(model)
    delta_dtype = _resolve_dtype(task_vector_dtype) if task_vector_dtype != "none" else None
    if delta_dtype is None:
        delta_dtype = torch.float32
    parameter_keys = [name for name, _ in model.named_parameters()]
    lora_entries: List[_LoraEntry] = []
    delta_entries: List[_DeltaEntry] = []
    if merge_impl == "fused_lora_linear":
        lora_entries = _build_source_lora_entries(
            context=context,
            parameter_keys=parameter_keys,
            merge_mode=context.merge_mode,
        )
    else:
        delta_entries = _build_source_delta_entries(
            context=context,
            parameter_keys=parameter_keys,
            merge_mode=context.merge_mode,
            dtype=delta_dtype,
        )

    alpha_init_raw = math.log(math.expm1(max(init_alpha, 1e-8)))
    alpha_raw = torch.nn.Parameter(torch.tensor(alpha_init_raw, device=device, dtype=torch.float32))
    lambda_raw = torch.nn.Parameter(torch.tensor(_logit(init_lambda), device=device, dtype=torch.float32))
    optimizer = torch.optim.Adam([alpha_raw, lambda_raw], lr=lr, betas=betas)
    coefficient_provider = TaskCoefficientProvider()

    def apply_coefficients() -> None:
        coeffs = _coefficients(alpha_raw, lambda_raw)
        coefficient_provider.set_coefficients(
            task_coeffs=coeffs,
            default_coeffs=None,
            layer_coeffs={},
        )

    apply_coefficients()
    streaming_handles = []
    fused_lora_handles = []
    if merge_impl == "fused_lora_linear":
        fused_lora_handles = register_fused_lora_linear_modules(
            model=model,
            entries=_to_streaming_lora_entries(lora_entries),
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
        print(
            "[continual_supermerge] merge_impl=fused_lora_linear "
            f"registered_modules={len(fused_lora_handles)} delta_residency={delta_residency}"
        )
    else:
        streaming_handles = register_streaming_parametrizations(
            model=model,
            entries=_to_streaming_entries(delta_entries),
            coefficient_provider=coefficient_provider,
            delta_residency=delta_residency,
            dtype_compute=dtype_compute,
        )
        print(
            "[continual_supermerge] merge_impl=streaming_parametrize "
            f"registered_params={len(streaming_handles)} delta_residency={delta_residency}"
        )

    task_iters: Dict[str, Optional[Iterator[Any]]] = {task: None for task in tasks}
    ce_baselines: Optional[Dict[str, float]] = None
    if ce_task_weighting == "baseline_normalized":
        ce_baselines = {}
        with torch.no_grad():
            apply_coefficients()
            for task in tasks:
                ce_sum = 0.0
                ce_count = 0
                for _ in range(ce_baseline_batches):
                    batch, task_iters[task] = _next_batch(loaders[task], task_iters[task])
                    batch = _to_device(batch, device, non_blocking=(non_blocking_transfer and device.type == "cuda"))
                    with torch.autocast(
                        device_type="cuda",
                        dtype=torch.bfloat16,
                        enabled=(use_autocast and device.type == "cuda"),
                    ):
                        outputs = model(**batch)
                        logits = _extract_logits(outputs)
                        labels = batch.get("labels") if isinstance(batch, Mapping) else None
                        model_loss = _extract_model_loss(outputs) if ce_use_model_loss else None
                        raw_loss = (
                            model_loss.float()
                            if model_loss is not None
                            else _compute_supervised_ce_loss(
                                logits,
                                labels,
                                ignore_index=ce_ignore_index,
                                label_smoothing=ce_label_smoothing,
                                reduction=ce_reduction,
                            )
                        )
                    ce_sum += float(raw_loss.detach().item())
                    ce_count += 1
                    del batch, outputs, logits, labels, raw_loss
                ce_baselines[task] = ce_sum / float(max(1, ce_count))
        print(
            "[continual_supermerge] CE baselines: "
            + ", ".join(f"{task}={value:.6f}" for task, value in ce_baselines.items())
        )
        task_iters = {task: None for task in tasks}

    ce_multipliers = _compute_task_ce_multipliers(
        tasks=tasks,
        ce_task_weighting=ce_task_weighting,
        ce_baselines=ce_baselines,
        ce_baseline_floor=ce_baseline_floor,
        ce_manual_task_weights=ce_manual_task_weights if isinstance(ce_manual_task_weights, Mapping) else None,
        ce_multiplier_cap=ce_multiplier_cap,
    )
    history: List[Dict[str, Any]] = []
    heldout_evaluator: Optional[PeriodicHeldoutEvaluator] = None
    if heldout_enabled and heldout_eval_cfg is not None:
        heldout_evaluator = PeriodicHeldoutEvaluator(
            model=model,
            processor=processor,
            tasks=tasks,
            config=heldout_eval_cfg,
            show_summary=True,
        )

    best_loss: Optional[float] = None
    best_params: Optional[ContinualScalarParameters] = None
    best_update_step: Optional[int] = None
    no_improve_steps = 0
    optimizer_steps_completed = 0
    update_loss_sum = 0.0
    update_loss_count = 0
    monitor_window: List[float] = []
    optimizer.zero_grad(set_to_none=True)
    step_iter = tqdm(range(steps), desc="continual_supermerge", total=steps) if progress_bar and tqdm else range(steps)

    try:
        for step in step_iter:
            raw_loss_sum = 0.0
            weighted_loss_sum = 0.0
            by_task: Dict[str, float] = {}
            for task in tasks:
                # Each per-task backward needs its own coefficient graph; reusing
                # the same sigmoid/softplus tensor across backward calls frees it
                # after the first task.
                apply_coefficients()
                batch, task_iters[task] = _next_batch(loaders[task], task_iters[task])
                batch = _to_device(batch, device, non_blocking=(non_blocking_transfer and device.type == "cuda"))
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(use_autocast and device.type == "cuda"),
                ):
                    outputs = model(**batch)
                    logits = _extract_logits(outputs)
                    labels = batch.get("labels") if isinstance(batch, Mapping) else None
                    model_loss = _extract_model_loss(outputs) if ce_use_model_loss else None
                    task_loss = (
                        model_loss.float()
                        if model_loss is not None
                        else _compute_supervised_ce_loss(
                            logits,
                            labels,
                            ignore_index=ce_ignore_index,
                            label_smoothing=ce_label_smoothing,
                            reduction=ce_reduction,
                        )
                    )
                raw_loss_sum += float(task_loss.detach().item())
                by_task[task] = float(task_loss.detach().item())
                weighted_task_loss = task_loss * float(ce_multipliers[task])
                weighted_loss_sum += float(weighted_task_loss.detach().item()) * float(len(tasks))
                (weighted_task_loss / float(gradient_accumulation_steps)).backward()
                del batch, outputs, logits, labels, task_loss

            avg_raw_loss = raw_loss_sum / float(len(tasks))
            avg_weighted_loss = weighted_loss_sum / float(len(tasks))
            update_loss_sum += avg_weighted_loss
            update_loss_count += 1
            should_step = (((step + 1) % gradient_accumulation_steps) == 0) or ((step + 1) == steps)
            if not should_step:
                continue

            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
            optimizer_steps_completed += 1
            apply_coefficients()
            scalars = _scalar_values(alpha_raw, lambda_raw)
            update_loss = update_loss_sum / float(max(1, update_loss_count))
            update_loss_sum = 0.0
            update_loss_count = 0
            monitor_window.append(float(update_loss))
            if len(monitor_window) > early_stopping_rolling_window:
                monitor_window.pop(0)
            monitor_loss = sum(monitor_window) / float(len(monitor_window))

            improved = best_loss is None or ((best_loss - monitor_loss) > early_stopping_threshold)
            if improved:
                best_loss = monitor_loss
                if not heldout_enabled:
                    best_params = scalars
                    best_update_step = optimizer_steps_completed
                if optimizer_steps_completed >= early_stopping_warmup_steps:
                    no_improve_steps = 0
            elif optimizer_steps_completed >= early_stopping_warmup_steps:
                no_improve_steps += 1

            heldout_entry = (
                heldout_evaluator.maybe_evaluate(
                    update_step=optimizer_steps_completed,
                    apply_coefficients_fn=apply_coefficients,
                )
                if heldout_evaluator is not None
                else None
            )
            if heldout_entry is not None and heldout_entry.get("best_selection_score") == heldout_entry.get("selection_score"):
                best_params = scalars
                best_update_step = optimizer_steps_completed
            entry = {
                "step": int(step + 1),
                "optimizer_step": int(optimizer_steps_completed),
                "raw_ce": float(avg_raw_loss),
                "weighted_ce": float(avg_weighted_loss),
                "update_weighted_ce": float(update_loss),
                "monitor_weighted_ce": float(monitor_loss),
                "monitor_window": int(len(monitor_window)),
                "best_monitor_weighted_ce": None if best_loss is None else float(best_loss),
                "no_improve_steps": int(no_improve_steps),
                "ce_by_task": by_task,
                "alpha": float(scalars.alpha),
                "lambda": float(scalars.lambda_weight),
            }
            if heldout_entry is not None:
                entry["heldout_eval"] = heldout_entry
            history.append(entry)
            if logging_steps > 0 and (
                optimizer_steps_completed % logging_steps == 0 or (step + 1) == steps
            ):
                print(
                    "[continual_supermerge] "
                    f"step={step + 1}/{steps} update_step={optimizer_steps_completed} "
                    f"weighted_ce={update_loss:.6f} monitor_ce={monitor_loss:.6f} "
                    f"best_monitor_ce={best_loss:.6f} no_improve={no_improve_steps} "
                    f"alpha={scalars.alpha:.6f} "
                    f"lambda={scalars.lambda_weight:.6f}"
                )
            if heldout_entry is not None and bool(heldout_entry.get("should_stop", False)):
                print("[continual_supermerge] early stopping on held-out selection.")
                break
            if (
                early_stopping_patience > 0
                and optimizer_steps_completed >= min_optimizer_steps_before_early_stop
                and no_improve_steps >= early_stopping_patience
            ):
                print("[continual_supermerge] early stopping on validation CE plateau.")
                break
    finally:
        if fused_lora_handles:
            unregister_fused_lora_linear_modules(fused_lora_handles)
        if streaming_handles:
            unregister_streaming_parametrizations(streaming_handles)

    final_scalars = _scalar_values(alpha_raw, lambda_raw)
    selected = best_params if (restore_best_checkpoint and best_params is not None) else final_scalars
    summary_dir.mkdir(parents=True, exist_ok=True)
    optimizer_provenance = {
        "type": "validation_ce_scalar_alpha_lambda",
        "optimized_tasks": tasks,
        "split": split,
        "best_update_step": best_update_step,
        "steps_completed": optimizer_steps_completed,
        "early_stopping": {
            "patience": early_stopping_patience,
            "threshold": early_stopping_threshold,
            "warmup_steps": early_stopping_warmup_steps,
            "min_optimizer_steps_before_stop": min_optimizer_steps_before_early_stop,
            "rolling_window": early_stopping_rolling_window,
            "monitor": "rolling_mean_update_weighted_ce",
            "best_monitor_weighted_ce": None if best_loss is None else float(best_loss),
        },
        "heldout_eval": {
            "enabled": bool(heldout_enabled),
            "split": (heldout_eval_cfg.split if heldout_eval_cfg is not None else None),
            "frequency_updates": (
                heldout_eval_cfg.frequency_updates if heldout_eval_cfg is not None else None
            ),
            "selection_criterion": (
                heldout_eval_cfg.pareto.selection_criterion if heldout_eval_cfg is not None else None
            ),
            "restore_best_checkpoint": (
                heldout_eval_cfg.restore_best_checkpoint if heldout_eval_cfg is not None else None
            ),
        },
        "parameterization": {
            "alpha": "softplus(raw_alpha)",
            "lambda": "sigmoid(raw_lambda)",
        },
        "gradient_checkpointing": gradient_checkpointing,
        "gradient_checkpointing_use_reentrant": gradient_checkpointing_use_reentrant,
        "merge_impl": merge_impl,
        "delta_residency": delta_residency,
        "dtype_compute": dtype_compute,
        "task_vector_dtype": task_vector_dtype,
        "model_dtype": model_dtype,
    }
    _write_optimizer_metadata(
        summary_dir / "continual_supermerge_optimizer_history.json",
        {
            "method": CONTINUAL_SUPERMERGE_METHOD,
            "optimizer": {"type": opt_type, "params": params},
            "optimizer_provenance": optimizer_provenance,
            "history": history,
            "selected_params": {"alpha": selected.alpha, "lambda": selected.lambda_weight},
            "final_params": {"alpha": final_scalars.alpha, "lambda": final_scalars.lambda_weight},
        },
    )
    run_dir = summary_dir.parent / "runs" / (
        f"run_0001_alpha{selected.alpha:g}_lambda{selected.lambda_weight:g}".replace(".", "p")
    )
    merge_result = continual_merge_sources_to_artifact(
        x_source=context.x_source,
        y_source=context.y_source,
        policy=ContinualMergePolicy(alpha=selected.alpha, lambda_weight=selected.lambda_weight),
        output_dir=run_dir,
        energy_threshold=context.energy_threshold,
        merge_mode=context.merge_mode,
        store_dtype=context.store_dtype,
    )
    merge_tag = build_continual_tag(
        source_tasks=context.eval_tasks,
        alpha=selected.alpha,
        lambda_weight=selected.lambda_weight,
    )
    results = evaluate_continual_artifact(
        artifact_path=merge_result.artifact_dir,
        eval_tasks=context.eval_tasks,
        split=config.split,
        batch_size=context.batch_size,
        enable_cache=context.use_cache,
        show_summary=True,
        compute_missing_interference_baselines=config.compute_missing_interference_baselines,
        save_results=True,
        eval_subset=config.eval_subset,
        merge_tag=merge_tag,
        alpha=selected.alpha,
        lambda_weight=selected.lambda_weight,
        method_name=CONTINUAL_SUPERMERGE_METHOD,
    )
    score, score_details = _score_min_interference(results, config.constraint_nonnegative)
    post_sweep_eval: Dict[str, Any] = {"enabled": False}
    post_cfg = config.post_sweep_eval
    if isinstance(post_cfg, Mapping) and bool(post_cfg.get("enabled", False)):
        post_split = str(post_cfg.get("split", "test"))
        post_eval_tasks_raw = post_cfg.get("eval_tasks")
        post_eval_tasks = (
            [str(task) for task in post_eval_tasks_raw]
            if isinstance(post_eval_tasks_raw, list)
            else list(context.eval_tasks)
        )
        post_results = evaluate_continual_artifact(
            artifact_path=merge_result.artifact_dir,
            eval_tasks=post_eval_tasks,
            split=post_split,
            batch_size=context.batch_size,
            enable_cache=context.use_cache,
            show_summary=True,
            compute_missing_interference_baselines=config.compute_missing_interference_baselines,
            save_results=True,
            merge_tag=merge_tag,
            alpha=selected.alpha,
            lambda_weight=selected.lambda_weight,
            method_name=CONTINUAL_SUPERMERGE_METHOD,
        )
        post_score, post_score_details = _score_min_interference(post_results, config.constraint_nonnegative)
        post_sweep_eval = {
            "enabled": True,
            "split": post_split,
            "save_merged": True,
            "params": {"alpha": selected.alpha, "lambda": selected.lambda_weight},
            "score": float(post_score),
            "score_details": post_score_details,
            "results": post_results,
            "artifact_dir": str(merge_result.artifact_dir),
            "manifest_path": str(merge_result.manifest_path),
        }
    started_at = datetime.now()
    summary = {
        "timestamp": started_at.isoformat(),
        "method": CONTINUAL_SUPERMERGE_METHOD,
        "adapters": config.adapters,
        "merge_mode": config.merge_mode,
        "optimizer": {"type": opt_type, "params": params},
        "optimizer_provenance": optimizer_provenance,
        "eval_tasks": config.eval_tasks,
        "optimization_tasks": tasks,
        "split": config.split,
        "eval_subset": config.eval_subset,
        "constraint_nonnegative": config.constraint_nonnegative,
        "best_index": 0,
        "best_score": float(score),
        "runs": [
            {
                "params": {"alpha": selected.alpha, "lambda": selected.lambda_weight},
                "score": float(score),
                "score_details": score_details,
                "results": results,
                "continual": {
                    "alpha": selected.alpha,
                    "lambda": selected.lambda_weight,
                    "artifact_dir": str(merge_result.artifact_dir),
                    "manifest_path": str(merge_result.manifest_path),
                    "num_merged_params": int(merge_result.num_merged_params),
                    "num_skipped_params": int(merge_result.num_skipped_params),
                },
                "optimizer": {
                    "history": history,
                    "final_params": {"alpha": final_scalars.alpha, "lambda": final_scalars.lambda_weight},
                    "selected_params": {"alpha": selected.alpha, "lambda": selected.lambda_weight},
                },
            }
        ],
        "post_sweep_eval": post_sweep_eval,
    }
    summary_path = summary_dir / f"sweep_{started_at.strftime('%Y%m%d_%H%M%S')}.json"
    _atomic_write_json(summary_path, summary)
    _write_optimizer_metadata(
        Path(merge_result.artifact_dir) / "continual_supermerge_optimizer.json",
        {
            "method": CONTINUAL_SUPERMERGE_METHOD,
            "optimizer": summary["optimizer"],
            "optimizer_provenance": summary["optimizer_provenance"],
            "history": history,
            "selected_params": {"alpha": selected.alpha, "lambda": selected.lambda_weight},
            "final_params": {"alpha": final_scalars.alpha, "lambda": final_scalars.lambda_weight},
        },
    )
    print(f"\n💾 Continual SuperMerge summary saved to {summary_path}")
    print(f"🏆 Selected alpha={selected.alpha:.6f}, lambda={selected.lambda_weight:.6f}")
    return summary


__all__ = [
    "ContinualScalarParameters",
    "run_continual_supermerge_optimizer",
]
