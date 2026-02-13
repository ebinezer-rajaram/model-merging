"""Regret smooth-max optimizer for static global LoRA merge coefficients."""

from __future__ import annotations

import math
import time
from typing import Any, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import torch

from core import load_config
from core.evaluation.eval_utils import load_model_and_processor
from experiments.evaluate_task import _get_model_path, _prepare_dataset_cache
from experiments.extract_vector import extract_task_vector_from_lora
from merging.config.specs import MergeSpec
from merging.plugins.adamerging_streaming import (
    TaskCoefficientProvider,
    register_fused_linear_modules,
    register_fused_lora_linear_modules,
    register_streaming_parametrizations,
    unregister_fused_linear_modules,
    unregister_fused_lora_linear_modules,
    unregister_streaming_parametrizations,
)
from merging.plugins.gradient_common import (
    _build_delta_entries,
    _build_lora_entries,
    _build_functional_params,
    _build_task_loader,
    _extract_logits,
    _force_zero_dropout,
    _functional_forward,
    _infer_input_device,
    _next_batch,
    _resolve_dtype,
    _resolve_eval_tasks,
    _to_device,
    _to_streaming_lora_entries,
    _to_streaming_entries,
)
from merging.plugins.optimizers import OptimizerContext, OptimizerResult
from merging.runtime.utils import PACKAGE_ROOT, get_task_module

try:
    from tqdm.auto import tqdm
except Exception:  # pragma: no cover - optional dependency
    tqdm = None


def _token_normalized_ce(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    *,
    ignore_index: int = -100,
) -> Tuple[torch.Tensor, int]:
    """Compute CE normalized by count of non-ignore labels."""
    if labels is None:
        raise ValueError("Label-token CE requires batch['labels'] to be present.")
    if logits.ndim < 3:
        raise ValueError(
            "Label-token CE expects token logits [B, T, V]. "
            f"Got shape={tuple(logits.shape)}"
        )
    if labels.shape[:2] != logits.shape[:2]:
        raise ValueError(
            "Label-token CE requires labels shape [B, T] matching logits [B, T, V]. "
            f"Got labels={tuple(labels.shape)} logits={tuple(logits.shape)}"
        )
    flat_logits = logits.float().reshape(-1, logits.shape[-1])
    flat_labels = labels.reshape(-1).long()
    valid = flat_labels != ignore_index
    valid_tokens = int(valid.sum().detach().item())
    if valid_tokens <= 0:
        raise ValueError("No valid label tokens found (all labels ignored).")
    ce_sum = torch.nn.functional.cross_entropy(
        flat_logits,
        flat_labels,
        ignore_index=ignore_index,
        reduction="sum",
    )
    return ce_sum / float(valid_tokens), valid_tokens


def _resolve_initial_logits(
    *,
    num_adapters: int,
    init_mode: str,
    init_values: Optional[List[float]],
    device: torch.device,
) -> torch.Tensor:
    mode = init_mode.strip().lower()
    if mode == "uniform":
        return torch.zeros((num_adapters,), device=device, dtype=torch.float32)
    if mode == "from_lambda":
        if init_values is None:
            raise ValueError("optimizer.params.init_lambda_values is required for init_mode=from_lambda.")
        if len(init_values) != num_adapters:
            raise ValueError(
                "optimizer.params.init_lambda_values length must equal number of adapters. "
                f"Got {len(init_values)} vs {num_adapters}."
            )
        values = torch.tensor([float(x) for x in init_values], device=device, dtype=torch.float32)
        if torch.any(values < 0.0):
            raise ValueError("optimizer.params.init_lambda_values must be >= 0.")
        total = float(values.sum().item())
        if total <= 0.0:
            raise ValueError("optimizer.params.init_lambda_values must sum to > 0.")
        values = values / total
        return torch.log(values.clamp_min(1e-8))
    raise ValueError("optimizer.params.init_mode must be one of: uniform|from_lambda.")


def _compute_fixed_losses(
    *,
    tasks: List[str],
    loaders: Mapping[str, Any],
    task_iters: Mapping[str, Optional[Iterator[Any]]],
    device: torch.device,
    use_autocast: bool,
    non_blocking_transfer: bool,
    ignore_index: int,
    baseline_batches: int,
    merge_impl: str,
    model: Any,
    base_params: Mapping[str, torch.Tensor],
    delta_entries: List[Any],
    coefficient_provider: TaskCoefficientProvider,
    adapter_index_by_task: Mapping[str, int],
    mode: str,
    num_adapters: int,
) -> Dict[str, float]:
    if baseline_batches <= 0:
        raise ValueError("optimizer.params.baseline_batches must be > 0.")
    if mode not in {"base", "best"}:
        raise ValueError(f"Unsupported fixed loss mode '{mode}'.")
    out: Dict[str, float] = {}
    with torch.no_grad():
        for task in tasks:
            ce_acc = 0.0
            count = 0
            iterator = task_iters.get(task)
            for _ in range(baseline_batches):
                if mode == "base":
                    coeffs = torch.zeros((num_adapters,), device=device, dtype=torch.float32)
                else:
                    coeffs = torch.zeros((num_adapters,), device=device, dtype=torch.float32)
                    coeffs[adapter_index_by_task[task]] = 1.0
                if merge_impl == "functional_clone_legacy":
                    func_params = _build_functional_params(
                        base_params=dict(base_params),
                        delta_entries=delta_entries,
                        task_coeffs=coeffs,
                        default_coeffs=None,
                        layer_coeffs={},
                    )
                else:
                    coefficient_provider.set_coefficients(
                        task_coeffs=coeffs,
                        default_coeffs=None,
                        layer_coeffs={},
                    )
                    func_params = None
                batch, iterator = _next_batch(loaders[task], iterator)
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
                    ce_token_norm, _ = _token_normalized_ce(logits, labels, ignore_index=ignore_index)
                ce_acc += float(ce_token_norm.detach().item())
                count += 1
                del func_params, batch, outputs, logits, labels, ce_token_norm
            if count <= 0:
                raise ValueError(f"Failed to compute fixed losses for task '{task}' (no batches).")
            out[task] = ce_acc / float(count)
    return out


def run_regret_smoothmax_optimizer(spec: MergeSpec, context: OptimizerContext) -> OptimizerResult:
    params = dict(spec.optimizer.params if spec.optimizer is not None else {})
    if context.method != "weighted_delta_n":
        raise ValueError("regret_smoothmax optimizer only supports method='weighted_delta_n'.")

    tasks = _resolve_eval_tasks(context, params)
    steps = int(params.get("steps", 500))
    split = str(params.get("split", "validation"))
    batch_size = int(params.get("batch_size", 1))
    lr = float(params.get("lr", 3e-4))
    betas = params.get("betas", [0.9, 0.999])
    if not isinstance(betas, (list, tuple)) or len(betas) != 2:
        raise ValueError("optimizer.params.betas must be a 2-item list/tuple.")
    betas = (float(betas[0]), float(betas[1]))
    tau = float(params.get("tau", 0.1))
    regret_eps = float(params.get("regret_eps", 1e-4))
    baseline_batches = int(params.get("baseline_batches", 64))
    ignore_index = int(params.get("ce_ignore_index", -100))
    force_cpu = bool(params.get("force_cpu", False))
    use_autocast = bool(params.get("use_autocast", True))
    gradient_checkpointing = bool(params.get("gradient_checkpointing", True))
    zero_dropout = bool(params.get("zero_dropout", True))
    allow_tf32 = bool(params.get("allow_tf32", True))
    merge_impl = str(params.get("merge_impl", "fused_linear")).strip().lower()
    delta_residency = str(params.get("delta_residency", "gpu_cache")).strip().lower()
    dtype_compute = str(params.get("dtype_compute", "auto")).strip().lower()
    task_vector_dtype = str(params.get("task_vector_dtype", "auto")).strip().lower()
    model_dtype = str(params.get("model_dtype", "auto")).strip().lower()
    logging_steps = int(params.get("logging_steps", 10))
    progress_bar = bool(params.get("progress_bar", True))
    non_blocking_transfer = bool(params.get("non_blocking_transfer", True))
    dataloader_num_workers = int(params.get("dataloader_num_workers", 0))
    dataloader_pin_memory = bool(params.get("dataloader_pin_memory", True))
    eval_subset = params.get("eval_subset")
    if eval_subset is not None and not isinstance(eval_subset, Mapping):
        raise ValueError("optimizer.params.eval_subset must be a mapping when provided.")
    sampling = params.get("sampling")
    if sampling is not None and not isinstance(sampling, Mapping):
        raise ValueError("optimizer.params.sampling must be a mapping when provided.")
    init_mode = str(params.get("init_mode", "uniform"))
    init_lambda_values_raw = params.get("init_lambda_values")
    init_lambda_values: Optional[List[float]]
    if init_lambda_values_raw is None:
        init_lambda_values = None
    else:
        if not isinstance(init_lambda_values_raw, list):
            raise ValueError("optimizer.params.init_lambda_values must be a list when provided.")
        init_lambda_values = [float(x) for x in init_lambda_values_raw]

    if split.strip().lower() != "validation":
        raise ValueError("regret_smoothmax requires optimizer.params.split='validation'.")
    if steps <= 0:
        raise ValueError("optimizer.params.steps must be > 0.")
    if batch_size <= 0:
        raise ValueError("optimizer.params.batch_size must be > 0.")
    if tau <= 0.0:
        raise ValueError("optimizer.params.tau must be > 0.")
    if regret_eps <= 0.0:
        raise ValueError("optimizer.params.regret_eps must be > 0.")
    if merge_impl not in {"streaming_parametrize", "functional_clone_legacy", "fused_linear", "fused_lora_linear"}:
        raise ValueError(
            "optimizer.params.merge_impl must be one of: "
            "streaming_parametrize|functional_clone_legacy|fused_linear|fused_lora_linear."
        )
    if delta_residency not in {"cpu_stream", "gpu_cache"}:
        raise ValueError("optimizer.params.delta_residency must be one of: cpu_stream|gpu_cache.")

    from merging.plugins.transforms import apply_transforms

    if merge_impl == "fused_lora_linear":
        task_vectors = []
    else:
        task_vectors = [
            apply_transforms(extract_task_vector_from_lora(path), spec.transforms)
            for path in context.adapter_paths
        ]
    num_adapters = len(context.adapter_paths)
    if len(tasks) != num_adapters:
        raise ValueError(
            "regret_smoothmax currently requires optimizer.params.tasks length to match number of adapters "
            f"(tasks={len(tasks)}, adapters={num_adapters})."
        )
    adapter_index_by_task = {task: i for i, task in enumerate(tasks)}

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
            "[regret_smoothmax] zero_dropout enabled: "
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
        lora_entries = _build_lora_entries(
            context.adapter_paths,
            parameter_keys=parameter_keys,
            merge_mode=context.merge_mode,
        )
    else:
        delta_entries = _build_delta_entries(task_vectors, parameter_keys=parameter_keys, merge_mode=context.merge_mode)

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

    coefficient_provider = TaskCoefficientProvider()
    streaming_handles = []
    fused_linear_handles = []
    fused_lora_handles = []
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

    task_iters: Dict[str, Optional[Iterator[Any]]] = {task: None for task in tasks}
    u_init = _resolve_initial_logits(
        num_adapters=num_adapters,
        init_mode=init_mode,
        init_values=init_lambda_values,
        device=device,
    )
    u = torch.nn.Parameter(u_init.clone())
    optimizer = torch.optim.Adam([u], lr=lr, betas=betas)

    best_j: Optional[float] = None
    best_step = 0
    best_u = u.detach().clone()
    final_j: Optional[float] = None
    steps_completed = 0
    fixed_base_losses: Dict[str, float] = {}
    fixed_best_losses: Dict[str, float] = {}

    try:
        fixed_base_losses = _compute_fixed_losses(
            tasks=tasks,
            loaders=loaders,
            task_iters=task_iters,
            device=device,
            use_autocast=use_autocast,
            non_blocking_transfer=non_blocking_transfer,
            ignore_index=ignore_index,
            baseline_batches=baseline_batches,
            merge_impl=merge_impl,
            model=model,
            base_params=base_params,
            delta_entries=delta_entries,
            coefficient_provider=coefficient_provider,
            adapter_index_by_task=adapter_index_by_task,
            mode="base",
            num_adapters=num_adapters,
        )
        fixed_best_losses = _compute_fixed_losses(
            tasks=tasks,
            loaders=loaders,
            task_iters=task_iters,
            device=device,
            use_autocast=use_autocast,
            non_blocking_transfer=non_blocking_transfer,
            ignore_index=ignore_index,
            baseline_batches=baseline_batches,
            merge_impl=merge_impl,
            model=model,
            base_params=base_params,
            delta_entries=delta_entries,
            coefficient_provider=coefficient_provider,
            adapter_index_by_task=adapter_index_by_task,
            mode="best",
            num_adapters=num_adapters,
        )

        step_iter: Iterable[int]
        if progress_bar and tqdm is not None:
            step_iter = tqdm(range(steps), desc="regret_smoothmax", total=steps)
        else:
            step_iter = range(steps)

        for step in step_iter:
            optimizer.zero_grad(set_to_none=True)
            per_task_loss_values: Dict[str, float] = {}
            per_task_valid_tokens: Dict[str, int] = {}
            active_task = "<none>"
            try:
                # Pass 1 (no grad): evaluate regrets to get smooth-max weights.
                lambdas_eval = torch.softmax(u.detach(), dim=0)
                for task in tasks:
                    active_task = task
                    if merge_impl == "functional_clone_legacy":
                        func_params = _build_functional_params(
                            base_params=base_params,
                            delta_entries=delta_entries,
                            task_coeffs=lambdas_eval,
                            default_coeffs=None,
                            layer_coeffs={},
                        )
                    else:
                        coefficient_provider.set_coefficients(
                            task_coeffs=lambdas_eval,
                            default_coeffs=None,
                            layer_coeffs={},
                        )
                        func_params = None
                    batch, task_iters[task] = _next_batch(loaders[task], task_iters[task])
                    batch = _to_device(batch, device, non_blocking=(non_blocking_transfer and device.type == "cuda"))
                    with torch.no_grad():
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
                            loss_t, valid_tokens = _token_normalized_ce(logits, labels, ignore_index=ignore_index)
                    per_task_loss_values[task] = float(loss_t.item())
                    per_task_valid_tokens[task] = int(valid_tokens)
                    del func_params, batch, outputs, logits, labels, loss_t
            except ValueError as exc:
                raise ValueError(
                    f"regret_smoothmax objective failed at step={step}, task='{active_task}': {exc}"
                ) from exc

            regret_values: Dict[str, float] = {}
            for task in tasks:
                denom = float(fixed_base_losses[task] - fixed_best_losses[task] + regret_eps)
                regret_values[task] = (per_task_loss_values[task] - float(fixed_best_losses[task])) / denom

            regret_vec = torch.tensor(
                [regret_values[task] for task in tasks],
                device=device,
                dtype=torch.float32,
            )
            smooth_weights = torch.softmax(regret_vec / tau, dim=0).detach()
            final_j = float((tau * torch.logsumexp(regret_vec / tau, dim=0)).item())

            # Pass 2 (with grad): accumulate exact gradient using fixed smooth-max weights.
            for i, task in enumerate(tasks):
                active_task = task
                lambdas_train = torch.softmax(u, dim=0)
                if merge_impl == "functional_clone_legacy":
                    func_params = _build_functional_params(
                        base_params=base_params,
                        delta_entries=delta_entries,
                        task_coeffs=lambdas_train,
                        default_coeffs=None,
                        layer_coeffs={},
                    )
                else:
                    coefficient_provider.set_coefficients(
                        task_coeffs=lambdas_train,
                        default_coeffs=None,
                        layer_coeffs={},
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
                    loss_t, _ = _token_normalized_ce(logits, labels, ignore_index=ignore_index)
                denom = float(fixed_base_losses[task] - fixed_best_losses[task] + regret_eps)
                regret_t = (loss_t - float(fixed_best_losses[task])) / denom
                (smooth_weights[i] * regret_t).backward()
                del func_params, batch, outputs, logits, labels, loss_t, regret_t

            optimizer.step()
            steps_completed = step + 1

            if best_j is None or final_j < best_j:
                best_j = final_j
                best_step = step + 1
                best_u = u.detach().clone()

            if logging_steps > 0 and (((step + 1) % logging_steps) == 0 or ((step + 1) == steps)):
                lambda_values = [float(x) for x in torch.softmax(u.detach(), dim=0).cpu().tolist()]
                print(
                    f"[regret_smoothmax] step {step + 1}/{steps} "
                    f"J={final_j:.6f} lambda={lambda_values}"
                )
                print(f"[regret_smoothmax][L] {per_task_loss_values}")
                print(f"[regret_smoothmax][r] {regret_values}")
                print(f"[regret_smoothmax][valid_tokens] {per_task_valid_tokens}")
            if tqdm is not None and hasattr(step_iter, "set_postfix"):
                step_iter.set_postfix({"J": f"{final_j:.4f}"})
    finally:
        if streaming_handles:
            unregister_streaming_parametrizations(streaming_handles)
        if fused_linear_handles:
            unregister_fused_linear_modules(fused_linear_handles)
        if fused_lora_handles:
            unregister_fused_lora_linear_modules(fused_lora_handles)

    with torch.no_grad():
        u.copy_(best_u)
    final_lambda = [float(x) for x in torch.softmax(u.detach(), dim=0).cpu().tolist()]
    final_u = [float(x) for x in u.detach().cpu().tolist()]

    method_params_overrides: Dict[str, Any] = {
        "task_coefficients": final_lambda,
        "normalize_coefficients": False,
        "allow_negative_coefficients": False,
    }
    provenance = {
        "optimizer": "regret_smoothmax",
        "status": "optimized",
        "tasks": tasks,
        "split": split,
        "steps": steps,
        "steps_completed": steps_completed,
        "batch_size": batch_size,
        "lr": lr,
        "betas": [betas[0], betas[1]],
        "tau": tau,
        "regret_eps": regret_eps,
        "baseline_batches": baseline_batches,
        "merge_impl": merge_impl,
        "delta_residency": delta_residency,
        "dtype_compute": dtype_compute,
        "task_vector_dtype": task_vector_dtype,
        "model_dtype": model_dtype,
        "force_cpu": force_cpu,
        "allow_tf32": allow_tf32,
        "use_autocast": use_autocast,
        "gradient_checkpointing": gradient_checkpointing,
        "zero_dropout": zero_dropout,
        "dataloader_num_workers": dataloader_num_workers,
        "dataloader_pin_memory": dataloader_pin_memory,
        "non_blocking_transfer": non_blocking_transfer,
        "sampling": dict(sampling) if isinstance(sampling, Mapping) else None,
        "ce_ignore_index": ignore_index,
        "init_mode": init_mode,
        "init_lambda_values": init_lambda_values,
        "fixed_base_losses": fixed_base_losses,
        "fixed_best_losses": fixed_best_losses,
        "best_j": best_j,
        "best_step": best_step,
        "final_j": final_j,
        "final_task_coefficients": final_lambda,
        "final_logits_u": final_u,
        "num_adapters": num_adapters,
        "num_delta_entries": len(delta_entries),
        "num_lora_entries": len(lora_entries),
        "eval_subset": dict(eval_subset) if isinstance(eval_subset, Mapping) else None,
        "elapsed_sec": time.time() - t0,
        "objective": "tau_logsumexp_headroom_normalized_regret",
    }
    return OptimizerResult(
        lambda_policy=context.lambda_policy,
        provenance=provenance,
        method_params_overrides=method_params_overrides,
    )


__all__ = ["run_regret_smoothmax_optimizer"]
