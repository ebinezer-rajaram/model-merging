"""Shared supervised-training helpers for gradient-style optimizers."""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Mapping, Optional

import torch

from merging.optimizers.core.streaming import TaskCoefficientProvider
from merging.optimizers.core.common import _extract_logits, _merge_coeffs, _next_batch, _to_device


def _compute_supervised_ce_loss(
    logits: torch.Tensor,
    labels: Optional[torch.Tensor],
    *,
    ignore_index: int = -100,
    label_smoothing: float = 0.0,
    reduction: str = "mean",
    return_stats: bool = False,
):
    if labels is None:
        raise ValueError("Supervised CE requires batch['labels'] to be present.")

    labels = labels.long()
    if logits.ndim == 2:
        if labels.ndim != 1 or labels.shape[0] != logits.shape[0]:
            raise ValueError(
                "For logits shape [B, C], labels must have shape [B]. "
                f"Got logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
            )
        valid = labels != ignore_index
        valid_count = int(valid.sum().detach().item())
        if valid_count == 0:
            raise ValueError("No valid labels found for supervised CE (all labels ignored).")
        loss = torch.nn.functional.cross_entropy(
            logits.float(),
            labels,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    elif logits.ndim >= 3:
        if labels.shape[:2] != logits.shape[:2]:
            raise ValueError(
                "For token logits shape [B, T, V], labels must have shape [B, T]. "
                f"Got logits={tuple(logits.shape)} labels={tuple(labels.shape)}"
            )
        flat_logits = logits.float().reshape(-1, logits.shape[-1])
        flat_labels = labels.reshape(-1)
        valid = flat_labels != ignore_index
        valid_count = int(valid.sum().detach().item())
        if valid_count == 0:
            raise ValueError("No valid labels found for supervised CE (all labels ignored).")
        loss = torch.nn.functional.cross_entropy(
            flat_logits,
            flat_labels,
            ignore_index=ignore_index,
            reduction=reduction,
            label_smoothing=label_smoothing,
        )
    else:
        raise ValueError(
            "Unsupported logits rank for supervised CE; expected rank 2 or >=3, "
            f"got shape={tuple(logits.shape)}"
        )

    if not return_stats:
        return loss
    stats = {
        "valid_label_tokens": valid_count,
        "num_ce_positions": valid_count,
    }
    return loss, stats


def _extract_model_loss(outputs: Any) -> Optional[torch.Tensor]:
    if isinstance(outputs, Mapping):
        loss = outputs.get("loss")
    else:
        loss = getattr(outputs, "loss", None)
    if loss is None:
        return None
    if not torch.is_tensor(loss):
        return None
    return loss


def _validate_selection_split(split: str, enforce_validation_only_selection: bool) -> None:
    if not enforce_validation_only_selection:
        return
    normalized = str(split).strip().lower()
    if normalized != "validation":
        raise ValueError(
            "optimizer.params.enforce_validation_only_selection=true requires "
            "optimizer.params.split='validation'."
        )


def _normalize_manual_weights(
    *,
    tasks: List[str],
    ce_manual_task_weights: Optional[Mapping[str, Any]],
) -> Dict[str, float]:
    if ce_manual_task_weights is None:
        raise ValueError(
            "optimizer.params.ce_manual_task_weights is required when "
            "optimizer.params.ce_task_weighting='manual'."
        )
    out: Dict[str, float] = {}
    for task in tasks:
        if task not in ce_manual_task_weights:
            raise ValueError(
                "optimizer.params.ce_manual_task_weights is missing task "
                f"'{task}'. Expected keys: {tasks}"
            )
        value = float(ce_manual_task_weights[task])
        if value < 0.0:
            raise ValueError(
                "optimizer.params.ce_manual_task_weights values must be >= 0. "
                f"Got {task}={value}."
            )
        out[task] = value
    total = float(sum(out.values()))
    if total <= 0.0:
        raise ValueError(
            "optimizer.params.ce_manual_task_weights must sum to > 0."
        )
    return {task: (value / total) for task, value in out.items()}


def _compute_task_ce_multipliers(
    *,
    tasks: List[str],
    ce_task_weighting: str,
    ce_baselines: Optional[Mapping[str, float]],
    ce_baseline_floor: float,
    ce_manual_task_weights: Optional[Mapping[str, Any]],
    ce_multiplier_cap: Optional[float],
) -> Dict[str, float]:
    num_tasks = len(tasks)
    if num_tasks <= 0:
        raise ValueError("tasks must be non-empty.")
    mode = str(ce_task_weighting).strip().lower()
    if mode == "equal":
        return {task: (1.0 / float(num_tasks)) for task in tasks}
    if mode == "manual":
        return _normalize_manual_weights(tasks=tasks, ce_manual_task_weights=ce_manual_task_weights)
    if mode != "baseline_normalized":
        raise ValueError(
            "optimizer.params.ce_task_weighting must be one of: baseline_normalized|equal|manual."
        )
    if ce_baselines is None:
        raise ValueError("ce_baselines must be provided when ce_task_weighting=baseline_normalized.")
    multipliers: Dict[str, float] = {}
    for task in tasks:
        baseline = float(ce_baselines.get(task, 0.0))
        denom = max(float(ce_baseline_floor), baseline)
        mult = 1.0 / (float(num_tasks) * denom)
        if ce_multiplier_cap is not None:
            mult = min(mult, float(ce_multiplier_cap))
        multipliers[task] = mult
    return multipliers


def _compute_single_task_ce_baselines(
    *,
    tasks: List[str],
    num_adapters: int,
    alpha_task: torch.Tensor,
    alpha_default: Optional[torch.Tensor],
    alpha_layer: Optional[torch.Tensor],
    layer_indices: List[int],
    coefficient_parameterization: str,
    normalize_coefficients: bool,
    coefficient_provider: TaskCoefficientProvider,
    merge_impl: str,
    model: Any,
    loaders: Mapping[str, Any],
    task_iters: Mapping[str, Optional[Iterator[Any]]],
    device: torch.device,
    use_autocast: bool,
    non_blocking_transfer: bool,
    ce_use_model_loss: bool,
    ce_ignore_index: int,
    ce_label_smoothing: float,
    ce_reduction: str,
    ce_baseline_batches: int,
) -> Dict[str, float]:
    if ce_baseline_batches <= 0:
        raise ValueError("optimizer.params.ce_baseline_batches must be > 0.")

    one_hots: List[torch.Tensor] = []
    for i in range(num_adapters):
        vec = torch.zeros((num_adapters,), device=device, dtype=alpha_task.dtype)
        vec[i] = 1.0
        one_hots.append(vec)

    baselines: Dict[str, float] = {}
    with torch.no_grad():
        for task_idx, task in enumerate(tasks):
            if task_idx >= num_adapters:
                raise ValueError(
                    "optimizer.params.tasks length exceeds number of adapters, "
                    f"cannot map task '{task}' to adapter index."
                )
            hot = one_hots[task_idx]
            task_coeffs = hot
            default_coeffs = hot if alpha_default is not None else None
            layer_coeffs = {int(layer): hot for layer in layer_indices} if alpha_layer is not None else {}

            coefficient_provider.set_coefficients(
                task_coeffs=task_coeffs,
                default_coeffs=default_coeffs,
                layer_coeffs=layer_coeffs,
            )

            ce_sum = 0.0
            ce_count = 0
            iterator = task_iters.get(task)
            for _ in range(ce_baseline_batches):
                batch, iterator = _next_batch(loaders[task], iterator)
                batch = _to_device(batch, device, non_blocking=(non_blocking_transfer and device.type == "cuda"))
                with torch.autocast(
                    device_type="cuda",
                    dtype=torch.bfloat16,
                    enabled=(use_autocast and device.type == "cuda"),
                ):
                    outputs = model(**batch) if merge_impl != "functional_clone_legacy" else None
                    if outputs is None:
                        raise ValueError(
                            "ce_baseline_source=single_task_eval does not support "
                            "optimizer.params.merge_impl='functional_clone_legacy'."
                        )
                    logits = _extract_logits(outputs)
                    labels = batch.get("labels") if isinstance(batch, Mapping) else None
                    model_loss = _extract_model_loss(outputs) if ce_use_model_loss else None
                    if model_loss is not None:
                        raw_loss = model_loss.float()
                    else:
                        raw_loss = _compute_supervised_ce_loss(
                            logits,
                            labels,
                            ignore_index=ce_ignore_index,
                            label_smoothing=ce_label_smoothing,
                            reduction=ce_reduction,
                        )
                ce_sum += float(raw_loss.detach().item())
                ce_count += 1
                del batch, outputs, logits, labels, raw_loss
            if ce_count <= 0:
                raise ValueError(f"Failed to compute CE baseline for task '{task}' (no batches).")
            baselines[task] = ce_sum / float(ce_count)
    merged_task, merged_default, merged_layer = _merge_coeffs(
        alpha_task=alpha_task,
        alpha_default=alpha_default,
        alpha_layer=alpha_layer,
        layer_indices=layer_indices,
        coefficient_parameterization=coefficient_parameterization,
        normalize_coefficients=normalize_coefficients,
    )
    coefficient_provider.set_coefficients(
        task_coeffs=merged_task,
        default_coeffs=merged_default,
        layer_coeffs=merged_layer,
    )
    return baselines


__all__ = [
    "_compute_supervised_ce_loss",
    "_extract_model_loss",
    "_validate_selection_split",
    "_normalize_manual_weights",
    "_compute_task_ce_multipliers",
    "_compute_single_task_ce_baselines",
]
