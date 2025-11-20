"""Unified training loop logic for all tasks."""

from __future__ import annotations

import os
import re
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from core.data.io_utils import ensure_dir
from core.evaluation.plotting import plot_confusion_matrix, plot_loss_and_wer

from .run_manager import RunManager
from .trainer import CustomTrainer, save_artifacts, save_history_to_csv


def find_latest_checkpoint(output_dir: str | Path) -> Optional[str]:
    """Find the latest checkpoint in the output directory.

    Args:
        output_dir: Directory to search for checkpoints

    Returns:
        Path to the latest checkpoint directory, or None if no checkpoints found
    """
    output_path = Path(output_dir)
    if not output_path.exists():
        return None

    # Find all checkpoint directories
    checkpoint_dirs = []
    checkpoint_pattern = re.compile(r"checkpoint-(\d+)$")

    for item in output_path.iterdir():
        if not item.is_dir():
            continue
        match = checkpoint_pattern.search(item.name)
        if match:
            step = int(match.group(1))
            # Verify checkpoint is complete by checking for required files
            if _is_checkpoint_complete(item):
                checkpoint_dirs.append((step, item))

    if not checkpoint_dirs:
        return None

    # Sort by step number and return the latest
    checkpoint_dirs.sort(key=lambda x: x[0])
    latest_checkpoint = checkpoint_dirs[-1][1]
    return str(latest_checkpoint)


def _is_checkpoint_complete(checkpoint_dir: Path) -> bool:
    """Check if a checkpoint directory contains all required files.

    Args:
        checkpoint_dir: Path to checkpoint directory

    Returns:
        True if checkpoint is complete, False otherwise
    """
    required_files = [
        "trainer_state.json",
        "optimizer.pt",
        "scheduler.pt",
    ]

    for file in required_files:
        if not (checkpoint_dir / file).exists():
            return False

    # Check for either adapter_model.safetensors or model weights
    has_model = (
        (checkpoint_dir / "adapter_model.safetensors").exists() or
        (checkpoint_dir / "adapter_model.bin").exists() or
        (checkpoint_dir / "pytorch_model.bin").exists() or
        (checkpoint_dir / "model.safetensors").exists()
    )

    return has_model


def resolve_checkpoint_path(
    resume_from_checkpoint: Optional[str],
    output_dir: str | Path,
) -> Optional[str]:
    """Resolve the checkpoint path to use for resuming training.

    Args:
        resume_from_checkpoint: Config value - can be "auto", a path, or None
        output_dir: Training output directory

    Returns:
        Resolved checkpoint path, or None if no checkpoint should be used
    """
    if resume_from_checkpoint is None:
        return None

    if resume_from_checkpoint == "auto":
        # Auto-detect latest checkpoint
        checkpoint_path = find_latest_checkpoint(output_dir)
        if checkpoint_path:
            print(f"🔄 Auto-detected checkpoint: {checkpoint_path}")
        else:
            print("ℹ️  No existing checkpoint found, starting from scratch")
        return checkpoint_path

    # Treat as explicit path
    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.exists():
        raise ValueError(f"Checkpoint path does not exist: {resume_from_checkpoint}")

    if not checkpoint_path.is_dir():
        raise ValueError(f"Checkpoint path is not a directory: {resume_from_checkpoint}")

    if not _is_checkpoint_complete(checkpoint_path):
        raise ValueError(f"Checkpoint is incomplete: {resume_from_checkpoint}")

    print(f"🔄 Resuming from checkpoint: {checkpoint_path}")
    return str(checkpoint_path)


def build_history_record(metrics: Dict[str, Any], *, step: int) -> Optional[Dict[str, Any]]:
    """Transform a metrics dictionary into a history row suitable for CSV export.

    This function extracts numeric metrics and creates a standardized history record.
    It handles any metric name dynamically (WER, F1, accuracy, etc.).

    Args:
        metrics: Dictionary of metrics from evaluation
        step: Training step number

    Returns:
        Dictionary with history record or None if no valid metrics
    """

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_metric_key(key: str) -> bool:
        """Check if a key represents a metric (same logic as save_history_to_csv)."""
        if key in {"step", "epoch"}:
            return False
        if key in {"loss", "learning_rate"}:
            return True
        if key.startswith(("eval_", "train_", "test_")):
            excluded_suffixes = ("runtime", "samples_per_second", "steps_per_second")
            if any(key.endswith(suffix) for suffix in excluded_suffixes):
                return False
            return True
        metric_tokens = ("loss", "accuracy", "precision", "recall", "f1", "wer", "rate", "perplexity")
        return any(token in key for token in metric_tokens)

    # Build record with step
    record: Dict[str, Any] = {"step": int(step)}

    # Add epoch if present
    epoch_value = metrics.get("epoch")
    if epoch_value is not None:
        epoch_float = _to_float(epoch_value)
        if epoch_float is not None:
            record["epoch"] = epoch_float

    # Extract all numeric metrics dynamically
    has_metric = False
    for key, value in metrics.items():
        if not _is_metric_key(key):
            continue
        metric_value = _to_float(value)
        if metric_value is None:
            continue
        record[key] = metric_value
        has_metric = True

    # Return None if no valid metrics found
    if not has_metric:
        return None

    return record


def run_training_with_evaluation(
    trainer: CustomTrainer,
    *,
    processor: Any,
    full_eval_dataset: Dataset,
    initial_eval: bool,
    history_csv_path: Path,
    loss_plot_path: Optional[Path],
    final_adapter_dir: Path,
    config: Optional[Dict[str, Any]] = None,
    config_path: Optional[Path] = None,
    metrics_dir: Optional[Path] = None,
    test_dataset: Optional[Dataset] = None,
    test_split_name: str = "test",
    confusion_matrix_path: Optional[Path] = None,
    confusion_matrix_normalize: bool = True,
    label_names: Optional[List[str]] = None,
    resume_from_checkpoint: Optional[str] = None,
) -> None:
    """Execute training loop with optional initial eval, training, final eval, and artifact saving.

    Args:
        trainer: CustomTrainer instance
        processor: Model processor/tokenizer
        full_eval_dataset: Full evaluation dataset (not truncated)
        initial_eval: Whether to run evaluation before training
        history_csv_path: Path to save training history CSV
        loss_plot_path: Optional path to save loss plot
        final_adapter_dir: Directory to save final adapter
        config: Full configuration dictionary (for run registration)
        config_path: Path to config file (for copying to run directory)
        metrics_dir: Base metrics directory (for copying best/latest metrics)
        test_dataset: Optional test dataset for final evaluation
        test_split_name: Name of the test split for logging (default: "test")
        confusion_matrix_path: Optional path to save confusion matrix plot
        confusion_matrix_normalize: Whether to normalize confusion matrix (default: True)
        label_names: List of label names for confusion matrix axis labels
        resume_from_checkpoint: Optional checkpoint path to resume from
    """
    seed_history_rows: List[Dict[str, Any]] = []

    # Optional initial evaluation
    if initial_eval:
        print("🧪 Running initial evaluation before training...")
        initial_metrics = trainer.evaluate()
        if isinstance(initial_metrics, dict) and initial_metrics:
            scalar_metrics = {
                key: value
                for key, value in initial_metrics.items()
                if isinstance(value, (int, float))
            }
            if scalar_metrics:
                formatted = ", ".join(f"{k}={v:.4f}" for k, v in scalar_metrics.items())
                print(f"📊 Initial metrics: {formatted}")

            # Log initial metrics to wandb if available
            if "wandb" in trainer.args.report_to:
                try:
                    import wandb
                    if wandb.run is not None:
                        # Log initial metrics with step 0
                        wandb_metrics = {k: v for k, v in scalar_metrics.items() if isinstance(v, (int, float))}
                        wandb.log(wandb_metrics, step=0)
                        print(f"📊 Logged initial metrics to wandb")
                except ImportError:
                    pass

            record = build_history_record(initial_metrics, step=0)
            if record is not None:
                seed_history_rows.append(record)

    # Training
    print("🚀 Starting LoRA fine-tuning...")
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)

    # Final evaluation on full validation dataset
    print(f"🎯 Running evaluation on validation split ({len(full_eval_dataset)} samples)...")
    final_eval_metrics = trainer.evaluate(eval_dataset=full_eval_dataset)
    if isinstance(final_eval_metrics, dict) and final_eval_metrics:
        scalar_final_metrics = {
            key: value for key, value in final_eval_metrics.items() if isinstance(value, (int, float))
        }
        if scalar_final_metrics:
            formatted = ", ".join(f"{k}={v:.4f}" for k, v in scalar_final_metrics.items())
            print(f"🏁 Final validation metrics: {formatted}")

    # IMPORTANT: Capture training history NOW, before test evaluation
    # This ensures test metrics never contaminate the training history CSV/plots
    # We make a deep copy of the log_history to freeze the state before test eval
    import copy
    training_log_history = copy.deepcopy(getattr(trainer.state, "log_history", []))

    # Optional test set evaluation (only if test dataset provided)
    # IMPORTANT: Test metrics should NOT be added to training history
    # We evaluate test set AFTER capturing training history to keep them separate
    test_metrics: Optional[Dict[str, Any]] = None
    if test_dataset is not None:
        print(f"🧪 Running evaluation on {test_split_name} split ({len(test_dataset)} samples)...")

        # Temporarily enable prediction storage for confusion matrix if needed
        original_compute_metrics = trainer.compute_metrics
        if confusion_matrix_path and label_names and original_compute_metrics is not None:
            # Wrap the compute_metrics to enable store_predictions
            def compute_metrics_with_predictions(eval_pred):
                # Call original with store_predictions=True if it supports it
                import inspect
                from functools import partial

                # Handle both regular functions and partial functions
                if hasattr(original_compute_metrics, 'func'):
                    # It's already a partial - extract the underlying function and keywords
                    base_func = original_compute_metrics.func
                    existing_keywords = original_compute_metrics.keywords.copy()
                    sig = inspect.signature(base_func)

                    if 'store_predictions' in sig.parameters:
                        # Add store_predictions to existing keywords
                        existing_keywords['store_predictions'] = True
                        temp_fn = partial(base_func, **existing_keywords)
                        return temp_fn(eval_pred)
                    else:
                        return original_compute_metrics(eval_pred)
                else:
                    # It's a regular function
                    sig = inspect.signature(original_compute_metrics)
                    if 'store_predictions' in sig.parameters:
                        temp_fn = partial(original_compute_metrics, store_predictions=True)
                        return temp_fn(eval_pred)
                    return original_compute_metrics(eval_pred)

            trainer.compute_metrics = compute_metrics_with_predictions

        # Evaluate test set - these metrics are captured separately and saved to test_metrics.json
        test_metrics = trainer.evaluate(eval_dataset=test_dataset)

        # Restore original compute_metrics
        if original_compute_metrics is not None:
            trainer.compute_metrics = original_compute_metrics

        # Clean up any non-scalar metrics that might have been added to log_history
        # This prevents TensorBoard from trying to log lists/arrays as "[...]"
        if hasattr(trainer.state, 'log_history'):
            for log_entry in trainer.state.log_history:
                keys_to_remove = [k for k in log_entry.keys() if k.startswith('_') or '/_' in k]
                for key in keys_to_remove:
                    log_entry.pop(key, None)

        if isinstance(test_metrics, dict) and test_metrics:
            scalar_test_metrics = {
                key: value for key, value in test_metrics.items() if isinstance(value, (int, float))
            }
            if scalar_test_metrics:
                formatted = ", ".join(f"{k}={v:.4f}" for k, v in scalar_test_metrics.items())
                print(f"📦 Test set metrics: {formatted}")

            # Generate confusion matrix if predictions and labels are available
            if confusion_matrix_path and label_names:
                predictions = test_metrics.get("_predictions")
                labels = test_metrics.get("_labels")
                if predictions is not None and labels is not None:
                    print("📊 Generating confusion matrix...")
                    plot_confusion_matrix(
                        y_true=labels,
                        y_pred=predictions,
                        label_names=label_names,
                        plot_path=confusion_matrix_path,
                        title=f"Confusion Matrix - {test_split_name.title()} Set",
                        normalize=confusion_matrix_normalize,
                    )

    # Restore the training history before saving (excluding test metrics)
    trainer.state.log_history = training_log_history

    # Determine if we're using the new run management system
    adapter_dir = final_adapter_dir.parent  # This is the adapter base directory
    use_run_manager = config is not None and trainer.args.metric_for_best_model is not None

    if use_run_manager:
        # New run management system
        metric_for_ranking = trainer.args.metric_for_best_model
        greater_is_better = getattr(trainer.args, "greater_is_better", False)

        # Create run manager
        run_manager = RunManager(
            adapter_dir=adapter_dir,
            metric_for_ranking=metric_for_ranking,
            greater_is_better=greater_is_better,
        )

        # Create new run directory
        run_dir = run_manager.create_run_directory()
        print(f"📁 Created run directory: {run_dir.name}")

        # Save training history to run directory (only contains validation metrics)
        run_history_csv = run_dir / "training_history.csv"
        history_rows = save_history_to_csv(trainer, run_history_csv, extra_rows=seed_history_rows)
        if history_rows and loss_plot_path:
            # Also save plot to run directory (only validation metrics)
            run_plot_path = run_dir / loss_plot_path.name
            plot_loss_and_wer(run_history_csv, run_plot_path)

        # Save confusion matrix to run directory if it was generated
        if confusion_matrix_path and confusion_matrix_path.exists():
            run_confusion_matrix_path = run_dir / confusion_matrix_path.name
            shutil.copy(confusion_matrix_path, run_confusion_matrix_path)

        # Save adapter artifacts to run directory
        print("💾 Saving LoRA adapter and processor...")
        save_artifacts(trainer, processor, run_dir)

        # Use test metrics for ranking if available, otherwise validation metrics
        metrics_for_ranking = test_metrics if test_metrics is not None else final_eval_metrics

        # Register the run (saves config, metrics for ranking, updates registry)
        run_manager.register_run(
            run_dir=run_dir,
            metrics=metrics_for_ranking,
            config=config,
            config_path=config_path,
        )

        # Save both validation and test metrics separately
        # These are saved independently and NOT included in training_history.csv
        import json

        # Save validation metrics (final eval on full validation set)
        validation_metrics_path = run_dir / "validation_metrics.json"
        with validation_metrics_path.open("w") as f:
            json.dump(final_eval_metrics, f, indent=2)

        # Save test metrics if available (separate from training history)
        if test_metrics is not None:
            test_metrics_path = run_dir / "test_metrics.json"
            with test_metrics_path.open("w") as f:
                json.dump(test_metrics, f, indent=2)
            print(f"💾 Saved validation and test metrics to run directory (ranked by {'test' if test_metrics is not None else 'validation'} set)")

        # Copy metrics to metrics/best and metrics/latest directories
        # This includes validation, test, history CSV, and plots
        if metrics_dir is not None:
            _copy_metrics_to_subdirs(
                run_manager,
                metrics_dir,
                history_csv_path,
                loss_plot_path,
                confusion_matrix_path,
                has_test_metrics=(test_metrics is not None),
            )

        # Clean up intermediate checkpoints
        _cleanup_checkpoints(adapter_dir)

        print(f"✅ Done: {run_dir.name} (registered and ranked)")
    else:
        # Legacy system (fallback)
        history_rows = save_history_to_csv(trainer, history_csv_path, extra_rows=seed_history_rows)
        if history_rows and loss_plot_path:
            plot_loss_and_wer(history_csv_path, loss_plot_path)

        print("💾 Saving LoRA adapter and processor...")
        ensure_dir(final_adapter_dir)
        save_artifacts(trainer, processor, final_adapter_dir)
        print(f"✅ Done: {final_adapter_dir}")


def _copy_metrics_to_subdirs(
    run_manager: RunManager,
    metrics_dir: Path,
    history_csv_path: Path,
    loss_plot_path: Optional[Path],
    confusion_matrix_path: Optional[Path],
    has_test_metrics: bool = False,
) -> None:
    """Copy metrics from best and latest runs to metrics/best and metrics/latest directories.

    Args:
        run_manager: RunManager instance
        metrics_dir: Base metrics directory
        history_csv_path: Path to training history CSV
        loss_plot_path: Optional path to loss plot
        confusion_matrix_path: Optional path to confusion matrix plot
        has_test_metrics: Whether test metrics were generated
    """
    best_run_path = run_manager.get_best_run_path()
    latest_run_path = run_manager.get_latest_run_path()

    # Copy best run metrics
    if best_run_path:
        best_metrics_dir = ensure_dir(metrics_dir / "best")
        _copy_run_metrics(
            best_run_path,
            best_metrics_dir,
            history_csv_path.name,
            loss_plot_path.name if loss_plot_path else None,
            confusion_matrix_path.name if confusion_matrix_path else None,
            has_test_metrics=has_test_metrics,
        )

    # Copy latest run metrics
    if latest_run_path:
        latest_metrics_dir = ensure_dir(metrics_dir / "latest")
        _copy_run_metrics(
            latest_run_path,
            latest_metrics_dir,
            history_csv_path.name,
            loss_plot_path.name if loss_plot_path else None,
            confusion_matrix_path.name if confusion_matrix_path else None,
            has_test_metrics=has_test_metrics,
        )


def _copy_run_metrics(
    run_dir: Path,
    dest_dir: Path,
    history_csv_name: str,
    loss_plot_name: Optional[str],
    confusion_matrix_name: Optional[str],
    has_test_metrics: bool = False,
) -> None:
    """Copy training history, plots, and metrics from run directory to destination.

    Args:
        run_dir: Source run directory
        dest_dir: Destination directory (e.g., metrics/best or metrics/latest)
        history_csv_name: Name for the training history CSV
        loss_plot_name: Optional name for the loss plot
        confusion_matrix_name: Optional name for the confusion matrix plot
        has_test_metrics: Whether to copy test metrics
    """
    # Copy training history CSV (validation metrics only)
    src_csv = run_dir / "training_history.csv"
    if src_csv.exists():
        dest_csv = dest_dir / history_csv_name
        shutil.copy(src_csv, dest_csv)

    # Copy loss plot (validation metrics only)
    if loss_plot_name:
        # First try exact match with the expected plot name
        src_plot = run_dir / loss_plot_name
        if src_plot.exists():
            dest_plot = dest_dir / loss_plot_name
            shutil.copy(src_plot, dest_plot)
        else:
            # Fallback: try to find the plot in the run directory
            # Look for PNG files that contain keywords like "loss", "accuracy", "f1", "wer", "bleu"
            # but exclude confusion matrix plots
            for item in run_dir.iterdir():
                if (item.suffix == ".png" and
                    "confusion" not in item.name.lower() and
                    any(keyword in item.name.lower() for keyword in ["plot", "loss", "accuracy", "f1", "wer", "bleu"])):
                    dest_plot = dest_dir / loss_plot_name
                    shutil.copy(item, dest_plot)
                    break

    # Copy confusion matrix plot if available
    if confusion_matrix_name:
        src_confusion_matrix = run_dir / confusion_matrix_name
        if src_confusion_matrix.exists():
            dest_confusion_matrix = dest_dir / confusion_matrix_name
            shutil.copy(src_confusion_matrix, dest_confusion_matrix)

    # Copy validation metrics JSON
    src_validation_metrics = run_dir / "validation_metrics.json"
    if src_validation_metrics.exists():
        dest_validation_metrics = dest_dir / "validation_metrics.json"
        shutil.copy(src_validation_metrics, dest_validation_metrics)

    # Copy test metrics JSON if available
    if has_test_metrics:
        src_test_metrics = run_dir / "test_metrics.json"
        if src_test_metrics.exists():
            dest_test_metrics = dest_dir / "test_metrics.json"
            shutil.copy(src_test_metrics, dest_test_metrics)


def _cleanup_checkpoints(adapter_dir: Path) -> None:
    """Remove intermediate checkpoint directories after training completes."""
    for item in adapter_dir.iterdir():
        if item.is_dir() and item.name.startswith("checkpoint-"):
            shutil.rmtree(item)
            print(f"🗑️  Removed checkpoint: {item.name}")


__all__ = [
    "build_history_record",
    "run_training_with_evaluation",
]
