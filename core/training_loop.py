"""Unified training loop logic for all tasks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Optional

from datasets import Dataset

from .io_utils import ensure_dir
from .plotting import plot_loss_and_wer
from .trainer import CustomTrainer, save_artifacts, save_history_to_csv


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

    # Extract common metrics with backward compatibility
    loss = _to_float(metrics.get("loss"))
    eval_loss = _to_float(metrics.get("eval_loss"))
    learning_rate = _to_float(metrics.get("learning_rate"))

    # Extract eval metric (could be WER, F1, accuracy, etc.)
    # Priority: eval_wer > wer > eval_f1 > f1 > eval_macro_f1 > macro_f1 > eval_accuracy > accuracy
    eval_metric = None
    for key in ["eval_wer", "wer", "eval_f1", "f1", "eval_macro_f1", "macro_f1", "eval_accuracy", "accuracy"]:
        if key in metrics:
            eval_metric = _to_float(metrics.get(key))
            if eval_metric is not None:
                break

    if all(value is None for value in (loss, eval_loss, eval_metric, learning_rate)):
        return None

    return {
        "step": int(step),
        "loss": loss,
        "eval_loss": eval_loss,
        "wer": eval_metric,  # Column name kept as "wer" for backward compatibility with plotting
        "learning_rate": learning_rate,
    }


def run_training_with_evaluation(
    trainer: CustomTrainer,
    *,
    processor: Any,
    full_eval_dataset: Dataset,
    initial_eval: bool,
    history_csv_path: Path,
    loss_plot_path: Optional[Path],
    final_adapter_dir: Path,
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
            record = build_history_record(initial_metrics, step=0)
            if record is not None:
                seed_history_rows.append(record)

    # Training
    print("🚀 Starting LoRA fine-tuning...")
    trainer.train()

    # Final evaluation on full dataset
    print(f"🎯 Running evaluation on held-out split ({len(full_eval_dataset)} samples)...")
    final_eval_metrics = trainer.evaluate(eval_dataset=full_eval_dataset)
    if isinstance(final_eval_metrics, dict) and final_eval_metrics:
        scalar_final_metrics = {
            key: value for key, value in final_eval_metrics.items() if isinstance(value, (int, float))
        }
        if scalar_final_metrics:
            formatted = ", ".join(f"{k}={v:.4f}" for k, v in scalar_final_metrics.items())
            print(f"🏁 Final evaluation metrics: {formatted}")

    # Save training history
    history_rows = save_history_to_csv(trainer, history_csv_path, extra_rows=seed_history_rows)
    if history_rows and loss_plot_path:
        plot_loss_and_wer(history_csv_path, loss_plot_path)

    # Save final artifacts
    print("💾 Saving LoRA adapter and processor...")
    ensure_dir(final_adapter_dir)
    save_artifacts(trainer, processor, final_adapter_dir)
    print(f"✅ Done: {final_adapter_dir}")


__all__ = [
    "build_history_record",
    "run_training_with_evaluation",
]
