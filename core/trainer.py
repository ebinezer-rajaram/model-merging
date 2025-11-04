"""Trainer configuration and utilities."""

import csv
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch
from transformers import Trainer


class CustomTrainer(Trainer):
    """Custom trainer that performs generation during evaluation."""

    def __init__(self, *args, generation_kwargs: Optional[Dict[str, Any]] = None, **kwargs):
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 128, "do_sample": False}
        super().__init__(*args, **kwargs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run generation to obtain predictions for metric computation."""
        has_labels = "labels" in inputs
        if prediction_loss_only and not has_labels:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs).detach() if has_labels else None

        gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}
        generated = model.generate(**gen_inputs, **self.generation_kwargs)

        preds = generated.detach().cpu()
        label_ids = inputs["labels"].detach().cpu() if has_labels else None

        return loss, preds, label_ids


def save_history_to_csv(trainer_instance: Trainer, csv_path: Path) -> List[Dict[str, Any]]:
    """Persist relevant training logs to CSV for downstream analysis."""
    log_history = getattr(trainer_instance.state, "log_history", [])
    relevant_rows: List[Dict[str, Any]] = []

    for entry in log_history:
        record = {
            "step": entry.get("step"),
            "loss": entry.get("loss"),
            "eval_loss": entry.get("eval_loss"),
            "wer": entry.get("eval_wer") or entry.get("wer"),
            "learning_rate": entry.get("learning_rate"),
        }

        has_metric = any(
            record[key] is not None for key in ("loss", "eval_loss", "wer", "learning_rate")
        )
        if has_metric:
            relevant_rows.append(record)

    if not relevant_rows:
        print("ℹ️ No training metrics available to write.")
        return []

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=["step", "loss", "eval_loss", "wer", "learning_rate"],
        )
        writer.writeheader()
        writer.writerows(relevant_rows)

    print(f"📈 Saved training history to {csv_path}")
    return relevant_rows


def save_artifacts(trainer_instance: Trainer, processor, output_dir: Path) -> None:
    """Save the trained adapter and processor."""
    trainer_instance.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
