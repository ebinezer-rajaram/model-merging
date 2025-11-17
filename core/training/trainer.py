"""Trainer configuration and utilities."""

import csv
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Sampler
from transformers import Trainer


class CustomTrainer(Trainer):
    """Custom trainer that optionally performs generation during evaluation."""

    def __init__(
        self,
        *args,
        generation_kwargs: Optional[Dict[str, Any]] = None,
        enable_generation: bool = True,
        constrained_decoding_fn: Optional[Callable] = None,
        custom_loss_fn: Optional[nn.Module] = None,
        custom_sampler: Optional[Sampler] = None,
        **kwargs,
    ):
        self.generation_kwargs = generation_kwargs or {"max_new_tokens": 128, "do_sample": False}
        self.enable_generation = enable_generation
        self.constrained_decoding_fn = constrained_decoding_fn
        self.custom_loss_fn = custom_loss_fn
        self.custom_sampler = custom_sampler
        super().__init__(*args, **kwargs)

    def log(self, logs: Dict[str, float], start_time: Optional[float] = None) -> None:
        """Override log to filter out non-scalar metrics before logging to TensorBoard.

        Filters out keys starting with '_' or containing '/_' which are used to pass
        non-scalar data (like predictions/labels arrays) between compute_metrics and
        evaluation code.
        """
        # Filter out keys starting with '_' or containing '/_' (non-scalar metrics)
        filtered_logs = {k: v for k, v in logs.items() if not k.startswith('_') and '/_' not in k}
        if start_time is not None:
            super().log(filtered_logs, start_time)
        else:
            super().log(filtered_logs)

    def prediction_step(
        self,
        model: torch.nn.Module,
        inputs: Dict[str, torch.Tensor],
        prediction_loss_only: bool,
        ignore_keys: Optional[Sequence[str]] = None,
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """Run generation to obtain predictions for metric computation."""
        if not self.enable_generation or not hasattr(model, "generate"):
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        has_labels = "labels" in inputs
        if prediction_loss_only and not has_labels:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys)

        inputs = self._prepare_inputs(inputs)
        with torch.no_grad():
            loss = self.compute_loss(model, inputs).detach() if has_labels else None

        prompt_lengths = None
        if has_labels:
            label_ids = inputs.get("labels")
            if label_ids is not None:
                prompt_lengths = []
                for row in label_ids:
                    non_mask = torch.nonzero(row != -100, as_tuple=False)
                    if non_mask.numel() == 0:
                        prompt_lengths.append(row.size(0))
                    else:
                        prompt_lengths.append(non_mask[0].item())

        gen_inputs = {k: v for k, v in inputs.items() if k != "labels"}

        if prompt_lengths is not None and "input_ids" in gen_inputs:
            input_ids = gen_inputs["input_ids"]
            attention_mask = gen_inputs.get("attention_mask")

            seq_len = input_ids.size(1)
            pad_token_id = getattr(model.config, "pad_token_id", None)
            if pad_token_id is None:
                tokenizer = getattr(self, "tokenizer", None)
                pad_token_id = getattr(tokenizer, "pad_token_id", 0)

            max_prompt_len = max(int(length) for length in prompt_lengths) if prompt_lengths else seq_len
            max_prompt_len = min(max_prompt_len, seq_len)

            new_input_ids = input_ids.new_full((input_ids.size(0), max_prompt_len), pad_token_id)
            new_attention_mask = None
            if attention_mask is not None:
                new_attention_mask = attention_mask.new_zeros((attention_mask.size(0), max_prompt_len))

            for row_idx, prompt_len in enumerate(prompt_lengths):
                prompt_len = int(prompt_len)
                if prompt_len <= 0:
                    continue
                prompt_len = min(prompt_len, seq_len)

                source_tokens = input_ids[row_idx, :prompt_len]
                dest_len = min(prompt_len, max_prompt_len)
                new_input_ids[row_idx, -dest_len:] = source_tokens[-dest_len:]

                if new_attention_mask is not None:
                    if attention_mask is not None:
                        source_mask = attention_mask[row_idx, :prompt_len]
                    else:
                        source_mask = source_tokens.new_ones((prompt_len,), dtype=new_attention_mask.dtype)
                    new_attention_mask[row_idx, -dest_len:] = source_mask[-dest_len:]

            gen_inputs["input_ids"] = new_input_ids
            if new_attention_mask is not None:
                gen_inputs["attention_mask"] = new_attention_mask

        # Apply constrained decoding if available
        final_gen_kwargs = dict(self.generation_kwargs)
        if self.constrained_decoding_fn is not None:
            final_gen_kwargs["prefix_allowed_tokens_fn"] = self.constrained_decoding_fn

        # Ensure EOS token handling is properly configured
        # Build list of EOS token IDs (matching reference test script)
        if "eos_token_id" not in final_gen_kwargs or final_gen_kwargs["eos_token_id"] is None:
            # Get tokenizer from processor
            tokenizer = getattr(self, "tokenizer", None) or getattr(self.processing_class, "tokenizer", None)
            if tokenizer is not None:
                # Start with base EOS token
                eos_token_ids = [tokenizer.eos_token_id]

                # Add newline token if it exists and is not UNK
                newline_id = tokenizer.convert_tokens_to_ids("\n")
                if newline_id is not None and newline_id != tokenizer.unk_token_id:
                    eos_token_ids.append(newline_id)

                # Add im_end token if it exists and is not UNK
                im_end_id = tokenizer.convert_tokens_to_ids("<|im_end|>")
                if im_end_id is not None and im_end_id != tokenizer.unk_token_id:
                    eos_token_ids.append(im_end_id)

                final_gen_kwargs["eos_token_id"] = eos_token_ids
            else:
                # Fallback to model's configured EOS token ID
                eos_token_id = getattr(model.config, "eos_token_id", None)
                if eos_token_id is not None:
                    final_gen_kwargs["eos_token_id"] = eos_token_id

        if "pad_token_id" not in final_gen_kwargs or final_gen_kwargs["pad_token_id"] is None:
            # Use model's configured pad token ID
            pad_token_id = getattr(model.config, "pad_token_id", None)
            if pad_token_id is not None:
                final_gen_kwargs["pad_token_id"] = pad_token_id

        # Enable early stopping for beam search to stop when EOS is generated
        if final_gen_kwargs.get("num_beams", 1) > 1:
            final_gen_kwargs.setdefault("early_stopping", True)

        generated = model.generate(**gen_inputs, **final_gen_kwargs)

        preds = generated.detach().cpu()
        label_ids = inputs["labels"].detach().cpu() if has_labels else None

        return loss, preds, label_ids

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """Compute loss using custom loss function if provided."""
        if self.custom_loss_fn is None:
            # Use default loss computation
            return super().compute_loss(model, inputs, return_outputs, num_items_in_batch)

        # Extract labels
        labels = inputs.get("labels")
        if labels is None:
            raise ValueError("Labels must be provided when using custom loss function")

        # Forward pass
        outputs = model(**inputs)

        # Get logits from model outputs
        logits = outputs.get("logits")
        if logits is None:
            raise ValueError("Model outputs must contain 'logits' when using custom loss function")

        # Compute custom loss
        loss = self.custom_loss_fn(logits, labels)

        return (loss, outputs) if return_outputs else loss

    def get_train_dataloader(self) -> DataLoader:
        """Override to support custom samplers (batch samplers or regular samplers)."""
        if self.custom_sampler is None:
            return super().get_train_dataloader()

        # Use custom sampler
        from torch.utils.data import BatchSampler

        if isinstance(self.custom_sampler, BatchSampler):
            # Batch sampler - handles batching internally
            return DataLoader(
                self.train_dataset,
                batch_sampler=self.custom_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )
        else:
            # Regular sampler - need to specify batch size
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=self.custom_sampler,
                collate_fn=self.data_collator,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
                drop_last=self.args.dataloader_drop_last,
            )


def save_history_to_csv(
    trainer_instance: Trainer,
    csv_path: Path,
    *,
    extra_rows: Optional[List[Dict[str, Any]]] = None,
) -> List[Dict[str, Any]]:
    """Persist relevant training logs to CSV for downstream analysis."""
    log_history = getattr(trainer_instance.state, "log_history", [])
    relevant_rows: List[Dict[str, Any]] = []
    seen_rows: set[Tuple[Tuple[str, Any], ...]] = set()

    def _to_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def _is_metric_key(key: str) -> bool:
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

    def _normalize(entry: Dict[str, Any]) -> Dict[str, Any]:
        record: Dict[str, Any] = {}

        step_value = entry.get("step")
        if step_value is not None:
            try:
                record["step"] = int(float(step_value))
            except (TypeError, ValueError):
                pass

        epoch_value = entry.get("epoch")
        if epoch_value is not None:
            epoch_float = _to_float(epoch_value)
            if epoch_float is not None:
                record["epoch"] = epoch_float

        for key, value in entry.items():
            if not _is_metric_key(key):
                continue
            metric_value = _to_float(value)
            if metric_value is None:
                continue
            # Normalize "loss" to "train_loss" for consistency
            normalized_key = "train_loss" if key == "loss" else key
            record[normalized_key] = metric_value

        return record

    def _append(entry: Dict[str, Any]) -> None:
        record = _normalize(entry)
        metric_keys = [key for key in record.keys() if key not in {"step", "epoch"}]
        has_metric = any(record.get(key) is not None for key in metric_keys)
        if not has_metric:
            return
        signature = tuple(sorted(record.items()))
        if signature in seen_rows:
            return
        seen_rows.add(signature)
        relevant_rows.append(record)

    if extra_rows:
        for row in extra_rows:
            if isinstance(row, dict):
                _append(row)

    for entry in log_history:
        _append(entry)

    if not relevant_rows:
        print("ℹ️ No training metrics available to write.")
        return []

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    field_priorities = ["step", "epoch", "train_loss", "eval_loss", "wer", "eval_wer", "learning_rate"]
    discovered_fields = {key for row in relevant_rows for key in row.keys()}
    ordered_fields = [field for field in field_priorities if field in discovered_fields]
    remaining_fields = sorted(discovered_fields - set(ordered_fields))
    fieldnames = ordered_fields + remaining_fields

    with csv_path.open("w", newline="") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=fieldnames)
        writer.writeheader()
        for row in relevant_rows:
            writer.writerow({field: row.get(field) for field in fieldnames})

    print(f"📈 Saved training history to {csv_path}")
    return relevant_rows


def save_artifacts(trainer_instance: Trainer, processor, output_dir: Path) -> None:
    """Save the trained adapter and processor."""
    trainer_instance.save_model(str(output_dir))
    processor.save_pretrained(str(output_dir))
