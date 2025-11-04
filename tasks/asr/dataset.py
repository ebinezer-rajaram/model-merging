"""Dataset loading and collation for ASR tasks."""

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from datasets import Dataset, load_dataset


def load_librispeech_10h() -> Tuple[Dataset, Dataset]:
    """Load a 10h subset of LibriSpeech clean for ASR fine-tuning."""
    base = load_dataset("librispeech_asr", "clean")
    train_full, val_ds = base["train.100"], base["validation"]

    def _add_dur(batch):
        arr, sr = batch["audio"]["array"], batch["audio"]["sampling_rate"]
        batch["duration"] = float(len(arr) / sr) if sr and arr is not None else 0.0
        return batch

    train_full = train_full.map(_add_dur)

    target_sec, cumulative, indices = 10 * 3600.0, 0.0, []
    for idx, duration in enumerate(train_full["duration"]):
        cumulative += duration
        indices.append(idx)
        if cumulative >= target_sec:
            break

    print(f"🎧 Selected {len(indices)} samples ≈ {cumulative/3600.0:.2f} h from train.100")
    return train_full.select(indices), val_ds


@dataclass
class OmniASRCollator:
    """Data collator for Omni ASR inputs."""

    processor: Any
    sampling_rate: int

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, torch.Tensor]:
        """Prepare batch tensors for the trainer."""
        audio_arrays = [feature["audio"]["array"] for feature in features]
        texts = [feature["text"] for feature in features]
        prompts = [self.processor.audio_token + text for text in texts]

        inputs = self.processor(
            audio=audio_arrays,
            sampling_rate=self.sampling_rate,
            text=prompts,
            return_tensors="pt",
            padding=True,
        )

        labels = inputs["input_ids"].clone()
        pad_id = self.processor.tokenizer.pad_token_id
        audio_token_id = self.processor.tokenizer.convert_tokens_to_ids(
            self.processor.audio_token
        )

        labels = labels.masked_fill(labels == pad_id, -100)
        labels = labels.masked_fill(labels == audio_token_id, -100)
        inputs["labels"] = labels
        return inputs
