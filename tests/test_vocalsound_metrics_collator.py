from __future__ import annotations

from typing import Any, Dict, List

import numpy as np
import torch

from tasks.vocalsound.dataset import VocalSoundCollator
from tasks.vocalsound.metrics import compute_vocalsound_metrics


class _DummyTokenizerForCollator:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.padding_side = "right"

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<audio>":
            return 1
        return 2

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [ord(ch) for ch in text]


class _DummyProcessorForCollator:
    def __init__(self) -> None:
        self.tokenizer = _DummyTokenizerForCollator()
        self.audio_token = "<audio>"

    def apply_chat_template(self, conversation: List[Dict[str, Any]], add_generation_prompt: bool, tokenize: bool) -> str:
        del add_generation_prompt, tokenize
        label = conversation[1]["content"][0]["text"]
        return f"user\nassistant\n{label}"

    def __call__(self, *, audio, sampling_rate: int, text, return_tensors: str, padding: bool) -> Dict[str, torch.Tensor]:
        del audio, sampling_rate, return_tensors, padding
        encoded = [self.tokenizer.encode(item, add_special_tokens=False) for item in text]
        max_len = max(len(row) for row in encoded)
        padded = []
        for row in encoded:
            pad_count = max_len - len(row)
            padded.append(([self.tokenizer.pad_token_id] * pad_count) + row)
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}


class _DummyTokenizerForMetrics:
    def __init__(self, token_map: Dict[int, str]):
        self._token_map = dict(token_map)

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        return "".join(self._token_map.get(int(token_id), "") for token_id in token_ids)


class _DummyProcessorForMetrics:
    def __init__(self, token_map: Dict[int, str]):
        self.tokenizer = _DummyTokenizerForMetrics(token_map)


def test_vocalsound_collator_builds_masked_labels() -> None:
    processor = _DummyProcessorForCollator()
    collator = VocalSoundCollator(
        processor=processor,
        sampling_rate=16000,
        label_names=["cough", "sneeze"],
    )

    features = [
        {"audio": {"array": [0.0] * 80}, "label": 0},
        {"audio": {"array": [0.0] * 120}, "label": 1},
    ]
    batch = collator(features)

    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == batch["labels"].shape
    assert batch["labels"].shape[0] == 2
    assert int((batch["labels"][0] != -100).sum().item()) > 0
    assert int((batch["labels"][1] != -100).sum().item()) > 0


def test_vocalsound_metrics_returns_classification_scores_and_predictions() -> None:
    token_map = {
        11: "assistant\ncough",
        12: "assistant\nsneeze",
        21: "assistant\ncough",
        22: "assistant\nsneeze",
    }
    processor = _DummyProcessorForMetrics(token_map)
    preds = np.array([[11], [12]])
    labels = np.array([[21, -100], [22, -100]])

    metrics = compute_vocalsound_metrics(
        (preds, labels),
        processor=processor,
        label_names=["cough", "sneeze"],
        store_predictions=True,
    )

    assert metrics["accuracy"] == 1.0
    assert metrics["macro_f1"] == 1.0
    assert metrics["weighted_f1"] == 1.0
    assert metrics["recognized_rate"] == 1.0
    assert metrics["num_samples"] == 2.0
    assert "_predictions" in metrics
    assert "_labels" in metrics
