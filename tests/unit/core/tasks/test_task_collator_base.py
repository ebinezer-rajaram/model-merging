from __future__ import annotations

from dataclasses import dataclass

import pytest
import torch

from core.tasks.collator import BaseClassificationCollator, BaseGenerationCollator, build_strict_label_mask
from tests.helpers.core import CoreDummyProcessor


@dataclass
class DemoClassificationCollator(BaseClassificationCollator):
    def _build_instruction(self, feature):
        return f"Classify: {feature.get('text', '')}"


@dataclass
class DemoGenerationCollator(BaseGenerationCollator):
    def _build_instruction(self, feature):
        return f"Transcribe: {feature.get('text', '')}"


def _feature(label=1):
    return {"audio": {"array": [0.0, 1.0]}, "label": label, "text": "hello", "answer": "free form"}


def test_strict_label_mask_keeps_rightmost_exact_label_span() -> None:
    input_ids = torch.tensor([5, 9, 8, 9, 8, 0])
    labels = input_ids.clone()
    stats = build_strict_label_mask(input_ids=input_ids, labels=labels, label_tokens=[9, 8])
    assert stats["matched"] is True
    assert torch.equal(labels, torch.tensor([-100, -100, -100, 9, 8, -100]))


def test_strict_label_mask_fallbacks_are_deterministic() -> None:
    empty = torch.tensor([1, 2, 3])
    stats = build_strict_label_mask(input_ids=empty, labels=empty.clone(), label_tokens=[])
    assert stats["used_fallback"] is True
    assert stats["fallback_reason"] == "empty_label_tokens"

    fallback_labels = torch.tensor([-100, 4, 5, 6])
    fallback = build_strict_label_mask(
        input_ids=torch.tensor([1, 2, 3, 4]),
        labels=fallback_labels,
        label_tokens=[8, 9],
    )
    assert fallback["used_fallback"] is True
    assert torch.equal(fallback_labels, torch.tensor([-100, -100, 3, 4]))


def test_base_classification_collator_builds_left_padded_batch_and_masks_labels() -> None:
    processor = CoreDummyProcessor()
    collator = DemoClassificationCollator(
        processor=processor,
        sampling_rate=16000,
        label_names=["negative", "positive"],
        warn_on_label_mask_fallback=False,
    )
    batch = collator([_feature(1), _feature("negative")])

    assert processor.tokenizer.padding_side == "left"
    assert batch["input_ids"].shape == batch["labels"].shape
    assert processor.calls[0]["sampling_rate"] == 16000
    assert processor.conversations[0][1]["content"][0]["text"] == "positive"
    assert torch.all(batch["labels"][batch["input_ids"] == processor.tokenizer.pad_token_id] == -100)
    assert (batch["labels"] != -100).sum().item() >= 2


def test_base_generation_collator_uses_common_target_fields() -> None:
    processor = CoreDummyProcessor()
    collator = DemoGenerationCollator(processor=processor, sampling_rate=8000, warn_on_label_mask_fallback=False)
    batch = collator([{"audio": {"array": [0.0]}, "answer": "free form"}])
    assert processor.conversations[0][1]["content"][0]["text"] == "free form"
    assert batch["attention_mask"].sum().item() > 0


def test_base_collator_filters_corrupted_audio_and_errors_when_all_bad() -> None:
    processor = CoreDummyProcessor()
    collator = DemoClassificationCollator(processor=processor, sampling_rate=16000, label_names=["x"])
    with pytest.raises(RuntimeError, match="All audio samples"):
        collator([{"audio": object(), "label": 0}])
