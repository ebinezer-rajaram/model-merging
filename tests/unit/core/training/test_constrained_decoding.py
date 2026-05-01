from __future__ import annotations

import torch

from core.training.constrained_decoding import create_classification_constraint, create_multi_token_constraint
from tests.helpers.core import CoreDummyProcessor


def test_classification_constraint_allows_label_space_and_eos() -> None:
    processor = CoreDummyProcessor()
    constraint = create_classification_constraint(processor, ["book flight", "play"], allow_eos=True)
    allowed = constraint(0, torch.tensor([1, 2]))
    assert processor.tokenizer.eos_token_id in allowed
    assert processor.tokenizer.encode("book", add_special_tokens=False)[0] in allowed


def test_multi_token_constraint_can_disable_eos() -> None:
    processor = CoreDummyProcessor()
    constraint = create_multi_token_constraint(processor, ["book flight", ""], allow_eos=False)
    allowed = constraint(0, torch.tensor([1]))
    assert processor.tokenizer.eos_token_id not in allowed
    assert processor.tokenizer.encode("flight", add_special_tokens=False)[0] in allowed
