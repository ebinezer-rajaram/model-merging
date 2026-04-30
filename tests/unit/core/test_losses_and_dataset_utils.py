from __future__ import annotations

import pytest
import torch
from datasets import Dataset

from core.data.dataset_utils import (
    add_duration,
    compute_split_hours,
    hours_key,
    hours_to_seconds,
    num_proc_map_kwargs,
    resolve_num_proc,
    select_indices_by_duration,
    select_random_indices,
)
from core.training.losses import FocalLoss, WeightedCrossEntropyLoss, compute_class_weights


def test_focal_loss_handles_sequence_layouts_and_ignore_index() -> None:
    logits_blc = torch.tensor([[[3.0, 0.1], [0.2, 2.0], [4.0, 0.1]]])
    targets = torch.tensor([[0, 1, -100]])

    loss_blc = FocalLoss(gamma=0.0)(logits_blc, targets)
    loss_bcl = FocalLoss(gamma=0.0)(logits_blc.transpose(1, 2), targets)

    assert loss_blc.item() > 0.0
    assert loss_bcl == pytest.approx(loss_blc)


def test_weighted_cross_entropy_and_class_weights() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, -100])
    loss = WeightedCrossEntropyLoss([1.0, 3.0])(logits, targets)

    weights = compute_class_weights([0, 0, 1], num_classes=3, method="inverse")

    assert loss.item() > 0.0
    assert weights[1] > weights[0]
    assert weights[2] == pytest.approx(1.0)
    with pytest.raises(ValueError, match="Unknown method"):
        compute_class_weights([0], num_classes=2, method="bad")


def test_dataset_duration_and_sampling_helpers_are_deterministic() -> None:
    row = add_duration({"audio": {"array": [0.0] * 160, "sampling_rate": 16000}})
    ds = Dataset.from_list(
        [
            {"id": "a", "duration": 1.0},
            {"id": "b", "duration": 2.0},
            {"id": "c", "duration": 3.0},
        ]
    )

    assert row["duration"] == pytest.approx(0.01)
    assert hours_to_seconds(0.5) == 1800.0
    assert hours_key(None) == "full"
    assert hours_key(1.25) == "001250"
    assert resolve_num_proc(0) == 1
    assert num_proc_map_kwargs(1) == {}
    assert num_proc_map_kwargs(2) == {"num_proc": 2}
    assert select_indices_by_duration([1.0, 2.0, 3.0], 0, seed=7) == []
    assert select_indices_by_duration([1.0, 2.0, 3.0], None, seed=7) == [0, 1, 2]
    assert select_random_indices(5, 0, seed=7) == []
    assert sorted(select_random_indices(5, 2, seed=7)) == sorted(select_random_indices(5, 2, seed=7))
    assert compute_split_hours(ds, [0, 2]) == pytest.approx(4.0 / 3600.0)
