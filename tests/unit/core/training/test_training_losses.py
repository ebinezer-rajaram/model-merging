from __future__ import annotations

import pytest
import torch

from core.training.losses import FocalLoss, WeightedCrossEntropyLoss, compute_class_weights


def test_focal_loss_handles_sequence_layouts_alpha_reductions_and_ignore_index() -> None:
    logits_blc = torch.tensor([[[3.0, 0.1], [0.2, 2.0], [4.0, 0.1]]])
    targets = torch.tensor([[0, 1, -100]])

    loss_blc = FocalLoss(gamma=0.0)(logits_blc, targets)
    loss_bcl = FocalLoss(gamma=0.0)(logits_blc.transpose(1, 2), targets)
    assert loss_blc.item() > 0.0
    assert loss_bcl == pytest.approx(loss_blc)

    per_class = FocalLoss(alpha=[1.0, 2.0], gamma=1.0, reduction="none")(logits_blc, targets)
    assert per_class.shape == targets.shape
    assert per_class[0, -1] == 0.0
    assert FocalLoss(alpha=0.5, gamma=1.0, reduction="sum")(logits_blc, targets).item() > 0.0


def test_loss_helpers_validate_shapes_and_weight_methods() -> None:
    targets = torch.tensor([0, 1, -100])
    with pytest.raises(ValueError, match="Expected inputs"):
        FocalLoss()(torch.ones(1, 1, 1, 1), targets)
    with pytest.raises(ValueError, match="Shape mismatch"):
        FocalLoss()(torch.ones(2, 2), torch.ones(3, dtype=torch.long))

    sequence_logits = torch.tensor([[[2.0, 0.1], [0.1, 2.0]]])
    sequence_targets = torch.tensor([[0, 1]])
    loss = WeightedCrossEntropyLoss(torch.tensor([1.0, 2.0]), reduction="sum")(sequence_logits, sequence_targets)
    assert loss.item() > 0.0
    with pytest.raises(ValueError, match="Expected inputs"):
        WeightedCrossEntropyLoss([1.0, 1.0])(torch.ones(1, 1, 1, 1), sequence_targets)

    labels = torch.tensor([0, 0, 1])
    assert compute_class_weights(labels, 3, method="sqrt_inverse")[1] > compute_class_weights(labels, 3, method="sqrt_inverse")[0]
    assert torch.equal(compute_class_weights(labels, 3, method="balanced"), compute_class_weights(labels, 3, method="inverse"))
    with pytest.raises(ValueError, match="Unknown method"):
        compute_class_weights([0], num_classes=2, method="bad")
