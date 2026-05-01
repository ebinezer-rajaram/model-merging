from __future__ import annotations

import pytest

from core.training.samplers import BalancedBatchSampler, WeightedClassSampler
from tests.helpers.core import TinyLabeledDataset


def test_balanced_batch_sampler_yields_homogeneous_class_counts() -> None:
    sampler = BalancedBatchSampler(TinyLabeledDataset([0, 0, 1, 1]), batch_size=2, num_classes=2, shuffle=False)
    batch = next(iter(sampler))
    assert len(batch) == 2
    assert len(sampler) == 2


def test_weighted_class_sampler_methods_and_validation() -> None:
    dataset = TinyLabeledDataset([0, 0, 0, 1])
    for method in ("inverse", "sqrt_inverse", "balanced"):
        sampler = WeightedClassSampler(dataset, num_samples=3, method=method)
        assert len(list(iter(sampler))) == 3
        assert len(sampler) == 3
    with pytest.raises(ValueError, match="Unknown weighting method"):
        WeightedClassSampler(dataset, method="bad")
