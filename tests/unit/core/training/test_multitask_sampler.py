from __future__ import annotations

import pytest

from core.training.multitask_sampler import (
    MultiTaskDataset,
    TemperatureMultiTaskBatchSampler,
    compute_task_probabilities,
    estimate_batches_per_epoch,
)


def test_multitask_dataset_adds_task_names_and_exposes_global_indices() -> None:
    dataset = MultiTaskDataset({"asr": [{"x": 1}], "emotion": [{"x": 2}, {"x": 3}]})
    assert len(dataset) == 3
    assert dataset[0] == {"x": 1, "__task_name": "asr"}
    assert dataset[2] == {"x": 3, "__task_name": "emotion"}
    assert dataset.task_to_global_indices == {"asr": [0], "emotion": [1, 2]}
    with pytest.raises(IndexError):
        _ = dataset[3]
    with pytest.raises(ValueError, match="non-empty"):
        MultiTaskDataset({})
    with pytest.raises(ValueError, match="no samples"):
        MultiTaskDataset({"empty": []})


def test_task_probability_temperature_and_validation() -> None:
    probs = compute_task_probabilities(task_sizes={"a": 1, "b": 3}, task_weights={"a": 2.0, "b": 1.0}, temperature=1.0)
    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["b"] > probs["a"]
    with pytest.raises(ValueError, match="temperature"):
        compute_task_probabilities(task_sizes={"a": 1}, task_weights={}, temperature=0.0)
    with pytest.raises(ValueError, match="non-positive"):
        compute_task_probabilities(task_sizes={"a": 1}, task_weights={"a": 0.0}, temperature=1.0)
    with pytest.raises(ValueError, match="No positive"):
        compute_task_probabilities(task_sizes={"a": 0}, task_weights={}, temperature=1.0)


def test_temperature_batch_sampler_yields_homogeneous_fixed_size_batches() -> None:
    sampler = TemperatureMultiTaskBatchSampler(
        task_to_indices={"a": [0, 1], "b": [2, 3, 4]},
        task_weights={"a": 1.0, "b": 1.0},
        batch_size=2,
        temperature=1.0,
        num_batches=4,
        seed=1,
    )
    batches = list(iter(sampler))
    assert len(batches) == 4
    assert all(len(batch) == 2 for batch in batches)
    assert len(sampler) == 4
    with pytest.raises(ValueError, match="batch_size"):
        TemperatureMultiTaskBatchSampler(task_to_indices={"a": [1]}, task_weights={}, batch_size=0, temperature=1, num_batches=1)


def test_estimate_batches_per_epoch_validation() -> None:
    assert estimate_batches_per_epoch(5, 2, drop_last=True) == 2
    assert estimate_batches_per_epoch(5, 2, drop_last=False) == 3
    with pytest.raises(ValueError, match="batch_size"):
        estimate_batches_per_epoch(5, 0)
    with pytest.raises(ValueError, match="total_examples"):
        estimate_batches_per_epoch(0, 2)
