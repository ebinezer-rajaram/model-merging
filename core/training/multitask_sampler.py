"""Batch samplers and dataset wrappers for joint multi-task training."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

import numpy as np
from torch.utils.data import BatchSampler, Dataset


@dataclass(frozen=True)
class TaskIndexRange:
    """Dense global-index range for one task in a concatenated dataset."""

    task_name: str
    start: int
    end: int


class MultiTaskDataset(Dataset):
    """Concatenate per-task datasets while preserving task routing metadata."""

    def __init__(self, datasets_by_task: Mapping[str, Dataset]):
        if not datasets_by_task:
            raise ValueError("datasets_by_task must be non-empty.")

        self.datasets_by_task: Dict[str, Dataset] = dict(datasets_by_task)
        self.task_ranges: List[TaskIndexRange] = []
        self._task_to_global_indices: Dict[str, List[int]] = {}

        cursor = 0
        for task, dataset in self.datasets_by_task.items():
            size = len(dataset)
            if size <= 0:
                raise ValueError(f"Task '{task}' has no samples in the selected split.")
            start = cursor
            end = cursor + size
            self.task_ranges.append(TaskIndexRange(task_name=task, start=start, end=end))
            self._task_to_global_indices[task] = list(range(start, end))
            cursor = end

        self._length = cursor

    def __len__(self) -> int:
        return self._length

    def __getitem__(self, index: int):
        idx = int(index)
        if idx < 0 or idx >= self._length:
            raise IndexError(f"Index {index} out of range for MultiTaskDataset with length {self._length}.")

        for entry in self.task_ranges:
            if entry.start <= idx < entry.end:
                local_idx = idx - entry.start
                row = dict(self.datasets_by_task[entry.task_name][local_idx])
                row["__task_name"] = entry.task_name
                return row

        raise IndexError(f"Failed to resolve index {index}.")

    @property
    def task_to_global_indices(self) -> Dict[str, List[int]]:
        return {task: list(indices) for task, indices in self._task_to_global_indices.items()}


class TemperatureMultiTaskBatchSampler(BatchSampler):
    """Sample homogeneous task batches using temperature-scaled task probabilities."""

    def __init__(
        self,
        *,
        task_to_indices: Mapping[str, Sequence[int]],
        task_weights: Mapping[str, float],
        batch_size: int,
        temperature: float,
        num_batches: int,
        drop_last: bool = True,
        seed: int = 0,
    ):
        if batch_size <= 0:
            raise ValueError("batch_size must be > 0")
        if temperature <= 0.0:
            raise ValueError("temperature must be > 0")
        if num_batches <= 0:
            raise ValueError("num_batches must be > 0")

        self.batch_size = int(batch_size)
        self.temperature = float(temperature)
        self.num_batches = int(num_batches)
        self.drop_last = bool(drop_last)
        self.seed = int(seed)

        self.task_names = [task for task, indices in task_to_indices.items() if len(indices) > 0]
        if not self.task_names:
            raise ValueError("No task has indices for sampling.")

        self.task_to_indices = {task: list(task_to_indices[task]) for task in self.task_names}
        self.task_weights = {task: float(task_weights.get(task, 1.0)) for task in self.task_names}
        for task, weight in self.task_weights.items():
            if weight <= 0.0:
                raise ValueError(f"Task '{task}' has non-positive weight: {weight}")

        self.task_probabilities = compute_task_probabilities(
            task_sizes={task: len(self.task_to_indices[task]) for task in self.task_names},
            task_weights=self.task_weights,
            temperature=self.temperature,
        )

    def __iter__(self) -> Iterator[List[int]]:
        rng = np.random.default_rng(self.seed)

        # Task-local shuffled pools to cycle through without immediate repeats.
        pools: Dict[str, List[int]] = {}
        cursors: Dict[str, int] = {}
        for task, indices in self.task_to_indices.items():
            shuffled = list(indices)
            rng.shuffle(shuffled)
            pools[task] = shuffled
            cursors[task] = 0

        for _ in range(self.num_batches):
            task = str(rng.choice(self.task_names, p=[self.task_probabilities[t] for t in self.task_names]))
            batch: List[int] = []

            while len(batch) < self.batch_size:
                cursor = cursors[task]
                pool = pools[task]
                if cursor >= len(pool):
                    pool = list(self.task_to_indices[task])
                    rng.shuffle(pool)
                    pools[task] = pool
                    cursors[task] = 0
                    cursor = 0
                batch.append(int(pool[cursor]))
                cursors[task] = cursor + 1

            if self.drop_last and len(batch) < self.batch_size:
                continue
            yield batch

    def __len__(self) -> int:
        return self.num_batches


def compute_task_probabilities(
    *,
    task_sizes: Mapping[str, int],
    task_weights: Mapping[str, float],
    temperature: float,
) -> Dict[str, float]:
    """Compute temperature-scaled task sampling probabilities."""
    if temperature <= 0.0:
        raise ValueError("temperature must be > 0")

    logits: Dict[str, float] = {}
    for task, size_raw in task_sizes.items():
        size = int(size_raw)
        if size <= 0:
            continue
        weight = float(task_weights.get(task, 1.0))
        if weight <= 0.0:
            raise ValueError(f"Task '{task}' has non-positive train_weight: {weight}")
        logits[task] = (float(size) ** (1.0 / float(temperature))) * weight

    if not logits:
        raise ValueError("No positive task sizes available for probability computation.")

    total = float(sum(logits.values()))
    if total <= 0.0:
        raise ValueError("Invalid probability denominator (sum <= 0).")

    return {task: float(value / total) for task, value in logits.items()}


def estimate_batches_per_epoch(total_examples: int, batch_size: int, drop_last: bool = True) -> int:
    """Estimate train batches per epoch for the mixed task dataset."""
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    if total_examples <= 0:
        raise ValueError("total_examples must be > 0")
    if drop_last:
        return max(1, total_examples // batch_size)
    return max(1, int(math.ceil(total_examples / float(batch_size))))


__all__ = [
    "TaskIndexRange",
    "MultiTaskDataset",
    "TemperatureMultiTaskBatchSampler",
    "compute_task_probabilities",
    "estimate_batches_per_epoch",
]
