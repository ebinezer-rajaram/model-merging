"""Custom data samplers for balanced training."""

from typing import Iterator, List, Optional

import torch
from torch.utils.data import Sampler, Dataset, BatchSampler
import numpy as np


class BalancedBatchSampler(BatchSampler):
    """Sampler that creates balanced batches with equal class representation.

    This sampler ensures each batch contains approximately equal numbers of samples
    from each class, which is especially useful for imbalanced datasets like MELD.

    Args:
        dataset: The dataset to sample from (must have 'label' column)
        batch_size: Number of samples per batch
        num_classes: Total number of classes
        drop_last: Whether to drop the last incomplete batch
        shuffle: Whether to shuffle samples within each class

    Example:
        >>> sampler = BalancedBatchSampler(train_dataset, batch_size=16, num_classes=7)
        >>> dataloader = DataLoader(dataset, batch_sampler=sampler)
    """

    def __init__(
        self,
        dataset: Dataset,
        batch_size: int,
        num_classes: int,
        drop_last: bool = False,
        shuffle: bool = True,
    ):
        # Create a dummy sampler to satisfy BatchSampler's requirements
        # We won't actually use it since we override __iter__
        dummy_sampler = Sampler(dataset)
        dummy_sampler.__len__ = lambda: len(dataset)
        dummy_sampler.__iter__ = lambda: iter(range(len(dataset)))

        # Initialize parent BatchSampler
        super().__init__(dummy_sampler, batch_size, drop_last)

        self.dataset = dataset
        self.num_classes = num_classes
        self.shuffle = shuffle

        # Group indices by class
        self.class_indices = [[] for _ in range(num_classes)]
        for idx in range(len(dataset)):
            label = dataset[idx]['label']
            if isinstance(label, torch.Tensor):
                label = label.item()
            self.class_indices[label].append(idx)

        # Calculate samples per class per batch
        self.samples_per_class = max(1, batch_size // num_classes)
        self.actual_batch_size = self.samples_per_class * num_classes

        # Compute number of batches
        min_class_size = min(len(indices) for indices in self.class_indices if len(indices) > 0)
        self.num_batches = min_class_size // self.samples_per_class

        if not self.drop_last and min_class_size % self.samples_per_class != 0:
            self.num_batches += 1

        print(f"📊 BalancedBatchSampler initialized:")
        print(f"   - Batch size: {batch_size} → Actual: {self.actual_batch_size}")
        print(f"   - Samples per class per batch: {self.samples_per_class}")
        print(f"   - Number of batches: {self.num_batches}")
        for class_id, indices in enumerate(self.class_indices):
            print(f"   - Class {class_id}: {len(indices)} samples")

    def __iter__(self) -> Iterator[List[int]]:
        """Generate balanced batches."""
        # Shuffle indices within each class if requested
        class_iters = []
        for indices in self.class_indices:
            if len(indices) == 0:
                class_iters.append(iter([]))
                continue

            if self.shuffle:
                # Shuffle and repeat to ensure we have enough samples
                indices_array = np.array(indices)
                np.random.shuffle(indices_array)
                # Repeat if necessary
                repeats = (self.num_batches * self.samples_per_class // len(indices)) + 1
                indices_repeated = np.tile(indices_array, repeats)
                np.random.shuffle(indices_repeated)
                class_iters.append(iter(indices_repeated.tolist()))
            else:
                # Just repeat without shuffling
                repeats = (self.num_batches * self.samples_per_class // len(indices)) + 1
                indices_repeated = indices * repeats
                class_iters.append(iter(indices_repeated))

        # Generate batches
        for _ in range(self.num_batches):
            batch = []
            for class_iter in class_iters:
                for _ in range(self.samples_per_class):
                    try:
                        batch.append(next(class_iter))
                    except StopIteration:
                        # This shouldn't happen with repetition, but handle gracefully
                        pass

            if len(batch) > 0 and (not self.drop_last or len(batch) == self.actual_batch_size):
                # Shuffle within batch to mix classes
                if self.shuffle:
                    np.random.shuffle(batch)
                yield batch

    def __len__(self) -> int:
        """Return number of batches."""
        return self.num_batches


class WeightedClassSampler(Sampler[int]):
    """Weighted sampler that oversamples minority classes.

    This sampler assigns weights to samples inversely proportional to their class
    frequency, ensuring the model sees minority classes more often during training.

    Args:
        dataset: The dataset to sample from (must have 'label' column)
        num_samples: Number of samples to draw per epoch (default: len(dataset))
        replacement: Whether to sample with replacement
        method: Weighting method - "inverse", "sqrt_inverse", or "balanced"

    Example:
        >>> sampler = WeightedClassSampler(train_dataset, method="sqrt_inverse")
        >>> dataloader = DataLoader(dataset, sampler=sampler, batch_size=16)
    """

    def __init__(
        self,
        dataset: Dataset,
        num_samples: Optional[int] = None,
        replacement: bool = True,
        method: str = "inverse",
    ):
        self.dataset = dataset
        self.num_samples = num_samples if num_samples is not None else len(dataset)
        self.replacement = replacement
        self.method = method

        # Count samples per class
        labels = [dataset[idx]['label'] for idx in range(len(dataset))]
        if isinstance(labels[0], torch.Tensor):
            labels = [label.item() for label in labels]

        class_counts = torch.bincount(torch.tensor(labels))
        num_classes = len(class_counts)

        # Compute weights based on method
        # Handle zero counts by replacing with a very small positive number
        # (these classes won't be sampled anyway since they have no samples)
        class_counts_safe = class_counts.float().clamp(min=1e-8)

        if method == "inverse":
            # Inverse of class frequency (full rebalancing)
            class_weights = 1.0 / class_counts_safe
        elif method == "sqrt_inverse":
            # Square root of inverse (gentler rebalancing)
            class_weights = 1.0 / torch.sqrt(class_counts_safe)
        elif method == "balanced":
            # Balanced: scale to make expected samples equal
            total_samples = class_counts.sum().float()
            class_weights = total_samples / (num_classes * class_counts_safe)
        else:
            raise ValueError(f"Unknown weighting method: {method}. Use 'inverse', 'sqrt_inverse', or 'balanced'")

        # Set weight to 0 for classes with no samples (can't be sampled anyway)
        class_weights[class_counts == 0] = 0.0

        # Assign weight to each sample based on its class
        self.weights = torch.tensor([class_weights[label] for label in labels])

        print(f"📊 WeightedClassSampler initialized (method={method}):")
        print(f"   - Dataset size: {len(dataset)}")
        print(f"   - Samples per epoch: {self.num_samples}")
        print(f"   - Class counts: {class_counts.tolist()}")
        print(f"   - Class weights: {class_weights.tolist()}")

    def __iter__(self) -> Iterator[int]:
        """Generate weighted random samples."""
        indices = torch.multinomial(
            self.weights,
            self.num_samples,
            replacement=self.replacement
        )
        return iter(indices.tolist())

    def __len__(self) -> int:
        """Return number of samples per epoch."""
        return self.num_samples


__all__ = [
    "BalancedBatchSampler",
    "WeightedClassSampler",
]
