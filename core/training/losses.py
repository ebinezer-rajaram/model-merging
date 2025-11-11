"""Custom loss functions for fine-tuning tasks."""

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class FocalLoss(nn.Module):
    """Focal Loss for addressing class imbalance.

    Focal loss down-weights easy examples and focuses training on hard negatives.
    Reference: Lin et al. "Focal Loss for Dense Object Detection" (2017)

    Args:
        alpha: Weighting factor in range (0,1) to balance positive/negative examples,
               or a list of weights for each class. Default: None (no weighting)
        gamma: Focusing parameter for modulating loss. Higher values increase focus
               on hard examples. Default: 2.0
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'
        ignore_index: Specifies a target value that is ignored. Default: -100

    Example:
        >>> loss_fn = FocalLoss(gamma=2.0, alpha=0.25)
        >>> logits = torch.randn(8, 7, 128, requires_grad=True)  # (batch, classes, seq_len)
        >>> targets = torch.randint(0, 7, (8, 128))  # (batch, seq_len)
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        alpha: Optional[float | list[float]] = None,
        gamma: float = 2.0,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.ignore_index = ignore_index

        # Convert alpha to tensor if it's a list
        if isinstance(alpha, list):
            self.alpha_tensor = torch.tensor(alpha)
        else:
            self.alpha_tensor = None

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute focal loss.

        Args:
            inputs: Predicted logits of shape (batch_size, num_classes, seq_len),
                   (batch_size, seq_len, num_classes), or (batch_size * seq_len, num_classes)
            targets: Ground truth labels of shape (batch_size, seq_len) or (batch_size * seq_len,)

        Returns:
            Computed focal loss
        """
        # Handle different input shapes
        if inputs.dim() == 3:
            # Could be (B, C, L) or (B, L, C)
            # The targets help us determine the correct interpretation
            # targets should be (B, L) where L is sequence length
            if targets.dim() == 2:
                # targets is (B, L), so inputs should be (B, L, C)
                # Check if we need to transpose
                if inputs.size(1) == targets.size(1):
                    # inputs is already (B, L, C) - correct format
                    pass
                elif inputs.size(2) == targets.size(1):
                    # inputs is (B, C, L) - need to transpose to (B, L, C)
                    inputs = inputs.transpose(1, 2)
                else:
                    # Fallback: assume smaller dimension is num_classes
                    if inputs.size(1) < inputs.size(2):
                        inputs = inputs.transpose(1, 2)
            else:
                # Fallback for other target shapes
                if inputs.size(1) < inputs.size(2):
                    inputs = inputs.transpose(1, 2)

            # Now inputs is (B, L, C)
            batch_size, seq_len, num_classes = inputs.shape
            inputs_flat = inputs.reshape(-1, num_classes)  # (B*L, C)
        elif inputs.dim() == 2:
            # Already flattened: (B*L, C)
            inputs_flat = inputs
            num_classes = inputs.size(1)
            batch_size = None  # Unknown, will be inferred
            seq_len = None
        else:
            raise ValueError(f"Expected inputs to be 2D or 3D, got shape {inputs.shape}")

        # Flatten targets
        targets_flat = targets.reshape(-1)  # (B*L,)

        # Verify shapes match
        if inputs_flat.size(0) != targets_flat.size(0):
            raise ValueError(
                f"Shape mismatch: inputs_flat has {inputs_flat.size(0)} samples "
                f"but targets_flat has {targets_flat.size(0)} samples. "
                f"Original inputs shape: {inputs.shape}, targets shape: {targets.shape}"
            )

        # Create mask for valid targets (not ignore_index)
        valid_mask = targets_flat != self.ignore_index

        # Compute cross entropy without reduction
        ce_loss = F.cross_entropy(
            inputs_flat,
            targets_flat,
            reduction="none",
            ignore_index=self.ignore_index,
        )

        # Compute log probabilities (more memory efficient and numerically stable)
        log_probs = F.log_softmax(inputs_flat, dim=1)

        # Extract log probability of the true class using gather
        valid_targets = targets_flat.clamp(min=0)  # Replace -100 with 0 temporarily
        log_pt = log_probs.gather(1, valid_targets.unsqueeze(1)).squeeze(1)

        # Convert to probability: pt = exp(log_pt)
        pt = torch.exp(log_pt)
        pt = pt.clamp(min=1e-8)  # For numerical stability

        # Apply focal term: (1 - pt)^gamma
        focal_weight = (1 - pt) ** self.gamma

        # Apply alpha if specified
        if self.alpha is not None:
            if self.alpha_tensor is not None:
                # Per-class alpha
                alpha_t = self.alpha_tensor.to(inputs.device)
                alpha_t = alpha_t[targets_flat.clamp(min=0)]
                alpha_t = alpha_t * valid_mask.float()
            else:
                # Single alpha value
                alpha_t = self.alpha
            focal_weight = alpha_t * focal_weight

        # Compute final loss
        focal_loss = focal_weight * ce_loss

        # Apply mask to ignore padding tokens
        focal_loss = focal_loss * valid_mask.float()

        # Apply reduction
        if self.reduction == "mean":
            return focal_loss.sum() / valid_mask.sum().clamp(min=1)
        elif self.reduction == "sum":
            return focal_loss.sum()
        else:  # "none"
            if batch_size is not None and seq_len is not None:
                return focal_loss.reshape(batch_size, seq_len)
            else:
                return focal_loss


class WeightedCrossEntropyLoss(nn.Module):
    """Weighted Cross-Entropy Loss for handling class imbalance.

    This loss function applies different weights to different classes,
    allowing you to penalize misclassification of minority classes more heavily.

    Args:
        weights: A tensor or list of weights for each class. Should have length
                equal to the number of classes.
        reduction: Specifies the reduction to apply to the output:
                  'none' | 'mean' | 'sum'. Default: 'mean'
        ignore_index: Specifies a target value that is ignored. Default: -100

    Example:
        >>> # For MELD with 7 emotions, weight rare emotions more heavily
        >>> weights = [1.0, 2.0, 1.5, 1.5, 2.5, 3.0, 2.0]  # adjust based on distribution
        >>> loss_fn = WeightedCrossEntropyLoss(weights=weights)
        >>> logits = torch.randn(8, 7, 128, requires_grad=True)
        >>> targets = torch.randint(0, 7, (8, 128))
        >>> loss = loss_fn(logits, targets)
    """

    def __init__(
        self,
        weights: list[float] | torch.Tensor,
        reduction: str = "mean",
        ignore_index: int = -100,
    ):
        super().__init__()
        if isinstance(weights, list):
            weights = torch.tensor(weights, dtype=torch.float32)
        self.register_buffer("weights", weights)
        self.reduction = reduction
        self.ignore_index = ignore_index

    def forward(self, inputs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """Compute weighted cross-entropy loss.

        Args:
            inputs: Predicted logits of shape (batch_size, num_classes, seq_len),
                   (batch_size, seq_len, num_classes), or (batch_size * seq_len, num_classes)
            targets: Ground truth labels of shape (batch_size, seq_len) or (batch_size * seq_len,)

        Returns:
            Computed weighted cross-entropy loss
        """
        # Handle different input shapes
        if inputs.dim() == 3:
            # Could be (B, C, L) or (B, L, C)
            # Use targets to determine the correct interpretation
            if targets.dim() == 2:
                # targets is (B, L), so inputs should be (B, L, C)
                if inputs.size(1) == targets.size(1):
                    # inputs is already (B, L, C) - correct format
                    pass
                elif inputs.size(2) == targets.size(1):
                    # inputs is (B, C, L) - need to transpose to (B, L, C)
                    inputs = inputs.transpose(1, 2)
                else:
                    # Fallback: assume smaller dimension is num_classes
                    if inputs.size(1) < inputs.size(2):
                        inputs = inputs.transpose(1, 2)
            else:
                # Fallback for other target shapes
                if inputs.size(1) < inputs.size(2):
                    inputs = inputs.transpose(1, 2)
            # Flatten for easier computation
            batch_size, seq_len, num_classes = inputs.shape
            inputs_flat = inputs.reshape(-1, num_classes)  # (B*L, C)
        elif inputs.dim() == 2:
            # Already flattened: (B*L, C)
            inputs_flat = inputs
        else:
            raise ValueError(f"Expected inputs to be 2D or 3D, got shape {inputs.shape}")

        # Flatten targets
        targets_flat = targets.reshape(-1)  # (B*L,)

        # Compute weighted cross entropy
        loss = F.cross_entropy(
            inputs_flat,
            targets_flat,
            weight=self.weights,
            reduction=self.reduction,
            ignore_index=self.ignore_index,
        )

        return loss


def compute_class_weights(
    labels: list[int] | torch.Tensor,
    num_classes: int,
    method: str = "inverse",
) -> torch.Tensor:
    """Compute class weights from label distribution.

    Args:
        labels: List or tensor of class labels
        num_classes: Total number of classes
        method: Method for computing weights:
               - "inverse": weight = total_samples / (num_classes * class_count)
               - "sqrt_inverse": weight = sqrt(total_samples / class_count)
               - "balanced": weight = total_samples / (num_classes * class_count) [sklearn style]

    Returns:
        Tensor of weights for each class

    Example:
        >>> labels = [0, 0, 0, 1, 1, 2]  # Class 0 is common, class 2 is rare
        >>> weights = compute_class_weights(labels, num_classes=3, method="inverse")
        >>> print(weights)  # Class 2 will have higher weight
    """
    if isinstance(labels, list):
        labels = torch.tensor(labels)

    # Count samples per class
    class_counts = torch.bincount(labels, minlength=num_classes).float()

    # Avoid division by zero
    class_counts = class_counts.clamp(min=1)

    total_samples = len(labels)

    if method == "inverse":
        # Standard inverse frequency
        weights = total_samples / (num_classes * class_counts)
    elif method == "sqrt_inverse":
        # Square root of inverse frequency (less aggressive)
        weights = torch.sqrt(total_samples / class_counts)
    elif method == "balanced":
        # Sklearn-style balanced weights
        weights = total_samples / (num_classes * class_counts)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'inverse', 'sqrt_inverse', or 'balanced'.")

    return weights


__all__ = [
    "FocalLoss",
    "WeightedCrossEntropyLoss",
    "compute_class_weights",
]
