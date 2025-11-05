"""Utilities for reproducibility control."""

from typing import Optional

import numpy as np
import torch


def set_global_seed(seed: Optional[int]) -> None:
    """Set RNG seeds for numpy and torch if provided."""
    if seed is None:
        return
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
