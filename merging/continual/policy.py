"""Policy helpers for two-source continual merge coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Dict


@dataclass(frozen=True)
class ContinualMergePolicy:
    """Global two-source policy for continual merge.

    delta_out = alpha * (lambda * delta_x + (1 - lambda) * delta_y)
    """

    alpha: float
    lambda_weight: float

    def validate(self) -> None:
        if not math.isfinite(float(self.alpha)):
            raise ValueError(f"alpha must be finite, got {self.alpha}")
        if float(self.alpha) < 0.0:
            raise ValueError(f"alpha must be >= 0, got {self.alpha}")
        if not math.isfinite(float(self.lambda_weight)):
            raise ValueError(f"lambda_weight must be finite, got {self.lambda_weight}")
        if not 0.0 <= float(self.lambda_weight) <= 1.0:
            raise ValueError(f"lambda_weight must be in [0,1], got {self.lambda_weight}")

    def source_coefficients(self) -> tuple[float, float]:
        self.validate()
        x_coeff = float(self.alpha) * float(self.lambda_weight)
        y_coeff = float(self.alpha) * (1.0 - float(self.lambda_weight))
        return x_coeff, y_coeff

    def to_dict(self) -> Dict[str, float]:
        return {
            "alpha": float(self.alpha),
            "lambda": float(self.lambda_weight),
        }


__all__ = ["ContinualMergePolicy"]
