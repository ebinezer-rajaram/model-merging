"""Policy helpers for two-source continual merge coefficients."""

from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Any, Dict, Optional

from merging.policies.lambda_policy import extract_layer_index


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


@dataclass(frozen=True)
class LayerWiseContinualMergePolicy:
    """Per-layer two-source policy for continual merge.

    Each transformer layer has independent alpha and lambda. Non-layer parameters
    (embeddings, norms, etc.) fall back to the global defaults.

    delta_out[layer] = alpha[layer] * (lambda[layer] * x + (1-lambda[layer]) * y)
    """

    default_alpha: float
    default_lambda: float
    layer_alpha: Dict[int, float]
    layer_lambda: Dict[int, float]

    def validate(self) -> None:
        if not math.isfinite(float(self.default_alpha)):
            raise ValueError(f"default_alpha must be finite, got {self.default_alpha}")
        if float(self.default_alpha) < 0.0:
            raise ValueError(f"default_alpha must be >= 0, got {self.default_alpha}")
        if not math.isfinite(float(self.default_lambda)):
            raise ValueError(f"default_lambda must be finite, got {self.default_lambda}")
        if not 0.0 <= float(self.default_lambda) <= 1.0:
            raise ValueError(f"default_lambda must be in [0,1], got {self.default_lambda}")
        for layer_idx, alpha in self.layer_alpha.items():
            if not math.isfinite(float(alpha)):
                raise ValueError(f"layer_alpha[{layer_idx}] must be finite, got {alpha}")
            if float(alpha) < 0.0:
                raise ValueError(f"layer_alpha[{layer_idx}] must be >= 0, got {alpha}")
        for layer_idx, lam in self.layer_lambda.items():
            if not math.isfinite(float(lam)):
                raise ValueError(f"layer_lambda[{layer_idx}] must be finite, got {lam}")
            if not 0.0 <= float(lam) <= 1.0:
                raise ValueError(f"layer_lambda[{layer_idx}] must be in [0,1], got {lam}")

    def source_coefficients_for_key(self, source_key: str) -> tuple[float, float]:
        self.validate()
        layer_idx: Optional[int] = extract_layer_index(source_key)
        if layer_idx is not None and layer_idx in self.layer_alpha:
            alpha = float(self.layer_alpha[layer_idx])
            lam = float(self.layer_lambda[layer_idx])
        else:
            alpha = float(self.default_alpha)
            lam = float(self.default_lambda)
        x_coeff = alpha * lam
        y_coeff = alpha * (1.0 - lam)
        return x_coeff, y_coeff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "default_alpha": float(self.default_alpha),
            "default_lambda": float(self.default_lambda),
            "layer_alpha": {str(k): float(v) for k, v in self.layer_alpha.items()},
            "layer_lambda": {str(k): float(v) for k, v in self.layer_lambda.items()},
        }


__all__ = ["ContinualMergePolicy", "LayerWiseContinualMergePolicy"]
