from __future__ import annotations

import math

import pytest

from merging.continual.policy import ContinualMergePolicy


def test_policy_coefficients() -> None:
    policy = ContinualMergePolicy(alpha=2.0, lambda_weight=0.25)
    x_coeff, y_coeff = policy.source_coefficients()
    assert math.isclose(x_coeff, 0.5)
    assert math.isclose(y_coeff, 1.5)


def test_policy_validation_errors() -> None:
    with pytest.raises(ValueError):
        ContinualMergePolicy(alpha=-1.0, lambda_weight=0.5).validate()

    with pytest.raises(ValueError):
        ContinualMergePolicy(alpha=1.0, lambda_weight=1.5).validate()
