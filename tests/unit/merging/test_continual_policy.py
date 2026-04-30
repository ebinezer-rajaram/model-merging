from __future__ import annotations

import math

from merging.continual.policy import ContinualMergePolicy


def test_policy_coefficients() -> None:
    policy = ContinualMergePolicy(alpha=2.0, lambda_weight=0.25)
    x_coeff, y_coeff = policy.source_coefficients()
    assert math.isclose(x_coeff, 0.5)
    assert math.isclose(y_coeff, 1.5)


def test_policy_validation_errors() -> None:
    try:
        ContinualMergePolicy(alpha=-1.0, lambda_weight=0.5).validate()
        raise AssertionError("Expected ValueError for negative alpha")
    except ValueError:
        pass

    try:
        ContinualMergePolicy(alpha=1.0, lambda_weight=1.5).validate()
        raise AssertionError("Expected ValueError for lambda out of range")
    except ValueError:
        pass
