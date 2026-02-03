"""Scaffold for Bayesian optimization-based merge sweeps."""

from __future__ import annotations

from typing import Any, Dict

from merging.evaluation.sweep import SweepConfig


def run_bayes_search(config: SweepConfig, search: Dict[str, Any]) -> Dict[str, Any]:
    """Placeholder for Bayesian optimization.

    Expected search config shape:
      search:
        type: bayes
        space:
          lambda:
            min: 0.0
            max: 1.0
            type: float
        budget: 20
        seed: 42
    """
    raise NotImplementedError(
        "Bayesian optimization scaffold is not implemented yet. "
        "Provide search.type=grid or implement run_bayes_search with a BO backend."
    )

