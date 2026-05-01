"""Base class for all experiment result collectors."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterator, List

from .schema import ExperimentResult


class ResultsCollector(ABC):
    """Abstract base for experiment-type-specific result collectors.

    Subclasses implement collect() as a generator. All file-level parsing
    should be wrapped in try/except so that a single bad file does not abort
    the entire collection run. Use logging.warning to surface failures.
    """

    @abstractmethod
    def collect(self) -> Iterator[ExperimentResult]:
        """Yield ExperimentResult objects.

        Must be robust to missing files and partial outputs.
        """
        ...

    def collect_all(self) -> List[ExperimentResult]:
        """Materialise the generator into a list."""
        return list(self.collect())
