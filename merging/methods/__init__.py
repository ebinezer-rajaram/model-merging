"""Concrete merge method implementations."""

from merging.methods.uniform import merge_adapters_uniform, merge_uniform
from merging.methods.weighted import merge_adapters_weighted, merge_task_vectors_weighted, merge_weighted
from merging.methods.weighted_delta_n import merge_task_vectors_weighted_n
from merging.methods.uniform_scalar_delta import merge_task_vectors_uniform_scalar
from merging.methods.task_vector import merge_uniform_via_task_vectors
from merging.methods.dare import merge_dare
from merging.methods.ties import merge_ties

__all__ = [
    "merge_uniform",
    "merge_adapters_uniform",
    "merge_weighted",
    "merge_adapters_weighted",
    "merge_task_vectors_weighted",
    "merge_task_vectors_weighted_n",
    "merge_task_vectors_uniform_scalar",
    "merge_uniform_via_task_vectors",
    "merge_dare",
    "merge_ties",
]
