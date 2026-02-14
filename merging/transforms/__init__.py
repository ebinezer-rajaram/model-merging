"""Pre-merge adapter transform registry and built-ins."""

from merging.transforms.registry import (
    TransformScaffoldNote,
    apply_transforms,
    get_transform,
    list_transforms,
    register_transform,
)

__all__ = [
    "register_transform",
    "get_transform",
    "list_transforms",
    "apply_transforms",
    "TransformScaffoldNote",
]
