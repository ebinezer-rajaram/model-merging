"""Lightweight import wrapper for continual-suite helpers.

This avoids importing the full `merging` package, which pulls heavyweight
runtime dependencies that are not needed for config generation or summary code.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


_MODULE_PATH = Path(__file__).resolve().parent / "merging" / "continual" / "suite.py"
_SPEC = importlib.util.spec_from_file_location("_continual_suite_impl", _MODULE_PATH)
if _SPEC is None or _SPEC.loader is None:
    raise ImportError(f"Could not load continual suite helpers from {_MODULE_PATH}")
_MODULE = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = _MODULE
_SPEC.loader.exec_module(_MODULE)


__all__ = list(getattr(_MODULE, "__all__", []))
globals().update({name: getattr(_MODULE, name) for name in __all__})
