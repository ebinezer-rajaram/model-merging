from __future__ import annotations

from pathlib import Path

import pytest


def pytest_collection_modifyitems(config, items):
    root = Path(config.rootpath)
    for item in items:
        try:
            rel_parts = Path(str(item.fspath)).resolve().relative_to(root).parts
        except ValueError:
            continue
        if "integration" in rel_parts:
            item.add_marker(pytest.mark.integration)
        elif "unit" in rel_parts:
            item.add_marker(pytest.mark.unit)
