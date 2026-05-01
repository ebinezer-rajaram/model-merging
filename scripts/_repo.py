"""Helpers for repository-root discovery in standalone scripts."""

from __future__ import annotations

from pathlib import Path


def find_repo_root(start: str | Path) -> Path:
    """Find the repository root from a script path or directory."""
    path = Path(start).resolve()
    current = path if path.is_dir() else path.parent

    for candidate in (current, *current.parents):
        if (
            (candidate / "README.md").is_file()
            and (candidate / "core").is_dir()
            and (candidate / "tasks").is_dir()
        ):
            return candidate

    raise RuntimeError(f"Could not locate repository root from {path}")
