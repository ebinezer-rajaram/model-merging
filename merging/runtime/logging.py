"""Lightweight logging helpers for merge workflows."""

from __future__ import annotations


def banner(title: str) -> None:
    line = "=" * 60
    print(f"\n{line}")
    print(title)
    print(line)


def section(title: str) -> None:
    print(f"\n{title}")
