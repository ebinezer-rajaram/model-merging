#!/usr/bin/env python3
"""Move merged per-task *_metrics.json into per-task subfolders.

This repo writes merged evaluation outputs under:
  artifacts/merged/<method>/<task_combo>/eval/<split>/

Historically, per-task metric files were written directly under the split folder,
which gets cluttered during sweeps:
  <split>/<task>_<label>_metrics.json

This script migrates those files to:
  <split>/per_task/<task>/<task>_<label>_metrics.json

It also repairs any broken symlinks under:
  artifacts/<task>/metrics/eval/<split>/merged/**/metrics.json
that used to point at the old location.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
KNOWN_TASKS = [
    "asr",
    "emotion",
    "intent",
    "speaker_id",
    "speaker_ver",
    "speech_qa",
    "st",
    "langid",
    "kws",
]


def _infer_task_from_filename(filename: str) -> str | None:
    if not filename.endswith("_metrics.json"):
        return None
    for task in sorted(KNOWN_TASKS, key=len, reverse=True):
        if filename.startswith(f"{task}_"):
            return task
    return None


def _iter_merged_split_dirs(merged_root: Path) -> list[Path]:
    split_dirs: list[Path] = []
    for eval_dir in merged_root.rglob("eval"):
        if not eval_dir.is_dir():
            continue
        for child in eval_dir.iterdir():
            if child.is_dir():
                split_dirs.append(child)
    return split_dirs


def _migrate_split_dir(split_dir: Path, *, dry_run: bool) -> int:
    moved = 0
    for path in sorted(split_dir.iterdir()):
        if not path.is_file():
            continue
        if path.name == "index.json":
            continue
        if not path.name.endswith("_metrics.json"):
            continue
        if path.parent.name == "per_task":
            continue

        task = _infer_task_from_filename(path.name)
        if task is None:
            continue

        dest_dir = split_dir / "per_task" / task
        dest_path = dest_dir / path.name
        if dest_path.exists():
            # If destination already exists, keep it and avoid clobbering.
            # Leave the original in place so the user can inspect.
            print(f"SKIP (exists): {path} -> {dest_path}")
            continue

        print(f"MOVE: {path} -> {dest_path}")
        moved += 1
        if dry_run:
            continue
        dest_dir.mkdir(parents=True, exist_ok=True)
        path.rename(dest_path)
    return moved


def _repair_broken_symlinks(repo_root: Path, *, dry_run: bool) -> int:
    repaired = 0
    artifacts_root = repo_root / "artifacts"
    if not artifacts_root.exists():
        return 0

    for dirpath, dirnames, filenames in os.walk(artifacts_root):
        # Skip merged artifacts; we only care about task-centric symlinks.
        if Path(dirpath).resolve() == (artifacts_root / "merged").resolve():
            dirnames[:] = []
            continue
        if "metrics.json" not in filenames:
            continue

        symlink_path = Path(dirpath) / "metrics.json"
        if not symlink_path.is_symlink():
            continue

        target_rel = os.readlink(symlink_path)
        target_abs = (symlink_path.parent / target_rel).resolve()
        if target_abs.exists():
            continue

        filename = Path(target_rel).name
        task = _infer_task_from_filename(filename)
        if task is None:
            continue

        # Old target layout: .../eval/<split>/<task>_<label>_metrics.json
        # New target layout: .../eval/<split>/per_task/<task>/<task>_<label>_metrics.json
        if "/per_task/" in str(Path(target_rel)).replace("\\", "/"):
            continue

        new_target_abs = target_abs.parent / "per_task" / task / filename
        if not new_target_abs.exists():
            continue

        new_target_rel = os.path.relpath(new_target_abs, start=symlink_path.parent)
        print(f"RELINK: {symlink_path} -> {new_target_rel}")
        repaired += 1
        if dry_run:
            continue
        symlink_path.unlink()
        os.symlink(new_target_rel, symlink_path)

    return repaired


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--merged-root",
        type=Path,
        default=Path("artifacts/merged"),
        help="Path to artifacts/merged root (default: artifacts/merged).",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print actions without changing files.")
    args = parser.parse_args()

    repo_root = Path.cwd()
    merged_root = (repo_root / args.merged_root).resolve()
    if not merged_root.exists():
        raise SystemExit(f"merged root does not exist: {merged_root}")

    split_dirs = _iter_merged_split_dirs(merged_root)
    if not split_dirs:
        print(f"No split dirs found under: {merged_root}")
        return 0

    moved_total = 0
    for split_dir in sorted(set(split_dirs)):
        moved_total += _migrate_split_dir(split_dir, dry_run=args.dry_run)

    repaired = _repair_broken_symlinks(repo_root, dry_run=args.dry_run)
    print(f"Done. moved={moved_total} relinked={repaired} dry_run={args.dry_run}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
