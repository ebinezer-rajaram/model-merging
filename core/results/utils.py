"""Shared utilities for results collection."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Task registries — import TASK_METRICS from the canonical source rather
# than duplicating it here.
# ---------------------------------------------------------------------------
try:
    from merging.evaluation.interference import TASK_METRICS
except Exception:
    # Fallback if running outside the full project environment.
    TASK_METRICS: Dict[str, tuple] = {
        "asr": ("wer", False),
        "emotion": ("macro_f1", True),
        "intent": ("accuracy", True),
        "speech_qa": ("accuracy", True),
        "kws": ("macro_f1", True),
        "langid": ("accuracy", True),
        "speaker_id": ("accuracy", True),
        "speaker_ver": ("accuracy", True),
        "vocalsound": ("accuracy", True),
        "st": ("wer", False),
    }

# Tasks treated as OOD / held-out when not explicitly trained on.
HELD_OUT_TASKS: frozenset = frozenset({"speech_qa", "st"})

# All known task names (used by SingleTaskCollector for directory scanning).
KNOWN_TASKS: frozenset = frozenset(TASK_METRICS.keys())

# Stub taxonomy — fill in when a grouping strategy is decided.
TASK_GROUPS: Dict[str, str] = {}

# Named metric fields emitted by TaskEvalResult (used by _result_to_rows).
NAMED_METRIC_FIELDS = ("accuracy", "macro_f1", "weighted_f1", "wer", "loss", "recognized_rate")


# ---------------------------------------------------------------------------
# eval_context derivation
# ---------------------------------------------------------------------------

def derive_eval_context(
    task: str,
    source_tasks: List[str],
    eval_tag: Optional[str],
) -> str:
    """Return the eval_context string for a (task, source_tasks, eval_tag) triple.

    Priority: subset > ood > cross_task > full
    """
    if eval_tag:
        return "subset"
    if task in HELD_OUT_TASKS and task not in source_tasks:
        return "ood"
    if source_tasks and task not in source_tasks:
        return "cross_task"
    return "full"


# ---------------------------------------------------------------------------
# experiment_id construction
# ---------------------------------------------------------------------------

def make_experiment_id(
    experiment_type: str,
    method: str,
    source_tasks: List[str],
    adapter_subdir: Optional[str] = None,
) -> str:
    """Return a stable, lambda-free experiment identifier.

    Format:
      single_task  ->  single_task:{task}:{adapter_subdir}
      mtl          ->  mtl:{source_tasks_key}
      merge        ->  merge:{method}:{source_tasks_key}
    """
    key = make_source_tasks_key(source_tasks)
    if experiment_type == "single_task":
        parts = ["single_task", method]
        if adapter_subdir:
            parts.append(adapter_subdir)
        return ":".join(parts)
    elif experiment_type == "mtl":
        return f"mtl:{key}" if key else "mtl"
    else:
        return f"merge:{method}:{key}" if key else f"merge:{method}"


def make_source_tasks_key(source_tasks: List[str]) -> str:
    """Return '+'.join(sorted(source_tasks)), empty string if no tasks."""
    return "+".join(sorted(source_tasks)) if source_tasks else ""


# ---------------------------------------------------------------------------
# Primary metric helpers
# ---------------------------------------------------------------------------

def is_primary_metric(task: str, metric_name: str) -> bool:
    """Return True if metric_name is the canonical primary metric for task."""
    entry = TASK_METRICS.get(task)
    if entry is None:
        return False
    return metric_name == entry[0]


# ---------------------------------------------------------------------------
# Safe JSON loading
# ---------------------------------------------------------------------------

def safe_load_json(path: Path) -> Optional[Dict[str, Any]]:
    """Load JSON from path; return None on any error."""
    try:
        with path.open("r") as fh:
            return json.load(fh)
    except Exception as exc:
        log.warning("Failed to load JSON from %s: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Wide-format pivot helper
# ---------------------------------------------------------------------------

def to_wide_df(long_df: "pd.DataFrame") -> "pd.DataFrame":
    """Pivot long-format results to wide format (one row per experiment×task×split).

    Each metric_name becomes its own column. Useful for paper tables.
    Requires pandas.
    """
    import pandas as pd  # local import to avoid hard dependency at module load

    identity_cols = [
        c for c in long_df.columns
        if c not in ("metric_name", "metric_value", "is_primary_metric")
    ]
    return long_df.pivot_table(
        index=identity_cols,
        columns="metric_name",
        values="metric_value",
        aggfunc="first",
    ).reset_index()
