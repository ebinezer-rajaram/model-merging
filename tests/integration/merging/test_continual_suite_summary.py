from __future__ import annotations

import csv
import json
import subprocess
import sys
from pathlib import Path


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def test_summary_script_writes_all_expected_csvs(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    suite_root = tmp_path / "artifacts" / "continual_suite" / "smoke_suite"
    stage_dir = suite_root / "compatible_acoustic" / "stage_02"
    sweeps_dir = stage_dir / "merge" / "sweeps"
    best_metrics_dir = stage_dir / "mtl" / "metrics" / "best"
    balanced_metrics_dir = stage_dir / "mtl" / "metrics" / "balanced_best"

    manifest = {
        "suite_id": "smoke_suite",
        "path_id": "compatible_acoustic",
        "stage_index": 2,
        "stage_name": "stage_02",
        "seen_tasks": ["speaker_ver", "langid"],
        "prior_tasks": ["speaker_ver"],
        "added_task": "langid",
        "eval_only_tasks": ["asr"],
        "report_tasks": ["speaker_ver", "langid", "asr", "speech_qa"],
        "selection_split": "validation",
        "report_split": "test",
        "merge": {
            "sweeps_dir": str(sweeps_dir),
        },
        "mtl": {
            "expected_best_metrics_dir": str(best_metrics_dir),
            "expected_metrics_dir": str(stage_dir / "mtl" / "metrics"),
            "balanced_best": {
                "metrics_dir": str(balanced_metrics_dir),
            },
        },
    }
    _write_json(stage_dir / "stage_manifest.json", manifest)

    merge_payload = {
        "post_sweep_eval": {
            "enabled": True,
            "split": "test",
            "results": {
                "speaker_ver": {"accuracy": 0.90, "interference_delta": 0.88},
                "langid": {"accuracy": 0.99, "interference_delta": 0.95},
                "asr": {"wer": 0.02, "interference_delta": 0.30},
                "speech_qa": {"accuracy": 0.63},
            },
        },
        "best_index": 0,
        "runs": [],
    }
    _write_json(sweeps_dir / "sweep_20260101_000000.json", merge_payload)

    mtl_best_payload = {
        "results": {
            "speaker_ver": {"accuracy": 0.91, "interference_delta": 0.90},
            "langid": {"accuracy": 0.995, "interference_delta": 0.97},
            "asr": {"wer": 0.025, "interference_delta": 0.28},
            "speech_qa": {"accuracy": 0.64},
        }
    }
    _write_json(best_metrics_dir / "eval_results_test.json", mtl_best_payload)

    mtl_balanced_payload = {
        "results": {
            "speaker_ver": {"accuracy": 0.89, "interference_delta": 0.92},
            "langid": {"accuracy": 0.992, "interference_delta": 0.94},
            "asr": {"wer": 0.024, "interference_delta": 0.29},
            "speech_qa": {"accuracy": 0.635},
        }
    }
    _write_json(balanced_metrics_dir / "eval_results_test.json", mtl_balanced_payload)

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_continual_suite_summary.py"),
            "--suite-root",
            str(suite_root),
        ],
        check=True,
    )

    expected = [
        suite_root / "continual_stage_metrics.csv",
        suite_root / "continual_growth_curves.csv",
        suite_root / "continual_forgetting_curves.csv",
        suite_root / "continual_final_comparison.csv",
    ]
    for path in expected:
        assert path.exists(), path

    with (suite_root / "continual_stage_metrics.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    checkpoint_views = {row["checkpoint_view"] for row in rows}
    assert {"best", "added_task_best", "balanced_best"} <= checkpoint_views


def test_summary_script_preserves_continual_supermerge_method_label(tmp_path: Path) -> None:
    repo_root = Path(__file__).resolve().parents[3]
    suite_root = tmp_path / "artifacts" / "continual_suite" / "smoke_suite"
    stage_dir = suite_root / "compatible_acoustic" / "stage_02"
    sweeps_dir = stage_dir / "merge" / "sweeps"
    manifest = {
        "suite_id": "smoke_suite",
        "path_id": "compatible_acoustic",
        "stage_index": 2,
        "stage_name": "stage_02",
        "seen_tasks": ["speaker_ver", "langid"],
        "prior_tasks": ["speaker_ver"],
        "added_task": "langid",
        "eval_only_tasks": [],
        "report_tasks": ["speaker_ver", "langid"],
        "selection_split": "validation",
        "report_split": "test",
        "merge": {"sweeps_dir": str(sweeps_dir)},
    }
    _write_json(stage_dir / "stage_manifest.json", manifest)
    _write_json(
        sweeps_dir / "sweep_20260101_000000.json",
        {
            "method": "continual_supermerge",
            "post_sweep_eval": {
                "enabled": True,
                "split": "test",
                "results": {
                    "speaker_ver": {"accuracy": 0.90, "interference_delta": 0.88},
                    "langid": {"accuracy": 0.99, "interference_delta": 0.95},
                },
            },
            "best_index": 0,
            "runs": [],
        },
    )

    subprocess.run(
        [
            sys.executable,
            str(repo_root / "scripts" / "build_continual_suite_summary.py"),
            "--suite-root",
            str(suite_root),
        ],
        check=True,
    )

    with (suite_root / "continual_stage_metrics.csv").open("r", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["method"] for row in rows} == {"continual_supermerge"}
