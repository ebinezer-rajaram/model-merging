from __future__ import annotations

import json
from pathlib import Path

import pytest

from continual_suite_lib import (
    StageResultRecord,
    compute_stage_tables,
    expand_stage_plans,
    load_suite_config,
    select_mtl_checkpoint_record,
)


def test_suite_config_and_stage_expansion_are_generic(tmp_path: Path) -> None:
    config_path = tmp_path / "suite.yaml"
    config_path.write_text(
        """
suite_id: test_suite
defaults:
  output_root: artifacts/continual_suite
  merge_template: configs/merge/continual/merge_continual_6task_materialized_plus_asr.yaml
  mtl_template: configs/mtl/continual/continual_6task_plus_asr.yaml
  include_speech_qa: true
paths:
  - path_id: short_path
    tasks: [speaker_ver, langid]
    eval_only_tasks: [asr]
  - path_id: long_path
    tasks: [asr, intent, emotion, speaker_ver]
        """.strip(),
        encoding="utf-8",
    )

    cfg = load_suite_config(config_path)
    plans = expand_stage_plans(cfg, start_stage=2)

    assert [plan.path_id for plan in plans] == ["short_path", "long_path", "long_path", "long_path"]
    assert plans[0].stage_index == 2
    assert plans[0].seen_tasks == ["speaker_ver", "langid"]
    assert plans[0].eval_only_tasks == ["asr"]
    assert plans[0].report_tasks == ["speaker_ver", "langid", "asr", "speech_qa"]
    assert plans[-1].stage_index == 4
    assert plans[-1].seen_tasks == ["asr", "intent", "emotion", "speaker_ver"]


def test_select_mtl_checkpoint_record_supports_added_and_balanced_views() -> None:
    rows = [
        {
            "step": 100,
            "eval_added_tasks_primary_oriented_mean": 0.40,
            "eval_asr_interference_delta": 0.60,
            "eval_intent_interference_delta": 0.55,
        },
        {
            "step": 200,
            "eval_added_tasks_primary_oriented_mean": 0.52,
            "eval_asr_interference_delta": 0.45,
            "eval_intent_interference_delta": 0.90,
        },
        {
            "step": 300,
            "eval_added_tasks_primary_oriented_mean": 0.48,
            "eval_asr_interference_delta": 0.70,
            "eval_intent_interference_delta": 0.72,
        },
    ]

    added = select_mtl_checkpoint_record(
        rows,
        seen_tasks=["asr", "intent"],
        mode="added_task_best",
        available_steps=[100, 200, 300],
    )
    balanced = select_mtl_checkpoint_record(
        rows,
        seen_tasks=["asr", "intent"],
        mode="balanced_best",
        available_steps=[100, 200, 300],
    )

    assert added is not None and int(added["step"]) == 200
    assert balanced is not None and int(balanced["step"]) == 300


def test_compute_stage_tables_tracks_forgetting_and_rollups() -> None:
    records = [
        StageResultRecord(
            suite_id="suite",
            path_id="path",
            stage_index=2,
            stage_name="stage_02",
            method="continual_mtl",
            checkpoint_view="balanced_best",
            split="test",
            seen_tasks=["asr", "intent"],
            prior_tasks=["asr"],
            added_task="intent",
            eval_only_tasks=[],
            report_tasks=["asr", "intent"],
            results={
                "asr": {"wer": 0.02, "interference_delta": 0.90},
                "intent": {"accuracy": 0.80, "interference_delta": 0.80},
            },
            source_path="stage2.json",
        ),
        StageResultRecord(
            suite_id="suite",
            path_id="path",
            stage_index=3,
            stage_name="stage_03",
            method="continual_mtl",
            checkpoint_view="balanced_best",
            split="test",
            seen_tasks=["asr", "intent", "emotion"],
            prior_tasks=["asr", "intent"],
            added_task="emotion",
            eval_only_tasks=[],
            report_tasks=["asr", "intent", "emotion"],
            results={
                "asr": {"wer": 0.03, "interference_delta": 0.70},
                "intent": {"accuracy": 0.82, "interference_delta": 0.85},
                "emotion": {"macro_f1": 0.50, "interference_delta": 0.60},
            },
            source_path="stage3.json",
        ),
    ]

    tables = compute_stage_tables(records)
    stage_rows = tables["stage_rows"]
    forgetting_rows = tables["forgetting_rows"]

    assert len(stage_rows) == 2
    assert stage_rows[1]["new_task_delta"] == 0.60
    assert stage_rows[1]["seen_min_delta"] == 0.60
    assert stage_rows[1]["prior_avg_delta"] == pytest.approx(0.775)
    assert round(float(stage_rows[1]["avg_prior_forgetting"]), 4) == -0.075

    forgetting_by_task = {row["prior_task"]: row["forgetting"] for row in forgetting_rows}
    assert forgetting_by_task["asr"] == pytest.approx(-0.20)
    assert forgetting_by_task["intent"] == pytest.approx(0.05)
