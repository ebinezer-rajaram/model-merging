#!/usr/bin/env python3
import csv
import json
import glob
import re
from pathlib import Path
from typing import Dict, List, Optional

ROOT = Path(".")
OUT_CSV = ROOT / "artifacts/merged/comparisons/speechqa_category_comparison.csv"

CATEGORIES = ["Reasoning", "Perception", "Detection", "Recognition", "Comparison", "Language"]

EXPLICIT_SUBTASK_CATEGORY = {
    "accent_identification": "Recognition",
    "age_prediction": "Recognition",
    "code_switch_question_answering": "Language",
    "content_grounding": "Language",
    "continuation_writing": "Language",
    "couplet_matching": "Language",
    "deixis_resolution": "Reasoning",
    "dialogue_turn_counting": "Perception",
    "gender_prediction": "Recognition",
    "long_speech_summarization": "Language",
    "plosive_sound_identification": "Recognition",
    "puns_interpretation": "Reasoning",
    "speech_act_classification": "Recognition",
    "speech_duration_estimation": "Perception",
    "speech_translation": "Language",
    "syntactic_structure_matching": "Language",
    "total_speaker_counting": "Perception",
}

BASELINE_MODEL_FILES = [
    ("base_model", ROOT / "artifacts/speech_qa/metrics/eval/test/base_model.json"),
    ("best_asr_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_asr_adapter.json"),
    ("best_emotion_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_emotion_adapter.json"),
    ("best_intent_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_intent_adapter.json"),
    ("best_kws_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_kws_adapter.json"),
    ("best_langid_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_langid_adapter.json"),
    ("best_speaker_ver_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_speaker_ver_adapter.json"),
    ("best_vocalsound_adapter", ROOT / "artifacts/speech_qa/metrics/eval/test/best_vocalsound_adapter.json"),
]


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def categorize_subtask(name: str) -> Optional[str]:
    n = name.strip().lower()
    if n in EXPLICIT_SUBTASK_CATEGORY:
        return EXPLICIT_SUBTASK_CATEGORY[n]
    # Suffix-first with keyword fallback.
    if n.endswith("_reasoning") or "reasoning" in n:
        return "Reasoning"
    if n.endswith("_perception") or "perception" in n:
        return "Perception"
    if n.endswith("_detection") or "detection" in n:
        return "Detection"
    if n.endswith("_recognition") or "recognition" in n:
        return "Recognition"
    if n.endswith("_comparison") or "comparison" in n:
        return "Comparison"
    if any(k in n for k in ("translation", "summarization", "writing", "matching", "grounding", "question_answering")):
        return "Language"
    return None


def _extract_speechqa_metrics(payload: Dict) -> Optional[Dict]:
    # Direct SpeechQA metrics JSON shape.
    if "subtask_accuracy" in payload and "accuracy" in payload:
        return payload
    # MTL eval_results shape.
    results = payload.get("results", {})
    if isinstance(results, dict):
        sq = results.get("speech_qa")
        if isinstance(sq, dict) and "subtask_accuracy" in sq:
            return sq
    # Per-task wrapper shape (if introduced later).
    if payload.get("task") == "speech_qa" and isinstance(payload.get("metrics"), dict):
        sq = payload["metrics"]
        if "subtask_accuracy" in sq:
            return sq
    return None


def _extract_6task_key_from_path(raw: Optional[str]) -> Optional[str]:
    if not isinstance(raw, str):
        return None
    m = re.search(r"artifacts/mtl/6_task/([^/]+)/adapters", raw)
    if m:
        return m.group(1)
    return None


def _find_continual_config_from_eval(source_path: Path) -> Optional[Dict]:
    # .../artifacts/mtl/continual/7_task/<setting>/metrics/<best|latest>/eval_results_test.json
    try:
        setting_root = source_path.parents[2]
    except IndexError:
        return None
    cfg_candidates = sorted(setting_root.glob("adapters/*/best/mtl_config_resolved.json"))
    if not cfg_candidates:
        return None
    try:
        return load_json(cfg_candidates[0])
    except Exception:
        return None


def _build_provenance(
    *,
    model_name: str,
    source_type: str,
    source_path: Path,
) -> Dict[str, Optional[str]]:
    rel = source_path.relative_to(ROOT).as_posix()
    out: Dict[str, Optional[str]] = {
        "regime": source_type,
        "task_combo": None,
        "base_6task": None,
        "added_task": None,
        "merge_method": None,
        "merge_alpha": None,
        "merge_lambda": None,
    }

    if source_type == "baseline":
        if model_name.startswith("best_") and model_name.endswith("_adapter"):
            out["regime"] = "single_task_adapter"
            out["added_task"] = model_name[len("best_") : -len("_adapter")]
        else:
            out["regime"] = "base_model"
        out["task_combo"] = "speech_qa"
        return out

    if source_type == "mtl":
        if "/artifacts/mtl/continual/7_task/" in f"/{rel}":
            out["regime"] = "mtl_continual"
            setting = source_path.parts[4] if len(source_path.parts) > 4 else None
            out["task_combo"] = setting
            cfg = _find_continual_config_from_eval(source_path)
            continual = (cfg or {}).get("continual", {}) if isinstance(cfg, dict) else {}
            if isinstance(continual, dict):
                out["base_6task"] = _extract_6task_key_from_path(continual.get("base_adapter"))
                added_tasks = continual.get("added_tasks")
                if isinstance(added_tasks, list) and added_tasks:
                    out["added_task"] = str(added_tasks[0])
            if out["base_6task"] is None and isinstance(setting, str) and "__added_" in setting:
                left, right = setting.split("__added_", 1)
                if left.startswith("base_"):
                    out["base_6task"] = left[len("base_") :]
                out["added_task"] = right or out["added_task"]
        else:
            out["regime"] = "mtl_joint"
            setting = source_path.parts[3] if len(source_path.parts) > 3 else None
            out["task_combo"] = setting
        return out

    if source_type == "merged":
        # artifacts/merged/<method>/<combo>/eval/test/per_task/speech_qa/<file>.json
        parts = source_path.parts
        method = parts[2] if len(parts) > 2 else None
        combo = parts[3] if len(parts) > 3 else None
        out["merge_method"] = method
        out["task_combo"] = combo
        out["regime"] = "merged_continual" if method == "continual" else "merged"

        fname = source_path.name
        m_alpha = re.search(r"_alpha([0-9]+(?:p[0-9]+)?)", fname)
        m_lambda = re.search(r"_lambda([0-9]+(?:p[0-9]+)?)", fname)
        if m_alpha:
            out["merge_alpha"] = m_alpha.group(1).replace("p", ".")
        if m_lambda:
            out["merge_lambda"] = m_lambda.group(1).replace("p", ".")

        # Infer added task for continual merged variants from filename pattern.
        if method == "continual":
            if "_asr_" in fname:
                out["added_task"] = "asr"
                out["base_6task"] = "emotion_intent_kws_langid_speaker_ver_vocalsound"
            elif "_added_emotion" in fname or "_emotion_" in fname:
                out["added_task"] = "emotion"
                out["base_6task"] = "asr_intent_kws_langid_speaker_ver_vocalsound"
            elif "_vocalsound_" in fname:
                out["added_task"] = "vocalsound"
                out["base_6task"] = "asr_emotion_intent_kws_langid_speaker_ver"
        return out

    return out


def build_row(model_name: str, source_type: str, payload: Dict, source_path: Path) -> Dict:
    metrics = _extract_speechqa_metrics(payload)
    if metrics is None:
        return {
            "model": model_name,
            "source_type": source_type,
            "overall_accuracy": None,
            "overall_exact_match": None,
            "overall_f1": None,
            "num_subtasks_total": 0,
            "num_subtasks_categorized": 0,
            "num_subtasks_unclassified": 0,
            "source_file": str(source_path),
            "notes": "no_speechqa_metrics_found",
            **_build_provenance(model_name=model_name, source_type=source_type, source_path=source_path),
        }

    subtask_acc = metrics.get("subtask_accuracy", {})
    subtask_n = metrics.get("subtask_num_samples", {})

    row = {
        "model": model_name,
        "source_type": source_type,
        "overall_accuracy": metrics.get("accuracy"),
        "overall_exact_match": metrics.get("exact_match"),
        "overall_f1": metrics.get("f1"),
        "num_subtasks_total": 0,
        "num_subtasks_categorized": 0,
        "num_subtasks_unclassified": 0,
        "source_file": str(source_path),
        "notes": "",
        **_build_provenance(model_name=model_name, source_type=source_type, source_path=source_path),
    }

    for cat in CATEGORIES:
        row[f"{cat.lower()}_macro_acc"] = None
        row[f"{cat.lower()}_weighted_acc"] = None
        row[f"{cat.lower()}_num_subtasks"] = 0
        row[f"{cat.lower()}_num_samples"] = 0

    by_cat_values: Dict[str, List[float]] = {cat: [] for cat in CATEGORIES}
    by_cat_weighted_num: Dict[str, float] = {cat: 0.0 for cat in CATEGORIES}
    by_cat_weighted_den: Dict[str, int] = {cat: 0 for cat in CATEGORIES}

    for task, acc in subtask_acc.items():
        if not isinstance(acc, (int, float)):
            continue
        row["num_subtasks_total"] += 1
        cat = categorize_subtask(task)
        if cat is None:
            row["num_subtasks_unclassified"] += 1
            continue

        row["num_subtasks_categorized"] += 1
        by_cat_values[cat].append(float(acc))
        row[f"{cat.lower()}_num_subtasks"] += 1

        n = subtask_n.get(task)
        if isinstance(n, int) and n > 0:
            by_cat_weighted_num[cat] += float(acc) * n
            by_cat_weighted_den[cat] += n
            row[f"{cat.lower()}_num_samples"] += n

    for cat in CATEGORIES:
        vals = by_cat_values[cat]
        if vals:
            row[f"{cat.lower()}_macro_acc"] = sum(vals) / len(vals)
        if by_cat_weighted_den[cat] > 0:
            row[f"{cat.lower()}_weighted_acc"] = (
                by_cat_weighted_num[cat] / by_cat_weighted_den[cat]
            )

    return row


def main() -> None:
    rows = []
    for model_name, path in BASELINE_MODEL_FILES:
        if not path.exists():
            rows.append(
                {
                    "model": model_name,
                    "source_type": "baseline",
                    "overall_accuracy": None,
                    "overall_exact_match": None,
                    "overall_f1": None,
                    "num_subtasks_total": 0,
                    "num_subtasks_categorized": 0,
                    "num_subtasks_unclassified": 0,
                    "source_file": f"MISSING:{path}",
                    "notes": "missing_file",
                    **_build_provenance(model_name=model_name, source_type="baseline", source_path=path),
                }
            )
            continue
        payload = load_json(path)
        rows.append(build_row(model_name, "baseline", payload, path))

    mtl_paths = sorted(
        Path(p)
        for p in glob.glob("artifacts/mtl/**/metrics/*/eval_results_test.json", recursive=True)
    )
    for path in mtl_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        if _extract_speechqa_metrics(payload) is None:
            continue
        rel = path.relative_to(ROOT)
        model_name = f"mtl::{rel.parts[2]}::{rel.parts[3]}"
        rows.append(build_row(model_name, "mtl", payload, path))

    merged_paths = sorted(
        Path(p)
        for p in glob.glob("artifacts/merged/**/eval/test/per_task/speech_qa/*.json", recursive=True)
    )
    for path in merged_paths:
        if not path.exists():
            continue
        payload = load_json(path)
        if _extract_speechqa_metrics(payload) is None:
            continue
        rel = path.relative_to(ROOT)
        # artifacts/merged/<method>/<combo>/...
        method = rel.parts[2] if len(rel.parts) > 2 else "unknown"
        combo = rel.parts[3] if len(rel.parts) > 3 else "unknown"
        model_name = f"merged::{method}::{combo}"
        rows.append(build_row(model_name, "merged", payload, path))

    fieldnames = [
        "model",
        "source_type",
        "regime",
        "task_combo",
        "base_6task",
        "added_task",
        "merge_method",
        "merge_alpha",
        "merge_lambda",
        "overall_accuracy",
        "overall_exact_match",
        "overall_f1",
        "reasoning_macro_acc",
        "reasoning_weighted_acc",
        "reasoning_num_subtasks",
        "reasoning_num_samples",
        "perception_macro_acc",
        "perception_weighted_acc",
        "perception_num_subtasks",
        "perception_num_samples",
        "detection_macro_acc",
        "detection_weighted_acc",
        "detection_num_subtasks",
        "detection_num_samples",
        "recognition_macro_acc",
        "recognition_weighted_acc",
        "recognition_num_subtasks",
        "recognition_num_samples",
        "comparison_macro_acc",
        "comparison_weighted_acc",
        "comparison_num_subtasks",
        "comparison_num_samples",
        "language_macro_acc",
        "language_weighted_acc",
        "language_num_subtasks",
        "language_num_samples",
        "num_subtasks_total",
        "num_subtasks_categorized",
        "num_subtasks_unclassified",
        "source_file",
        "notes",
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f"Wrote {OUT_CSV}")


if __name__ == "__main__":
    main()
