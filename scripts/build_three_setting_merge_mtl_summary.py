#!/usr/bin/env python3
import csv
import json
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Dict, List, Optional, Tuple

ROOT = Path('.')
DATE_STAMP = datetime.now().strftime('%Y%m%d')
OUT_CSV = ROOT / 'artifacts/merged/comparisons' / f'three_setting_merge_mtl_summary_{DATE_STAMP}.csv'

TASK_METRIC_MAP = {
    'emotion': ('macro_f1', 'emotion_macro_f1'),
    'intent': ('accuracy', 'intent_acc'),
    'kws': ('macro_f1', 'kws_macro_f1'),
    'langid': ('accuracy', 'langid_acc'),
    'vocalsound': ('accuracy', 'vocalsound_acc'),
    'speaker_ver': ('accuracy', 'speaker_ver_acc'),
    'asr': ('wer', 'asr_wer'),
    'speech_qa': ('accuracy', 'spoken_qa_acc'),
}

INTERFERENCE_DELTA_COLS = [
    "emotion_macro_f1_interference_delta",
    "intent_acc_interference_delta",
    "kws_macro_f1_interference_delta",
    "langid_acc_interference_delta",
    "vocalsound_acc_interference_delta",
    "speaker_ver_acc_interference_delta",
    "asr_wer_interference_delta",
    "spoken_qa_acc_interference_delta",
]

DELTA_TO_METRIC_COL = {
    "emotion_macro_f1_interference_delta": "emotion_macro_f1",
    "intent_acc_interference_delta": "intent_acc",
    "kws_macro_f1_interference_delta": "kws_macro_f1",
    "langid_acc_interference_delta": "langid_acc",
    "vocalsound_acc_interference_delta": "vocalsound_acc",
    "speaker_ver_acc_interference_delta": "speaker_ver_acc",
    "asr_wer_interference_delta": "asr_wer",
    "spoken_qa_acc_interference_delta": "spoken_qa_acc",
}

METRIC_ORIENTATION = {
    "emotion_macro_f1": "higher",
    "intent_acc": "higher",
    "kws_macro_f1": "higher",
    "langid_acc": "higher",
    "vocalsound_acc": "higher",
    "speaker_ver_acc": "higher",
    "asr_wer": "lower",
    "spoken_qa_acc": "higher",
}

SETTINGS = {
    '7_task': {
        'merge_uniform_summary': ROOT / 'artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/summary.json',
        'merge_weighted_summary': ROOT / 'artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/summary.json',
        'mtl_candidates': [
            ROOT / 'artifacts/mtl/7_task/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/latest/test_metrics.json',
            ROOT / 'artifacts/mtl/7_task/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/best/test_metrics.json',
            ROOT / 'artifacts/mtl/7_task/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/best/metrics.json',
        ],
    },
    '6_task_minus_asr': {
        'merge_uniform_summary': ROOT / 'artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/summary.json',
        'merge_weighted_summary': ROOT / 'artifacts/merged/weighted_delta_n/emotion_intent_kws_langid_speaker_ver_vocalsound/eval/test/summary.json',
        'mtl_candidates': [
            ROOT / 'artifacts/mtl/6_task/emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/latest/test_metrics.json',
            ROOT / 'artifacts/mtl/6_task/emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/best/test_metrics.json',
            ROOT / 'artifacts/mtl/6_task/emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/best/metrics.json',
        ],
    },
    '6_task_minus_emotion': {
        'merge_uniform_summary': ROOT / 'artifacts/merged/uniform_scalar_delta/asr_intent_kws_langid_speaker_ver_vocalsound/eval/test/summary.json',
        'merge_weighted_summary': ROOT / 'artifacts/merged/weighted_delta_n/asr_intent_kws_langid_speaker_ver_vocalsound/eval/test/summary.json',
        'mtl_candidates': [
            ROOT / 'artifacts/mtl/6_task/asr_intent_kws_langid_speaker_ver_vocalsound/metrics/best/test_metrics.json',
            ROOT / 'artifacts/mtl/6_task/asr_intent_kws_langid_speaker_ver_vocalsound/metrics/latest/test_metrics.json',
            ROOT / 'artifacts/mtl/6_task/asr_intent_kws_langid_speaker_ver_vocalsound/metrics/best/metrics.json',
        ],
    },
    '6_task_minus_vocalsound': {
        'merge_uniform_summary': ROOT / 'artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver/eval/test/summary.json',
        'merge_weighted_summary': ROOT / 'artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver/eval/test/summary.json',
        'mtl_candidates': [
            ROOT / 'artifacts/mtl/6_task/asr_emotion_intent_kws_langid_speaker_ver/metrics/best/test_metrics.json',
            ROOT / 'artifacts/mtl/6_task/asr_emotion_intent_kws_langid_speaker_ver/metrics/latest/test_metrics.json',
            ROOT / 'artifacts/mtl/6_task/asr_emotion_intent_kws_langid_speaker_ver/metrics/best/metrics.json',
        ],
    },
}

CONTINUAL_7TASK = {
    "plus_asr": {
        "setting": "7_task_continual_plus_asr",
        "mtl_candidates": [
            ROOT
            / "artifacts/mtl/continual/7_task/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/best/test_metrics.json",
            ROOT
            / "artifacts/mtl/continual/7_task/asr_emotion_intent_kws_langid_speaker_ver_vocalsound/metrics/latest/test_metrics.json",
        ],
        "merge_summary": ROOT
        / "artifacts/merged/continual/6task_plus_asr/eval/alpha1_lambda075/continual_sweep_summary.json",
    },
    "plus_emotion": {
        "setting": "7_task_continual_plus_emotion",
        "mtl_candidates": [
            ROOT
            / "artifacts/mtl/continual/7_task/base_intent_kws_langid_speaker_ver_asr_vocalsound__added_emotion/metrics/best/test_metrics.json",
            ROOT
            / "artifacts/mtl/continual/7_task/base_intent_kws_langid_speaker_ver_asr_vocalsound__added_emotion/metrics/latest/test_metrics.json",
        ],
        "merge_sweep_dir": ROOT / "artifacts/merged/continual/6task_supermerge_20260316_emotion/sweeps",
    },
}


def load_json(path: Path) -> Dict:
    with path.open('r', encoding='utf-8') as f:
        return json.load(f)


def maybe_float(v):
    if isinstance(v, (int, float)):
        return float(v)
    return None


def _compute_interference_delta(
    *,
    measured: Optional[float],
    base: Optional[float],
    best: Optional[float],
    orientation: str,
) -> Optional[float]:
    if measured is None or base is None or best is None:
        return None
    if orientation == "higher":
        denom = best - base
        if abs(denom) < 1e-12:
            return None
        return (measured - base) / denom
    if orientation == "lower":
        denom = base - best
        if abs(denom) < 1e-12:
            return None
        return (base - measured) / denom
    return None


def _build_delta_baselines() -> Dict[str, Dict[str, Optional[float]]]:
    base_row = baseline_row("base_model", "")
    best_row = baseline_row("best_single_task", "")
    baselines: Dict[str, Dict[str, Optional[float]]] = {}
    for metric_col in METRIC_ORIENTATION:
        baselines[metric_col] = {
            "base": maybe_float(base_row.get(metric_col)),
            "best": maybe_float(best_row.get(metric_col)),
        }
    return baselines


def _fill_missing_interference_deltas(rows: List[Dict]) -> None:
    baselines = _build_delta_baselines()
    for row in rows:
        for delta_col, metric_col in DELTA_TO_METRIC_COL.items():
            if row.get(delta_col) is not None:
                continue
            measured = maybe_float(row.get(metric_col))
            baseline = baselines.get(metric_col, {})
            delta_val = _compute_interference_delta(
                measured=measured,
                base=maybe_float(baseline.get("base")),
                best=maybe_float(baseline.get("best")),
                orientation=METRIC_ORIENTATION[metric_col],
            )
            if delta_val is not None:
                row[delta_col] = delta_val

        deltas = [maybe_float(row.get(col)) for col in INTERFERENCE_DELTA_COLS]
        deltas = [d for d in deltas if d is not None]
        if deltas:
            row["min_interference_delta"] = min(deltas)
            row["mean_interference_delta"] = mean(deltas)


def load_per_task_metric(summary_path: Path, task: str, metric_key: str) -> Optional[float]:
    per_task_dir = summary_path.parent / "per_task" / task
    if not per_task_dir.exists():
        return None
    matches = sorted(per_task_dir.glob("*.json"))
    if not matches:
        return None
    payload = load_json(matches[0])
    return maybe_float(payload.get(metric_key))


def pick_existing(paths: List[Path]) -> Tuple[Optional[Path], Optional[str]]:
    for i, p in enumerate(paths):
        if p.exists():
            note = None if i == 0 else f'fallback_used={p}'
            return p, note
    return None, 'missing_all_candidates'


def baseline_row(method: str, note: str) -> Dict:
    row = {
        'method': method,
        'emotion_macro_f1': maybe_float(load_json(ROOT / 'artifacts/emotion/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/emotion/metrics/eval/test/best_emotion_adapter.json').get('macro_f1')),
        'intent_acc': maybe_float(load_json(ROOT / 'artifacts/intent/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/intent/metrics/eval/test/best_intent_adapter.json').get('accuracy')),
        'kws_macro_f1': maybe_float(load_json(ROOT / 'artifacts/kws/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/kws/metrics/eval/test/best_kws_adapter.json').get('macro_f1')),
        'langid_acc': maybe_float(load_json(ROOT / 'artifacts/langid/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/langid/metrics/eval/test/best_langid_adapter.json').get('accuracy')),
        'vocalsound_acc': maybe_float(load_json(ROOT / 'artifacts/vocalsound/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/vocalsound/metrics/eval/test/best_vocalsound_adapter.json').get('accuracy')),
        'speaker_ver_acc': maybe_float(load_json(ROOT / 'artifacts/speaker_ver/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/speaker_ver/metrics/eval/test/best_speaker_ver_adapter.json').get('accuracy')),
        'asr_wer': maybe_float(load_json(ROOT / 'artifacts/asr/metrics/eval/test/base_model.json' if method == 'base_model' else ROOT / 'artifacts/asr/metrics/eval/test/best_asr_adapter.json').get('wer')),
        'spoken_qa_acc': maybe_float(load_json(ROOT / 'artifacts/speech_qa/metrics/eval/test/base_model.json').get('accuracy')) if method == 'base_model' else None,
        'emotion_macro_f1_interference_delta': None,
        'intent_acc_interference_delta': None,
        'kws_macro_f1_interference_delta': None,
        'langid_acc_interference_delta': None,
        'vocalsound_acc_interference_delta': None,
        'speaker_ver_acc_interference_delta': None,
        'asr_wer_interference_delta': None,
        'spoken_qa_acc_interference_delta': None,
        'min_interference_delta': None,
        'mean_interference_delta': None,
        'notes': note,
    }
    return row


def merge_row(method: str, summary_path: Path) -> Dict:
    payload = load_json(summary_path)
    results = payload.get('results', {})
    row = {
        'method': method,
        'emotion_macro_f1': None,
        'intent_acc': None,
        'kws_macro_f1': None,
        'langid_acc': None,
        'vocalsound_acc': None,
        'speaker_ver_acc': None,
        'asr_wer': None,
        'spoken_qa_acc': None,
        'emotion_macro_f1_interference_delta': None,
        'intent_acc_interference_delta': None,
        'kws_macro_f1_interference_delta': None,
        'langid_acc_interference_delta': None,
        'vocalsound_acc_interference_delta': None,
        'speaker_ver_acc_interference_delta': None,
        'asr_wer_interference_delta': None,
        'spoken_qa_acc_interference_delta': None,
        'min_interference_delta': None,
        'mean_interference_delta': None,
        'notes': str(summary_path),
    }

    deltas = []
    for task, (metric_key, col) in TASK_METRIC_MAP.items():
        metric_val = load_per_task_metric(summary_path, task, metric_key)
        if metric_val is None:
            t = results.get(task, {}).get('test', {})
            metric_val = maybe_float(t.get(metric_key))
        row[col] = metric_val

        t = results.get(task, {}).get('test', {})
        delta_val = maybe_float(t.get('interference_delta'))
        row[f'{col}_interference_delta'] = delta_val
        if delta_val is not None:
            deltas.append(delta_val)

    if deltas:
        row['min_interference_delta'] = min(deltas)
        row['mean_interference_delta'] = mean(deltas)
    return row


def mtl_row(setting_name: str, candidates: List[Path]) -> Dict:
    selected, note = pick_existing(candidates)
    row = {
        'method': 'mtl',
        'emotion_macro_f1': None,
        'intent_acc': None,
        'kws_macro_f1': None,
        'langid_acc': None,
        'vocalsound_acc': None,
        'speaker_ver_acc': None,
        'asr_wer': None,
        'spoken_qa_acc': None,
        'emotion_macro_f1_interference_delta': None,
        'intent_acc_interference_delta': None,
        'kws_macro_f1_interference_delta': None,
        'langid_acc_interference_delta': None,
        'vocalsound_acc_interference_delta': None,
        'speaker_ver_acc_interference_delta': None,
        'asr_wer_interference_delta': None,
        'spoken_qa_acc_interference_delta': None,
        'min_interference_delta': None,
        'mean_interference_delta': None,
        'notes': '' if selected is None else str(selected),
    }

    if selected is None:
        row['notes'] = f'MISSING_MTL_METRICS setting={setting_name} note={note}'
        return row

    payload = load_json(selected)

    # Metric columns from eval_<task>_<metric>
    key_map = {
        'emotion_macro_f1': ('emotion', 'macro_f1'),
        'intent_acc': ('intent', 'accuracy'),
        'kws_macro_f1': ('kws', 'macro_f1'),
        'langid_acc': ('langid', 'accuracy'),
        'vocalsound_acc': ('vocalsound', 'accuracy'),
        'speaker_ver_acc': ('speaker_ver', 'accuracy'),
        'asr_wer': ('asr', 'wer'),
    }
    for col, (task, metric) in key_map.items():
        row[col] = maybe_float(payload.get(f'eval_{task}_{metric}'))

    # Interference deltas
    delta_key_map = {
        'emotion_macro_f1_interference_delta': 'emotion',
        'intent_acc_interference_delta': 'intent',
        'kws_macro_f1_interference_delta': 'kws',
        'langid_acc_interference_delta': 'langid',
        'vocalsound_acc_interference_delta': 'vocalsound',
        'speaker_ver_acc_interference_delta': 'speaker_ver',
        'asr_wer_interference_delta': 'asr',
    }
    deltas = []
    for col, task in delta_key_map.items():
        val = maybe_float(payload.get(f'eval_{task}_interference_delta'))
        row[col] = val
        if val is not None:
            deltas.append(val)

    # Backfill missing metrics from sibling eval_results_test.json when present.
    # This covers held-out task evaluations run post-hoc via scripts/eval_mtl_adapter.py.
    supplemented_from = None
    eval_results_path = selected.parent / "eval_results_test.json"
    if eval_results_path.exists():
        eval_payload = load_json(eval_results_path)
        eval_results = eval_payload.get("results", {}) if isinstance(eval_payload, dict) else {}
        if isinstance(eval_results, dict):
            for col, (task, metric) in key_map.items():
                if row[col] is None:
                    task_payload = eval_results.get(task, {})
                    if isinstance(task_payload, dict):
                        row[col] = maybe_float(task_payload.get(metric))
                        if row[col] is not None:
                            supplemented_from = eval_results_path

            if row["spoken_qa_acc"] is None:
                speech_qa_payload = eval_results.get("speech_qa", {})
                if isinstance(speech_qa_payload, dict):
                    row["spoken_qa_acc"] = maybe_float(speech_qa_payload.get("accuracy"))
                    if row["spoken_qa_acc"] is not None:
                        supplemented_from = eval_results_path

            for col, task in delta_key_map.items():
                if row[col] is not None:
                    continue
                task_payload = eval_results.get(task, {})
                if isinstance(task_payload, dict):
                    row[col] = maybe_float(task_payload.get("interference_delta"))
                    if row[col] is not None:
                        supplemented_from = eval_results_path

            if row["spoken_qa_acc_interference_delta"] is None:
                speech_qa_payload = eval_results.get("speech_qa", {})
                if isinstance(speech_qa_payload, dict):
                    row["spoken_qa_acc_interference_delta"] = maybe_float(
                        speech_qa_payload.get("interference_delta")
                    )
                    if row["spoken_qa_acc_interference_delta"] is not None:
                        supplemented_from = eval_results_path

    all_delta_cols = [
        "emotion_macro_f1_interference_delta",
        "intent_acc_interference_delta",
        "kws_macro_f1_interference_delta",
        "langid_acc_interference_delta",
        "vocalsound_acc_interference_delta",
        "speaker_ver_acc_interference_delta",
        "asr_wer_interference_delta",
        "spoken_qa_acc_interference_delta",
    ]
    all_deltas = [row[c] for c in all_delta_cols if isinstance(row.get(c), float)]
    if all_deltas:
        row['min_interference_delta'] = min(all_deltas)
        row['mean_interference_delta'] = mean(all_deltas)

    if note:
        row['notes'] = f"{row['notes']} ; {note}"
    if supplemented_from is not None:
        row['notes'] = f"{row['notes']} ; supplemented_from={supplemented_from}"

    return row


def mtl_row_with_method(setting_name: str, candidates: List[Path], method_name: str) -> Dict:
    row = mtl_row(setting_name, candidates)
    row["method"] = method_name
    return row


def continual_merge_row_from_eval_results(method: str, eval_results_path: Path, note: str) -> Dict:
    payload = load_json(eval_results_path)
    results = payload.get("results", {})
    row = {
        "method": method,
        "emotion_macro_f1": None,
        "intent_acc": None,
        "kws_macro_f1": None,
        "langid_acc": None,
        "vocalsound_acc": None,
        "speaker_ver_acc": None,
        "asr_wer": None,
        "spoken_qa_acc": None,
        "emotion_macro_f1_interference_delta": None,
        "intent_acc_interference_delta": None,
        "kws_macro_f1_interference_delta": None,
        "langid_acc_interference_delta": None,
        "vocalsound_acc_interference_delta": None,
        "speaker_ver_acc_interference_delta": None,
        "asr_wer_interference_delta": None,
        "spoken_qa_acc_interference_delta": None,
        "min_interference_delta": None,
        "mean_interference_delta": None,
        "notes": note,
    }
    deltas = []
    for task, (metric_key, col) in TASK_METRIC_MAP.items():
        task_payload = results.get(task, {})
        row[col] = maybe_float(task_payload.get(metric_key))
        delta_val = maybe_float(task_payload.get("interference_delta"))
        row[f"{col}_interference_delta"] = delta_val
        if delta_val is not None:
            deltas.append(delta_val)
    if deltas:
        row["min_interference_delta"] = min(deltas)
        row["mean_interference_delta"] = mean(deltas)
    return row


def main() -> None:
    rows = []

    for setting_name, cfg in SETTINGS.items():
        rows.append({
            'setting': setting_name,
            **baseline_row('base_model', 'Per-task base_model.json files.')
        })
        rows.append({
            'setting': setting_name,
            **baseline_row('best_single_task', 'Per-task best_{task}_adapter.json files; spoken_qa best-single not available.')
        })
        rows.append({
            'setting': setting_name,
            **merge_row('uniform_scalar_delta_merge', cfg['merge_uniform_summary'])
        })
        rows.append({
            'setting': setting_name,
            **merge_row('weighted_delta_n_merge', cfg['merge_weighted_summary'])
        })
        rows.append({
            'setting': setting_name,
            **mtl_row(setting_name, cfg['mtl_candidates'])
        })

    # Continual 7-task comparisons (MTL continual vs continual merge best lambda).
    # plus_asr
    asr_summary = load_json(CONTINUAL_7TASK["plus_asr"]["merge_summary"])
    asr_best = asr_summary.get("best", {}) if isinstance(asr_summary, dict) else {}
    asr_best_lambda = asr_best.get("lambda")
    asr_artifact_dir = asr_best.get("artifact_dir")
    asr_eval = Path(asr_artifact_dir) / "eval_results_test.json" if asr_artifact_dir else None
    if asr_eval is not None and asr_eval.exists():
        rows.append({
            "setting": CONTINUAL_7TASK["plus_asr"]["setting"],
            **baseline_row("base_model", "Per-task base_model.json files."),
        })
        rows.append({
            "setting": CONTINUAL_7TASK["plus_asr"]["setting"],
            **baseline_row(
                "best_single_task",
                "Per-task best_{task}_adapter.json files; spoken_qa best-single not available.",
            ),
        })
        rows.append({
            "setting": CONTINUAL_7TASK["plus_asr"]["setting"],
            **continual_merge_row_from_eval_results(
                "continual_merge_best_lambda",
                asr_eval,
                f"best_lambda={asr_best_lambda}; source={CONTINUAL_7TASK['plus_asr']['merge_summary']}",
            ),
        })
        rows.append({
            "setting": CONTINUAL_7TASK["plus_asr"]["setting"],
            **mtl_row_with_method(
                CONTINUAL_7TASK["plus_asr"]["setting"],
                CONTINUAL_7TASK["plus_asr"]["mtl_candidates"],
                "mtl_continual",
            ),
        })

    # plus_emotion
    emotion_sweep_files = sorted(CONTINUAL_7TASK["plus_emotion"]["merge_sweep_dir"].glob("sweep_*.json"), reverse=True)
    chosen_emotion_sweep = None
    chosen_emotion_lambda = None
    chosen_emotion_artifact = None
    for sf in emotion_sweep_files:
        payload = load_json(sf)
        post = payload.get("post_sweep_eval", {}) if isinstance(payload, dict) else {}
        params = post.get("params", {}) if isinstance(post, dict) else {}
        art = post.get("artifact_dir")
        if isinstance(params, dict) and "lambda" in params and art:
            candidate_eval = Path(art) / "eval_results_test.json"
            if candidate_eval.exists():
                chosen_emotion_sweep = sf
                chosen_emotion_lambda = params.get("lambda")
                chosen_emotion_artifact = candidate_eval
                break
    if chosen_emotion_artifact is not None:
        rows.append({
            "setting": CONTINUAL_7TASK["plus_emotion"]["setting"],
            **baseline_row("base_model", "Per-task base_model.json files."),
        })
        rows.append({
            "setting": CONTINUAL_7TASK["plus_emotion"]["setting"],
            **baseline_row(
                "best_single_task",
                "Per-task best_{task}_adapter.json files; spoken_qa best-single not available.",
            ),
        })
        rows.append({
            "setting": CONTINUAL_7TASK["plus_emotion"]["setting"],
            **continual_merge_row_from_eval_results(
                "continual_merge_best_lambda",
                chosen_emotion_artifact,
                f"best_lambda={chosen_emotion_lambda}; source={chosen_emotion_sweep}",
            ),
        })
        rows.append({
            "setting": CONTINUAL_7TASK["plus_emotion"]["setting"],
            **mtl_row_with_method(
                CONTINUAL_7TASK["plus_emotion"]["setting"],
                CONTINUAL_7TASK["plus_emotion"]["mtl_candidates"],
                "mtl_continual",
            ),
        })

    _fill_missing_interference_deltas(rows)

    primary_metric_columns = [
        'emotion_macro_f1',
        'intent_acc',
        'kws_macro_f1',
        'langid_acc',
        'vocalsound_acc',
        'speaker_ver_acc',
        'asr_wer',
        'spoken_qa_acc',
    ]
    for row in rows:
        missing = [col for col in primary_metric_columns if row.get(col) is None]
        row['missing_eval'] = "|".join(missing)

    fieldnames = [
        'setting', 'method',
        'emotion_macro_f1', 'intent_acc', 'kws_macro_f1', 'langid_acc', 'vocalsound_acc', 'speaker_ver_acc', 'asr_wer', 'spoken_qa_acc',
        'emotion_macro_f1_interference_delta', 'intent_acc_interference_delta', 'kws_macro_f1_interference_delta',
        'langid_acc_interference_delta', 'vocalsound_acc_interference_delta', 'speaker_ver_acc_interference_delta', 'asr_wer_interference_delta', 'spoken_qa_acc_interference_delta',
        'min_interference_delta', 'mean_interference_delta', 'missing_eval', 'notes'
    ]

    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    print(f'Wrote {OUT_CSV}')


if __name__ == '__main__':
    main()
