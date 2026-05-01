#!/usr/bin/env python3
import csv
import json
from pathlib import Path
from statistics import mean

TARGET_TASKS = ["emotion", "intent", "kws", "langid"]

# User-requested sources.
INPUTS = [
    # uniform_delta (both run folders)
    "artifacts/merged/uniform_delta/emotion_intent_kws_langid/runs/run_20260209_223613/eval_results_test.json",
    "artifacts/merged/uniform_delta/emotion_intent_kws_langid/runs/run_20260216_124047/eval_results_test.json",
    # uniform_scalar_delta (two explicit runs)
    "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/runs/run_20260213_201016/eval_results_test.json",
    "artifacts/merged/uniform_scalar_delta/emotion_intent_kws_langid/runs/run_20260216_130805/eval_results_test.json",
    # weighted_delta_n: 4-task runs opened/requested
    "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260214_145251/eval_results_test.json",
    "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260214_193409/eval_results_test.json",
    "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260214_225951/eval_results_test.json",
    "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/runs/run_supermerge_layer_wise_20260215_045019/eval_results_test.json",
    # weighted_delta_n scalar+simplex eval (asr+speaker_ver)
    "artifacts/merged/weighted_delta_n/emotion_intent_kws_langid/eval/test/eval_results_merged_weighted_delta_n_emotion_intent_kws_langid_test.json",
]

OUT_DIR = Path("analysis/merge_comparison")
OUT_DIR.mkdir(parents=True, exist_ok=True)


def run_id_from_path(path: Path) -> str:
    for part in path.parts:
        if part.startswith("run_"):
            return part
    return path.stem


rows = []
missing = []
for p in INPUTS:
    path = Path(p)
    if not path.exists():
        missing.append(str(path))
        continue

    with path.open() as f:
        payload = json.load(f)

    method = payload.get("merge_method", "unknown")
    run_id = run_id_from_path(path)
    timestamp = payload.get("timestamp")

    for task, metrics in payload.get("results", {}).items():
        meta = metrics.get("interference_delta_meta", {}) or {}
        primary_metric = meta.get("metric")
        primary_metric_value = meta.get("merged")

        rows.append(
            {
                "merge_method": method,
                "run_id": run_id,
                "timestamp": timestamp,
                "task": task,
                "source_path": str(path),
                "interference_delta": metrics.get("interference_delta"),
                "interference_metric": primary_metric,
                "interference_base": meta.get("base"),
                "interference_task_adapter": meta.get("task_adapter"),
                "interference_merged": primary_metric_value,
                "accuracy": metrics.get("accuracy"),
                "macro_f1": metrics.get("macro_f1"),
                "weighted_f1": metrics.get("weighted_f1"),
                "wer": metrics.get("wer"),
                "loss": metrics.get("loss"),
                "recognized_rate": metrics.get("recognized_rate"),
                "num_samples": metrics.get("num_samples"),
                "runtime": metrics.get("runtime"),
                "samples_per_second": metrics.get("samples_per_second"),
                "steps_per_second": metrics.get("steps_per_second"),
            }
        )

rows.sort(key=lambda r: (r["merge_method"], r["run_id"], r["task"]))

all_csv = OUT_DIR / "merge_results_all_requested_tasks.csv"
fieldnames = [
    "merge_method",
    "run_id",
    "timestamp",
    "task",
    "source_path",
    "interference_delta",
    "interference_metric",
    "interference_base",
    "interference_task_adapter",
    "interference_merged",
    "accuracy",
    "macro_f1",
    "weighted_f1",
    "wer",
    "loss",
    "recognized_rate",
    "num_samples",
    "runtime",
    "samples_per_second",
    "steps_per_second",
]

with all_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(rows)


target_rows = [r for r in rows if r["task"] in TARGET_TASKS]
target_csv = OUT_DIR / "merge_results_emotion_intent_kws_langid.csv"
with target_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(target_rows)

# Run-level summary over the 4 target tasks.
summary_rows = []
keyed = {}
for r in target_rows:
    k = (r["merge_method"], r["run_id"], r["timestamp"])
    keyed.setdefault(k, []).append(r)

for (method, run_id, ts), vals in sorted(keyed.items()):
    deltas = [v["interference_delta"] for v in vals if isinstance(v["interference_delta"], (int, float))]
    summary_rows.append(
        {
            "merge_method": method,
            "run_id": run_id,
            "timestamp": ts,
            "num_target_tasks": len(vals),
            "avg_interference_delta": mean(deltas) if deltas else None,
            "min_interference_delta": min(deltas) if deltas else None,
            "max_interference_delta": max(deltas) if deltas else None,
        }
    )

summary_csv = OUT_DIR / "merge_summary_emotion_intent_kws_langid.csv"
summary_fields = [
    "merge_method",
    "run_id",
    "timestamp",
    "num_target_tasks",
    "avg_interference_delta",
    "min_interference_delta",
    "max_interference_delta",
]
with summary_csv.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=summary_fields)
    writer.writeheader()
    writer.writerows(summary_rows)

# Gnuplot helper data: use stable x labels.
interference_dat = OUT_DIR / "plot_interference.dat"
primary_dat = OUT_DIR / "plot_primary_metric.dat"

with interference_dat.open("w") as f1, primary_dat.open("w") as f2:
    f1.write("label task value\n")
    f2.write("label task value metric\n")
    for r in target_rows:
        label = f"{r['merge_method']}|{r['run_id']}"
        value = r["interference_delta"]
        if value is not None:
            f1.write(f'"{label}" "{r["task"]}" {value}\n')

        pm = r["interference_merged"]
        if pm is not None:
            metric_name = r["interference_metric"] or "n/a"
            f2.write(f'"{label}" "{r["task"]}" {pm} "{metric_name}"\n')

report = OUT_DIR / "README.txt"
with report.open("w") as f:
    f.write("Generated files:\n")
    f.write(f"- {all_csv}\n")
    f.write(f"- {target_csv}\n")
    f.write(f"- {summary_csv}\n")
    f.write(f"- {interference_dat}\n")
    f.write(f"- {primary_dat}\n")
    if missing:
        f.write("\\nMissing input files:\n")
        for m in missing:
            f.write(f"- {m}\n")

print(f"Wrote {all_csv}")
print(f"Wrote {target_csv}")
print(f"Wrote {summary_csv}")
print(f"Wrote {interference_dat}")
print(f"Wrote {primary_dat}")
if missing:
    print("Missing inputs:")
    for m in missing:
        print(m)
