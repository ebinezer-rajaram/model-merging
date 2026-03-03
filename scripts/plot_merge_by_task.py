#!/usr/bin/env python3
import csv
from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

IN_CSV = Path('analysis/merge_comparison/merge_results_all_requested_tasks.csv')
OUT_DIR = Path('analysis/merge_comparison')
OUT_DIR.mkdir(parents=True, exist_ok=True)

TASKS = ['emotion', 'intent', 'kws', 'langid', 'speaker_ver', 'asr']
METHOD_ORDER = ['uniform_delta', 'uniform_scalar_delta', 'weighted_delta_n']
METHOD_LABELS = {
    'uniform_delta': 'Uniform (1/T)',
    'uniform_scalar_delta': 'Uniform scalar',
    'weighted_delta_n': 'SuperMerge layer-wise (scalar+simplex)',
}

rows = []
with IN_CSV.open() as f:
    reader = csv.DictReader(f)
    for r in reader:
        rows.append(r)

agg = defaultdict(lambda: {
    'interference_delta': [],
    'interference_merged': [],
    'interference_metric': set(),
})

for r in rows:
    task = r['task']
    method = r['merge_method']
    if task not in TASKS:
        continue
    k = (task, method)

    idel = r.get('interference_delta', '')
    if idel != '':
        agg[k]['interference_delta'].append(float(idel))

    imer = r.get('interference_merged', '')
    if imer != '':
        agg[k]['interference_merged'].append(float(imer))

    metric = (r.get('interference_metric') or '').strip()
    if metric:
        agg[k]['interference_metric'].add(metric)

summary_path = OUT_DIR / 'method_task_summary.csv'
with summary_path.open('w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow([
        'task', 'merge_method', 'num_runs',
        'avg_interference_delta', 'avg_primary_metric', 'primary_metric_names'
    ])
    for task in TASKS:
        for method in METHOD_ORDER:
            d = agg[(task, method)]
            n = max(len(d['interference_delta']), len(d['interference_merged']))
            avg_d = (sum(d['interference_delta']) / len(d['interference_delta'])) if d['interference_delta'] else ''
            avg_p = (sum(d['interference_merged']) / len(d['interference_merged'])) if d['interference_merged'] else ''
            metrics = ';'.join(sorted(d['interference_metric']))
            writer.writerow([task, method, n, avg_d, avg_p, metrics])


def grouped_bar(values_by_method, ylabel, title, out_file, ylim=None, footer_note=None):
    x = np.arange(len(TASKS))
    width = 0.24

    fig, ax = plt.subplots(figsize=(11, 6.2), dpi=150)
    fig.patch.set_facecolor('#dcdcdc')
    ax.set_facecolor('#e9e9e9')
    colors = plt.cm.tab10.colors

    for i, method in enumerate(METHOD_ORDER):
        vals = [values_by_method[method].get(task, np.nan) for task in TASKS]
        bars = ax.bar(
            x + (i - 1) * width,
            vals,
            width=width,
            label=METHOD_LABELS.get(method, method),
            color=colors[i],
            alpha=0.88,
        )
        for bar in bars:
            h = bar.get_height()
            if np.isnan(h):
                continue
            ax.annotate(
                f'{h:.3f}',
                xy=(bar.get_x() + bar.get_width() / 2, h),
                xytext=(0, 3),
                textcoords='offset points',
                ha='center',
                va='bottom',
                fontsize=10,
                color='#333333',
            )

    ax.set_xticks(x)
    ax.set_xticklabels([t.upper() for t in TASKS], fontsize=12)
    ax.set_xlabel('Task', fontsize=13, fontweight='bold')
    ax.set_ylabel(ylabel, fontsize=13, fontweight='bold')
    ax.set_title(title, fontsize=17, fontweight='bold', pad=10)
    if ylim:
        ax.set_ylim(*ylim)
    ax.set_axisbelow(True)
    ax.grid(True, which='major', axis='y', alpha=0.45, linestyle='-', linewidth=0.8, color='#9a9a9a')
    ax.grid(True, which='minor', axis='y', alpha=0.25, linestyle=':', linewidth=0.5, color='#a7a7a7')
    ax.minorticks_on()
    ax.tick_params(axis='y', labelsize=12)
    for spine in ax.spines.values():
        spine.set_linewidth(1.0)
        spine.set_color('#333333')
    ax.legend(loc='upper left', framealpha=0.95, fontsize=11)
    if footer_note:
        fig.text(0.01, 0.012, footer_note, fontsize=9, color='#333333')
    fig.tight_layout()
    fig.savefig(out_file)
    plt.close(fig)

interference_values = {m: {} for m in METHOD_ORDER}
primary_values = {m: {} for m in METHOD_ORDER}
for task in TASKS:
    for method in METHOD_ORDER:
        d = agg[(task, method)]
        if d['interference_delta']:
            interference_values[method][task] = sum(d['interference_delta']) / len(d['interference_delta'])
        if d['interference_merged']:
            primary_values[method][task] = sum(d['interference_merged']) / len(d['interference_merged'])

grouped_bar(
    interference_values,
    ylabel='Interference delta',
    title='Interference Delta Comparison by Task',
    out_file=OUT_DIR / 'interference_delta_by_task_method.png',
    ylim=None,
)

grouped_bar(
    primary_values,
    ylabel='Primary metric value',
    title='Primary Metric Comparison by Task',
    out_file=OUT_DIR / 'primary_metric_by_task_method.png',
    ylim=(0.0, 1.05),
    footer_note='Note: asr uses WER (lower is better); other tasks use accuracy or macro_f1 (higher is better).',
)

print(f'Wrote {summary_path}')
print(f'Wrote {OUT_DIR / "interference_delta_by_task_method.png"}')
print(f'Wrote {OUT_DIR / "primary_metric_by_task_method.png"}')
