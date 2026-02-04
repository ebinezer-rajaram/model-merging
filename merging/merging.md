# Merging Framework Overview

This package implements a modular framework for merging LoRA adapters, evaluating merged models, and running parameter sweeps. It is designed to be extensible (new merge techniques) and consistent in output artifacts.

## High‑Level Flow

1. **Resolve adapters** (task names or explicit paths)
2. **Run merge method** (registry dispatch)
3. **Optionally save merged adapter** (LoRA weights + metadata)
4. **Evaluate merged adapter** on one or more tasks
5. **Record outputs** into unified directories (run bundle + eval index)

The canonical CLI entrypoint is `main.py`. Legacy CLI is `experiments/merge_vectors.py` (shim).

## Package Structure

- `merging/techniques/`  
  Method implementations (uniform, weighted, task_vector, etc.)

- `merging/core/`  
  Registry, runner, metadata helpers, output paths

- `merging/evaluation/`  
  Evaluation + sweeps (grid now, bayes scaffold)

- `merging/cli.py`  
  Merge CLI wrapper used by `main.py merge`

## Registry + Method Interface

Merge methods are registered in `merging/core/methods.py` using `MergeMethod`:

- `name`
- `required_params`
- `params_defaults`
- `params_validator` (optional)
- `min_adapters` / `max_adapters`
- `saveable`
- `merge_in_memory(...) -> MergeOutput`
- `save_fn(...)` (optional)

This makes adding a new technique a single‑file change:

1. Implement in `merging/techniques/<new_method>.py`
2. Register in `merging/core/methods.py`

## Runner (Single Source of Truth)

`merging/core/runner.run_merge(...)` handles:

- adapter resolution (`resolve_adapter_specs`)
- parameter normalization (`normalize_params`)
- method dispatch
- optional saving (LoRA adapter + metadata)
- returns a `MergeResult` (method, params, tag, output path, metadata)

This is used by the CLI and is the preferred entrypoint for new integrations.

## Evaluation

`merging/evaluation/evaluate.py` handles:

- **In‑memory evaluation** (no adapter saved)
- **Saved adapter evaluation**
- Writes results to unified eval directories
- Adds entries to the eval results index

Key metric logging includes the task’s primary metric plus `interference_delta`.

## Sweeps

`merging/evaluation/sweep.py` runs sweeps over parameter grids:

- Default: `search.type = grid`
- Bayes: `search.type = bayes` (Gaussian Process + Expected Improvement)

Ranking uses **max‑min interference delta** across evaluated tasks.

### Bayesian optimization config

Example (`weighted_delta` / `weighted` lambda search):
```yaml
adapters: [asr, intent]
method: weighted_delta
split: test
eval_subset:
  # Optional: speed up sweeps by evaluating on a small, deterministic subset.
  # Results are namespaced so they won't overwrite full-split metrics.
  # Subset indices are cached on disk for consistency and reuse.
  enabled: true
  seed: 0
  shuffle: true
  stratified: true
  max_samples: 1000
  per_task:
    asr:
      max_samples: 200
search:
  type: bayes
  budget: 20
  init_points: 6
  seed: 42
  space:
    lambda:
      type: float
      min: 0.0
      max: 1.0
```

To reuse results from an existing sweep (grid or bayes) **without re-evaluating**, you can warm-start from one or more
previous sweep summary JSONs (a file, a directory containing `sweep_*.json`, or a glob):
```yaml
search:
  type: bayes
  warm_start_sweeps:
    - artifacts/merged/weighted_delta/asr_intent/sweeps/
  warm_start_max_points: 8
```
Warm-start sweeps are only used when they match the current `method`, `adapters`, `merge_mode`, `split`, `eval_tasks`,
and `constraint_nonnegative`.

## Output Layout (Unified)

### Run Bundles
Always created (even for in‑memory merges):
```
artifacts/merged/<method>/<task_combo>/runs/run_YYYYMMDD_HHMMSS/
  merge_metadata.json
  eval_results_<split>.json
  summary.json
  adapter_model.safetensors (if saved)
  adapter_config.json (if saved)
```

### Unified Eval Outputs
```
artifacts/merged/<method>/<task_combo>/eval/<split>/
  eval_results_<merge_tag>_<split>.json

artifacts/merged/<method>/<task_combo>/eval/index.json
```
The index is append‑only and stores a history of all evaluations.

### Per‑Task Metrics (Symlinked)
Merged metrics are saved in `artifacts/merged/...` and **symlinked** into task folders to avoid clutter:
```
artifacts/<task>/metrics/eval/<split>/merged/<other_tasks>/<method_or_tag>/metrics.json ->
  artifacts/merged/<method>/<task_combo>/eval/<split>/per_task/<task>/<task>_<label>_metrics.json
```

## Extending the Framework

To add a new technique:

1. Create `merging/techniques/<name>.py` with a `merge_in_memory` function.
2. Register it in `merging/core/methods.py` with:
   - `required_params`
   - `params_defaults`
   - `params_validator` (optional)
3. Use `main.py merge --method <name>`.

For parameter sweeps, add a config under `configs/merge/` and run:
```
python main.py merge-sweep --config <path>
```

## Notes

- `weighted_delta` is **not saveable** (in‑memory only).
- Params are stored under `metadata["params"]` and indexed in `eval/index.json`.
- Use `resolve_merge_eval_dir` and `update_results_index` for consistent outputs.
