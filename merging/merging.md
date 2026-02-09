# Merging Framework Overview

This package implements a modular framework for merging LoRA adapters, evaluating merged models, and running parameter sweeps. It is designed to be extensible (new methods, lambda policies, transforms, optimizers) and consistent in output artifacts.

## High‑Level Flow

1. **Resolve adapters** (task names or explicit paths)
2. **Run merge method** (registry dispatch)
3. **Optionally save merged adapter** (LoRA weights + metadata)
4. **Evaluate merged adapter** on one or more tasks
5. **Record outputs** into unified directories (run bundle + eval index)

The canonical CLI entrypoint is `main.py`. Legacy CLI is `experiments/merge_vectors.py` (shim).

## Package Structure

- `merging/methods/`  
  Method implementations (uniform, weighted, task_vector, etc.)

- `merging/engine/`  
  Registry, runner, built-in method registration

- `merging/runtime/`  
  Metadata helpers, output paths, logging

- `merging/config/`, `merging/policies/`, `merging/plugins/`  
  Merge spec model, lambda policies, transforms/optimizers

- `merging/evaluation/`  
  Evaluation + sweeps (grid now, bayes scaffold)

- `merging/cli.py`  
  Merge CLI wrapper used by `main.py merge`

## Registry + Method Interface

Merge methods are registered in `merging/engine/builtin_methods.py` using `MergeMethod`:

- `name`
- `required_params`
- `params_defaults`
- `params_validator` (optional)
- `min_adapters` / `max_adapters`
- `saveable`
- `merge_in_memory(...) -> MergeOutput`
- `save_fn(...)` (optional)

This makes adding a new technique a single-file change:

1. Implement in `merging/methods/<new_method>.py`
2. Register in `merging/engine/builtin_methods.py`

## Runner (Single Source of Truth)

`merging/engine/runner.run_merge(...)` handles:

- adapter resolution (`resolve_adapter_specs`)
- spec/parameter normalization (`MergeSpec` + `normalize_params`)
- method dispatch
- lambda optimizer hook (scaffold-friendly)
- pre-merge transforms
- optional saving (LoRA adapter + metadata)
- returns a `MergeResult` (method, params, tag, output path, metadata)

This is used by the CLI and is the preferred entrypoint for new integrations.

## Unified Merge Config

Merge and sweep now share one schema and one loader:

- loader: `merging.config.load_merge_config(path)`
- model: `MergeConfig` (convertible to `MergeSpec` for merge execution)
- commands: `main.py merge` and `main.py merge-sweep` both consume this schema

Direct merge fields:

- `adapters`: list of task names or adapter paths
- `method`: `uniform | weighted | task_vector | weighted_delta | ties`
- `merge_mode`: `common | strict`
- `method_params`: method-specific params (`lambda`, etc.)
- `transforms`: pre-merge transform pipeline (`identity` available by default)
- `lambda_policy`: `scalar` or `per_layer`
- `optimizer`: `none | bayes | adamerging`

Optional sweep fields:

- `search`
- `constraint_nonnegative`
- `eval_tasks`
- `split`
- `save_merged`
- `eval_subset`
- `output_dir`

Use either legacy CLI args (`--method --lambda`) or `--config` YAML.
Legacy merge/sweep schemas are still accepted and emit `DeprecationWarning`.

## Lambda Policies

Supported policies:

- `scalar`: one lambda for every parameter
- `per_layer`: default lambda + layer overrides (layer index parsed from LoRA key paths)

Weighted merge accepts per-key lambda resolution while preserving scalar behavior.

## Transforms

Transforms are pre-merge plugins run on each source adapter weight dict before method composition.

- Built-in: `identity`
- Scaffold hook: `ties_scaffold` (no-op placeholder for future transform experiments)

## Optimizers

Optimizers are registry plugins that can set/adjust lambda policy.

- `none`: no-op
- `bayes`: adapter scaffold for direct merge; active Bayesian optimization remains in `merge-sweep`
- `adamerging`: entropy-minimization optimizer for classification tasks with low-VRAM defaults
  - `optimizer.params.merge_impl`: `streaming_parametrize` (default) or `functional_clone_legacy`
  - `optimizer.params.delta_residency`: `cpu_stream` (default) or `gpu_cache`
  - `optimizer.params.dtype_compute`: `auto` (default), `bf16`, `fp16`, `fp32`
  - `optimizer.params.force_cpu=true` now loads directly on CPU (`device_map={"": "cpu"}`, `torch_dtype=float32`)

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

Sweep configs can now also include:

- `lambda_policy`
- `optimizer`

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

1. Create `merging/methods/<name>.py` with a `merge_in_memory` function.
2. Register it in `merging/engine/builtin_methods.py` with:
   - `required_params`
   - `params_defaults`
   - `params_validator` (optional)
3. Use `main.py merge --method <name>` or a merge YAML config.

For parameter sweeps, add a config under `configs/merge/` and run:
```
python main.py merge-sweep --config <path>
```

### Canonical Config Set

- `configs/merge/merge_weighted_scalar.yaml`
- `configs/merge/merge_weighted_per_layer.yaml`
- `configs/merge/merge_ties.yaml`
- `configs/merge/merge_weighted_delta_sweep_bayes.yaml`

### Migration Table

| Old key/file pattern | New key/file pattern |
| --- | --- |
| top-level `grid` | `search.grid` with `search.type: grid` |
| top-level `lambda` | `method_params.lambda` |
| top-level `params` | `method_params` |
| `configs/merge/sweep_weighted_delta_bayes.yaml` | `configs/merge/merge_weighted_delta_sweep_bayes.yaml` |
| `configs/merge/sweep_weighted_delta_example.yaml` | use `configs/merge/merge_weighted_delta_sweep_bayes.yaml` + CLI `--search-type grid --grid ...` overrides |
| `configs/merge/merge_ties_scaffold.yaml` | `configs/merge/merge_ties.yaml` |
| `configs/merge/merge_weighted_bayes_optimizer.yaml` | `configs/merge/merge_weighted_scalar.yaml` for direct merge, or `merge_weighted_delta_sweep_bayes.yaml` for active bayes sweep |
| `merge_weighted_delta_n_adamerging_*` variants | one canonical config + explicit CLI/YAML overrides for `optimizer.params` |

## Notes

- `weighted_delta` is **not saveable** (in‑memory only).
- `ties` is runnable and uses paper-core TIES in task-vector space:
  - `method_params.k` (percent in `[0,100]`, default `20`)
  - `method_params.lambda` (final scaling factor, default `1.0`)
- `ties` is **not saveable** (in‑memory only), but works with `evaluate-merged` and `merge-sweep`.
- Params are stored under `metadata["params"]` and indexed in `eval/index.json`.
- Extended metadata also includes `method_params`, `lambda_policy`, `transforms`, and `optimizer`.
- Use `resolve_merge_eval_dir` and `update_results_index` for consistent outputs.
