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
- `method`: `uniform | uniform_delta | uniform_scalar_delta | weighted | task_vector | weighted_delta | weighted_delta_n | dare | ties`
- `merge_mode`: `common | strict`
- `method_params`: method-specific params (`lambda`, etc.)
- `transforms`: pre-merge transform pipeline (`identity` available by default)
- `lambda_policy`: `scalar` or `per_layer`
- `optimizer`: `none | bayes | adamerging | gradient | supermerge | regret_smoothmax`

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
- Built-in: `layer_l2_normalize`
  - Per-layer task-vector normalization to target L2 (Frobenius) norm.
  - Params:
    - `target_norm` (float, default `1.0`)
    - `eps` (float, default `1e-12`)
    - `include_non_layer_keys` (bool, default `true`)
  - Example:
    - `transforms: [{name: layer_l2_normalize, params: {target_norm: 1.0}}]`
- Scaffold hook: `ties_scaffold` (no-op placeholder for future transform experiments)

## Optimizers

Optimizers are registry plugins that can set/adjust lambda policy.

- `none`: no-op
- `bayes`: adapter scaffold for direct merge; active Bayesian optimization remains in `merge-sweep`
- `adamerging`: entropy-minimization optimizer for classification tasks with low-VRAM defaults
  - `optimizer.params.merge_impl`: `streaming_parametrize` (default), `functional_clone_legacy`, `fused_linear`, or `fused_lora_linear`
    - `fused_lora_linear` applies LoRA factors directly at runtime (`xA^T` then `·B^T`) for `nn.Linear` targets and enforces strict LoRA coverage (no dense fallback)
  - `optimizer.params.delta_residency`: `cpu_stream` (default) or `gpu_cache`
  - `optimizer.params.model_dtype`: model load dtype for optimization (`auto` default; use `bf16`/`fp16` to reduce VRAM)
  - `optimizer.params.dtype_compute`: `auto` (default), `bf16`, `fp16`, `fp32`
  - `optimizer.params.progress_bar`: show tqdm progress bar during optimization (`true` by default when tqdm is installed)
  - `optimizer.params.logging_steps`: print optimization metrics every N optimizer update steps (`0` disables periodic logs)
  - `optimizer.params.dataloader_num_workers`: DataLoader worker count (`0` default; increase to improve throughput)
  - `optimizer.params.dataloader_pin_memory`: pin host memory for faster H2D transfers (`true` default)
  - `optimizer.params.non_blocking_transfer`: use non-blocking tensor transfer to GPU (`true` default)
  - `optimizer.params.gradient_accumulation_steps`: gradient accumulation factor for larger effective batch (`1` default)
  - `optimizer.params.early_stopping_patience`: stop after N non-improving steps (`0` disables)
  - `optimizer.params.early_stopping_threshold`: minimum entropy improvement to reset patience (`0.0` default)
  - Backward-compatible aliases: `log_every`, `grad_accum_steps`, `early_stopping_min_delta`
  - `optimizer.params.force_cpu=true` now loads directly on CPU (`device_map={"": "cpu"}`, `torch_dtype=float32`)
- `gradient`: supervised CE optimizer for `weighted_delta_n` (classification scope)
  - `optimizer.params.ce_task_weighting`: `baseline_normalized` (default), `equal`, or `manual`
  - `optimizer.params.ce_baseline_source`: `single_task_eval` (default)
  - `optimizer.params.ce_baseline_floor`: denominator floor for baseline-normalized CE (`1e-6` default)
  - `optimizer.params.ce_baseline_batches`: batches per task for baseline CE estimation (`32` default)
  - `optimizer.params.ce_manual_task_weights`: mapping used when `ce_task_weighting=manual`
  - `optimizer.params.min_optimizer_steps_before_early_stop`: minimum optimizer updates before early stop can trigger (`20` default)
  - `optimizer.params.early_stopping_warmup_steps`: warmup optimizer updates before patience tracking (`20` default)
  - `optimizer.params.restore_best_checkpoint`: restore minibatch-selected best coefficients at the end (`false` default)
  - `optimizer.params.enforce_validation_only_selection`: require `optimizer.params.split=validation` (`true` default)
- `supermerge`: exact SuperMerge-style supervised optimizer for `weighted_delta_n`
  - Defaults: `optimizer.params.coefficient_parameterization=tanh_alpha`, `optimizer.params.split=validation`, `optimizer.params.normalize_coefficients=false`
  - Emits signed coefficients in `(-1, 1)` and injects `method_params.allow_negative_coefficients=true`
  - Uses model-provided supervised `loss` when available, with CE fallback from logits/labels
  - Supports `variant=task_wise|layer_wise` (layer-wise mirrors the paper's per-layer formulation)
  - `optimizer.params.restore_best_checkpoint`: restore minibatch-selected best coefficients at the end (`false` default)
  - Hierarchical keys under `optimizer.params.hierarchical.*` are accepted as config stubs, but `hierarchical.enabled=true` is not implemented yet
- `regret_smoothmax`: supervised label-token CE optimizer for `weighted_delta_n` that learns global static coefficients using softmax logits and smooth-max regret objective
  - Key params: `optimizer.params.tau`, `optimizer.params.regret_eps`, `optimizer.params.baseline_batches`
  - Uses token-normalized CE on `labels != -100`
  - Default/global sampling can be set for all tasks with:
    - `optimizer.params.sampling.default.enabled`
    - `optimizer.params.sampling.default.strategy` (`uniform|inverse|sqrt_inverse|balanced|none`)
    - optional overrides under `optimizer.params.sampling.per_task.<task>`

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

### Single-scalar uniform-delta search

`uniform_scalar_delta` applies one global scalar on top of the summed task vectors:

`delta_merged = scale * (delta_1 + ... + delta_n)`

Example (grid):
```yaml
adapters: [emotion, intent, kws, langid]
method: uniform_scalar_delta
merge_mode: common
eval_tasks: [emotion, intent, kws, langid]
split: test
search:
  type: grid
  grid:
    scale: [0.0, 0.05, 0.1, 0.2, 0.3, 0.4, 0.5]
```

Example (bayes):
```yaml
search:
  type: bayes
  budget: 20
  init_points: 6
  space:
    scale:
      type: float
      min: 0.0
      max: 2.0
```

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
- `configs/merge/merge_dare.yaml`
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

- `uniform_delta` is **not saveable** (in‑memory only).
- `weighted_delta` is **not saveable** (in‑memory only).
- `weighted_delta_n` is **not saveable** (in‑memory only).
- `dare` is runnable in task-vector space with deterministic random masking:
  - `method_params.drop_rate` in `[0,1)` (default `0.9`)
  - `method_params.seed` integer seed (default `42`)
  - operation: random drop + rescale by `1/(1-drop_rate)`, then uniform fusion
- `dare` is **not saveable** (in‑memory only), but works with `evaluate-merged` and `merge-sweep`.
- `ties` is runnable and uses paper-core TIES in task-vector space:
  - `method_params.k` (percent in `[0,100]`, default `20`)
  - `method_params.lambda` (final scaling factor, default `1.0`)
- `ties` is **not saveable** (in‑memory only), but works with `evaluate-merged` and `merge-sweep`.
- Params are stored under `metadata["params"]` and indexed in `eval/index.json`.
- Extended metadata also includes `method_params`, `lambda_policy`, `transforms`, and `optimizer`.
- Use `resolve_merge_eval_dir` and `update_results_index` for consistent outputs.
