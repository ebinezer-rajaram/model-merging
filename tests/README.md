# Test Layout

Tests are grouped first by level, then by the behavior they protect.

- `unit/` contains local, synthetic tests for one concern at a time. Large domains can use subfolders, such as `unit/core/{config,data,evaluation,tasks,training}`.
- `integration/` contains offline workflow tests that cross module boundaries, use temporary artifacts, or exercise registry/loader wiring.
- `helpers/` contains small reusable fakes and builders. Keep helpers subsystem-specific when they are not broadly useful.

Task tests keep unit files flat because the concerns are small and stable: collators, metrics, and dataset helpers. Integration task loaders are grouped by loader family, with classification loaders kept together so every classification-style task is visible in one place.

Coverage should omit only CLI entrypoints, full training/model orchestration, external dataset bundles, or optimizer workflows that cannot be exercised offline without real models or long runs.
