# Speech Merging Research Codebase

This project provides a modular framework for fine-tuning, analysing, and merging task-specific adapters built on top of Qwen Omni models.

## Quick Start

Create and activate your environment, install dependencies (not included here), then run:

```bash
cd speech_merging
```

### Unified CLI

`main.py` wraps the experiment scripts behind a simple interface.

```bash
# Run LoRA fine-tuning for the ASR task (default config)
python3 main.py train

# Specify task and config explicitly
python3 main.py train --task asr --config asr.yaml
```

### Direct Experiment Scripts

You can also call the experiment modules directly.

```bash
# Train an ASR adapter using the config specified (defaults to configs/asr.yaml)
python3 experiments/train_task.py --task asr --config asr.yaml
```

The remaining scripts are placeholders for future stages and currently raise `NotImplementedError`:

- `python experiments/extract_vector.py`
- `python experiments/evaluate_task.py`
- `python experiments/analyze_overfitting.py`
- `python experiments/merge_vectors.py`

## Configuration

Task configs live in `configs/` and are merged with `base.yaml`. Override values either by editing the task config or passing a different file via `--config`.

## Artifacts & Logs

Artifacts are written under `artifacts/<task>/`, and experiment logs live under `runs/`. Adjust paths through the config files as needed.
