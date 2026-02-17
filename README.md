# Speech Merging Research Codebase

This repository is a research framework for training task-specific LoRA adapters on speech tasks, evaluating cross-task behavior, and merging adapters with multiple strategies. The main goal is to study task overfitting and whether adapter merging can recover stronger multi-task generalization in speech-language models.

## Table of Contents

- [Research Context and Objectives](#research-context-and-objectives)
- [Repository Overview](#repository-overview)
- [Setup](#setup)
- [Model Prerequisite](#model-prerequisite)
- [Quickstart](#quickstart)
- [Core CLI Workflows](#core-cli-workflows)
- [Supported Tasks](#supported-tasks)
- [Merge Config Usage](#merge-config-usage)
- [Outputs and Artifact Locations](#outputs-and-artifact-locations)
- [Metrics Interpretation](#metrics-interpretation)
- [Reproducibility Checklist](#reproducibility-checklist)

## Research Context and Objectives

Modern speech-language models can be adapted effectively for individual tasks using LoRA, but those task-specific adapters often encode narrow task priors. In practice, this leads to strong in-domain gains with weak cross-task transfer, especially when tasks have different supervision structure (for example, sequence generation for ASR/ST vs. classification for intent/emotion/langid). This repository is built to study that gap directly.

The central research question is whether merging independently trained speech adapters can recover broader capability without full multitask retraining. Instead of jointly training one large model on all tasks, we train each task adapter independently and then combine adapters in parameter space. This setup makes it possible to test whether useful task signals are complementary, conflicting, or redundant.

### Why this is important

1. Full multitask training is expensive and sensitive to data mixing strategies.
2. Single-task adapters are efficient but can overfit to task-specific patterns.
3. If merge-time composition works reliably, we gain a practical path to modular capability building: train once per task, combine later based on deployment needs.

### Core objectives

1. Quantify task overfitting after single-task adapter training.
2. Measure cross-task transfer for both unmerged and merged adapters.
3. Compare merge strategies under consistent evaluation conditions.
4. Characterize interference vs. synergy across tasks using unified metrics.
5. Identify which merge methods and coefficient policies are robust across heterogeneous speech tasks.

### Experimental framing in this codebase

The repository supports a consistent experimental loop:

1. Fine-tune separate adapters per task.
2. Evaluate per-task and cross-task performance for each adapter.
3. Merge adapters with a selected method and configuration.
4. Evaluate merged adapters on one or more tasks.
5. Compare outcomes using primary task metrics and interference-oriented summaries.

This structure is intended to separate training effects from merge effects, so merge quality can be analyzed as a first-class experimental variable rather than as a side effect of multitask training.

### What this repository aims to contribute

1. A reproducible workflow for adapter-level speech merging experiments.
2. A unified CLI (`main.py`) so training, evaluation, merging, and sweep experiments share one interface.
3. A practical benchmark harness across multiple speech tasks with different output formats.
4. A modular merging layer where methods, optimizer-backed coefficient learning, and sweep-based search can be compared under matched conditions.

In short, the project is not only about obtaining one high-performing merged adapter, but about understanding when and why merging helps, when it hurts, and how that behavior changes across speech task families.

## Repository Overview

- `main.py`: canonical CLI surface for all supported workflows.
- `configs/`: task configs (`asr.yaml`, `emotion.yaml`, etc.) and merge configs.
- `core/`: shared training and evaluation infrastructure.
- `tasks/`: task-specific dataset loading, metrics, and config logic.
- `merging/`: merge methods, optimizer-backed merging, sweep/evaluation utilities.
- `artifacts/`: saved model/adapters and related experiment outputs.
- `runs/`: experiment logs and run bundles.

## Setup

Minimal baseline setup:

```bash
cd speech_merging
python3 -m pip install -r requirements.txt
```

## Model Prerequisite

Download **Qwen2.5-Omni-3B** and place it at:

```text
data/models/Qwen2.5-Omni-3B
```

Several task configs use this path by default (for example, `configs/asr.yaml`).

## Quickstart

Train one task adapter, evaluate it, then run one merge config:

```bash
# Train ASR adapter
python3 main.py train --task asr --config asr.yaml

# Evaluate on validation split
python3 main.py evaluate --task asr --config asr.yaml --split validation

# Run merge from config
python3 main.py merge --config configs/merge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr.yaml
```

## Core CLI Workflows

`main.py` supports these commands:

- `train`
- `evaluate`
- `merge`
- `evaluate-merged`
- `merge-sweep`

### 1) Train

```bash
# Default task/config behavior
python3 main.py train

# Explicit task + config
python3 main.py train --task emotion --config emotion.yaml
```

### 2) Evaluate

```bash
# Evaluate base model on a task split
python3 main.py evaluate --task intent --config intent.yaml --split validation

# Evaluate a specific adapter path
python3 main.py evaluate --task intent --config intent.yaml --adapter artifacts/intent/<adapter_dir> --split test
```

### 3) Merge

Config-driven merge:

```bash
python3 main.py merge --config configs/merge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr.yaml
```

Direct argument merge:

```bash
python3 main.py merge --adapters asr emotion --method weighted --lambda 0.5 --evaluate --eval-split test
```

### 4) Evaluate Merged Adapter

```bash
# Evaluate by explicit merged adapter path
python3 main.py evaluate-merged --adapter-path artifacts/merged/<run_dir> --split test

# Or resolve by method/tasks/run-id
python3 main.py evaluate-merged --method weighted --tasks asr emotion --run-id latest --split test
```

### 5) Merge Sweep

Config-driven sweep:

```bash
python3 main.py merge-sweep --config configs/merge/merge_uniform_scalar_delta_emotion_intent_kws_langid_speaker_ver_asr.yaml
```

Override-driven sweep:

```bash
python3 main.py merge-sweep --adapters asr emotion --method weighted --search-type grid --grid lambda=0.2,0.5,0.8 --split validation
```

## Supported Tasks

| Task key | Config file | Default dataset | Notes |
| --- | --- | --- | --- |
| `asr` | `configs/asr.yaml` | `librispeech_asr` (`clean`) | ASR with duration filtering and configurable train/val/test hours. |
| `emotion` | `configs/emotion.yaml` | `AbstractTTS/IEMOCAP` | Emotion classification with configurable split behavior. |
| `intent` | `configs/intent.yaml` | `slurp` | Intent classification with optional prompt metadata controls. |
| `kws` | `configs/kws.yaml` | `speech_commands` | Keyword spotting with sampling and filtering controls. |
| `langid` | `configs/langid.yaml` | `google/fleurs` | Language ID across configurable language sets. |
| `speaker_id` | `configs/speaker_id.yaml` | `speechcolab/voxceleb1` | Speaker identification with subset controls for scale. |
| `speaker_ver` | `configs/speaker_ver.yaml` | `speechcolab/voxceleb1` | Speaker verification with positive/negative pair controls. |
| `speech_qa` | `configs/speech_qa.yaml` | `local_spoken_squad` (`data/datasets/Spoken-SQuAD`) | Local Spoken-SQuAD loader (JSON + wav files) with optional noisy test variants. |
| `st` | `configs/st.yaml` | `fixie-ai/covost2` | Speech translation with configurable language pair/splits. |

### Local Spoken-SQuAD Setup

```bash
git clone https://github.com/Chia-Hsuan-Lee/Spoken-SQuAD.git data/datasets/Spoken-SQuAD
bash scripts/prepare_spoken_squad_local.sh data/datasets/Spoken-SQuAD
```

Expected extracted layout:

- `data/datasets/Spoken-SQuAD/wav/train`
- `data/datasets/Spoken-SQuAD/wav/test`

## Merge Config Usage

Current valid merge config files in this repo:

- `configs/merge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr.yaml`
- `configs/merge/merge_uniform_scalar_delta_emotion_intent_kws_langid_speaker_ver_asr.yaml`

Use them directly with `merge` and `merge-sweep`:

```bash
python3 main.py merge --config configs/merge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr.yaml
python3 main.py merge-sweep --config configs/merge/merge_uniform_scalar_delta_emotion_intent_kws_langid_speaker_ver_asr.yaml
```

For method internals and full merge framework notes, see `merging/merging.md`.

## Outputs and Artifact Locations

- General outputs are written under `artifacts/`.
- Run logs and run-level bundles are written under `runs/`.

`artifacts/` and `runs/` are not tracked in git in this repository. Treat them as local experiment outputs.

## Metrics Interpretation

- Each task has task-specific primary metrics (for example, WER for ASR, accuracy/F1-style metrics for classification tasks).
- For merged evaluations, `interference_delta` captures transfer quality relative to reference baselines; higher is generally better.
- Compare metrics at fixed split (`train`, `validation`, or `test`) to avoid invalid conclusions.

## Reproducibility Checklist

- Record the exact command you ran.
- Record the config file path used.
- Record the seed value(s) in config.
- Record the evaluation split.
- Record the output path(s) under `artifacts/` and/or `runs/`.
