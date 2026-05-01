# Speech Adapter Merging

This repository contains the research code for a thesis project on LoRA adapter composition for speech-language models. It trains task-specific adapters for Qwen2.5-Omni-3B, evaluates their in-task and cross-task behavior, and studies whether independently trained adapters can be combined to recover useful multi-task performance without full retraining.

## Research Question

Single-task LoRA adapters are efficient to train, but they can overfit to narrow task formats. In speech-language systems, that problem is especially visible because tasks vary widely: ASR and speech translation are generation tasks, while intent, language ID, speaker, emotion, and vocal sound tasks are classification-style tasks.

The project asks:

> Can independently trained speech adapters be combined after training to improve multi-task generalization while avoiding the cost and sensitivity of full joint multi-task training?

The codebase supports experiments that compare:

- the base Qwen2.5-Omni-3B model,
- single-task LoRA adapters,
- joint multi-task adapters,
- merged adapter variants,
- continual updates that add new task capability to existing adapter sets.

## What The Code Does

The main workflow is:

1. Train one LoRA adapter per speech task.
2. Evaluate adapters on their own task and, where useful, across other tasks.
3. Run joint multi-task training as a comparison baseline.
4. Search merge coefficients and evaluate merged adapters.
5. Run continual merge/evaluation experiments for adding tasks after an initial adapter set.
6. Save metrics, summaries, and run artifacts for later analysis.

The canonical entry point is `main.py`.

## Setup

Install dependencies:

```bash
python3 -m pip install -r requirements.txt
```

The default configs expect the base model at:

```text
data/models/Qwen2.5-Omni-3B
```

Several datasets are loaded from Hugging Face. Some configs also expect local data under `data/datasets/`, including MELD, VocalSound, and optional Spoken-SQuAD compatibility data.

## Quickstart

Train and evaluate a single-task adapter:

```bash
python3 main.py train --task asr --config asr.yaml
python3 main.py evaluate --task asr --config asr.yaml --split validation
```

Evaluate a trained adapter:

```bash
python3 main.py evaluate \
  --task intent \
  --config intent.yaml \
  --adapter artifacts/intent/adapters/qwen2_5_omni_lora_intent/best \
  --split test
```

Run joint multi-task training:

```bash
python3 main.py mtl --config configs/mtl/joint/mtl_intent_kws_langid_asr_emotion_vocalsound.yaml
```

Run a merge sweep:

```bash
python3 main.py merge-sweep \
  --config configs/merge/supermerge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml
```

Evaluate a merged adapter:

```bash
python3 main.py evaluate-merged \
  --config configs/merge/supermerge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml \
  --eval-tasks emotion intent kws langid speaker_ver asr vocalsound \
  --split test
```

Run a continual merge experiment:

```bash
python3 main.py continual-merge \
  --x-source artifacts/continual/<existing_artifact> \
  --y-source asr \
  --alpha 1.0 \
  --lambda 0.75
```

Evaluate a continual artifact:

```bash
python3 main.py evaluate-continual \
  --artifact-path artifacts/continual/<artifact_dir> \
  --eval-tasks emotion intent kws langid speaker_ver asr \
  --split test
```

## Supported Tasks

| Task key | Config | Default data source | Primary purpose |
| --- | --- | --- | --- |
| `asr` | `configs/asr.yaml` | `librispeech_asr` | Automatic speech recognition |
| `emotion` | `configs/emotion.yaml` | local MELD data | Emotion classification |
| `intent` | `configs/intent.yaml` | `marcel-gohsen/slurp` | Spoken intent classification |
| `kws` | `configs/kws.yaml` | `google/speech_commands` | Keyword spotting |
| `langid` | `configs/langid.yaml` | `google/fleurs` | Spoken language identification |
| `speaker_id` | `configs/speaker_id.yaml` | `acul3/voxceleb2` | Speaker identification |
| `speaker_ver` | `configs/speaker_ver.yaml` | `acul3/voxceleb2` | Speaker verification |
| `speech_qa` | `configs/speech_qa.yaml` | `ddwang2000/MMSU` | Speech question answering |
| `st` | `configs/st.yaml` | `fixie-ai/covost2` | Speech translation |
| `vocalsound` | `configs/vocalsound.yaml` | local VocalSound data | Vocal sound classification |

## Repository Structure

```text
main.py                 Unified CLI entry point
configs/                Task, merge, joint MTL, and continual experiment configs
core/                   Shared config, data, training, evaluation, and output utilities
tasks/                  Task-specific dataset loaders, collators, configs, and metrics
merging/                Adapter sources, merge methods, optimizers, sweeps, and continual logic
scripts/                Grouped experiment, evaluation, analysis, data, and maintenance scripts
scripts/continual/      Continual-suite runners and summaries
scripts/eval/           Task/model evaluation launchers
scripts/merging/        Merge evaluation, comparison, and weighted-merge helpers
scripts/analysis/       Reusable analysis and report-generation entrypoints
scripts/plots/          Plot-generation scripts
scripts/data/           Dataset inspection, preparation, and stats scripts
scripts/training/       Training launchers and focused training utilities
scripts/maintenance/    Backfills and migrations
scripts/archive/        Historical or one-off experiment scripts
tests/                  Unit and integration tests for datasets, metrics, merging, and continual flows
data/                   Local models and datasets (generated/local, not tracked)
artifacts/              Adapter checkpoints, metrics, summaries, and merge outputs
runs/                   Run bundles and experiment outputs
logs/                   Runtime, TensorBoard, and wandb logs
```

## Outputs And Results

Experiment outputs are generated locally and are generally ignored by git. Common locations are:

- `artifacts/<task>/adapters/` for trained task adapters,
- `artifacts/<task>/metrics/` for task-level evaluations,
- `artifacts/merged/` for merged-adapter evaluations,
- `artifacts/mtl/` for joint multi-task runs,
- `artifacts/continual/` and `artifacts/continual_suite/` for continual experiments,
- `runs/` and `logs/` for run bundles and logging output.

Some local summary CSVs may be present under `artifacts/`, such as comparisons between base-model, single-task, and merged-adapter metrics. Treat these as generated experiment outputs tied to the commands, configs, and checkpoints used to create them.

## Reproducibility Notes

For each experiment, record:

- the exact command,
- the config file path,
- the model checkpoint path,
- the adapter or artifact path,
- the evaluation split,
- the random seed in the config,
- the output directory under `artifacts/`, `runs/`, or `logs/`.

For final comparisons, use matched tasks, splits, checkpoints, and metric definitions. ASR uses WER-oriented metrics, classification tasks report accuracy/F1-style metrics, and Speech-QA uses option-letter accuracy for MMSU-style evaluation.
