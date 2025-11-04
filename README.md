# Speech Merging Research Codebase

This project provides a modular framework for **fine-tuning, analyzing, and merging task-specific adapters** built on top of **Qwen Omni** models (e.g., Qwen2.5-Omni-3B). It is part of the broader research effort ***“Model Merging and Task Overfitting in Speech LLMs”***, which investigates whether model merging can mitigate task overfitting and improve cross-task generalization in multimodal speech-language models.

---

## 🧭 Project Description

Modern speech LLMs excel at single downstream tasks after fine-tuning (e.g., ASR, speech translation, emotion recognition), but often suffer from **task overfitting** and poor **cross-task transfer**.
This research explores whether **model merging**—combining task-specific LoRA adapters through parameter arithmetic or subspace merging—can:

* **Preserve** individual task performance
* **Enhance** generalization to unseen or composite tasks
* **Reveal** emergent multimodal or multilingual capabilities

Using **parameter-efficient fine-tuning (PEFT)** and **task vectors**, we fine-tune separate adapters for each speech-related task, extract their parameter deltas, and experiment with **weighted merging**, **low-rank subspace merging**, and **calibrated post-merging**.
The codebase is designed to make these experiments **reproducible**, **modular**, and **extendable** across tasks and datasets.

---

## 🚀 Quick Start

Create and activate your environment, install dependencies (not included here), then run:

```bash
cd speech_merging
```

---

### Unified CLI

`main.py` wraps the experiment scripts behind a simple interface.

```bash
# Run LoRA fine-tuning for the ASR task (default config)
python3 main.py train

# Specify task and config explicitly
python3 main.py train --task asr --config asr.yaml
```

---

### Direct Experiment Scripts

You can also call the experiment modules directly.

```bash
# Train an ASR adapter using the config specified (defaults to configs/asr.yaml)
python3 experiments/train_task.py --task asr --config asr.yaml
```

The remaining scripts are placeholders for future stages and currently raise `NotImplementedError`:

* `python experiments/extract_vector.py`
* `python experiments/evaluate_task.py`
* `python experiments/analyze_overfitting.py`
* `python experiments/merge_vectors.py`

---

## ⚙️ Configuration

Task configs live in `configs/` and are merged with `base.yaml`.
Override values either by editing the task config or passing a different file via `--config`.

---

## 📂 Artifacts & Logs

* **Artifacts** are written under `artifacts/<task>/`
* **Experiment logs** live under `runs/`
  Adjust paths through the config files as needed.

---

## 🧩 Research Context

This repository supports experiments inspired by recent work on **model merging** and **task vector arithmetic**, including:

* **Yang et al. (2024)** — *Model Merging in LLMs, MLLMs, and Beyond*
* **Zhao et al. (2025)** — *Low-Rank and Sparse Model Merging for Multilingual Speech*
* **Ilharco et al. (2023)** — *Task Arithmetic in the Tangent Space*

These studies show that merging LoRA or adapter modules can recover multi-task behavior, reduce overfitting, and enable zero-shot transfer—motivating our exploration of **speech-specific LLM merging**.

---

## 🧠 Future Directions

Planned extensions include:

* Implementing task vector extraction and merging workflows
* Cross-task and cross-lingual merging for speech-to-text and speech translation
* Overfitting analysis across different LoRA ranks and merge weights
* Integration with **SALMONN** and **Qwen2-Audio** backbones

