#!/usr/bin/env python3
"""Analyze adjacent-layer representation similarity with branch-level tracking.

This script evaluates three model arms per task:
1) base model (no adapter)
2) full LoRA adapter
3) aligner-ablated adapter (audio_tower.proj LoRA tensors zeroed)

It computes adjacent-layer cosine similarity for:
- block residual stream outputs
- attention branch outputs
- MLP branch outputs

and saves CSV summaries + plots.
"""

from __future__ import annotations

import argparse
import json
import os
import random
import re
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from safetensors import safe_open
from torch.utils.data import DataLoader

PROJECT_ROOT = find_repo_root(__file__)
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from core import load_config, load_model_and_processor, prepare_task_for_evaluation
from tasks.asr import get_artifact_directories as get_asr_artifact_directories
from tasks.asr import get_config_path as get_asr_config_path
from tasks.emotion import get_artifact_directories as get_emotion_artifact_directories
from tasks.emotion import get_config_path as get_emotion_config_path
from tasks.intent import get_artifact_directories as get_intent_artifact_directories
from tasks.intent import get_config_path as get_intent_config_path
from tasks.kws import get_artifact_directories as get_kws_artifact_directories
from tasks.kws import get_config_path as get_kws_config_path
from tasks.langid import get_artifact_directories as get_langid_artifact_directories
from tasks.langid import get_config_path as get_langid_config_path
from tasks.speaker_ver import get_artifact_directories as get_speaker_ver_artifact_directories
from tasks.speaker_ver import get_config_path as get_speaker_ver_config_path
from tasks.vocalsound import get_artifact_directories as get_vocalsound_artifact_directories
from tasks.vocalsound import get_config_path as get_vocalsound_config_path


DEFAULT_TASKS = ["asr", "emotion", "intent", "kws", "langid", "speaker_ver", "vocalsound"]
SIGNAL_TYPES = ("block", "attention", "mlp")
ARM_BASE = "base"
ARM_FULL = "full"
ARM_ABLATED = "ablated_aligner"

ATTENTION_TOKENS = ("q_proj", "k_proj", "v_proj", "o_proj")
MLP_TOKENS = ("gate_proj", "up_proj", "down_proj")
ALIGNER_TOKEN = "audio_tower.proj"

TASK_CONFIG_HELPERS = {
    "asr": (get_asr_config_path, get_asr_artifact_directories),
    "emotion": (get_emotion_config_path, get_emotion_artifact_directories),
    "intent": (get_intent_config_path, get_intent_artifact_directories),
    "kws": (get_kws_config_path, get_kws_artifact_directories),
    "langid": (get_langid_config_path, get_langid_artifact_directories),
    "speaker_ver": (get_speaker_ver_config_path, get_speaker_ver_artifact_directories),
    "vocalsound": (get_vocalsound_config_path, get_vocalsound_artifact_directories),
}


@dataclass
class TaskContext:
    task: str
    config: Dict[str, Any]
    model_path: Path
    adapter_path: Path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze adjacent-layer output similarity with mandatory attention/MLP branch tracking."
    )
    parser.add_argument("--tasks", nargs="+", default=DEFAULT_TASKS, help="Task keys to analyze.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--max-samples", type=int, default=128, help="Max examples per task.")
    parser.add_argument("--batch-size", type=int, default=2, help="Batch size for forward passes.")
    parser.add_argument("--seed", type=int, default=0, help="Seed for deterministic sampling/bootstrap.")
    parser.add_argument(
        "--output-dir",
        default="analysis/results/layer_output_similarity",
        help="Directory for CSV outputs.",
    )
    parser.add_argument(
        "--plot-dir",
        default="analysis/plots/layer_output_similarity",
        help="Directory for plots.",
    )
    parser.add_argument(
        "--skip-plots",
        action="store_true",
        help="Skip plot generation (useful if matplotlib is unavailable).",
    )
    parser.add_argument("--bootstrap-samples", type=int, default=1000, help="Bootstrap draws for CI.")
    parser.add_argument(
        "--hf-cache-dir",
        default=None,
        help="Optional HuggingFace cache dir override (sets HF_HOME/HF_DATASETS_CACHE/TRANSFORMERS_CACHE).",
    )
    return parser.parse_args()


def _resolve_device(model: Any) -> torch.device:
    # For HF models with device_map, input ids must match embedding weight device.
    if hasattr(model, "get_input_embeddings"):
        try:
            emb = model.get_input_embeddings()
            if emb is not None and hasattr(emb, "weight"):
                return emb.weight.device
        except Exception:
            pass

    model_device = getattr(model, "device", None)
    if isinstance(model_device, torch.device):
        return model_device
    for param in model.parameters():
        return param.device
    return torch.device("cpu")


def _move_to_device(value: Any, device: torch.device) -> Any:
    if torch.is_tensor(value):
        return value.to(device)
    if hasattr(value, "to") and callable(getattr(value, "to")) and not isinstance(value, (dict, list, tuple)):
        try:
            return value.to(device)
        except Exception:
            pass
    if isinstance(value, dict):
        return {k: _move_to_device(v, device) for k, v in value.items()}
    if isinstance(value, list):
        return [_move_to_device(v, device) for v in value]
    if isinstance(value, tuple):
        return tuple(_move_to_device(v, device) for v in value)
    return value


def _force_input_tensors_to_embedding_device(batch: Any, model: Any) -> Any:
    """Force all input-id style tensors to embedding device.

    Some collators return containers that don't fully follow our recursive move logic.
    This guard ensures tensors used by token embedding are on the correct device.
    """
    try:
        emb = model.get_input_embeddings()
        if emb is None or not hasattr(emb, "weight"):
            return batch
        emb_device = emb.weight.device
    except Exception:
        return batch

    if isinstance(batch, dict):
        fixed = {}
        for key, value in batch.items():
            if torch.is_tensor(value) and ("input_ids" in key or key in {"attention_mask", "position_ids"}):
                fixed[key] = value.to(emb_device)
            elif isinstance(value, dict):
                fixed[key] = _force_input_tensors_to_embedding_device(value, model)
            else:
                fixed[key] = value
        return fixed

    return batch


def _extract_tensor_from_output(output: Any) -> Optional[torch.Tensor]:
    if torch.is_tensor(output):
        return output
    if isinstance(output, (tuple, list)):
        for item in output:
            if torch.is_tensor(item) and item.ndim >= 3:
                return item
    if isinstance(output, dict):
        for item in output.values():
            if torch.is_tensor(item) and item.ndim >= 3:
                return item
    return None


def _layer_index_from_name(module_name: str) -> Optional[int]:
    m = re.search(r"(?:^|\.|_)layers\.(\d+)(?:$|\.)", module_name)
    if not m:
        return None
    return int(m.group(1))


def _discover_decoder_layers(model: Any) -> Dict[int, Any]:
    layers: Dict[int, Any] = {}
    for name, module in model.named_modules():
        if hasattr(module, "self_attn") and hasattr(module, "mlp"):
            layer_idx = _layer_index_from_name(name)
            if layer_idx is not None and layer_idx not in layers:
                layers[layer_idx] = module
    if not layers:
        raise RuntimeError("Could not discover decoder layers with self_attn and mlp modules.")
    return dict(sorted(layers.items(), key=lambda item: item[0]))


class BranchCollector:
    def __init__(self) -> None:
        self.cache: Dict[str, Dict[int, torch.Tensor]] = {
            "attention": {},
            "mlp": {},
        }

    def make_hook(self, signal: str, layer_idx: int):
        def _hook(_module, _inp, out):
            tensor = _extract_tensor_from_output(out)
            if tensor is not None:
                self.cache[signal][layer_idx] = tensor

        return _hook

    def clear(self) -> None:
        self.cache["attention"].clear()
        self.cache["mlp"].clear()


def _dataset_subset(dataset: Any, max_samples: int, seed: int) -> Any:
    if max_samples <= 0 or len(dataset) <= max_samples:
        return dataset
    rng = random.Random(seed)
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    picked = sorted(indices[:max_samples])
    return dataset.select(picked)


def _prepare_dataset_cache(config: Dict[str, Any], task: str) -> Dict[str, Any]:
    dataset_cfg = config.setdefault("dataset", {})
    _, artifact_fn = TASK_CONFIG_HELPERS[task]
    artifact_dirs = artifact_fn(PROJECT_ROOT)
    cache_dir = dataset_cfg.get("cache_dir")
    if cache_dir is None:
        dataset_cfg["cache_dir"] = artifact_dirs["datasets"]
        return config
    cache_path = Path(cache_dir)
    if not cache_path.is_absolute():
        cache_path = artifact_dirs["base"] / cache_path
    dataset_cfg["cache_dir"] = cache_path
    return config


def _resolve_task_context(task: str) -> TaskContext:
    if task not in TASK_CONFIG_HELPERS:
        supported = ", ".join(sorted(TASK_CONFIG_HELPERS.keys()))
        raise ValueError(f"Unsupported task '{task}'. Supported: {supported}")

    cfg_fn, artifact_fn = TASK_CONFIG_HELPERS[task]
    config_path = cfg_fn(PROJECT_ROOT, None)
    config = load_config(config_path)
    config = _prepare_dataset_cache(config, task)

    model_rel = config.get("model", {}).get("path", "data/models/Qwen2.5-Omni-3B")
    model_path = (PROJECT_ROOT / model_rel).resolve()

    artifacts_cfg = config.get("artifacts", {})
    adapter_subdir = artifacts_cfg.get("adapter_subdir")
    if not adapter_subdir:
        raise ValueError(f"Task '{task}' config is missing artifacts.adapter_subdir.")

    adapter_base = artifact_fn(PROJECT_ROOT)["adapters"] / adapter_subdir
    adapter_path = adapter_base / "best"
    if not adapter_path.exists():
        raise FileNotFoundError(f"Missing adapter path for {task}: {adapter_path}")

    return TaskContext(task=task, config=config, model_path=model_path, adapter_path=adapter_path)


def _bootstrap_ci(values: Sequence[float], n_boot: int, seed: int) -> Tuple[float, float]:
    if not values:
        return (float("nan"), float("nan"))
    if len(values) == 1:
        v = float(values[0])
        return (v, v)
    arr = np.asarray(values, dtype=np.float64)
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot, dtype=np.float64)
    n = len(arr)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        means[i] = float(arr[idx].mean())
    return (float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5)))


def _adjacent_example_means(
    states: Sequence[torch.Tensor],
    mask: torch.Tensor,
) -> Dict[int, List[Tuple[float, int]]]:
    out: Dict[int, List[Tuple[float, int]]] = {}
    if len(states) < 2:
        return out
    valid_mask = mask.bool()
    for layer_idx in range(len(states) - 1):
        h1 = states[layer_idx]
        h2 = states[layer_idx + 1]
        cos = F.cosine_similarity(h1.float(), h2.float(), dim=-1)
        layer_values: List[Tuple[float, int]] = []
        for b in range(cos.shape[0]):
            token_mask = valid_mask[b]
            selected = cos[b][token_mask]
            if selected.numel() == 0:
                continue
            layer_values.append((float(selected.mean().item()), int(selected.numel())))
        out[layer_idx] = layer_values
    return out


def _summarize_signal_records(
    rows: List[Dict[str, Any]],
    *,
    bootstrap_samples: int,
    seed: int,
) -> pd.DataFrame:
    grouped: Dict[Tuple[str, str, str, int], List[float]] = defaultdict(list)
    token_counts: Dict[Tuple[str, str, str, int], int] = defaultdict(int)
    for row in rows:
        key = (row["task"], row["arm"], row["signal_type"], int(row["layer_pair"]))
        grouped[key].append(float(row["example_mean_cos"]))
        token_counts[key] += int(row["n_tokens"])

    summary_rows: List[Dict[str, Any]] = []
    for key, values in sorted(grouped.items()):
        task, arm, signal, layer_pair = key
        arr = np.asarray(values, dtype=np.float64)
        mean = float(arr.mean()) if arr.size else float("nan")
        std = float(arr.std(ddof=1)) if arr.size > 1 else 0.0
        ci_low, ci_high = _bootstrap_ci(values, bootstrap_samples, seed + (layer_pair * 997))
        summary_rows.append(
            {
                "task": task,
                "arm": arm,
                "signal_type": signal,
                "layer_pair": layer_pair,
                "mean_cos": mean,
                "std_cos": std,
                "ci_low": ci_low,
                "ci_high": ci_high,
                "n_examples": int(arr.size),
                "n_tokens": int(token_counts[key]),
            }
        )
    return pd.DataFrame(summary_rows)


def _split_signal_frames(summary_df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
    frames: Dict[str, pd.DataFrame] = {}
    for signal in SIGNAL_TYPES:
        part = summary_df[summary_df["signal_type"] == signal].copy()
        part = part.sort_values(["task", "arm", "layer_pair"]).reset_index(drop=True)
        frames[signal] = part
    return frames


def _extract_task_vector(adapter_path: Path) -> Dict[str, torch.Tensor]:
    config_path = adapter_path / "adapter_config.json"
    with config_path.open("r") as handle:
        config = json.load(handle)
    scaling = float(config["lora_alpha"]) / float(config["r"])

    weights: Dict[str, torch.Tensor] = {}
    with safe_open(adapter_path / "adapter_model.safetensors", framework="pt", device="cpu") as f:
        for key in f.keys():
            weights[key] = f.get_tensor(key).clone()

    grouped: Dict[str, Dict[str, torch.Tensor]] = {}
    for key, tensor in weights.items():
        if ".lora_A." in key:
            grouped.setdefault(key.replace(".lora_A.weight", ""), {})["A"] = tensor
        elif ".lora_B." in key:
            grouped.setdefault(key.replace(".lora_B.weight", ""), {})["B"] = tensor

    task_vector: Dict[str, torch.Tensor] = {}
    for base_key, pair in grouped.items():
        if "A" in pair and "B" in pair:
            task_vector[base_key] = (pair["B"] @ pair["A"]) * scaling
    return task_vector


def _module_group(param_name: str) -> str:
    if ALIGNER_TOKEN in param_name:
        return "audio_tower"
    if any(token in param_name for token in ATTENTION_TOKENS):
        return "attention"
    if any(token in param_name for token in MLP_TOKENS):
        return "mlp"
    return "other"


def _layer_of_param(param_name: str) -> int:
    if ALIGNER_TOKEN in param_name:
        return -1
    m = re.search(r"\.layers\.(\d+)\.", param_name)
    if m:
        return int(m.group(1))
    return -2


def _compute_weight_stats(task: str, adapter_path: Path) -> pd.DataFrame:
    task_vector = _extract_task_vector(adapter_path)
    rows: List[Dict[str, Any]] = []
    for key, delta in task_vector.items():
        group = _module_group(key)
        layer = _layer_of_param(key)
        l2 = float(torch.norm(delta.float()).item())
        numel = int(delta.numel())
        rows.append(
            {
                "task": task,
                "arm": ARM_FULL,
                "module_group": group,
                "layer": layer,
                "delta_l2_norm": l2,
                "num_params": numel,
            }
        )
        rows.append(
            {
                "task": task,
                "arm": ARM_ABLATED,
                "module_group": group,
                "layer": layer,
                "delta_l2_norm": 0.0 if group == "audio_tower" else l2,
                "num_params": numel,
            }
        )
    return pd.DataFrame(rows)


def _ablate_aligner_inplace(model: Any) -> Dict[str, Any]:
    changed_names: List[str] = []
    unchanged_non_aligner = 0
    changed_non_aligner = 0
    aligner_before = 0.0
    aligner_after = 0.0
    for name, param in model.named_parameters():
        if "lora_" not in name:
            continue
        before = float(torch.norm(param.detach().float()).item())
        if ALIGNER_TOKEN in name:
            with torch.no_grad():
                param.zero_()
            after = float(torch.norm(param.detach().float()).item())
            changed_names.append(name)
            aligner_before += before
            aligner_after += after
        else:
            after = float(torch.norm(param.detach().float()).item())
            if abs(after - before) > 1e-12:
                changed_non_aligner += 1
            else:
                unchanged_non_aligner += 1
    return {
        "num_aligner_lora_params_zeroed": len(changed_names),
        "aligner_norm_before": aligner_before,
        "aligner_norm_after": aligner_after,
        "non_aligner_unchanged_count": unchanged_non_aligner,
        "non_aligner_changed_count": changed_non_aligner,
    }


def _build_collapse_summary(
    summary_df: pd.DataFrame,
    weight_df: pd.DataFrame,
) -> pd.DataFrame:
    rows: List[Dict[str, Any]] = []
    tasks = sorted(summary_df["task"].unique())

    for task in tasks:
        task_df = summary_df[summary_df["task"] == task]
        for signal in SIGNAL_TYPES:
            signal_df = task_df[task_df["signal_type"] == signal]
            if signal_df.empty:
                continue
            per_arm: Dict[str, Dict[str, Any]] = {}
            for arm in (ARM_BASE, ARM_FULL, ARM_ABLATED):
                arm_df = signal_df[signal_df["arm"] == arm]
                if arm_df.empty:
                    continue
                idxmax = arm_df["mean_cos"].idxmax()
                peak_row = arm_df.loc[idxmax]
                per_arm[arm] = {
                    "peak": float(peak_row["mean_cos"]),
                    "peak_pair": int(peak_row["layer_pair"]),
                    "count_gt_095": int((arm_df["mean_cos"] > 0.95).sum()),
                    "count_gt_098": int((arm_df["mean_cos"] > 0.98).sum()),
                }

            full_vs_base = np.nan
            full_vs_ablated = np.nan
            if ARM_FULL in per_arm and ARM_BASE in per_arm:
                full_vs_base = per_arm[ARM_FULL]["peak"] - per_arm[ARM_BASE]["peak"]
            if ARM_FULL in per_arm and ARM_ABLATED in per_arm:
                full_vs_ablated = per_arm[ARM_FULL]["peak"] - per_arm[ARM_ABLATED]["peak"]

            corr = np.nan
            if signal in ("attention", "mlp"):
                full = signal_df[signal_df["arm"] == ARM_FULL][["layer_pair", "mean_cos"]]
                base = signal_df[signal_df["arm"] == ARM_BASE][["layer_pair", "mean_cos"]]
                merged = full.merge(base, on="layer_pair", suffixes=("_full", "_base"))
                merged["lift"] = merged["mean_cos_full"] - merged["mean_cos_base"]

                weights = weight_df[
                    (weight_df["task"] == task)
                    & (weight_df["arm"] == ARM_FULL)
                    & (weight_df["module_group"] == signal)
                    & (weight_df["layer"] >= 0)
                ][["layer", "delta_l2_norm"]]
                merged = merged.merge(weights, left_on="layer_pair", right_on="layer", how="inner")
                if len(merged) >= 2:
                    x = merged["delta_l2_norm"].to_numpy()
                    y = merged["lift"].to_numpy()
                    if np.std(x) > 0 and np.std(y) > 0:
                        corr = float(np.corrcoef(x, y)[0, 1])

            rows.append(
                {
                    "task": task,
                    "signal_type": signal,
                    "base_peak_mean_cos": per_arm.get(ARM_BASE, {}).get("peak", np.nan),
                    "full_peak_mean_cos": per_arm.get(ARM_FULL, {}).get("peak", np.nan),
                    "ablated_peak_mean_cos": per_arm.get(ARM_ABLATED, {}).get("peak", np.nan),
                    "full_minus_base_peak": full_vs_base,
                    "full_minus_ablated_peak": full_vs_ablated,
                    "full_peak_layer_pair": per_arm.get(ARM_FULL, {}).get("peak_pair", np.nan),
                    "base_num_pairs_gt_095": per_arm.get(ARM_BASE, {}).get("count_gt_095", 0),
                    "full_num_pairs_gt_095": per_arm.get(ARM_FULL, {}).get("count_gt_095", 0),
                    "ablated_num_pairs_gt_095": per_arm.get(ARM_ABLATED, {}).get("count_gt_095", 0),
                    "base_num_pairs_gt_098": per_arm.get(ARM_BASE, {}).get("count_gt_098", 0),
                    "full_num_pairs_gt_098": per_arm.get(ARM_FULL, {}).get("count_gt_098", 0),
                    "ablated_num_pairs_gt_098": per_arm.get(ARM_ABLATED, {}).get("count_gt_098", 0),
                    "weight_activation_corr": corr,
                }
            )

    return pd.DataFrame(rows)


def _plot_task_signals(summary_df: pd.DataFrame, plot_dir: Path) -> None:
    import matplotlib.pyplot as plt

    plot_dir.mkdir(parents=True, exist_ok=True)
    tasks = sorted(summary_df["task"].unique())
    colors = {
        ARM_BASE: "#1f77b4",
        ARM_FULL: "#d62728",
        ARM_ABLATED: "#2ca02c",
    }
    for task in tasks:
        task_df = summary_df[summary_df["task"] == task]
        fig, axes = plt.subplots(3, 1, figsize=(12, 11), sharex=True)
        for ax, signal in zip(axes, SIGNAL_TYPES):
            sig_df = task_df[task_df["signal_type"] == signal]
            for arm in (ARM_BASE, ARM_FULL, ARM_ABLATED):
                arm_df = sig_df[sig_df["arm"] == arm].sort_values("layer_pair")
                if arm_df.empty:
                    continue
                x = arm_df["layer_pair"].to_numpy()
                y = arm_df["mean_cos"].to_numpy()
                lo = arm_df["ci_low"].to_numpy()
                hi = arm_df["ci_high"].to_numpy()
                ax.plot(x, y, color=colors[arm], label=arm, linewidth=2)
                ax.fill_between(x, lo, hi, color=colors[arm], alpha=0.2)
            ax.set_ylabel("Mean cosine")
            ax.set_title(f"{task} - {signal}")
            ax.grid(alpha=0.3)
        axes[-1].set_xlabel("Adjacent layer pair (l -> l+1)")
        handles, labels = axes[0].get_legend_handles_labels()
        if handles:
            fig.legend(handles, labels, loc="upper center", ncol=3, frameon=True)
        fig.tight_layout(rect=(0, 0, 1, 0.97))
        fig.savefig(plot_dir / f"{task}_adjacent_similarity.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _run_arm_analysis(
    *,
    task: str,
    arm: str,
    model: Any,
    setup: Any,
    max_samples: int,
    batch_size: int,
    seed: int,
) -> List[Dict[str, Any]]:
    device = _resolve_device(model)
    subset_ds = _dataset_subset(setup.dataset, max_samples=max_samples, seed=seed)
    dataloader = DataLoader(
        subset_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=setup.data_collator,
    )

    layers = _discover_decoder_layers(model)
    collector = BranchCollector()
    hooks = []
    for idx, layer in layers.items():
        hooks.append(layer.self_attn.register_forward_hook(collector.make_hook("attention", idx)))
        hooks.append(layer.mlp.register_forward_hook(collector.make_hook("mlp", idx)))

    rows: List[Dict[str, Any]] = []
    model.eval()
    with torch.no_grad():
        saw_attention = False
        saw_mlp = False
        for batch in dataloader:
            collector.clear()
            batch_on_device = _move_to_device(batch, device)
            batch_on_device = _force_input_tensors_to_embedding_device(batch_on_device, model)
            outputs = model(**batch_on_device, output_hidden_states=True, return_dict=True)

            block_states = list(outputs.hidden_states[1:])  # drop embedding output
            if not block_states:
                continue
            mask = batch_on_device.get("attention_mask")
            if mask is None:
                mask = torch.ones(block_states[0].shape[:2], dtype=torch.bool, device=block_states[0].device)
            else:
                mask = mask.to(block_states[0].device)

            signal_state_map: Dict[str, List[torch.Tensor]] = {
                "block": block_states,
                "attention": [collector.cache["attention"][i] for i in sorted(collector.cache["attention"].keys())],
                "mlp": [collector.cache["mlp"][i] for i in sorted(collector.cache["mlp"].keys())],
            }
            saw_attention = saw_attention or bool(signal_state_map["attention"])
            saw_mlp = saw_mlp or bool(signal_state_map["mlp"])

            for signal, states in signal_state_map.items():
                means_by_pair = _adjacent_example_means(states, mask)
                for layer_pair, example_vals in means_by_pair.items():
                    for value, n_tokens in example_vals:
                        rows.append(
                            {
                                "task": task,
                                "arm": arm,
                                "signal_type": signal,
                                "layer_pair": layer_pair,
                                "example_mean_cos": value,
                                "n_tokens": n_tokens,
                            }
                        )

        if not saw_attention or not saw_mlp:
            raise RuntimeError(
                f"Mandatory branch capture failed for task={task}, arm={arm}. "
                f"saw_attention={saw_attention}, saw_mlp={saw_mlp}."
            )

    for hook in hooks:
        hook.remove()
    return rows


def _save_outputs(
    *,
    output_dir: Path,
    plot_dir: Path,
    summary_df: pd.DataFrame,
    collapse_df: pd.DataFrame,
    weight_df: pd.DataFrame,
    ablation_df: pd.DataFrame,
    skip_plots: bool,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = _split_signal_frames(summary_df)
    frames["block"].to_csv(output_dir / "adjacent_similarity_block.csv", index=False)
    frames["attention"].to_csv(output_dir / "adjacent_similarity_attention_branch.csv", index=False)
    frames["mlp"].to_csv(output_dir / "adjacent_similarity_mlp_branch.csv", index=False)
    collapse_df.to_csv(output_dir / "collapse_summary.csv", index=False)
    weight_df.sort_values(["task", "arm", "module_group", "layer"]).to_csv(
        output_dir / "weight_layer_stats.csv", index=False
    )
    ablation_df.sort_values(["task"]).to_csv(output_dir / "ablation_validation.csv", index=False)
    if not skip_plots:
        try:
            _plot_task_signals(summary_df, plot_dir)
        except ModuleNotFoundError as exc:
            raise ModuleNotFoundError(
                "Plot generation requires matplotlib. Install it or rerun with --skip-plots."
            ) from exc


def main() -> None:
    args = parse_args()
    if args.hf_cache_dir:
        cache_dir = str((PROJECT_ROOT / args.hf_cache_dir).resolve())
        os.environ["HF_HOME"] = cache_dir
        os.environ["HF_DATASETS_CACHE"] = str(Path(cache_dir) / "datasets")
        os.environ["TRANSFORMERS_CACHE"] = str(Path(cache_dir) / "transformers")

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    all_rows: List[Dict[str, Any]] = []
    weight_frames: List[pd.DataFrame] = []
    ablation_rows: List[Dict[str, Any]] = []

    print("=" * 100)
    print("LAYER OUTPUT SIMILARITY ANALYSIS")
    print("=" * 100)

    for task in args.tasks:
        print(f"\n{'-' * 100}\nTask: {task}\n{'-' * 100}")
        context = _resolve_task_context(task)
        setup_model_path = context.model_path

        # Base arm
        print("  [1/3] Running base arm...")
        base_model, processor = load_model_and_processor(setup_model_path, adapter_path=None)
        setup = prepare_task_for_evaluation(task, processor, split=args.split, config=context.config)
        base_rows = _run_arm_analysis(
            task=task,
            arm=ARM_BASE,
            model=base_model,
            setup=setup,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_rows.extend(base_rows)
        del base_model

        # Full + ablated arms share one model load
        print("  [2/3] Running full adapter arm...")
        full_model, processor = load_model_and_processor(setup_model_path, adapter_path=context.adapter_path)
        setup = prepare_task_for_evaluation(task, processor, split=args.split, config=context.config)
        full_rows = _run_arm_analysis(
            task=task,
            arm=ARM_FULL,
            model=full_model,
            setup=setup,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_rows.extend(full_rows)

        print("  [3/3] Running aligner-ablated arm...")
        ablation_info = _ablate_aligner_inplace(full_model)
        ablation_info["task"] = task
        ablation_rows.append(ablation_info)
        ablated_rows = _run_arm_analysis(
            task=task,
            arm=ARM_ABLATED,
            model=full_model,
            setup=setup,
            max_samples=args.max_samples,
            batch_size=args.batch_size,
            seed=args.seed,
        )
        all_rows.extend(ablated_rows)
        del full_model

        weight_frames.append(_compute_weight_stats(task, context.adapter_path))

    summary_df = _summarize_signal_records(
        all_rows,
        bootstrap_samples=args.bootstrap_samples,
        seed=args.seed,
    )
    weight_df = pd.concat(weight_frames, ignore_index=True) if weight_frames else pd.DataFrame()
    ablation_df = pd.DataFrame(ablation_rows)
    collapse_df = _build_collapse_summary(summary_df, weight_df)

    output_dir = (PROJECT_ROOT / args.output_dir).resolve()
    plot_dir = (PROJECT_ROOT / args.plot_dir).resolve()
    _save_outputs(
        output_dir=output_dir,
        plot_dir=plot_dir,
        summary_df=summary_df,
        collapse_df=collapse_df,
        weight_df=weight_df,
        ablation_df=ablation_df,
        skip_plots=args.skip_plots,
    )

    print(f"\nSaved CSV outputs to: {output_dir}")
    print(f"Saved plots to: {plot_dir}")
    print("Done.")


if __name__ == "__main__":
    main()
