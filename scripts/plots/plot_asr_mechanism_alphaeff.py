#!/usr/bin/env python3
"""Mechanism plot: alpha_eff projection vs ASR WER for merged models.

alpha_eff(m) = <Delta_m, tau_asr> / ||tau_asr||^2
where:
  - tau_asr is the ASR task vector (LoRA-space effective delta)
  - Delta_m is the merged model delta reconstructed from merge metadata
"""

from __future__ import annotations

import csv
import importlib.util
import json
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, List, Mapping

import matplotlib.pyplot as plt
import torch

PACKAGE_ROOT = find_repo_root(__file__)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from merging.runtime.task_vectors import extract_task_vector_from_lora
from merging.evaluation.evaluate import _merge_in_memory


PAIRWISE_SUMMARY = PACKAGE_ROOT / "analysis/results/asr_pairwise/summary.csv"
SWEEP_VALIDATION_CSV = PACKAGE_ROOT / "analysis/results/asr_scalar_sweep_old/asr_only_scalar_sweep.csv"
SWEEP_TEST_CSV = PACKAGE_ROOT / "analysis/results/asr_scalar_sweep/asr_only_scalar_sweep.csv"
SWEEP_SUMMARY_JSON = PACKAGE_ROOT / "analysis/results/asr_scalar_sweep/summary.json"
OUT_DIR = PACKAGE_ROOT / "analysis/results/asr_mechanism"
OUT_CSV = OUT_DIR / "alphaeff_mechanism_points.csv"
OUT_PLOT = OUT_DIR / "alphaeff_vs_wer_mechanism_plot.png"


@dataclass
class MergePoint:
    method: str
    run_dir: Path
    wer_test: float
    alpha_eff: float


def _load_inputs_by_method() -> Dict[str, Path]:
    script_path = PACKAGE_ROOT / "scripts/merging/build_merge_comparison_6task_merge_itself.py"
    spec = importlib.util.spec_from_file_location("build_merge_comparison_6task_merge_itself", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed loading {script_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    raw = getattr(module, "INPUTS_BY_METHOD", {})
    out: Dict[str, Path] = {}
    for method, paths in raw.items():
        if not paths:
            continue
        eval_json = PACKAGE_ROOT / Path(paths[0])
        out[str(method)] = eval_json.parent
    return out


def _load_pairwise_test_wers() -> Dict[str, float]:
    if not PAIRWISE_SUMMARY.exists():
        raise FileNotFoundError(PAIRWISE_SUMMARY)
    mapping: Dict[str, float] = {}
    with PAIRWISE_SUMMARY.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pair = str(row.get("pair", ""))
            if "__vs__" not in pair:
                continue
            method = pair.split("__vs__", 1)[1].strip()
            wer_target = row.get("wer_target")
            if wer_target is None or wer_target == "":
                continue
            mapping[method] = float(wer_target)
    return mapping


def _prepare_replay_params(params: Mapping[str, Any]) -> Dict[str, Any]:
    replay = dict(params)
    optimizer = replay.get("optimizer")
    if isinstance(optimizer, Mapping):
        optimizer_type = str(optimizer.get("type", "none")).strip().lower()
        if optimizer_type != "none":
            replay["optimizer"] = {
                "type": "none",
                "params": {
                    "replay_from_merge_metadata": True,
                    "original_optimizer_type": optimizer_type,
                },
            }
    return replay


def _reconstruct_merged_delta(run_dir: Path) -> Dict[str, torch.Tensor]:
    metadata_path = run_dir / "merge_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        md = json.load(f)

    method = str(md.get("merge_method", "")).strip()
    merge_mode = str(md.get("merge_mode", "common"))
    source_adapters = md.get("source_adapters", [])
    if not method or not isinstance(source_adapters, list) or not source_adapters:
        raise ValueError(f"Invalid merge metadata in {metadata_path}")

    adapter_paths = [Path(item["path"]) for item in source_adapters]
    source_metadata = source_adapters
    raw_params = md.get("method_params")
    if not isinstance(raw_params, Mapping):
        raw_params = md.get("params", {})
    if not isinstance(raw_params, Mapping):
        raw_params = {}
    replay_params = _prepare_replay_params(raw_params)

    merged_delta, _, _ = _merge_in_memory(
        method=method,
        adapter_paths=adapter_paths,
        source_metadata=source_metadata,
        params=replay_params,
        merge_mode=merge_mode,
    )
    return merged_delta


def _alpha_eff(delta_m: Dict[str, torch.Tensor], tau_asr: Dict[str, torch.Tensor]) -> float:
    num = 0.0
    den = 0.0
    for k, t in tau_asr.items():
        td = t.detach().to(dtype=torch.float64, device="cpu")
        den += float(torch.sum(td * td).item())
        dm = delta_m.get(k)
        if dm is None:
            continue
        dd = dm.detach().to(dtype=torch.float64, device="cpu")
        if dd.shape != td.shape:
            continue
        num += float(torch.sum(dd * td).item())
    if den <= 0:
        raise ValueError("||tau_asr||^2 is zero or invalid.")
    return num / den


def _load_asr_adapter_source() -> Path:
    if SWEEP_SUMMARY_JSON.exists():
        with SWEEP_SUMMARY_JSON.open("r", encoding="utf-8") as f:
            d = json.load(f)
        src = d.get("adapter_source")
        if src:
            return Path(str(src))
    return PACKAGE_ROOT / "artifacts/asr/adapters/qwen2_5_omni_lora_asr_100h/best"


def _load_sweep_points(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    rows: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            if not r.get("alpha") or not r.get("wer"):
                continue
            rows.append(
                {
                    "alpha": float(r["alpha"]),
                    "wer": float(r["wer"]),
                    "split": r.get("split", ""),
                    "strategy": r.get("strategy", ""),
                }
            )
    return rows


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    method_to_run = _load_inputs_by_method()
    method_to_wer = _load_pairwise_test_wers()
    adapter_src = _load_asr_adapter_source()
    if not adapter_src.is_absolute():
        adapter_src = (PACKAGE_ROOT / adapter_src).resolve()

    print(f"Loading tau_asr from: {adapter_src}")
    tau_asr = extract_task_vector_from_lora(adapter_src)

    points: List[MergePoint] = []
    for method, run_dir in method_to_run.items():
        if method not in method_to_wer:
            continue
        print(f"Reconstructing merged delta: {method} ({run_dir})")
        delta_m = _reconstruct_merged_delta(run_dir)
        alpha = _alpha_eff(delta_m, tau_asr)
        points.append(
            MergePoint(
                method=method,
                run_dir=run_dir,
                wer_test=float(method_to_wer[method]),
                alpha_eff=float(alpha),
            )
        )

    points_sorted = sorted(points, key=lambda x: x.alpha_eff)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "alpha_eff", "wer_test", "run_dir"],
        )
        writer.writeheader()
        for p in points_sorted:
            writer.writerow(
                {
                    "method": p.method,
                    "alpha_eff": p.alpha_eff,
                    "wer_test": p.wer_test,
                    "run_dir": str(p.run_dir),
                }
            )

    sweep_val = _load_sweep_points(SWEEP_VALIDATION_CSV)
    sweep_test = _load_sweep_points(SWEEP_TEST_CSV)

    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=150)

    if sweep_val:
        val_sorted = sorted(sweep_val, key=lambda r: r["alpha"])
        ax.plot(
            [r["alpha"] for r in val_sorted],
            [r["wer"] for r in val_sorted],
            color="#1f77b4",
            linewidth=2.0,
            label="ASR-only scalar sweep (validation)",
            alpha=0.9,
        )
        ax.scatter(
            [r["alpha"] for r in val_sorted],
            [r["wer"] for r in val_sorted],
            color="#1f77b4",
            s=24,
            alpha=0.8,
        )

    if sweep_test:
        ax.scatter(
            [r["alpha"] for r in sweep_test],
            [r["wer"] for r in sweep_test],
            marker="*",
            color="black",
            s=120,
            label="ASR-only scalar point(s) (test)",
            zorder=4,
        )

    colors = {
        "uniform_delta": "#d62728",
        "uniform_scalar_delta": "#ff7f0e",
        "supermerge_scalar_simplex": "#2ca02c",
        "supermerge_fixed_global_scalar_redistribution": "#9467bd",
    }
    for p in points_sorted:
        c = colors.get(p.method, "#d62728")
        ax.scatter([p.alpha_eff], [p.wer_test], color=c, s=62, zorder=5)
        ax.annotate(
            p.method,
            (p.alpha_eff, p.wer_test),
            textcoords="offset points",
            xytext=(6, 5),
            fontsize=8,
            color=c,
        )

    ax.set_xlabel(r"Effective ASR Scale $\alpha_{eff}=\langle \Delta_m,\tau_{ASR}\rangle / \|\tau_{ASR}\|^2$")
    ax.set_ylabel("ASR WER (lower is better)")
    ax.set_title("Mechanism Check: ASR-Subspace Scale vs WER")
    ax.grid(True, linestyle="--", alpha=0.35)
    ax.legend(loc="best", frameon=True)

    note = "Blue curve uses validation sweep; labeled merge points use test WER."
    ax.text(0.01, 0.01, note, transform=ax.transAxes, fontsize=8, alpha=0.8)

    fig.tight_layout()
    fig.savefig(OUT_PLOT)
    print(f"Saved points CSV: {OUT_CSV}")
    print(f"Saved mechanism plot: {OUT_PLOT}")


if __name__ == "__main__":
    main()
