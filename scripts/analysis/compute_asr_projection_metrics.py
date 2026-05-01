#!/usr/bin/env python3
"""Compute global ASR projection metrics for selected merged runs.

alpha_eff = <Delta_merged, Delta_asr> / ||Delta_asr||^2
cosine    = <Delta_merged, Delta_asr> / (||Delta_merged|| * ||Delta_asr||)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Any, Dict, Mapping

import torch

PACKAGE_ROOT = find_repo_root(__file__)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from merging.evaluation.evaluate import _merge_in_memory
from merging.runtime.task_vectors import extract_task_vector_from_lora


DEFAULT_UNIFORM_RUN_DIR = (
    PACKAGE_ROOT
    / "artifacts/merged/uniform_scalar_delta/asr_emotion_intent_kws_langid_speaker_ver/runs/run_20260215_230158"
)
DEFAULT_WEIGHTED_RUN_DIR = (
    PACKAGE_ROOT
    / "artifacts/merged/weighted_delta_n/asr_emotion_intent_kws_langid_speaker_ver/runs/run_supermerge_layer_wise_20260217_005010"
)
DEFAULT_ASR_ADAPTER_DIR = (
    PACKAGE_ROOT
    / "artifacts/asr/adapters/qwen2_5_omni_lora_asr_100h/runs/run_20260114_093047"
)
DEFAULT_EXPECTED_UNIFORM_SCALE = 0.651053423634474

DEFAULT_OUT_DIR = PACKAGE_ROOT / "analysis/results/asr_projection_metrics"


@dataclass
class ProjectionMetrics:
    method: str
    run_dir: str
    dot: float
    norm2_merged: float
    norm2_asr: float
    alpha_eff: float
    cosine: float

    def as_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "run_dir": self.run_dir,
            "dot": self.dot,
            "norm2_merged": self.norm2_merged,
            "norm2_asr": self.norm2_asr,
            "alpha_eff": self.alpha_eff,
            "cosine": self.cosine,
        }


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


def _load_metadata(run_dir: Path) -> Dict[str, Any]:
    metadata_path = run_dir / "merge_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(metadata_path)
    with metadata_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _reconstruct_merged_delta(run_dir: Path) -> Dict[str, torch.Tensor]:
    md = _load_metadata(run_dir)
    method = str(md.get("merge_method", "")).strip()
    merge_mode = str(md.get("merge_mode", "common"))
    source_adapters = md.get("source_adapters", [])
    if not method or not isinstance(source_adapters, list) or not source_adapters:
        raise ValueError(f"Invalid merge metadata in {run_dir}")

    adapter_paths = [Path(item["path"]) for item in source_adapters]
    raw_params = md.get("method_params")
    if not isinstance(raw_params, Mapping):
        raw_params = md.get("params", {})
    if not isinstance(raw_params, Mapping):
        raw_params = {}
    replay_params = _prepare_replay_params(raw_params)

    merged_delta, _, _ = _merge_in_memory(
        method=method,
        adapter_paths=adapter_paths,
        source_metadata=source_adapters,
        params=replay_params,
        merge_mode=merge_mode,
    )
    return merged_delta


def _compute_projection_metrics(
    method: str,
    run_dir: Path,
    tau_asr: Dict[str, torch.Tensor],
) -> ProjectionMetrics:
    delta_m = _reconstruct_merged_delta(run_dir)

    dot = 0.0
    norm2_merged = 0.0
    norm2_asr = 0.0

    keys = set(delta_m.keys()) | set(tau_asr.keys())
    for key in keys:
        dm = delta_m.get(key)
        ta = tau_asr.get(key)

        if dm is not None:
            dm64 = dm.detach().to(dtype=torch.float64, device="cpu")
            norm2_merged += float(torch.sum(dm64 * dm64).item())
        if ta is not None:
            ta64 = ta.detach().to(dtype=torch.float64, device="cpu")
            norm2_asr += float(torch.sum(ta64 * ta64).item())

        if dm is None or ta is None:
            continue

        dm64 = dm.detach().to(dtype=torch.float64, device="cpu")
        ta64 = ta.detach().to(dtype=torch.float64, device="cpu")
        if dm64.shape != ta64.shape:
            continue
        dot += float(torch.sum(dm64 * ta64).item())

    if norm2_merged <= 0 or norm2_asr <= 0:
        raise ValueError(f"Invalid norm(s) for {method}: merged={norm2_merged}, asr={norm2_asr}")

    alpha_eff = dot / norm2_asr
    cosine = dot / math.sqrt(norm2_merged * norm2_asr)
    return ProjectionMetrics(
        method=method,
        run_dir=str(run_dir),
        dot=dot,
        norm2_merged=norm2_merged,
        norm2_asr=norm2_asr,
        alpha_eff=alpha_eff,
        cosine=cosine,
    )


def _assert_close(name: str, a: float, b: float, tol: float = 1e-10) -> None:
    if abs(a - b) > tol:
        raise AssertionError(f"{name}: mismatch {a} vs {b} (tol={tol})")


def _validate_uniform_scale(uniform_run_dir: Path, expected_uniform_scale: float | None) -> None:
    if expected_uniform_scale is None:
        return
    md = _load_metadata(uniform_run_dir)
    method_params = md.get("method_params", {})
    scale = float(method_params.get("scale"))
    _assert_close("uniform scale", scale, expected_uniform_scale, tol=1e-12)


def _validate_metrics(metrics: ProjectionMetrics) -> None:
    # Formula consistency check.
    _assert_close(
        f"{metrics.method} formula consistency",
        metrics.alpha_eff * metrics.norm2_asr,
        metrics.dot,
        tol=1e-8,
    )
    # Sanity bounds.
    if not (-1.0 <= metrics.cosine <= 1.0):
        raise AssertionError(f"{metrics.method} cosine out of bounds: {metrics.cosine}")
    if metrics.norm2_merged <= 0 or metrics.norm2_asr <= 0:
        raise AssertionError(f"{metrics.method} norms must be positive")


def _determinism_check(
    method: str,
    run_dir: Path,
    tau_asr: Dict[str, torch.Tensor],
) -> None:
    m1 = _compute_projection_metrics(method, run_dir, tau_asr)
    m2 = _compute_projection_metrics(method, run_dir, tau_asr)
    _assert_close(f"{method} deterministic dot", m1.dot, m2.dot, tol=1e-10)
    _assert_close(f"{method} deterministic norm2_merged", m1.norm2_merged, m2.norm2_merged, tol=1e-10)
    _assert_close(f"{method} deterministic norm2_asr", m1.norm2_asr, m2.norm2_asr, tol=1e-10)
    _assert_close(f"{method} deterministic alpha_eff", m1.alpha_eff, m2.alpha_eff, tol=1e-12)
    _assert_close(f"{method} deterministic cosine", m1.cosine, m2.cosine, tol=1e-12)


def _write_outputs(
    results: Dict[str, ProjectionMetrics],
    out_dir: Path,
    asr_adapter_dir: Path,
    expected_uniform_scale: float | None,
) -> tuple[Path, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / "summary.json"
    out_csv = out_dir / "metrics.csv"

    with out_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["method", "run_dir", "dot", "norm2_merged", "norm2_asr", "alpha_eff", "cosine"],
        )
        writer.writeheader()
        for method in sorted(results):
            writer.writerow(results[method].as_dict())

    payload = {
        "formula": {
            "alpha_eff": "<Delta_merged, Delta_asr> / ||Delta_asr||^2",
            "cosine": "<Delta_merged, Delta_asr> / (||Delta_merged|| * ||Delta_asr||)",
        },
        "inputs": {
            "uniform_run_dir": results["uniform_scalar_delta"].run_dir,
            "weighted_run_dir": results["weighted_delta_n"].run_dir,
            "asr_adapter_dir": str(asr_adapter_dir),
            "expected_uniform_scale": expected_uniform_scale,
        },
        "results": {k: v.as_dict() for k, v in results.items()},
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out_csv, out_json


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute global ASR projection metrics for two merged runs.")
    parser.add_argument("--uniform-run-dir", type=Path, default=DEFAULT_UNIFORM_RUN_DIR)
    parser.add_argument("--weighted-run-dir", type=Path, default=DEFAULT_WEIGHTED_RUN_DIR)
    parser.add_argument("--asr-adapter-dir", type=Path, default=DEFAULT_ASR_ADAPTER_DIR)
    parser.add_argument("--expected-uniform-scale", type=float, default=DEFAULT_EXPECTED_UNIFORM_SCALE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    uniform_run_dir = args.uniform_run_dir.resolve()
    weighted_run_dir = args.weighted_run_dir.resolve()
    asr_adapter_dir = args.asr_adapter_dir.resolve()
    out_dir = args.output_dir.resolve()

    print(f"Loading ASR task vector from: {asr_adapter_dir}")
    tau_asr = extract_task_vector_from_lora(asr_adapter_dir)

    _validate_uniform_scale(uniform_run_dir, args.expected_uniform_scale)

    _determinism_check("uniform_scalar_delta", uniform_run_dir, tau_asr)
    _determinism_check("weighted_delta_n", weighted_run_dir, tau_asr)

    uniform_metrics = _compute_projection_metrics("uniform_scalar_delta", uniform_run_dir, tau_asr)
    weighted_metrics = _compute_projection_metrics("weighted_delta_n", weighted_run_dir, tau_asr)

    _validate_metrics(uniform_metrics)
    _validate_metrics(weighted_metrics)

    results = {
        "uniform_scalar_delta": uniform_metrics,
        "weighted_delta_n": weighted_metrics,
    }
    out_csv, out_json = _write_outputs(
        results,
        out_dir,
        asr_adapter_dir=asr_adapter_dir,
        expected_uniform_scale=args.expected_uniform_scale,
    )

    print("\nComputed ASR projection metrics:")
    for method in ("uniform_scalar_delta", "weighted_delta_n"):
        m = results[method]
        print(
            f"- {method}: alpha_eff={m.alpha_eff:.16f}, cosine={m.cosine:.16f}, "
            f"dot={m.dot:.16f}, norm2_merged={m.norm2_merged:.16f}, norm2_asr={m.norm2_asr:.16f}"
        )
    print(f"\nWrote: {out_csv}")
    print(f"Wrote: {out_json}")


if __name__ == "__main__":
    main()
