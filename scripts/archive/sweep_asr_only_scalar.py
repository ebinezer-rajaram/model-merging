#!/usr/bin/env python3
"""ASR-only scalar sweep: evaluate base + alpha * delta_asr.

Uses the standard evaluation path (`core.evaluation.evaluate_task.evaluate`) so
dataset loading, filtering, decoding, and metric computation match production.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime
from pathlib import Path

_SCRIPT_ROOT = Path(__file__).resolve().parents[1]
if str(_SCRIPT_ROOT) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_ROOT))

from _repo import find_repo_root
from typing import Dict, List

import numpy as np
import torch
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, Matern, WhiteKernel

PACKAGE_ROOT = find_repo_root(__file__)
if str(PACKAGE_ROOT) not in sys.path:
    sys.path.insert(0, str(PACKAGE_ROOT))

from core.evaluation.evaluate_task import evaluate
from merging.runtime.task_vectors import extract_task_vector_from_lora


DEFAULT_ASR_ADAPTER = "artifacts/asr/adapters/qwen2_5_omni_lora_asr_100h/best"


def _parse_scalars(raw: str) -> List[float]:
    vals: List[float] = []
    for part in str(raw).split(","):
        p = part.strip()
        if not p:
            continue
        vals.append(float(p))
    if not vals:
        raise ValueError("No scalar values parsed from --scalars.")
    return vals


def _scale_delta(base_delta: Dict[str, torch.Tensor], alpha: float) -> Dict[str, torch.Tensor]:
    a = float(alpha)
    return {k: (v * a) for k, v in base_delta.items()}


def _write_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        if not rows:
            writer = csv.writer(f)
            writer.writerow(["empty"])
            return
        headers = list(rows[0].keys())
        writer = csv.DictWriter(f, fieldnames=headers)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _evaluate_alpha(
    *,
    alpha: float,
    base_delta: Dict[str, torch.Tensor],
    split: str,
    batch_size: int | None,
) -> Dict[str, object]:
    scaled = _scale_delta(base_delta, alpha)
    result = evaluate(
        task="asr",
        adapter=None,
        split=split,
        batch_size=batch_size,
        enable_cache=False,  # force fresh inference
        show_summary=True,
        generate_confusion_matrix=False,
        delta_weights=scaled,
        adapter_label=f"asr_scalar_{alpha:.6f}",
        merged_tasks=["asr"],
        merged_method="asr_only_scalar_sweep",
    )
    m = result.metrics
    return {
        "alpha": float(alpha),
        "wer": m.get("wer"),
        "loss": m.get("loss"),
        "runtime": m.get("runtime"),
        "samples_per_second": m.get("samples_per_second"),
        "steps_per_second": m.get("steps_per_second"),
        "split": split,
    }


def _default_kernel() -> object:
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(length_scale=0.2, nu=2.5) + WhiteKernel(
        noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1)
    )


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float = 0.01) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    imp = best - mu - xi  # minimize objective (WER)
    z = imp / sigma
    return imp * norm.cdf(z) + sigma * norm.pdf(z)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Sweep scalar on ASR task vector only.")
    parser.add_argument("--split", default="validation", choices=("train", "validation", "test"))
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument(
        "--adapter",
        default=DEFAULT_ASR_ADAPTER,
        help="ASR adapter path used to extract task vector.",
    )
    parser.add_argument(
        "--search",
        default="bayes",
        choices=("bayes", "grid"),
        help="Search strategy for alpha.",
    )
    parser.add_argument("--alpha-min", type=float, default=0.0, help="Minimum alpha for search.")
    parser.add_argument("--alpha-max", type=float, default=1.2, help="Maximum alpha for search.")
    parser.add_argument("--iterations", type=int, default=12, help="Total evaluations for bayes mode.")
    parser.add_argument("--init-points", type=int, default=4, help="Initial random points for bayes mode.")
    parser.add_argument("--seed", type=int, default=0, help="Random seed for bayes mode.")
    parser.add_argument(
        "--candidate-pool",
        type=int,
        default=2048,
        help="Number of random alpha candidates per BO step.",
    )
    parser.add_argument(
        "--scalars",
        default="0.00,0.25,0.50,0.65,0.80,1.00,1.20",
        help="Comma-separated scalar list (used in grid mode).",
    )
    parser.add_argument(
        "--output-dir",
        default="analysis/results/asr_scalar_sweep",
        help="Output directory for sweep artifacts.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    out_dir = (PACKAGE_ROOT / args.output_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    adapter_path = Path(args.adapter).expanduser()
    if not adapter_path.is_absolute():
        adapter_path = (PACKAGE_ROOT / adapter_path).resolve()
    if not adapter_path.exists():
        raise FileNotFoundError(f"Adapter path does not exist: {adapter_path}")

    print(f"Extracting ASR task vector from: {adapter_path}")
    base_delta = extract_task_vector_from_lora(adapter_path)
    if args.alpha_max <= args.alpha_min:
        raise ValueError("--alpha-max must be greater than --alpha-min.")

    rows: List[Dict[str, object]] = []
    if args.search == "grid":
        scalars = _parse_scalars(args.scalars)
        print(f"Grid sweeping {len(scalars)} scalar values on split='{args.split}'")
        for i, alpha in enumerate(scalars, start=1):
            print(f"\n[{i}/{len(scalars)}] alpha={alpha:.6f}")
            row = _evaluate_alpha(
                alpha=float(alpha),
                base_delta=base_delta,
                split=args.split,
                batch_size=args.batch_size,
            )
            row["strategy"] = "grid"
            rows.append(row)
    else:
        budget = int(args.iterations)
        init_points = int(args.init_points)
        if budget <= 0:
            raise ValueError("--iterations must be > 0.")
        if init_points <= 0:
            raise ValueError("--init-points must be > 0.")
        if init_points > budget:
            init_points = budget

        rng = np.random.default_rng(int(args.seed))
        tried: set[float] = set()
        xs: List[float] = []
        ys: List[float] = []

        print(
            f"Bayesian optimization on split='{args.split}' "
            f"range=[{args.alpha_min}, {args.alpha_max}] budget={budget}"
        )

        # Initial random exploration.
        for i in range(init_points):
            alpha = float(rng.uniform(args.alpha_min, args.alpha_max))
            alpha = round(alpha, 8)
            while alpha in tried:
                alpha = round(float(rng.uniform(args.alpha_min, args.alpha_max)), 8)
            tried.add(alpha)
            print(f"\n[init {i + 1}/{init_points}] alpha={alpha:.6f}")
            row = _evaluate_alpha(
                alpha=alpha,
                base_delta=base_delta,
                split=args.split,
                batch_size=args.batch_size,
            )
            row["strategy"] = "bayes_init"
            rows.append(row)
            if isinstance(row.get("wer"), (int, float)):
                xs.append(alpha)
                ys.append(float(row["wer"]))

        # BO loop.
        for step in range(init_points, budget):
            if len(xs) < 2:
                alpha = round(float(rng.uniform(args.alpha_min, args.alpha_max)), 8)
            else:
                x_train = np.array(xs, dtype=np.float64).reshape(-1, 1)
                y_train = np.array(ys, dtype=np.float64)
                gp = GaussianProcessRegressor(
                    kernel=_default_kernel(),
                    normalize_y=True,
                    random_state=int(args.seed),
                    n_restarts_optimizer=2,
                )
                gp.fit(x_train, y_train)

                candidates = rng.uniform(args.alpha_min, args.alpha_max, size=int(args.candidate_pool))
                candidates = np.asarray(candidates, dtype=np.float64).reshape(-1, 1)
                mu, sigma = gp.predict(candidates, return_std=True)
                best_so_far = float(min(ys))
                ei = _expected_improvement(mu, sigma, best_so_far, xi=0.005)
                alpha = float(candidates[int(np.argmax(ei)), 0])
                alpha = round(alpha, 8)

            while alpha in tried:
                alpha = round(float(rng.uniform(args.alpha_min, args.alpha_max)), 8)
            tried.add(alpha)
            print(f"\n[bo {step + 1}/{budget}] alpha={alpha:.6f}")
            row = _evaluate_alpha(
                alpha=alpha,
                base_delta=base_delta,
                split=args.split,
                batch_size=args.batch_size,
            )
            row["strategy"] = "bayes_bo"
            rows.append(row)
            if isinstance(row.get("wer"), (int, float)):
                xs.append(alpha)
                ys.append(float(row["wer"]))

    rows_sorted = sorted(rows, key=lambda r: float(r["alpha"]))
    csv_path = out_dir / "asr_only_scalar_sweep.csv"
    _write_csv(csv_path, rows_sorted)

    best = min(
        [r for r in rows_sorted if isinstance(r.get("wer"), (int, float))],
        key=lambda r: float(r["wer"]),
    )
    summary = {
        "timestamp": datetime.now().isoformat(),
        "split": args.split,
        "search": args.search,
        "alpha_min": float(args.alpha_min),
        "alpha_max": float(args.alpha_max),
        "iterations": int(args.iterations),
        "init_points": int(args.init_points),
        "seed": int(args.seed),
        "adapter_source": str(adapter_path),
        "scalars": [float(r["alpha"]) for r in rows_sorted],
        "best_by_wer": best,
        "results_csv": str(csv_path),
    }
    summary_path = out_dir / "summary.json"
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"\nSaved sweep CSV: {csv_path}")
    print(f"Saved summary: {summary_path}")
    print(f"Best alpha by WER: {best['alpha']} (WER={best['wer']})")


if __name__ == "__main__":
    main()
