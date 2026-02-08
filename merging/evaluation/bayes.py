"""Bayesian optimization-based merge sweeps.

This implements a lightweight Bayesian Optimization loop using:
  - scikit-learn GaussianProcessRegressor as a surrogate model
  - Expected Improvement (EI) as an acquisition function

It is designed for the small dimensional, mostly-continuous spaces used by
merge hyperparameters (e.g., weighted / weighted_delta lambda).
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

import glob
import json

import numpy as np
from scipy.stats import norm
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel,
)

from merging.engine.registry import get_merge_method, normalize_params
from merging.runtime.utils import PACKAGE_ROOT
from merging.evaluation.evaluate import evaluate_merged_adapter
from merging.evaluation.sweep import SweepConfig, _atomic_write_json, _score_min_interference


_FLOAT_KINDS = {"float", "continuous"}
_INT_KINDS = {"int", "integer"}
_CAT_KINDS = {"categorical", "cat", "choice", "enum"}


@dataclass(frozen=True)
class _DimSpec:
    name: str
    kind: str
    min_value: Optional[float] = None
    max_value: Optional[float] = None
    choices: Optional[Tuple[Any, ...]] = None
    log: bool = False

    def validate(self) -> None:
        kind = self.kind.lower()
        if kind in _FLOAT_KINDS | _INT_KINDS:
            if self.min_value is None or self.max_value is None:
                raise ValueError(f"space.{self.name} requires min/max for kind={self.kind}.")
            if not (self.max_value > self.min_value):
                raise ValueError(f"space.{self.name} requires max > min.")
            if self.log and self.min_value <= 0:
                raise ValueError(f"space.{self.name} log scale requires min > 0.")
        elif kind in _CAT_KINDS:
            if not self.choices:
                raise ValueError(f"space.{self.name} requires non-empty values for kind={self.kind}.")
        else:
            raise ValueError(f"Unsupported space kind for {self.name}: {self.kind}")


class _SpaceEncoder:
    def __init__(self, dims: List[_DimSpec]) -> None:
        self.dims = list(dims)
        self.names = [d.name for d in self.dims]

        offsets: List[Tuple[int, int]] = []
        cursor = 0
        cat_maps: Dict[str, Dict[Any, int]] = {}
        for dim in self.dims:
            kind = dim.kind.lower()
            if kind in _CAT_KINDS:
                n = len(dim.choices or ())
                mapping = {v: i for i, v in enumerate(dim.choices or ())}
                cat_maps[dim.name] = mapping
            else:
                n = 1
            offsets.append((cursor, cursor + n))
            cursor += n

        self._offsets = offsets
        self._cat_maps = cat_maps
        self.encoded_dim = cursor

    def _key(self, params: Mapping[str, Any]) -> Tuple[Any, ...]:
        key: List[Any] = []
        for dim in self.dims:
            v = params[dim.name]
            if isinstance(v, float):
                key.append(round(v, 12))
            else:
                key.append(v)
        return tuple(key)

    def sample(self, rng: np.random.Generator) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        for dim in self.dims:
            kind = dim.kind.lower()
            if kind in _FLOAT_KINDS:
                if dim.log:
                    lo = float(np.log(dim.min_value))  # type: ignore[arg-type]
                    hi = float(np.log(dim.max_value))  # type: ignore[arg-type]
                    out[dim.name] = float(np.exp(rng.uniform(lo, hi)))
                else:
                    out[dim.name] = float(rng.uniform(dim.min_value, dim.max_value))  # type: ignore[arg-type]
            elif kind in _INT_KINDS:
                # inclusive bounds
                lo = int(dim.min_value)  # type: ignore[arg-type]
                hi = int(dim.max_value)  # type: ignore[arg-type]
                out[dim.name] = int(rng.integers(lo, hi + 1))
            elif kind in _CAT_KINDS:
                choices = list(dim.choices or ())
                out[dim.name] = choices[int(rng.integers(0, len(choices)))]
            else:  # pragma: no cover
                raise AssertionError(f"Unhandled kind: {dim.kind}")
        return out

    def encode_many(self, params_list: Iterable[Mapping[str, Any]]) -> np.ndarray:
        params_list = list(params_list)
        x = np.zeros((len(params_list), self.encoded_dim), dtype=np.float64)
        for row, params in enumerate(params_list):
            for dim, (start, end) in zip(self.dims, self._offsets):
                kind = dim.kind.lower()
                v = params[dim.name]
                if kind in _FLOAT_KINDS:
                    assert dim.min_value is not None and dim.max_value is not None
                    if dim.log:
                        v = float(np.log(float(v)))
                        lo = float(np.log(dim.min_value))
                        hi = float(np.log(dim.max_value))
                    else:
                        v = float(v)
                        lo = float(dim.min_value)
                        hi = float(dim.max_value)
                    x[row, start] = (v - lo) / (hi - lo)
                elif kind in _INT_KINDS:
                    assert dim.min_value is not None and dim.max_value is not None
                    v = float(int(v))
                    lo = float(int(dim.min_value))
                    hi = float(int(dim.max_value))
                    x[row, start] = (v - lo) / (hi - lo)
                elif kind in _CAT_KINDS:
                    mapping = self._cat_maps[dim.name]
                    idx = mapping[v]
                    x[row, start + idx] = 1.0
                    if end - start != len(mapping):  # pragma: no cover
                        raise AssertionError("categorical slice mismatch")
                else:  # pragma: no cover
                    raise AssertionError(f"Unhandled kind: {dim.kind}")
        return x


def _parse_space(space: Mapping[str, Any]) -> List[_DimSpec]:
    dims: List[_DimSpec] = []
    for name, spec in space.items():
        if not isinstance(spec, Mapping):
            raise ValueError(f"space.{name} must be a mapping.")
        kind = str(spec.get("type", "float")).lower()
        log = bool(spec.get("log", False) or spec.get("scale", "") == "log")
        if kind in _FLOAT_KINDS | _INT_KINDS:
            if "min" not in spec or "max" not in spec:
                raise ValueError(f"space.{name} requires both 'min' and 'max'.")
            dims.append(
                _DimSpec(
                    name=name,
                    kind=kind,
                    min_value=float(spec["min"]),
                    max_value=float(spec["max"]),
                    log=log,
                )
            )
        elif kind in _CAT_KINDS:
            values = spec.get("values", spec.get("choices"))
            if not isinstance(values, list):
                raise ValueError(f"space.{name}.values must be a list for kind={kind}.")
            dims.append(_DimSpec(name=name, kind=kind, choices=tuple(values)))
        else:
            raise ValueError(f"Unsupported space kind for {name}: {kind}")

    if not dims:
        raise ValueError("search.space must define at least one dimension.")

    for dim in dims:
        dim.validate()

    return dims


def _expected_improvement(mu: np.ndarray, sigma: np.ndarray, best: float, xi: float) -> np.ndarray:
    sigma = np.maximum(sigma, 1e-12)
    improvement = mu - best - xi
    z = improvement / sigma
    return improvement * norm.cdf(z) + sigma * norm.pdf(z)


def _default_kernel(d: int) -> Any:
    return ConstantKernel(1.0, (1e-3, 1e3)) * Matern(
        length_scale=np.ones(d),
        length_scale_bounds=(1e-2, 1e2),
        nu=2.5,
    ) + WhiteKernel(noise_level=1e-5, noise_level_bounds=(1e-10, 1e-1))


def _resolve_search_path(raw: str) -> Path:
    path = Path(raw)
    if not path.is_absolute():
        path = PACKAGE_ROOT / path
    return path


def _iter_sweep_json_paths(raw: str) -> List[Path]:
    """Resolve a sweep file/dir/glob to concrete sweep_*.json paths."""
    if any(ch in raw for ch in ("*", "?", "[")):
        resolved = str(_resolve_search_path(raw))
        return sorted(Path(p) for p in glob.glob(resolved))

    path = _resolve_search_path(raw)
    if path.is_dir():
        return sorted(path.glob("sweep_*.json"))
    return [path]


def _compatible_sweep(config: SweepConfig, sweep: Mapping[str, Any]) -> bool:
    if str(sweep.get("method")) != str(config.method):
        return False
    sweep_adapters = sweep.get("adapters")
    if not isinstance(sweep_adapters, list) or sorted(map(str, sweep_adapters)) != sorted(map(str, config.adapters)):
        return False
    if str(sweep.get("split")) != str(config.split):
        return False
    if str(sweep.get("merge_mode", "common")) != str(config.merge_mode):
        return False
    if bool(sweep.get("constraint_nonnegative", True)) != bool(config.constraint_nonnegative):
        return False

    sweep_eval_tasks = sweep.get("eval_tasks")
    if (config.eval_tasks is None) != (sweep_eval_tasks is None):
        return False
    if config.eval_tasks is not None:
        if not isinstance(sweep_eval_tasks, list):
            return False
        if sorted(map(str, sweep_eval_tasks)) != sorted(map(str, config.eval_tasks)):
            return False

    sweep_eval_subset = sweep.get("eval_subset")
    if (config.eval_subset is None) != (sweep_eval_subset is None):
        return False
    if config.eval_subset is not None and sweep_eval_subset != config.eval_subset:
        return False

    if sweep.get("lambda_policy") != config.lambda_policy:
        return False
    if sweep.get("optimizer") != config.optimizer:
        return False

    return True


def run_bayes_search(config: SweepConfig, search: Dict[str, Any]) -> Dict[str, Any]:
    """Bayesian optimization sweep runner.

    Expected search config shape:
      search:
        type: bayes
        space:
          lambda:
            min: 0.0
            max: 1.0
            type: float
        budget: 20
        seed: 42
    """
    space_raw = search.get("space")
    if not isinstance(space_raw, Mapping):
        raise ValueError("Bayes search requires search.space to be a mapping.")

    dims = _parse_space(space_raw)
    encoder = _SpaceEncoder(dims)

    budget = int(search.get("budget", 20))
    if budget <= 0:
        raise ValueError("search.budget must be > 0.")

    seed = search.get("seed")
    rng = np.random.default_rng(None if seed is None else int(seed))

    init_points = int(search.get("init_points", min(8, budget)))
    init_points = max(1, min(init_points, budget))

    n_candidates = int(search.get("n_candidates", 2048))
    n_candidates = max(128, n_candidates)

    xi = float(search.get("xi", 0.01))
    invalid_score = float(search.get("invalid_score", -1e9))

    # Output path mirrors grid sweeps.
    started_at = datetime.now()
    timestamp = started_at.strftime("%Y%m%d_%H%M%S")
    summary_dir = config.output_dir
    if summary_dir is None:
        summary_dir = (
            PACKAGE_ROOT
            / "artifacts"
            / "merged"
            / config.method
            / "_".join(sorted(config.adapters))
            / "sweeps"
        )
    summary_dir.mkdir(parents=True, exist_ok=True)
    summary_path = summary_dir / f"sweep_{timestamp}.json"

    runs: List[Dict[str, Any]] = []
    seen: set[Tuple[Any, ...]] = set()

    best_idx: Optional[int] = None
    best_score = float("-inf")
    best_tiebreak = float("-inf")

    summary: Dict[str, Any] = {
        "timestamp": started_at.isoformat(),
        "method": config.method,
        "adapters": config.adapters,
        "merge_mode": config.merge_mode,
        "lambda_policy": config.lambda_policy,
        "optimizer": config.optimizer,
        "eval_tasks": config.eval_tasks,
        "split": config.split,
        "eval_subset": config.eval_subset,
        "grid": config.grid,
        "search": search,
        "constraint_nonnegative": config.constraint_nonnegative,
        "best_index": best_idx,
        "best_score": best_score,
        "runs": runs,
    }
    _atomic_write_json(summary_path, summary)

    def _flush_summary() -> None:
        summary["best_index"] = best_idx
        summary["best_score"] = best_score
        _atomic_write_json(summary_path, summary)

    def _record_run(params: Dict[str, Any], results: Dict[str, Dict] | None, score: float, details: Dict[str, Any]) -> None:
        nonlocal best_idx, best_score, best_tiebreak
        entry: Dict[str, Any] = {
            "params": dict(params),
            "score": float(score),
            "score_details": dict(details),
            "results": results or {},
        }
        runs.append(entry)

        tiebreak = details.get("mean_interference_delta", float("-inf"))
        if score > best_score or (score == best_score and tiebreak > best_tiebreak):
            best_score = float(score)
            best_tiebreak = float(tiebreak) if isinstance(tiebreak, (int, float)) else float("-inf")
            best_idx = len(runs) - 1
            print(f"🏆 Best so far: params={params} score={best_score:.4f}")
        _flush_summary()

    def _evaluate(params: Dict[str, Any]) -> None:
        method_impl = get_merge_method(config.method)
        try:
            effective_params = normalize_params(method_impl, params=params)
            if config.lambda_policy is not None:
                effective_params["lambda_policy"] = config.lambda_policy
            if config.optimizer is not None:
                effective_params["optimizer"] = config.optimizer
            method_impl.validate(len(config.adapters), effective_params)
            results = evaluate_merged_adapter(
                adapter_path=None,
                method=config.method,
                task_names=config.adapters,
                params=effective_params,
                eval_tasks=config.eval_tasks,
                split=config.split,
                save_merged=config.save_merged,
                save_results=True,
                show_summary=True,
                merge_mode=config.merge_mode,
                compute_missing_interference_baselines=config.compute_missing_interference_baselines,
                eval_subset=config.eval_subset,
            )
            raw_score, score_details = _score_min_interference(results, config.constraint_nonnegative)
            score = raw_score if np.isfinite(raw_score) else invalid_score
            if not np.isfinite(score):
                score = invalid_score
            _record_run(params, results, float(score), score_details)
        except Exception as exc:
            _record_run(params, None, invalid_score, {"error": str(exc)})

    # Warm-start from previous sweep summaries (no re-evaluation).
    warm_start_sweeps = search.get("warm_start_sweeps") or search.get("warm_start_from")
    if warm_start_sweeps is not None:
        if isinstance(warm_start_sweeps, (str, Path)):
            warm_start_sweeps = [str(warm_start_sweeps)]
        if not isinstance(warm_start_sweeps, list) or not all(isinstance(p, (str, Path)) for p in warm_start_sweeps):
            raise ValueError("search.warm_start_sweeps must be a path string or a list of path strings.")

        max_points = int(search.get("warm_start_max_points", budget))
        max_points = max(0, min(max_points, budget))

        files_read = 0
        compatible_files = 0
        candidates: List[Tuple[float, float, Dict[str, Any], Dict[str, Any], Dict[str, Dict] | None]] = []
        for raw in warm_start_sweeps:
            for path in _iter_sweep_json_paths(str(raw)):
                if not path.exists():
                    raise FileNotFoundError(f"Warm-start sweep not found: {path}")
                try:
                    with path.open("r") as handle:
                        sweep_obj = json.load(handle)
                except Exception as exc:
                    raise ValueError(f"Failed to read warm-start sweep JSON at {path}: {exc}") from exc
                files_read += 1
                if not isinstance(sweep_obj, Mapping):
                    raise ValueError(f"Warm-start sweep JSON at {path} must be an object.")
                if not _compatible_sweep(config, sweep_obj):
                    continue
                compatible_files += 1
                sweep_runs = sweep_obj.get("runs")
                if not isinstance(sweep_runs, list):
                    continue
                for entry in sweep_runs:
                    if not isinstance(entry, Mapping):
                        continue
                    params_raw = entry.get("params")
                    if not isinstance(params_raw, Mapping):
                        continue
                    params = {k: params_raw[k] for k in encoder.names if k in params_raw}
                    if set(params.keys()) != set(encoder.names):
                        continue
                    try:
                        score = float(entry.get("score"))
                    except Exception:
                        score = invalid_score
                    if not np.isfinite(score):
                        score = invalid_score
                    details = entry.get("score_details") if isinstance(entry.get("score_details"), Mapping) else {}
                    tiebreak_raw = details.get("mean_interference_delta", float("-inf")) if isinstance(details, Mapping) else float("-inf")
                    tiebreak = float(tiebreak_raw) if isinstance(tiebreak_raw, (int, float)) else float("-inf")
                    results = entry.get("results") if isinstance(entry.get("results"), Mapping) else None
                    candidates.append((score, tiebreak, dict(params), dict(details) if isinstance(details, Mapping) else {}, results))  # type: ignore[arg-type]

        # Prefer best points first, but keep only unique params.
        candidates.sort(key=lambda t: (t[0], t[1]), reverse=True)
        added = 0
        for score, _tiebreak, params, details, results in candidates:
            if added >= max_points or len(runs) >= budget:
                break
            key = encoder._key(params)
            if key in seen:
                continue
            seen.add(key)
            _record_run(params, results, score, details)
            added += 1

        if files_read == 0:
            print("♻️  Warm-start sweeps: no sweep JSON files found.")
        elif compatible_files == 0:
            print(f"♻️  Warm-start sweeps: 0/{files_read} sweep files were compatible; skipping warm-start.")
        else:
            print(f"♻️  Warm-started BO with {added} point(s) from {compatible_files}/{files_read} compatible sweep file(s).")

    try:
        warm_start = search.get("initial_points")
        if warm_start is not None:
            if not isinstance(warm_start, list):
                raise ValueError("search.initial_points must be a list of param dicts.")
            for point in warm_start:
                if not isinstance(point, Mapping):
                    raise ValueError("search.initial_points entries must be mappings.")
                params = {k: point[k] for k in encoder.names if k in point}
                if set(params.keys()) != set(encoder.names):
                    missing = sorted(set(encoder.names) - set(params.keys()))
                    raise ValueError(f"search.initial_points entry missing keys: {missing}")
                key = encoder._key(params)
                if key in seen:
                    continue
                seen.add(key)
                print(f"\n[warm-start {len(runs)+1}/{budget}] Evaluating params: {params}")
                _evaluate(params)
                if len(runs) >= budget:
                    break

        # Random initialization.
        while len(runs) < min(init_points, budget):
            params = encoder.sample(rng)
            key = encoder._key(params)
            if key in seen:
                continue
            seen.add(key)
            print(f"\n[init {len(runs)+1}/{budget}] Evaluating params: {params}")
            _evaluate(params)

        # BO loop.
        while len(runs) < budget:
            x_train = encoder.encode_many([r["params"] for r in runs])
            y_train = np.asarray([float(r["score"]) for r in runs], dtype=np.float64)

            kernel = _default_kernel(encoder.encoded_dim)
            gp = GaussianProcessRegressor(
                kernel=kernel,
                alpha=float(search.get("alpha", 1e-6)),
                normalize_y=bool(search.get("normalize_y", True)),
                n_restarts_optimizer=int(search.get("n_restarts_optimizer", 3)),
                random_state=None if seed is None else int(seed),
            )

            fit_ok = True
            try:
                gp.fit(x_train, y_train)
            except Exception:
                fit_ok = False

            # Candidate pool (random sampling).
            candidates: List[Dict[str, Any]] = []
            attempts = 0
            while len(candidates) < n_candidates and attempts < n_candidates * 10:
                attempts += 1
                p = encoder.sample(rng)
                k = encoder._key(p)
                if k in seen:
                    continue
                candidates.append(p)

            if not candidates:
                # Extremely small discrete space exhausted.
                break

            if fit_ok and best_idx is not None:
                x_cand = encoder.encode_many(candidates)
                mu, sigma = gp.predict(x_cand, return_std=True)
                acq = _expected_improvement(mu, sigma, best=float(best_score), xi=xi)
                best_cand_idx = int(np.argmax(acq))
                next_params = candidates[best_cand_idx]
                phase = "bo"
            else:
                next_params = candidates[0]
                phase = "rand"

            seen.add(encoder._key(next_params))
            print(f"\n[{phase} {len(runs)+1}/{budget}] Evaluating params: {next_params}")
            _evaluate(next_params)
    except KeyboardInterrupt:
        _flush_summary()
        print(f"\n⏹️  Sweep interrupted. Partial summary saved to {summary_path}")
        if best_idx is not None:
            best_params = runs[best_idx]["params"]
            print(f"🏆 Best params so far: {best_params} (score={best_score:.4f})")
        return summary

    print(f"\n💾 Sweep summary saved to {summary_path}")
    if best_idx is not None:
        best_params = runs[best_idx]["params"]
        print(f"🏆 Best params: {best_params} (score={best_score:.4f})")

    return summary
