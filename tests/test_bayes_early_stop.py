from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence, Tuple

from merging.config.unified import MergeConfig
from merging.evaluation import bayes as bayes_module


class _FakeMethod:
    def validate(self, num_adapters: int, params: Mapping[str, Any]) -> None:
        return None


class _FakeGP:
    def __init__(self, *args, **kwargs) -> None:
        return None

    def fit(self, x_train, y_train) -> "_FakeGP":
        return self

    def predict(self, x_cand, return_std: bool = True):
        size = len(x_cand)
        mu = [float(size - idx) for idx in range(size)]
        sigma = [1.0 for _ in range(size)]
        if return_std:
            return mu, sigma
        return mu


def _params_key(params: Mapping[str, Any]) -> Tuple[Tuple[str, Any], ...]:
    return tuple(sorted((str(k), v) for k, v in params.items()))


def _lambda_sequence(targets: Sequence[float], *, filler_start: float = 0.2) -> List[Dict[str, float]]:
    sequence: List[Dict[str, float]] = []
    cursor = filler_start
    for target in targets:
        sequence.append({"lambda": float(target)})
        for _ in range(127):
            sequence.append({"lambda": round(cursor, 9)})
            cursor += 0.001
    return sequence


def _int_sequence(targets: Sequence[int], *, filler_start: int = 10) -> List[Dict[str, int]]:
    sequence: List[Dict[str, int]] = []
    cursor = filler_start
    for target in targets:
        sequence.append({"steps": int(target)})
        for _ in range(127):
            sequence.append({"steps": cursor})
            cursor += 1
    return sequence


def _run_search(
    *,
    tmp_path: Path,
    monkeypatch,
    search: Dict[str, Any],
    sample_sequence: Sequence[Dict[str, Any]],
    score_by_params: Mapping[Tuple[Tuple[str, Any], ...], float],
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    config = MergeConfig(
        adapters=["emotion", "intent"],
        method="fake_method",
        output_dir=tmp_path,
        split="validation",
        eval_tasks=["emotion"],
        save_merged=False,
        constraint_nonnegative=False,
    )

    sample_index = {"value": 0}
    post_eval_calls: List[Dict[str, Any]] = []

    def fake_sample(self, rng) -> Dict[str, Any]:
        idx = sample_index["value"]
        if idx >= len(sample_sequence):  # pragma: no cover - indicates broken test setup
            raise AssertionError("Test sample sequence exhausted.")
        sample_index["value"] = idx + 1
        return dict(sample_sequence[idx])

    def fake_evaluate_merged_adapter(**kwargs):
        params = kwargs["params"]
        score = score_by_params[_params_key(params)]
        return {"emotion": {"interference_delta": score}}

    def fake_post_sweep_eval(**kwargs):
        post_eval_calls.append(dict(kwargs))
        return {"enabled": True, "params": dict(kwargs["best_params"])}

    monkeypatch.setattr(bayes_module, "GaussianProcessRegressor", _FakeGP)
    monkeypatch.setattr(bayes_module._SpaceEncoder, "sample", fake_sample)
    monkeypatch.setattr(bayes_module, "get_merge_method", lambda method: _FakeMethod())
    monkeypatch.setattr(bayes_module, "normalize_params", lambda method_impl, params: dict(params))
    monkeypatch.setattr(bayes_module, "evaluate_merged_adapter", fake_evaluate_merged_adapter)
    monkeypatch.setattr(bayes_module, "_run_post_sweep_eval_for_best", fake_post_sweep_eval)
    monkeypatch.setattr("merging.evaluation.sweep._maybe_regen_plot", lambda summary_path: None)

    summary = bayes_module.run_bayes_search(config, search)
    return summary, post_eval_calls


def test_bayes_early_stop_disabled_uses_full_budget(tmp_path: Path, monkeypatch) -> None:
    search = {
        "type": "bayes",
        "budget": 5,
        "init_points": 1,
        "n_candidates": 128,
        "space": {"lambda": {"type": "float", "min": 0.0, "max": 1.0}},
        "initial_points": [{"lambda": 0.1}],
    }
    bo_targets = [0.31, 0.32, 0.33, 0.34]
    sample_sequence = _lambda_sequence(bo_targets)
    score_map = {_params_key({"lambda": 0.1}): 0.1}
    score_map.update({_params_key({"lambda": value}): value for value in bo_targets})

    summary, post_eval_calls = _run_search(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        search=search,
        sample_sequence=sample_sequence,
        score_by_params=score_map,
    )

    assert len(summary["runs"]) == 5
    assert summary["early_stopped"] is False
    assert summary["early_stop_reason"] is None
    assert len(post_eval_calls) == 1


def test_bayes_plateau_alone_does_not_stop_when_resolution_keeps_moving(tmp_path: Path, monkeypatch) -> None:
    search = {
        "type": "bayes",
        "budget": 5,
        "init_points": 1,
        "n_candidates": 128,
        "space": {"lambda": {"type": "float", "min": 0.0, "max": 1.0}},
        "initial_points": [{"lambda": 0.1}],
        "early_stop": {
            "enabled": True,
            "min_evals": 3,
            "patience": 3,
            "min_score_delta": 5e-4,
            "param_resolution": {"lambda": 0.001},
            "require_both": True,
        },
    }
    bo_targets = [0.301, 0.302, 0.303, 0.304]
    sample_sequence = _lambda_sequence(bo_targets)
    score_map = {_params_key({"lambda": 0.1}): 0.1}
    score_map.update({_params_key({"lambda": value}): 0.5 for value in bo_targets})

    summary, _ = _run_search(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        search=search,
        sample_sequence=sample_sequence,
        score_by_params=score_map,
    )

    assert len(summary["runs"]) == 5
    assert summary["early_stopped"] is False
    assert summary["early_stop_state"]["plateau_triggered"] is True
    assert summary["early_stop_state"]["resolution_triggered"] is False


def test_bayes_resolution_alone_does_not_stop_when_scores_keep_improving(tmp_path: Path, monkeypatch) -> None:
    search = {
        "type": "bayes",
        "budget": 5,
        "init_points": 1,
        "n_candidates": 128,
        "space": {"lambda": {"type": "float", "min": 0.0, "max": 1.0}},
        "initial_points": [{"lambda": 0.1}],
        "early_stop": {
            "enabled": True,
            "min_evals": 3,
            "patience": 3,
            "min_score_delta": 5e-4,
            "param_resolution": {"lambda": 0.001},
            "require_both": True,
        },
    }
    bo_targets = [0.3011, 0.3012, 0.3013, 0.3014]
    sample_sequence = _lambda_sequence(bo_targets)
    score_map = {_params_key({"lambda": 0.1}): 0.1}
    score_map.update(
        {
            _params_key({"lambda": 0.3011}): 0.2,
            _params_key({"lambda": 0.3012}): 0.2003,
            _params_key({"lambda": 0.3013}): 0.2008,
            _params_key({"lambda": 0.3014}): 0.2015,
        }
    )

    summary, _ = _run_search(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        search=search,
        sample_sequence=sample_sequence,
        score_by_params=score_map,
    )

    assert len(summary["runs"]) == 5
    assert summary["early_stopped"] is False
    assert summary["early_stop_state"]["plateau_triggered"] is False
    assert summary["early_stop_state"]["resolution_triggered"] is True


def test_bayes_combined_early_stop_records_metadata_and_runs_post_eval(tmp_path: Path, monkeypatch) -> None:
    search = {
        "type": "bayes",
        "budget": 8,
        "init_points": 1,
        "n_candidates": 128,
        "space": {"lambda": {"type": "float", "min": 0.0, "max": 1.0}},
        "initial_points": [{"lambda": 0.1}],
        "early_stop": {
            "enabled": True,
            "min_evals": 3,
            "patience": 3,
            "min_score_delta": 5e-4,
            "param_resolution": {"lambda": 0.001},
            "require_both": True,
        },
    }
    bo_targets = [0.3011, 0.3012, 0.3013, 0.45]
    sample_sequence = _lambda_sequence(bo_targets)
    score_map = {_params_key({"lambda": 0.1}): 0.1}
    score_map.update(
        {
            _params_key({"lambda": 0.3011}): 0.2,
            _params_key({"lambda": 0.3012}): 0.2001,
            _params_key({"lambda": 0.3013}): 0.2002,
            _params_key({"lambda": 0.45}): 0.8,
        }
    )

    summary, post_eval_calls = _run_search(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        search=search,
        sample_sequence=sample_sequence,
        score_by_params=score_map,
    )

    assert len(summary["runs"]) == 4
    assert summary["early_stopped"] is True
    assert summary["early_stop_reason"] == "score_plateau,param_resolution"
    assert summary["early_stop_state"]["evaluations"] == 3
    assert summary["early_stop_state"]["patience_used"] == 3
    assert summary["early_stop_state"]["resolution_triggered"] is True
    assert summary["early_stop_state"]["plateau_triggered"] is True
    assert summary["best_index"] == 3
    assert summary["runs"][summary["best_index"]]["params"] == {"lambda": 0.3013}
    assert len(post_eval_calls) == 1
    assert post_eval_calls[0]["best_params"] == {"lambda": 0.3013}


def test_bayes_integer_dimensions_do_not_trigger_resolution_stop(tmp_path: Path, monkeypatch) -> None:
    search = {
        "type": "bayes",
        "budget": 5,
        "init_points": 1,
        "n_candidates": 128,
        "space": {"steps": {"type": "int", "min": 0, "max": 1000}},
        "initial_points": [{"steps": 1}],
        "early_stop": {
            "enabled": True,
            "min_evals": 3,
            "patience": 3,
            "min_score_delta": 5e-4,
            "param_resolution": {"steps": 1},
            "require_both": True,
        },
    }
    bo_targets = [2, 3, 4, 5]
    sample_sequence = _int_sequence(bo_targets)
    score_map = {_params_key({"steps": 1}): 0.1}
    score_map.update({_params_key({"steps": value}): 0.5 for value in bo_targets})

    summary, _ = _run_search(
        tmp_path=tmp_path,
        monkeypatch=monkeypatch,
        search=search,
        sample_sequence=sample_sequence,
        score_by_params=score_map,
    )

    assert len(summary["runs"]) == 5
    assert summary["early_stopped"] is False
    assert summary["early_stop_state"]["plateau_triggered"] is True
    assert summary["early_stop_state"]["resolution_triggered"] is False
