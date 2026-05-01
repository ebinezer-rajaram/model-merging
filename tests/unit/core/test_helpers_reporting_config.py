from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
import torch
from datasets import Dataset

from core.config.loader import _deep_merge, _load_yaml_config, get_artifact_directories, get_config_path
from core.config.registry import get_task_info, list_tasks
from core.data.io_utils import dump_json, ensure_dir, load_config
from core.data.dataset_utils import (
    build_manifest,
    build_split_metadata,
    compute_speaker_stats,
    filter_dataset_columns,
    load_cached_split,
    normalize_audio,
    normalize_split_metadata,
    save_cached_split,
    subset_dataset_by_metadata,
)
from core.evaluation.metrics import compute_wer_from_texts, decode_tokens, sanitize_token_array
from core.output import summary_writer
from core.tasks.collator import build_strict_label_mask
from core.tasks.config import BaseTaskConfig, create_simple_task_config
from core.training.samplers import BalancedBatchSampler, WeightedClassSampler
from core.training.losses import FocalLoss, WeightedCrossEntropyLoss, compute_class_weights
from core.training.training_config import build_early_stopping_kwargs, parse_training_config
from core.utils.logger import setup_logger
from core.utils.seed_utils import set_global_seed


def test_config_and_io_helpers_merge_and_write(tmp_path: Path) -> None:
    base = tmp_path / "base.yaml"
    override = tmp_path / "override.yaml"
    base.write_text("a: 1\nnested:\n  x: 2\n  y: 3\n", encoding="utf-8")
    override.write_text("nested:\n  y: 9\nb: 4\n", encoding="utf-8")

    assert _deep_merge({"a": {"x": 1}}, {"a": {"y": 2}}) == {"a": {"x": 1, "y": 2}}
    assert _load_yaml_config(override, base)["nested"] == {"x": 2, "y": 9}
    assert load_config(override, base)["b"] == 4

    out_dir = ensure_dir(tmp_path / "created")
    assert out_dir.exists()
    json_path = out_dir / "payload.json"
    dump_json({"z": 1}, json_path)
    assert json.loads(json_path.read_text(encoding="utf-8")) == {"z": 1}

    assert get_config_path("asr").name == "asr.yaml"
    assert get_artifact_directories("asr", tmp_path)["adapters"] == tmp_path / "adapters"
    assert get_task_info("asr").default_adapter_subdir
    assert "asr" in list_tasks()
    with pytest.raises(ValueError, match="Unknown task"):
        get_task_info("missing")


def test_dataset_helpers_cover_manifests_cache_audio_and_columns(tmp_path: Path) -> None:
    ds = Dataset.from_list(
        [
            {"id": "a", "duration": 1.0, "speaker": "s1", "audio": {"array": [1.0, -1.0], "sampling_rate": 2}},
            {"id": "b", "duration": 2.0, "speaker": "s1", "audio": {"array": [0.5, -0.5], "sampling_rate": 2}},
        ]
    )

    stats = compute_speaker_stats(ds, speaker_column="speaker")
    assert stats["s1"]["count"] == 2
    normalized = normalize_audio({"audio": {"array": np.array([0.5, -0.5]), "sampling_rate": 2}})
    assert np.max(np.abs(normalized["audio"]["array"])) <= 1.0

    manifest = build_manifest(ds, [1, 0], fields=["id"], include_index=True)
    assert manifest[0]["id"] == "b"
    assert manifest[0]["index"] == 1
    metadata = build_split_metadata(ds, [0, 1], manifest_fields=["id"])
    assert metadata["hours"] == pytest.approx(3.0 / 3600.0)
    assert len(subset_dataset_by_metadata(ds, {"indices": [1]})) == 1
    assert len(subset_dataset_by_metadata(ds, {"indices": []})) == 0
    assert normalize_split_metadata({"train": {"indices": ["1"], "hours": "2.5"}})["train"]["indices"] == [1]

    cache_path = tmp_path / "cache" / "split.json"
    assert load_cached_split(cache_path) is None
    save_cached_split(cache_path, {"indices": [0]})
    assert load_cached_split(cache_path) == {"indices": [0]}
    assert filter_dataset_columns(ds, ["id"], always_keep=["duration"]).column_names == ["id", "duration"]


def test_metrics_collator_training_seed_sampler_and_logger(tmp_path: Path) -> None:
    assert sanitize_token_array([1, -100, 3], pad_id=0) == [1, 0, 3]

    class FallbackTokenizer:
        pad_token_id = 0
        eos_token_id = 0
        unk_token_id = 99
        vocab_size = 10
        all_special_tokens = ["<pad>"]

        def decode(self, tokens, skip_special_tokens=True):
            raise TypeError("fallback")

        def convert_ids_to_tokens(self, tokens):
            return ["A" if int(tok) == 1 else "<pad>" for tok in tokens]

        def convert_tokens_to_string(self, tokens):
            return " ".join(tokens)

    assert decode_tokens([1, 999], FallbackTokenizer()) == "A"
    assert compute_wer_from_texts(["Hello, world"], ["hello world"], normalization="aggressive") == 0.0

    input_ids = torch.tensor([5, 9, 8, 9, 8, 0])
    labels = input_ids.clone()
    stats = build_strict_label_mask(input_ids=input_ids, labels=labels, label_tokens=[9, 8])
    assert stats["matched"] is True
    assert torch.equal(labels, torch.tensor([-100, -100, -100, 9, 8, -100]))

    fallback_labels = torch.tensor([-100, 4, 5, 6])
    fallback = build_strict_label_mask(
        input_ids=torch.tensor([1, 2, 3, 4]),
        labels=fallback_labels,
        label_tokens=[8, 9],
    )
    assert fallback["used_fallback"] is True
    assert torch.equal(fallback_labels, torch.tensor([-100, -100, 3, 4]))

    cfg = parse_training_config(
        {"learning_rate": "0.001", "max_steps": 10, "warmup_ratio": 0.2, "early_stopping_threshold": 0.01},
        num_train_examples=100,
    )
    assert cfg.warmup_steps == 2
    assert build_early_stopping_kwargs(cfg)["early_stopping_threshold"] == 0.01
    with pytest.raises(ValueError, match="learning_rate"):
        parse_training_config({"learning_rate": "bad"}, num_train_examples=1)

    rows = [{"label": 0}, {"label": 0}, {"label": 1}, {"label": 1}]
    balanced = BalancedBatchSampler(rows, batch_size=2, num_classes=2, shuffle=False)
    assert len(next(iter(balanced))) == 2
    weighted = WeightedClassSampler(rows, num_samples=3, method="balanced")
    assert len(list(iter(weighted))) == 3
    with pytest.raises(ValueError, match="Unknown weighting method"):
        WeightedClassSampler(rows, method="bad")

    set_global_seed(123)
    first = torch.rand(1).item()
    set_global_seed(123)
    assert torch.rand(1).item() == pytest.approx(first)

    logger = setup_logger("unit-test-logger", tmp_path / "test.log")
    logger.info("hello")
    assert logger is setup_logger("unit-test-logger", tmp_path / "ignored.log")


def test_loss_helpers_cover_alpha_reductions_and_shape_errors() -> None:
    logits = torch.tensor([[2.0, 0.1], [0.1, 2.0], [1.0, 1.0]])
    targets = torch.tensor([0, 1, -100])

    per_class = FocalLoss(alpha=[1.0, 2.0], gamma=1.0, reduction="none")(logits, targets)
    assert per_class.shape == targets.shape
    assert per_class[-1] == 0.0
    summed = FocalLoss(alpha=0.5, gamma=1.0, reduction="sum")(logits, targets)
    assert summed.item() > 0.0
    with pytest.raises(ValueError, match="Expected inputs"):
        FocalLoss()(torch.ones(1, 1, 1, 1), targets)
    with pytest.raises(ValueError, match="Shape mismatch"):
        FocalLoss()(torch.ones(2, 2), torch.ones(3, dtype=torch.long))

    sequence_logits = torch.tensor([[[2.0, 0.1], [0.1, 2.0]]])
    sequence_targets = torch.tensor([[0, 1]])
    loss = WeightedCrossEntropyLoss(torch.tensor([1.0, 2.0]), reduction="sum")(sequence_logits, sequence_targets)
    assert loss.item() > 0.0
    with pytest.raises(ValueError, match="Expected inputs"):
        WeightedCrossEntropyLoss([1.0, 1.0])(torch.ones(1, 1, 1, 1), sequence_targets)

    labels = torch.tensor([0, 0, 1])
    assert compute_class_weights(labels, 3, method="sqrt_inverse")[1] > compute_class_weights(labels, 3, method="sqrt_inverse")[0]
    assert torch.equal(compute_class_weights(labels, 3, method="balanced"), compute_class_weights(labels, 3, method="inverse"))


def test_task_config_factory_and_summary_writer(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    with pytest.raises(NotImplementedError):
        BaseTaskConfig.get_config_path(tmp_path)

    task_name, default_file, get_path, get_dirs = create_simple_task_config("demo", "demo.yaml")
    assert task_name == "demo"
    assert default_file == "demo.yaml"
    assert get_path(tmp_path) == tmp_path / "configs" / "demo.yaml"
    assert get_dirs(tmp_path)["base"] == tmp_path / "artifacts" / "demo"

    assert summary_writer._clean_metrics({"eval_accuracy": 0.5, "runtime": 1.0}) == {"accuracy": 0.5}
    assert summary_writer._order_splits(["z", "validation", "test"]) == ["test", "validation", "z"]
    assert summary_writer.build_selection("best", "accuracy", 0.9, "validation")["policy"] == "best"
    assert summary_writer.build_hyperparameters(learning_rate=1e-4, num_tasks=2)["num_tasks"] == 2
    assert summary_writer._extract_delta_info(
        {"interference_delta_meta": {"metric": "accuracy", "base": 0.1, "task_adapter": 0.2, "merged": 0.3}},
        {},
    ) == {"metric": "accuracy", "base": 0.1, "task_adapter": 0.2, "merged": 0.3}

    analysis_mod = types.ModuleType("analysis")
    collect_mod = types.ModuleType("analysis.collect")
    utils_mod = types.ModuleType("analysis.collect.utils")
    utils_mod.derive_eval_context = lambda task, source_tasks, eval_tag: f"{task}:{eval_tag or 'none'}"
    monkeypatch.setitem(sys.modules, "analysis", analysis_mod)
    monkeypatch.setitem(sys.modules, "analysis.collect", collect_mod)
    monkeypatch.setitem(sys.modules, "analysis.collect.utils", utils_mod)

    output = tmp_path / "experiment_summary.json"
    summary_writer.write_experiment_summary(
        output_path=output,
        experiment_type="merge",
        run_id="run1",
        timestamp="2026-01-01T00:00:00",
        config_name="cfg",
        source_tasks=["asr"],
        method="weighted",
        results={"asr": {"test": {"eval_wer": 0.1, "eval_tag": "heldout"}}},
    )
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == "2"
    assert payload["results"]["asr"]["test"]["wer"] == 0.1
