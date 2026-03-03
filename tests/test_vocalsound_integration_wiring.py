from __future__ import annotations

from dataclasses import dataclass, field

from datasets import Dataset

from core.config.loader import load_task_config
from core.config.registry import TASK_REGISTRY
from core.config.schemas import TaskConfig
from core.evaluation.eval_utils import get_registered_eval_tasks, prepare_task_for_evaluation
from core.training.train_task import CLASSIFICATION_TASK_SPECS
from merging.config.unified import load_merge_config
from merging.evaluation.interference import TASK_METRICS
from merging.runtime.utils import TASK_REGISTRY as MERGE_TASK_REGISTRY


def test_vocalsound_present_in_task_and_merge_registries() -> None:
    assert "vocalsound" in TASK_REGISTRY
    assert "vocalsound" in MERGE_TASK_REGISTRY
    assert "vocalsound" in TASK_METRICS
    assert "vocalsound" in CLASSIFICATION_TASK_SPECS
    assert "vocalsound" in get_registered_eval_tasks()


def test_vocalsound_task_config_and_merge_presets_parse() -> None:
    task_cfg = load_task_config("vocalsound", "vocalsound.yaml", validate=False)
    assert task_cfg["task"] == "vocalsound"
    schema_cfg = TaskConfig(task="vocalsound", artifacts={"adapter_subdir": "tmp"})
    assert schema_cfg.task == "vocalsound"

    supermerge_cfg = load_merge_config(
        "configs/merge/merge_supermerge_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml"
    )
    assert "vocalsound" in supermerge_cfg.adapters
    assert supermerge_cfg.eval_tasks is not None
    assert "vocalsound" in supermerge_cfg.eval_tasks

    scalar_cfg = load_merge_config(
        "configs/merge/merge_uniform_scalar_delta_emotion_intent_kws_langid_speaker_ver_asr_vocalsound.yaml"
    )
    assert "vocalsound" in scalar_cfg.adapters
    assert scalar_cfg.eval_tasks is not None
    assert "vocalsound" in scalar_cfg.eval_tasks


@dataclass
class _DummyTokenizer:
    padding_side: str = "right"
    pad_token_id: int = 0


@dataclass
class _DummyFeatureExtractor:
    sampling_rate: int = 16000


@dataclass
class _DummyProcessor:
    tokenizer: _DummyTokenizer = field(default_factory=_DummyTokenizer)
    feature_extractor: _DummyFeatureExtractor = field(default_factory=_DummyFeatureExtractor)


def test_prepare_task_for_evaluation_resolves_vocalsound_builder(monkeypatch) -> None:
    def _fake_loader(**kwargs):
        del kwargs
        ds = Dataset.from_list(
            [{"audio": "dummy.wav", "label": 0, "duration": 0.1, "label_name": "cough", "audio_path": "dummy.wav"}]
        )
        return ds, ds, ds, ["cough", "sneeze", "throat clearing", "sniff", "laugh", "exhale"]

    monkeypatch.setattr("tasks.vocalsound.load_vocalsound_dataset", _fake_loader)

    setup = prepare_task_for_evaluation(
        "vocalsound",
        _DummyProcessor(),
        split="validation",
        config={"dataset": {}},
    )

    assert len(setup.dataset) == 1
    assert setup.label_names is not None
    assert "cough" in setup.label_names
