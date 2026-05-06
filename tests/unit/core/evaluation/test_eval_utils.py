from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

from datasets import Dataset

from core.evaluation import eval_utils
from tests.helpers.core import FakeEvalSetup


class _FakeTrainingArguments:
    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.output_dir = kwargs["output_dir"]


class _FakeCustomTrainer:
    instances = []

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        self.args = kwargs["args"]
        self.instances.append(self)

    def evaluate(self):
        return {"eval_accuracy": 1.0}


def test_run_evaluation_default_output_dir_uses_system_temp(
    monkeypatch,
    tmp_path: Path,
) -> None:
    temp_root = tmp_path / "system_tmp"
    monkeypatch.setattr(eval_utils.tempfile, "gettempdir", lambda: str(temp_root))
    monkeypatch.setattr(eval_utils, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(eval_utils, "CustomTrainer", _FakeCustomTrainer)
    _FakeCustomTrainer.instances.clear()

    metrics = eval_utils.run_evaluation(
        model=SimpleNamespace(),
        setup=FakeEvalSetup(Dataset.from_list([{"x": 1}])),
        processor=None,
    )

    assert metrics == {"accuracy": 1.0}
    output_dir = Path(_FakeCustomTrainer.instances[-1].args.output_dir)
    assert output_dir == temp_root / "speech_merging_eval" / f"pid_{eval_utils.os.getpid()}"
    assert output_dir.exists()
    assert output_dir != Path("runs/eval")


def test_run_evaluation_preserves_explicit_output_dir(
    monkeypatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setattr(eval_utils, "TrainingArguments", _FakeTrainingArguments)
    monkeypatch.setattr(eval_utils, "CustomTrainer", _FakeCustomTrainer)
    _FakeCustomTrainer.instances.clear()
    explicit_output_dir = tmp_path / "custom_eval"

    eval_utils.run_evaluation(
        model=SimpleNamespace(),
        setup=FakeEvalSetup(Dataset.from_list([{"x": 1}])),
        output_dir=explicit_output_dir,
        processor=None,
    )

    output_dir = Path(_FakeCustomTrainer.instances[-1].args.output_dir)
    assert output_dir == explicit_output_dir
    assert output_dir.exists()
