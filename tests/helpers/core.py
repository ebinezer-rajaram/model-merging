from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Iterable, Mapping, Sequence

import torch
from datasets import Dataset, DatasetDict


class CoreDummyTokenizer:
    def __init__(self, token_map: Mapping[int, str] | None = None, *, pad_token_id: int = 0) -> None:
        self._token_map = dict(token_map or {})
        self._reverse = {value: key for key, value in self._token_map.items() if value}
        self.pad_token_id = pad_token_id
        self.eos_token_id = 2
        self.bos_token_id = 1
        self.unk_token_id = 3
        self.vocab_size = 4096
        self.padding_side = "right"
        self.all_special_tokens = ["<pad>"]

    def _id_for_text(self, text: str) -> int:
        if text in self._reverse:
            return int(self._reverse[text])
        token_id = sum(ord(ch) for ch in text) % 3000 + 10
        self._token_map[token_id] = text
        self._reverse[text] = token_id
        return token_id

    def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
        del add_special_tokens
        if text == "":
            return []
        if text in self._reverse:
            return [self._reverse[text]]
        return [self._id_for_text(part) for part in str(text).split()]

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return " ".join(self._token_map.get(int(token_id), "") for token_id in token_ids).strip()

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens) for sequence in sequences]

    def convert_ids_to_tokens(self, token_ids, skip_special_tokens: bool = True) -> list[str]:
        del skip_special_tokens
        if isinstance(token_ids, int):
            token_ids = [token_ids]
        return [self._token_map.get(int(token_id), "<unk>") for token_id in token_ids]

    def convert_tokens_to_string(self, tokens: Iterable[str]) -> str:
        return " ".join(str(token) for token in tokens).strip()

    def convert_tokens_to_ids(self, token: str) -> int:
        if token == "<audio>":
            return 999
        if token == "\n":
            return 10
        if token == "<|im_end|>":
            return 11
        return self._id_for_text(token)


class CoreDummyProcessor:
    audio_token = "<audio>"

    def __init__(self, token_map: Mapping[int, str] | None = None) -> None:
        self.tokenizer = CoreDummyTokenizer(token_map)
        self.feature_extractor = SimpleNamespace(sampling_rate=16000)
        self.conversations: list[list[dict[str, Any]]] = []
        self.prompts: list[str] = []
        self.calls: list[dict[str, Any]] = []

    def apply_chat_template(self, conversation, add_generation_prompt: bool = False, tokenize: bool = False):
        del add_generation_prompt, tokenize
        self.conversations.append(conversation)
        user_text = conversation[0]["content"][1]["text"]
        label_text = conversation[1]["content"][0]["text"]
        prompt = f"user <audio> {user_text} assistant {label_text}"
        self.prompts.append(prompt)
        return prompt

    def __call__(self, *, audio, sampling_rate: int, text: Sequence[str], return_tensors: str, padding: bool):
        del padding
        assert return_tensors == "pt"
        self.calls.append({"audio": audio, "sampling_rate": sampling_rate, "text": list(text)})
        encoded = []
        audio_id = self.tokenizer.convert_tokens_to_ids(self.audio_token)
        for prompt in text:
            ids = []
            for part in prompt.split():
                ids.append(audio_id if part == "<audio>" else self.tokenizer._id_for_text(part))
            encoded.append(ids)
        max_len = max(len(row) for row in encoded)
        padded = []
        masks = []
        for row in encoded:
            pad_len = max_len - len(row)
            padded.append([self.tokenizer.pad_token_id] * pad_len + row)
            masks.append([0] * pad_len + [1] * len(row))
        return {
            "input_ids": torch.tensor(padded, dtype=torch.long),
            "attention_mask": torch.tensor(masks, dtype=torch.long),
        }

    def batch_decode(self, sequences, skip_special_tokens: bool = True) -> list[str]:
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)


class TinyLabeledDataset:
    def __init__(self, labels: Sequence[int]) -> None:
        self.rows = [{"label": int(label)} for label in labels]

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, int]:
        return dict(self.rows[index])


@dataclass
class FakeTrainerState:
    log_history: list[dict[str, Any]] = field(default_factory=list)


@dataclass
class FakeTrainerForCsv:
    state: FakeTrainerState


@dataclass
class FakeEvalSetup:
    dataset: Any
    data_collator: Any = None
    compute_metrics: Any = None
    applied_indices: list[int] | None = None

    def apply_subset_indices(self, indices: Sequence[int]) -> None:
        self.applied_indices = [int(index) for index in indices]


class TinyParameterModel(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = torch.nn.Module()
        self.model.linear = torch.nn.Linear(2, 2, bias=False)
        self.config = SimpleNamespace(pad_token_id=None, eos_token_id=None, bos_token_id=None)
        self.generation_config = SimpleNamespace(pad_token_id=None, eos_token_id=None, bos_token_id=None)


def audio_dataset_dict() -> DatasetDict:
    return DatasetDict(
        {
            "train": Dataset.from_list(
                [
                    {"speech": {"array": [0.0, 1.0], "sampling_rate": 2}, "intent": "book", "sentence": "book it"},
                    {"speech": {"array": [0.0, 1.0], "sampling_rate": 2}, "intent": "play", "sentence": "play it"},
                    {"speech": {"array": [0.0, 1.0], "sampling_rate": 2}, "intent": "book", "sentence": "book again"},
                    {"speech": {"array": [0.0, 1.0], "sampling_rate": 2}, "intent": "play", "sentence": "play again"},
                ]
            ),
            "validation": Dataset.from_list(
                [{"speech": {"array": [0.0], "sampling_rate": 1}, "intent": "book", "sentence": "validate"}]
            ),
            "test": Dataset.from_list(
                [{"speech": {"array": [0.0], "sampling_rate": 1}, "intent": "play", "sentence": "test"}]
            ),
        }
    )


def complete_checkpoint(path: Path, *, step: int = 1) -> Path:
    checkpoint = path / f"checkpoint-{step}"
    checkpoint.mkdir(parents=True)
    for name in ("trainer_state.json", "optimizer.pt", "scheduler.pt", "adapter_model.safetensors"):
        (checkpoint / name).write_text("{}", encoding="utf-8")
    return checkpoint
