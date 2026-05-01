from __future__ import annotations

from typing import Any, Dict, Iterable, List

import numpy as np
import pytest
import torch

from tasks.asr.dataset import OmniASRCollator
from tasks.emotion.dataset import EmotionRecognitionCollator
from tasks.intent.dataset import IntentClassificationCollator
from tasks.kws.dataset import KeywordSpottingCollator
from tasks.langid.dataset import LanguageIdentificationCollator
from tasks.speaker_id.dataset import SpeakerIdentificationCollator
from tasks.speaker_ver.dataset import SpeakerVerCollator
from tasks.speech_qa.dataset import SpeechQACollator
from tasks.st.dataset import STCollator
from tasks.vocalsound.dataset import VocalSoundCollator


class _CollatorTokenizer:
    def __init__(self) -> None:
        self.pad_token_id = 0
        self.eos_token_id = 0
        self.padding_side = "right"
        self._audio_token_id = 1
        self._char_offset = 10

    def convert_tokens_to_ids(self, token: str) -> int:
        return self._audio_token_id if token == "<audio>" else 2

    def encode(self, text: str, add_special_tokens: bool = False) -> List[int]:
        del add_special_tokens
        return [ord(ch) + self._char_offset for ch in str(text)]

    def decode(self, token_ids: Iterable[int], skip_special_tokens: bool = False) -> str:
        del skip_special_tokens
        values = []
        for token_id in token_ids:
            token = int(token_id)
            if token in {self.pad_token_id, self._audio_token_id}:
                continue
            values.append(chr(token - self._char_offset))
        return "".join(values)


class _CollatorProcessor:
    def __init__(self) -> None:
        self.tokenizer = _CollatorTokenizer()
        self.audio_token = "<audio>"
        self.prompts: List[str] = []
        self.conversations: List[List[Dict[str, Any]]] = []
        self.audio_lengths: List[int] = []

    def apply_chat_template(
        self,
        conversation: List[Dict[str, Any]],
        add_generation_prompt: bool,
        tokenize: bool,
    ) -> str:
        del add_generation_prompt, tokenize
        self.conversations.append(conversation)
        user_text = conversation[0]["content"][1]["text"]
        assistant_text = conversation[1]["content"][0]["text"]
        return f"user:{self.audio_token}\n{user_text}\nassistant\n{assistant_text}"

    def __call__(
        self,
        *,
        audio,
        sampling_rate: int,
        text,
        return_tensors: str,
        padding: bool,
    ) -> Dict[str, torch.Tensor]:
        del sampling_rate, return_tensors, padding
        self.prompts.extend(text)
        self.audio_lengths.extend(len(item) for item in audio)
        encoded = [self.tokenizer.encode(item, add_special_tokens=False) for item in text]
        encoded = [[self.tokenizer.convert_tokens_to_ids(self.audio_token)] + row for row in encoded]
        max_len = max(len(row) for row in encoded)
        padded = []
        for row in encoded:
            pad_count = max_len - len(row)
            if self.tokenizer.padding_side == "left":
                padded.append([self.tokenizer.pad_token_id] * pad_count + row)
            else:
                padded.append(row + [self.tokenizer.pad_token_id] * pad_count)
        return {"input_ids": torch.tensor(padded, dtype=torch.long)}


def _assert_labels_keep_exact_text(batch: Dict[str, torch.Tensor], processor: _CollatorProcessor, expected: List[str]) -> None:
    assert "input_ids" in batch
    assert "labels" in batch
    assert batch["input_ids"].shape == batch["labels"].shape
    for row, expected_text in enumerate(expected):
        kept = batch["labels"][row][batch["labels"][row] != -100].tolist()
        assert processor.tokenizer.decode(kept) == expected_text
    assert processor.tokenizer.padding_side == "left"


def _audio(length: int = 80) -> Dict[str, np.ndarray]:
    return {"array": np.linspace(0.0, 0.1, num=length, dtype=np.float32)}


@pytest.mark.parametrize(
    ("collator_cls", "features", "label_names", "expected_labels", "expected_prompt_bits"),
    [
        (
            EmotionRecognitionCollator,
            [{"audio": _audio(), "text": "hello there", "label": 1}],
            ["angry", "happy"],
            ["happy"],
            ["What emotion", "Transcript: hello there"],
        ),
        (
            IntentClassificationCollator,
            [{"audio": _audio(), "text": "turn on lights", "label": 0, "scenario": "home", "action": "activate"}],
            ["switch_on", "switch_off"],
            ["switch_on"],
            ["user's intent", "Scenario: home", "Action: activate", "Transcript: turn on lights"],
        ),
        (
            KeywordSpottingCollator,
            [{"audio": _audio(), "label": 1}],
            ["yes", "no"],
            ["no"],
            ["What word is spoken", "yes, no"],
        ),
        (
            LanguageIdentificationCollator,
            [{"audio": _audio(), "label": 0}],
            ["english", "german"],
            ["english"],
            ["What language", "english, german"],
        ),
        (
            SpeakerIdentificationCollator,
            [{"audio": _audio(120), "text": "speaker transcript", "label": 1}],
            ["1001", "1002"],
            ["1002"],
            ["Who is the speaker", "Transcript: speaker transcript"],
        ),
        (
            VocalSoundCollator,
            [{"audio": _audio(), "label": 0}],
            ["cough", "sneeze"],
            ["cough"],
            ["Identify the human vocal sound", "cough, sneeze"],
        ),
    ],
)
def test_classification_collators_build_prompts_and_mask_only_labels(
    collator_cls,
    features: List[Dict[str, Any]],
    label_names: List[str],
    expected_labels: List[str],
    expected_prompt_bits: List[str],
) -> None:
    processor = _CollatorProcessor()
    kwargs: Dict[str, Any] = {
        "processor": processor,
        "sampling_rate": 16000,
        "label_names": label_names,
    }
    if collator_cls is IntentClassificationCollator:
        kwargs["prepend_scenario"] = True
    if collator_cls is SpeakerIdentificationCollator:
        kwargs["include_transcript"] = True
        kwargs["max_audio_length"] = 0.004
    collator = collator_cls(**kwargs)

    batch = collator(features)

    _assert_labels_keep_exact_text(batch, processor, expected_labels)
    prompt = processor.prompts[0]
    for expected in expected_prompt_bits:
        assert expected in prompt
    if collator_cls is SpeakerIdentificationCollator:
        assert processor.audio_lengths == [64]


def test_asr_collator_builds_transcription_prompt_and_masks_transcript() -> None:
    processor = _CollatorProcessor()
    collator = OmniASRCollator(processor=processor, sampling_rate=16000)

    batch = collator([{"audio": _audio(), "text": "hello world"}])

    _assert_labels_keep_exact_text(batch, processor, ["hello world"])
    assert "Only output the transcription" in processor.prompts[0]


def test_st_collator_derives_language_prompt_and_masks_translation() -> None:
    processor = _CollatorProcessor()
    collator = STCollator(processor=processor, sampling_rate=16000, language_pair="en_de")

    batch = collator([{"audio": _audio(), "translation": "guten tag"}])

    _assert_labels_keep_exact_text(batch, processor, ["guten tag"])
    assert collator.instruction == "Translate this audio to German."
    assert "Translate this audio to German." in processor.prompts[0]


def test_speech_qa_collator_choice_toggle_and_label_selection() -> None:
    feature = {
        "audio": _audio(),
        "question": "What is the capital of France?",
        "answers": ["Paris"],
        "label_text": "A",
        "choice_a": "Paris",
        "choice_b": "London",
        "choice_c": "Berlin",
        "choice_d": "Madrid",
        "transcript": "question audio",
    }
    question_only = SpeechQACollator(
        processor=None,
        sampling_rate=16000,
        include_choices_in_prompt=False,
    )._build_instruction(feature)
    with_choices = SpeechQACollator(
        processor=None,
        sampling_rate=16000,
        include_choices_in_prompt=True,
    )._build_instruction(feature)

    assert "Question: What is the capital of France?" in question_only
    assert "Choices:" not in question_only
    assert "Choices:" in with_choices
    assert "A. Paris" in with_choices
    assert "D. Madrid" in with_choices
    assert "Respond with exactly one uppercase letter: A, B, C, or D." in with_choices
    assert "Do not output any other text." in with_choices

    processor = _CollatorProcessor()
    collator = SpeechQACollator(
        processor=processor,
        sampling_rate=16000,
        include_choices_in_prompt=True,
        include_transcript=True,
    )
    batch = collator([feature])

    _assert_labels_keep_exact_text(batch, processor, ["A"])
    assert "Transcript: question audio" in processor.prompts[0]


def test_speaker_ver_collator_concatenates_pair_audio_and_masks_binary_label() -> None:
    processor = _CollatorProcessor()
    collator = SpeakerVerCollator(
        processor=processor,
        sampling_rate=16000,
        label_names=["no", "yes"],
        max_audio_length=0.002,
        audio_gap_seconds=0.001,
    )

    batch = collator(
        [
            {
                "audio_a": {"array": np.ones(80, dtype=np.float32)},
                "audio_b": {"array": np.ones(90, dtype=np.float64)},
                "label": 1,
            }
        ]
    )

    _assert_labels_keep_exact_text(batch, processor, ["yes"])
    assert processor.audio_lengths == [80]
    assert "Do they belong to the same speaker" in processor.prompts[0]
