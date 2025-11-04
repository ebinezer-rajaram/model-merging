"""Shared metric helper functions."""

from typing import Iterable, List, Sequence

import numpy as np
from jiwer import wer


def sanitize_token_array(array: Sequence[int], pad_id: int) -> List[int]:
    """Clamp invalid token ids to pad value."""
    arr = np.asarray(list(array))
    arr = np.where((arr < 0) | ~np.isfinite(arr), pad_id, arr)
    return arr.astype(np.int64).tolist()


def decode_tokens(batch: Sequence[int], tokenizer) -> str:
    """Decode a token batch into text."""
    tokens = sanitize_token_array(batch, tokenizer.pad_token_id or tokenizer.eos_token_id or 0)
    return tokenizer.decode(tokens, skip_special_tokens=True)


def compute_wer_from_texts(reference_texts: Iterable[str], predicted_texts: Iterable[str]) -> float:
    """Compute WER score given reference and predicted texts."""
    return wer(list(reference_texts), list(predicted_texts))
