"""Shared metric helper functions."""

from typing import Iterable, List, Literal, Sequence

import numpy as np
from jiwer import wer, wer_default, wer_standardize


def sanitize_token_array(array: Sequence[int], pad_id: int) -> List[int]:
    """Clamp invalid token ids to pad value."""
    arr = np.asarray(list(array))
    arr = np.where((arr < 0) | ~np.isfinite(arr), pad_id, arr)
    return arr.astype(np.int64).tolist()


def decode_tokens(batch: Sequence[int], tokenizer) -> str:
    """Decode a token batch into text."""
    pad_id = tokenizer.pad_token_id or tokenizer.eos_token_id or 0
    tokens = sanitize_token_array(batch, pad_id)

    vocab_size = getattr(tokenizer, "vocab_size", None)
    if vocab_size is not None:
        unk_id = tokenizer.unk_token_id if tokenizer.unk_token_id is not None else pad_id
        tokens = [
            token if 0 <= int(token) < vocab_size else unk_id
            for token in tokens
        ]

    try:
        return tokenizer.decode(tokens, skip_special_tokens=True)
    except TypeError:
        token_strs = tokenizer.convert_ids_to_tokens(tokens)
        token_strs = [tok for tok in token_strs if isinstance(tok, str)]
        if not token_strs:
            return ""
        if getattr(tokenizer, "all_special_tokens", None):
            token_strs = [tok for tok in token_strs if tok not in tokenizer.all_special_tokens]
        return tokenizer.convert_tokens_to_string(token_strs)


def compute_wer_from_texts(
    reference_texts: Iterable[str],
    predicted_texts: Iterable[str],
    normalization: Literal["default", "standardize"] = "default",
) -> float:
    """Compute WER score given reference and predicted texts.

    Args:
        reference_texts: Ground truth texts
        predicted_texts: Model predicted texts
        normalization: Normalization mode
            - 'default': Minimal normalization (remove extra spaces, strip whitespace)
            - 'standardize': Aggressive normalization (lowercase, remove punctuation,
                           expand contractions, remove Kaldi non-words)

    Returns:
        Word error rate as a float
    """
    transformation = wer_standardize if normalization == "standardize" else wer_default
    return wer(list(reference_texts), list(predicted_texts), truth_transform=transformation, hypothesis_transform=transformation)
