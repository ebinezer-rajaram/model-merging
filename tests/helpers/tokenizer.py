from __future__ import annotations

from numbers import Integral
from typing import Iterable


class DummyTokenizer:
    def __init__(self, token_map: dict[int, str] | None = None, *, pad_token_id: int = 0) -> None:
        self._token_map = token_map or {}
        self.pad_token_id = pad_token_id
        self.eos_token_id = pad_token_id

    def decode(self, token_ids, skip_special_tokens: bool = True) -> str:
        del skip_special_tokens
        if isinstance(token_ids, Integral):
            token_ids = [token_ids]
        return "".join(self._token_map.get(int(token_id), "") for token_id in token_ids)

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        return [self.decode(sequence, skip_special_tokens=skip_special_tokens) for sequence in sequences]

    def convert_ids_to_tokens(self, token_ids, skip_special_tokens: bool = True):
        del skip_special_tokens
        if isinstance(token_ids, Integral):
            token_ids = [token_ids]
        return [self._token_map.get(int(token_id), "") for token_id in token_ids]

    def convert_tokens_to_string(self, tokens: Iterable[str]) -> str:
        return "".join(tokens)


class DummyProcessor:
    def __init__(self, token_map: dict[int, str] | None = None, *, pad_token_id: int = 0) -> None:
        self.tokenizer = DummyTokenizer(token_map, pad_token_id=pad_token_id)

    def batch_decode(self, sequences, skip_special_tokens: bool = True):
        return self.tokenizer.batch_decode(sequences, skip_special_tokens=skip_special_tokens)
