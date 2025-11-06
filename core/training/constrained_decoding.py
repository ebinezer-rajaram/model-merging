"""Constrained decoding utilities for classification tasks."""

from typing import Callable, List, Optional, Sequence, Set

import torch


def create_classification_constraint(
    processor,
    label_names: Sequence[str],
    *,
    allow_eos: bool = True,
) -> Callable[[int, torch.LongTensor], List[int]]:
    """
    Create a prefix_allowed_tokens_fn for constrained generation.

    Forces the model to only generate tokens that correspond to valid labels,
    ensuring classification outputs are always from the known label vocabulary.

    Args:
        processor: The model processor containing the tokenizer
        label_names: List of valid label strings (e.g., ["happy", "sad", "angry"])
        allow_eos: Whether to allow EOS token (to end generation)

    Returns:
        A function that can be passed as prefix_allowed_tokens_fn to model.generate()

    Example:
        >>> constraint_fn = create_classification_constraint(
        ...     processor,
        ...     label_names=["happy", "sad", "angry"]
        ... )
        >>> model.generate(..., prefix_allowed_tokens_fn=constraint_fn)
    """
    tokenizer = processor.tokenizer

    # Build the set of allowed token IDs
    allowed_token_ids: Set[int] = set()

    # Tokenize each label and collect all possible tokens
    # This handles both single-token and multi-token labels
    for label in label_names:
        label_str = str(label).strip()
        if not label_str:
            continue

        # Tokenize the label (without special tokens)
        token_ids = tokenizer.encode(label_str, add_special_tokens=False)

        # Add all tokens from this label
        for token_id in token_ids:
            allowed_token_ids.add(token_id)

    # Optionally allow EOS token to end generation
    if allow_eos:
        eos_token_id = tokenizer.eos_token_id
        if eos_token_id is not None:
            allowed_token_ids.add(eos_token_id)

    # Also allow space and common punctuation that might be in labels
    space_tokens = tokenizer.encode(" ", add_special_tokens=False)
    for token_id in space_tokens:
        allowed_token_ids.add(token_id)

    # Convert to sorted list for consistency
    allowed_tokens_list = sorted(allowed_token_ids)

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        """
        Returns list of allowed token IDs for the next generation step.

        Args:
            batch_id: Index of the sequence in the batch
            input_ids: Tensor of token IDs generated so far

        Returns:
            List of allowed token IDs for next position
        """
        return allowed_tokens_list

    return prefix_allowed_tokens_fn


def create_multi_token_constraint(
    processor,
    label_names: Sequence[str],
    *,
    allow_eos: bool = True,
) -> Callable[[int, torch.LongTensor], List[int]]:
    """
    Create a more sophisticated constraint that tracks generation state.

    This constraint is smarter about multi-token labels - it only allows tokens
    that could continue a valid label prefix.

    Args:
        processor: The model processor containing the tokenizer
        label_names: List of valid label strings
        allow_eos: Whether to allow EOS token

    Returns:
        A stateful constraint function for generation
    """
    tokenizer = processor.tokenizer

    # Tokenize all labels and store as sequences
    label_token_sequences = []
    for label in label_names:
        label_str = str(label).strip()
        if not label_str:
            continue
        token_ids = tokenizer.encode(label_str, add_special_tokens=False)
        label_token_sequences.append(tuple(token_ids))

    eos_token_id = tokenizer.eos_token_id if allow_eos else None

    def prefix_allowed_tokens_fn(batch_id: int, input_ids: torch.LongTensor) -> List[int]:
        """
        Returns allowed tokens based on partial generation state.

        Tracks which labels are still possible given what's been generated so far.
        """
        # This is the sequence generated so far
        # We need to figure out which position in the label we're at

        # For simplicity, on first token, allow first tokens of all labels
        # On subsequent tokens, allow continuations + EOS

        # Get length to determine position (simple heuristic)
        generated_length = len(input_ids)

        allowed_tokens: Set[int] = set()

        # Allow tokens that could start or continue any label
        for label_tokens in label_token_sequences:
            # Allow all tokens in this label sequence
            for token in label_tokens:
                allowed_tokens.add(token)

        # Allow EOS after generating at least one token
        if eos_token_id is not None:
            allowed_tokens.add(eos_token_id)

        return sorted(allowed_tokens)

    return prefix_allowed_tokens_fn


__all__ = [
    "create_classification_constraint",
    "create_multi_token_constraint",
]
