"""Model loading and adaptation utilities."""

from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple
import logging
import warnings

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

DEFAULT_LORA_TARGET_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "out_proj",
    "fc1",
    "fc2",
]


_SUPPRESSED_LOG_SNIPPETS = (
    "System prompt modified, audio output may not work as expected.",
    "Audio output mode only works when using default system prompt",
)


class _QwenAudioOutputWarningFilter(logging.Filter):
    """Suppress known non-actionable Qwen audio-output warning spam."""

    def filter(self, record: logging.LogRecord) -> bool:
        message = record.getMessage()
        return not any(snippet in message for snippet in _SUPPRESSED_LOG_SNIPPETS)


_LOG_FILTER_INSTALLED = False


def _install_qwen_log_filter_once() -> None:
    global _LOG_FILTER_INSTALLED
    if _LOG_FILTER_INSTALLED:
        return
    logging.getLogger().addFilter(_QwenAudioOutputWarningFilter())
    _LOG_FILTER_INSTALLED = True


def create_lora_config_from_dict(lora_dict: Dict[str, Any]) -> LoraConfig:
    """
    Create a LoraConfig object from a dictionary (e.g., from YAML config).

    Args:
        lora_dict: Dictionary containing LoRA configuration parameters

    Returns:
        LoraConfig object

    Example:
        >>> lora_dict = {
        ...     "r": 32,
        ...     "alpha": 64,
        ...     "dropout": 0.1,
        ...     "target_modules": ["q_proj", "k_proj"]
        ... }
        >>> config = create_lora_config_from_dict(lora_dict)
    """
    # Map common alternative keys
    params = {}
    params["r"] = lora_dict.get("r", 32)
    params["lora_alpha"] = lora_dict.get("alpha", lora_dict.get("lora_alpha", 64))
    params["lora_dropout"] = lora_dict.get("dropout", lora_dict.get("lora_dropout", 0.1))
    params["bias"] = lora_dict.get("bias", "none")
    params["task_type"] = lora_dict.get("task_type", "CAUSAL_LM")
    params["target_modules"] = lora_dict.get("target_modules", DEFAULT_LORA_TARGET_MODULES)

    return LoraConfig(**params)


def _configure_special_tokens(model, processor) -> None:
    """Align special token ids between model and processor."""
    tokenizer = processor.tokenizer
    model.config.pad_token_id = tokenizer.pad_token_id
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id

    if hasattr(model, "generation_config"):
        model.generation_config.pad_token_id = tokenizer.pad_token_id
        model.generation_config.eos_token_id = tokenizer.eos_token_id
        model.generation_config.bos_token_id = tokenizer.bos_token_id

    if hasattr(model.config, "use_cache"):
        model.config.use_cache = False


def load_qwen_model(
    model_path: Path,
    *,
    torch_dtype: Optional[torch.dtype] = None,
    device_map: Optional[str] = "auto",
    use_fast_tokenizer: bool = False,
    apply_lora: bool = True,
    lora_config: Optional[LoraConfig] = None,
    lora_r: int = 32,
    lora_alpha: int = 64,
    lora_dropout: float = 0.1,
    lora_bias: str = "none",
    lora_target_modules: Optional[Sequence[str]] = None,
    lora_task_type: str = "CAUSAL_LM",
    print_trainable_parameters: bool = True,
) -> Tuple[
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
]:
    """Load a Qwen Omni model and optionally attach a LoRA adapter."""
    _install_qwen_log_filter_once()
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), use_fast=use_fast_tokenizer)

    tokenizer = getattr(processor, "tokenizer", None)
    if tokenizer is not None and getattr(tokenizer, "padding_side", None) != "left":
        tokenizer.padding_side = "left"

    resolved_dtype = (
        torch_dtype if torch_dtype is not None else (
            torch.bfloat16 if torch.cuda.is_available() else torch.float32
        )
    )

    model_kwargs: Dict[str, object] = {"torch_dtype": resolved_dtype}
    if device_map is not None:
        model_kwargs["device_map"] = device_map

    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        **model_kwargs,
    )

    _configure_special_tokens(model, processor)

    if apply_lora:
        config = lora_config or LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=lora_bias,
            task_type=lora_task_type,
            target_modules=list(lora_target_modules or DEFAULT_LORA_TARGET_MODULES),
        )
        model = get_peft_model(model, config)
        if print_trainable_parameters and hasattr(model, "print_trainable_parameters"):
            model.print_trainable_parameters()

    return model, processor


def load_qwen_asr_model(
    model_path: Path,
    **kwargs,
) -> Tuple[
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
]:
    """Backward-compatible wrapper for legacy training scripts."""
    warnings.warn(
        "load_qwen_asr_model is deprecated; use load_qwen_model instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    return load_qwen_model(model_path, **kwargs)
