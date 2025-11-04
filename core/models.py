"""Model loading and adaptation utilities."""

from pathlib import Path
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import (
    Qwen2_5OmniProcessor,
    Qwen2_5OmniThinkerForConditionalGeneration,
)

TARGET_LINEAR_MODULES = [
    "q_proj",
    "k_proj",
    "v_proj",
    "out_proj",
    "o_proj",
    "fc1",
    "fc2",
]


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


def load_qwen_asr_model(model_path: Path) -> Tuple[
    Qwen2_5OmniThinkerForConditionalGeneration,
    Qwen2_5OmniProcessor,
]:
    """Load the Qwen Omni model and processor with LoRA adapters enabled."""
    processor = Qwen2_5OmniProcessor.from_pretrained(str(model_path), use_fast=False)
    model = Qwen2_5OmniThinkerForConditionalGeneration.from_pretrained(
        str(model_path),
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map="auto",
    )

    _configure_special_tokens(model, processor)

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=TARGET_LINEAR_MODULES,
    )

    model = get_peft_model(model, lora_config)
    model.print_trainable_parameters()

    return model, processor
