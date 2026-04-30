from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file

from merging.delta_sources.lora_source import LoRADeltaSource


def test_lora_delta_source_materializes_dense_delta(tmp_path: Path) -> None:
    adapter_dir = tmp_path / "adapter"
    adapter_dir.mkdir(parents=True, exist_ok=True)

    with (adapter_dir / "adapter_config.json").open("w") as handle:
        json.dump({"r": 2, "lora_alpha": 4}, handle)

    a = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float32)
    b = torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32)
    save_file(
        {
            "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight": a,
            "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight": b,
        },
        str(adapter_dir / "adapter_model.safetensors"),
    )

    source = LoRADeltaSource(adapter_dir, task="asr")
    specs = source.list_target_params()
    assert len(specs) == 1

    dense = source.materialize_dense_param_delta(specs[0].source_key, dtype=torch.float32)
    # scaling = alpha / r = 4 / 2 = 2
    expected = (b @ a) * 2.0
    assert torch.allclose(dense, expected)
