from __future__ import annotations

import json
from pathlib import Path

import torch
from safetensors.torch import save_file


def create_tiny_checkpoint(path: str | Path) -> Path:
    root = Path(path)
    root.mkdir(parents=True, exist_ok=True)
    config = {
        "model_type": "qwen3_5_moe",
        "text_config": {
            "moe_intermediate_size": 2,
            "shared_expert_intermediate_size": 2,
            "num_experts": 2,
            "num_experts_per_tok": 1,
            "hidden_size": 4,
        },
    }
    (root / "config.json").write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    tensors = {
        "model.layers.0.mlp.experts.0.gate_up_proj.weight": torch.arange(16, dtype=torch.float32).reshape(4, 4),
        "model.layers.0.mlp.experts.0.down_proj.weight": torch.arange(8, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.mlp.experts.1.gate_up_proj.weight": torch.arange(16, 32, dtype=torch.float32).reshape(4, 4),
        "model.layers.0.mlp.experts.1.down_proj.weight": torch.arange(8, 16, dtype=torch.float32).reshape(4, 2),
        "model.layers.0.mlp.router.weight": torch.ones(2, 4, dtype=torch.float32),
        "model.layers.0.mlp.shared_expert.gate_up_proj.weight": torch.ones(4, 4, dtype=torch.float32),
        "model.layers.0.mlp.shared_expert.down_proj.weight": torch.ones(4, 2, dtype=torch.float32),
        "vision.encoder.layers.0.weight": torch.eye(4, dtype=torch.float32),
    }
    save_file(tensors, str(root / "model.safetensors"))
    logits = [[0.1, 0.2, 0.3, 0.4], [0.4, 0.3, 0.2, 0.1]]
    (root / "fixture_logits.json").write_text(json.dumps(logits) + "\n", encoding="utf-8")
    return root
