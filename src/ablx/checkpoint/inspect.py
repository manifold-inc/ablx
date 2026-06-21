from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .qwen36_map import tag_tensor
from .safetensors_io import tensor_manifest


def inspect_checkpoint(model: str | Path) -> dict[str, Any]:
    root = Path(model).expanduser()
    config_path = root / "config.json" if root.is_dir() else None
    config = {}
    if config_path and config_path.exists():
        config = json.loads(config_path.read_text(encoding="utf-8"))

    tensors = []
    total_params = 0
    role_counts: dict[str, int] = {}
    for info in tensor_manifest(root):
        tag = tag_tensor(info.name)
        params = 1
        for dim in info.shape:
            params *= dim
        total_params += params
        role_counts[tag.role] = role_counts.get(tag.role, 0) + 1
        tensors.append(
            {
                "name": info.name,
                "shape": list(info.shape),
                "dtype": info.dtype,
                "shard": info.shard,
                "role": tag.role,
                "expert_index": tag.expert_index,
                "is_mtp": tag.is_mtp,
                "is_shared": tag.is_shared,
                "is_vision": tag.is_vision,
            }
        )

    return {
        "model": str(root),
        "config_model_type": config.get("model_type") or config.get("text_config", {}).get("model_type"),
        "tensor_count": len(tensors),
        "total_params": total_params,
        "role_counts": role_counts,
        "tensors": tensors,
    }
