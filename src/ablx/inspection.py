from __future__ import annotations

import json
from pathlib import Path
from typing import List

from .errors import CheckpointFormatError
from .hf import resolve_model_path
from .models import ModelSpec, TensorInfo
from .safetensors_io import iter_safetensor_files, list_tensor_infos, load_weight_index


def inspect_model(model: str | Path) -> ModelSpec:
    model_dir = resolve_model_path(str(model))
    if not model_dir.is_dir():
        raise CheckpointFormatError(f"{model_dir} is not a checkpoint directory")

    config_path = model_dir / "config.json"
    if not config_path.exists():
        raise CheckpointFormatError(f"{model_dir} does not contain config.json")
    config = json.loads(config_path.read_text(encoding="utf-8"))

    weight_map, _, _ = load_weight_index(model_dir)
    tensors: List[TensorInfo] = []
    for shard in iter_safetensor_files(model_dir):
        tensors.extend(list_tensor_infos(shard))

    if not tensors:
        raise CheckpointFormatError(f"{model_dir} does not contain .safetensors files")

    if weight_map:
        tensor_names = {tensor.name for tensor in tensors}
        missing = sorted(set(weight_map) - tensor_names)
        if missing:
            raise CheckpointFormatError(
                f"weight index references {len(missing)} tensors missing from shards; first={missing[:3]}"
            )
    else:
        weight_map = {tensor.name: tensor.shard for tensor in tensors}

    total = sum(tensor.nbytes for tensor in tensors)
    return ModelSpec(
        path=str(model_dir),
        config=config,
        tensors=sorted(tensors, key=lambda item: item.name),
        weight_map=weight_map,
        total_tensor_bytes=total,
    )


def tensor_by_name(spec: ModelSpec) -> Dict[str, TensorInfo]:
    return {tensor.name: tensor for tensor in spec.tensors}
