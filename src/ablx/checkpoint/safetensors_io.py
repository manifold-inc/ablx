from __future__ import annotations

import json
import shutil
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable

import torch
from safetensors import safe_open
from safetensors.torch import load_file, save_file

from ablx.errors import CheckpointFormatError
from ablx.utils import log_progress


@dataclass(frozen=True)
class TensorInfo:
    name: str
    shape: tuple[int, ...]
    dtype: str
    shard: str


def checkpoint_files(model: str | Path) -> list[Path]:
    root = Path(model).expanduser()
    if root.is_file() and root.suffix == ".safetensors":
        return [root]
    if not root.exists():
        raise CheckpointFormatError(f"checkpoint path does not exist: {model}")
    files = sorted(root.glob("*.safetensors"))
    if not files:
        raise CheckpointFormatError(f"no .safetensors files found under {root}")
    return files


def tensor_manifest(model: str | Path) -> list[TensorInfo]:
    infos: list[TensorInfo] = []
    for shard in checkpoint_files(model):
        with safe_open(shard, framework="pt", device="cpu") as handle:
            for key in handle.keys():
                tensor = handle.get_tensor(key)
                infos.append(TensorInfo(key, tuple(tensor.shape), str(tensor.dtype), shard.name))
    return infos


def load_shard(path: str | Path) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu")


TensorTransform = Callable[[str, torch.Tensor], Iterable[tuple[str, torch.Tensor]]]


def rewrite_checkpoint(
    source: str | Path,
    out: str | Path,
    transform: TensorTransform,
    *,
    copy_non_tensor_files: bool = True,
    progress_label: str | None = None,
) -> list[str]:
    src = Path(source).expanduser().resolve()
    dst = Path(out).expanduser().resolve()
    dst.mkdir(parents=True, exist_ok=True)
    written: list[str] = []

    if copy_non_tensor_files and src.is_dir():
        for path in src.iterdir():
            if path.suffix == ".safetensors":
                continue
            target = dst / path.name
            if path.is_file():
                shutil.copy2(path, target)
            elif path.is_dir() and path.name not in {".git", "__pycache__"}:
                if target.exists():
                    shutil.rmtree(target)
                shutil.copytree(path, target)

    shards = checkpoint_files(src)
    for index, shard in enumerate(shards, start=1):
        if progress_label:
            log_progress(f"{progress_label}: loading shard {index}/{len(shards)} {shard.name}")
        tensors = load_shard(shard)
        if progress_label:
            log_progress(f"{progress_label}: transforming {len(tensors)} tensors from {shard.name}")
        new_tensors: dict[str, torch.Tensor] = {}
        for name, tensor in tensors.items():
            for out_name, out_tensor in transform(name, tensor):
                new_tensors[out_name] = out_tensor.contiguous()
        shard_name = shard.name
        if progress_label:
            log_progress(f"{progress_label}: writing shard {index}/{len(shards)} {shard_name}")
        save_file(new_tensors, str(dst / shard_name))
        written.append(shard_name)
        if progress_label:
            log_progress(f"{progress_label}: completed shard {index}/{len(shards)} {shard_name}")

    if progress_label:
        log_progress(f"{progress_label}: rebuilding safetensors index")
    rebuild_index_if_present(src, dst)
    return written


def rebuild_index_if_present(source: Path, out: Path) -> None:
    index_path = source / "model.safetensors.index.json"
    if not index_path.exists():
        return
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        return
    new_map: dict[str, str] = {}
    for shard in checkpoint_files(out):
        tensors = load_file(str(shard), device="cpu")
        for name in tensors:
            new_map[name] = shard.name
    data["weight_map"] = new_map
    data.setdefault("metadata", {})["format"] = "pt"
    (out / "model.safetensors.index.json").write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
