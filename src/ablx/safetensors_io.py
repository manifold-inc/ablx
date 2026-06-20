from __future__ import annotations

import hashlib
import json
import os
import shutil
import struct
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Callable, Dict, Iterable, Iterator, List, Mapping, Optional, Tuple

import numpy as np

from .errors import CheckpointFormatError
from .models import TensorInfo


DTYPE_ITEMSIZE = {
    "BOOL": 1,
    "U8": 1,
    "I8": 1,
    "I16": 2,
    "U16": 2,
    "F16": 2,
    "BF16": 2,
    "F8_E4M3": 1,
    "F8_E4M3FN": 1,
    "F8_E4M3FNUZ": 1,
    "F8_E5M2": 1,
    "F8_E5M2FNUZ": 1,
    "F8_E8M0": 1,
    "I32": 4,
    "U32": 4,
    "F32": 4,
    "F64": 8,
    "I64": 8,
    "U64": 8,
}

NUMPY_DTYPES = {
    "BOOL": np.bool_,
    "U8": np.uint8,
    "I8": np.int8,
    "I16": np.int16,
    "U16": np.uint16,
    "F16": np.float16,
    "BF16": np.uint16,
    "F8_E4M3": np.uint8,
    "F8_E4M3FN": np.uint8,
    "F8_E4M3FNUZ": np.uint8,
    "F8_E5M2": np.uint8,
    "F8_E5M2FNUZ": np.uint8,
    "F8_E8M0": np.uint8,
    "I32": np.int32,
    "U32": np.uint32,
    "F32": np.float32,
    "F64": np.float64,
    "I64": np.int64,
    "U64": np.uint64,
}


@dataclass(frozen=True)
class TensorPayload:
    name: str
    dtype: str
    shape: List[int]
    data: bytes

    @property
    def nbytes(self) -> int:
        return len(self.data)


def dtype_itemsize(dtype: str) -> int:
    try:
        return DTYPE_ITEMSIZE[dtype]
    except KeyError as exc:
        raise CheckpointFormatError(f"unsupported safetensors dtype {dtype!r}") from exc


def expected_nbytes(dtype: str, shape: Iterable[int]) -> int:
    count = 1
    for dim in shape:
        count *= int(dim)
    return count * dtype_itemsize(dtype)


def read_header(path: str | Path) -> Dict[str, object]:
    tensor_path = Path(path)
    with tensor_path.open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise CheckpointFormatError(f"{tensor_path} is too small to be safetensors")
        header_len = struct.unpack("<Q", raw_len)[0]
        header_raw = handle.read(header_len)
        if len(header_raw) != header_len:
            raise CheckpointFormatError(f"{tensor_path} has a truncated safetensors header")
    try:
        header = json.loads(header_raw.decode("utf-8"))
    except json.JSONDecodeError as exc:
        raise CheckpointFormatError(f"{tensor_path} has an invalid safetensors header: {exc}") from exc
    if not isinstance(header, dict):
        raise CheckpointFormatError(f"{tensor_path} header is not a JSON object")
    return header


def header_length(path: str | Path) -> int:
    with Path(path).open("rb") as handle:
        raw_len = handle.read(8)
        if len(raw_len) != 8:
            raise CheckpointFormatError(f"{path} is too small to be safetensors")
        return int(struct.unpack("<Q", raw_len)[0])


def tensor_data_start(path: str | Path) -> int:
    return 8 + header_length(path)


def list_tensor_infos(shard: str | Path) -> List[TensorInfo]:
    shard_path = Path(shard)
    header = read_header(shard_path)
    infos: List[TensorInfo] = []
    for name, metadata in header.items():
        if name == "__metadata__":
            continue
        if not isinstance(metadata, Mapping):
            raise CheckpointFormatError(f"tensor {name} metadata is not a mapping")
        dtype = str(metadata.get("dtype"))
        shape = [int(dim) for dim in metadata.get("shape", [])]
        offsets = [int(dim) for dim in metadata.get("data_offsets", [])]
        if len(offsets) != 2:
            raise CheckpointFormatError(f"tensor {name} has invalid data_offsets")
        nbytes = offsets[1] - offsets[0]
        expected = expected_nbytes(dtype, shape)
        if nbytes != expected:
            raise CheckpointFormatError(
                f"tensor {name} has {nbytes} bytes but shape/dtype imply {expected}"
            )
        infos.append(
            TensorInfo(
                name=name,
                dtype=dtype,
                shape=shape,
                data_offsets=offsets,
                shard=shard_path.name,
            )
        )
    return infos


def read_tensor_bytes(shard_path: str | Path, info: TensorInfo) -> bytes:
    start = tensor_data_start(shard_path) + int(info.data_offsets[0])
    with Path(shard_path).open("rb") as handle:
        handle.seek(start)
        data = handle.read(info.nbytes)
    if len(data) != info.nbytes:
        raise CheckpointFormatError(f"tensor {info.name} data is truncated")
    return data


def copy_tensor_bytes(source: BinaryIO, output: BinaryIO, base_offset: int, info: TensorInfo, chunk_size: int = 1024 * 1024 * 16) -> None:
    source.seek(base_offset + int(info.data_offsets[0]))
    remaining = info.nbytes
    while remaining:
        chunk = source.read(min(chunk_size, remaining))
        if not chunk:
            raise CheckpointFormatError(f"tensor {info.name} data ended early while copying")
        output.write(chunk)
        remaining -= len(chunk)


def tensor_to_numpy(payload: TensorPayload) -> np.ndarray:
    dtype = NUMPY_DTYPES.get(payload.dtype)
    if dtype is None:
        raise CheckpointFormatError(f"unsupported dtype {payload.dtype!r}")
    arr = np.frombuffer(payload.data, dtype=dtype)
    return arr.reshape(payload.shape)


def read_tensor_payload(shard_path: str | Path, info: TensorInfo) -> TensorPayload:
    return TensorPayload(
        name=info.name,
        dtype=info.dtype,
        shape=list(info.shape),
        data=read_tensor_bytes(shard_path, info),
    )


def as_tensor_bytes(array: np.ndarray, dtype: str) -> bytes:
    if dtype == "BF16":
        return np.asarray(array, dtype=np.uint16).tobytes(order="C")
    target_dtype = NUMPY_DTYPES.get(dtype)
    if target_dtype is None:
        raise CheckpointFormatError(f"unsupported dtype {dtype!r}")
    return np.asarray(array, dtype=target_dtype).tobytes(order="C")


def bf16_noise(shape: Iterable[int], std: float, seed: int) -> bytes:
    values = f32_noise(shape, std, seed)
    as_u32 = values.view(np.uint32)
    rounded = as_u32 + np.uint32(0x00008000)
    return (rounded >> np.uint32(16)).astype(np.uint16).tobytes(order="C")


def f32_noise(shape: Iterable[int], std: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.normal(loc=0.0, scale=float(std), size=tuple(int(dim) for dim in shape)).astype(np.float32)


def deterministic_noise_bytes(dtype: str, shape: Iterable[int], std: float, seed: int) -> bytes:
    if float(std) == 0.0:
        return zero_bytes(dtype, shape)
    if dtype == "BF16":
        return bf16_noise(shape, std, seed)
    if dtype == "F16":
        return f32_noise(shape, std, seed).astype(np.float16).tobytes(order="C")
    if dtype == "F32":
        return f32_noise(shape, std, seed).astype(np.float32).tobytes(order="C")
    return zero_bytes(dtype, shape)


def zero_bytes(dtype: str, shape: Iterable[int]) -> bytes:
    return b"\x00" * expected_nbytes(dtype, shape)


def write_safetensors(path: str | Path, tensors: Iterable[TensorPayload], metadata: Optional[Mapping[str, str]] = None) -> Dict[str, object]:
    path = Path(path)
    tensor_list = list(tensors)
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for tensor in tensor_list:
        expected = expected_nbytes(tensor.dtype, tensor.shape)
        if tensor.nbytes != expected:
            raise CheckpointFormatError(
                f"tensor {tensor.name} has {tensor.nbytes} bytes but shape/dtype imply {expected}"
            )
        header[tensor.name] = {
            "dtype": tensor.dtype,
            "shape": [int(dim) for dim in tensor.shape],
            "data_offsets": [offset, offset + tensor.nbytes],
        }
        offset += tensor.nbytes

    header_raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_raw)))
        handle.write(header_raw)
        for tensor in tensor_list:
            handle.write(tensor.data)
    return header


def write_streamed_safetensors(
    path: str | Path,
    tensor_infos: Iterable[TensorInfo],
    write_payload: Callable[[TensorInfo, BinaryIO], None],
    metadata: Optional[Mapping[str, str]] = None,
) -> Dict[str, object]:
    path = Path(path)
    infos = list(tensor_infos)
    header: Dict[str, object] = {}
    if metadata:
        header["__metadata__"] = {str(k): str(v) for k, v in metadata.items()}
    offset = 0
    for info in infos:
        nbytes = expected_nbytes(info.dtype, info.shape)
        header[info.name] = {
            "dtype": info.dtype,
            "shape": [int(dim) for dim in info.shape],
            "data_offsets": [offset, offset + nbytes],
        }
        offset += nbytes

    header_raw = json.dumps(header, separators=(",", ":"), sort_keys=True).encode("utf-8")
    with path.open("wb") as handle:
        handle.write(struct.pack("<Q", len(header_raw)))
        handle.write(header_raw)
        for info in infos:
            before = handle.tell()
            write_payload(info, handle)
            written = handle.tell() - before
            expected = expected_nbytes(info.dtype, info.shape)
            if written != expected:
                raise CheckpointFormatError(
                    f"writer emitted {written} bytes for {info.name}, expected {expected}"
                )
    return header


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024 * 16) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def copy_non_weight_files(source_dir: str | Path, out_dir: str | Path, skip_names: Optional[set[str]] = None) -> None:
    source_dir = Path(source_dir)
    out_dir = Path(out_dir)
    skip = set(skip_names or set())
    for root, dirs, files in os.walk(source_dir):
        rel_root = Path(root).relative_to(source_dir)
        if ".git" in rel_root.parts:
            continue
        target_root = out_dir / rel_root
        target_root.mkdir(parents=True, exist_ok=True)
        dirs[:] = [name for name in dirs if name != ".git"]
        for filename in files:
            if filename in skip or filename.endswith(".safetensors") or filename.endswith(".safetensors.index.json"):
                continue
            src = Path(root) / filename
            dst = target_root / filename
            if src.resolve() == dst.resolve():
                continue
            shutil.copy2(src, dst)


def iter_safetensor_files(model_dir: str | Path) -> Iterator[Path]:
    for path in sorted(Path(model_dir).glob("*.safetensors")):
        if path.is_file():
            yield path


def load_weight_index(model_dir: str | Path) -> Tuple[Dict[str, str], Optional[Path], Dict[str, object]]:
    model_dir = Path(model_dir)
    candidates = sorted(model_dir.glob("*.safetensors.index.json"))
    if not candidates:
        return {}, None, {}
    index_path = candidates[0]
    data = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = data.get("weight_map", {})
    if not isinstance(weight_map, dict):
        raise CheckpointFormatError(f"{index_path} has invalid weight_map")
    return {str(k): str(v) for k, v in weight_map.items()}, index_path, data
