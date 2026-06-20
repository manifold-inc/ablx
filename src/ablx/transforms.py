from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Optional

import numpy as np

from .errors import CheckpointFormatError, RecipeError
from .inspection import inspect_model
from .models import ExpansionReport, ModelSpec, TensorInfo, TensorMapping, TransformOp, UpscaleRecipe
from .safetensors_io import (
    copy_non_weight_files,
    copy_tensor_bytes,
    deterministic_noise_bytes,
    expected_nbytes,
    iter_safetensor_files,
    load_weight_index,
    read_tensor_payload,
    tensor_data_start,
    tensor_to_numpy,
    write_streamed_safetensors,
)


@dataclass(frozen=True)
class PlannedTensor:
    info: TensorInfo
    source_info: TensorInfo
    transform: Optional["TensorTransform"] = None


class TensorTransform:
    name = "copy"

    def target_info(self, source: TensorInfo) -> TensorInfo:
        return source

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        raise NotImplementedError

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
        )


class ExpandFusedGateUp(TensorTransform):
    name = "expand_moe_intermediate:fused_gate_up"

    def __init__(self, old_i: int, new_i: int, noise_std: float, seed: int) -> None:
        self.old_i = int(old_i)
        self.new_i = int(new_i)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if len(shape) != 3:
            raise CheckpointFormatError(f"{source.name} expected [experts, 2I, hidden], got {shape}")
        if shape[1] != 2 * self.old_i:
            raise CheckpointFormatError(f"{source.name} axis 1 is {shape[1]}, expected {2 * self.old_i}")
        shape[1] = 2 * self.new_i
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new_shape = [old.shape[0], 2 * self.new_i, old.shape[2]]
        if payload.dtype == "BF16":
            new = np.zeros(new_shape, dtype=np.uint16)
        else:
            new = np.zeros(new_shape, dtype=old.dtype)
        new[:, : self.old_i, :] = old[:, : self.old_i, :]
        new[:, self.new_i : self.new_i + self.old_i, :] = old[:, self.old_i : 2 * self.old_i, :]
        output.write(new.tobytes(order="C"))

        # Patch deterministic noise into the silent rows after writing by rebuilding locally.
        # For huge tensors, this avoids Python loops and keeps the copy path vectorized.
        if self.noise_std:
            output.seek(output.tell() - new.nbytes)
            new_bytes = bytearray(new.tobytes(order="C"))
            self._insert_noise(new_bytes, payload.dtype, old.shape[0], old.shape[2])
            output.write(bytes(new_bytes))

    def _insert_noise(self, buf: bytearray, dtype: str, experts: int, hidden: int) -> None:
        itemsize = expected_nbytes(dtype, [1])
        gate_shape = [experts, self.new_i - self.old_i, hidden]
        up_shape = [experts, self.new_i - self.old_i, hidden]
        gate_noise = deterministic_noise_bytes(dtype, gate_shape, self.noise_std, self.seed)
        up_noise = deterministic_noise_bytes(dtype, up_shape, self.noise_std, self.seed + 1)
        row_stride = 2 * self.new_i * hidden * itemsize
        col_stride = hidden * itemsize
        for expert in range(experts):
            base = expert * row_stride
            gate_start = base + self.old_i * col_stride
            gate_end = base + self.new_i * col_stride
            up_start = base + (self.new_i + self.old_i) * col_stride
            up_end = base + (2 * self.new_i) * col_stride
            g0 = expert * (self.new_i - self.old_i) * hidden * itemsize
            g1 = g0 + (self.new_i - self.old_i) * hidden * itemsize
            buf[gate_start:gate_end] = gate_noise[g0:g1]
            buf[up_start:up_end] = up_noise[g0:g1]

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[
                f"axis1[{self.old_i}:{self.new_i}] gate expansion",
                f"axis1[{self.new_i + self.old_i}:{2 * self.new_i}] up expansion",
            ],
            notes=["old gate/up rows are copied; new down projection starts silent"],
        )


class ExpandExpertDown(TensorTransform):
    name = "expand_moe_intermediate:expert_down"

    def __init__(self, old_i: int, new_i: int) -> None:
        self.old_i = int(old_i)
        self.new_i = int(new_i)

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if len(shape) != 3:
            raise CheckpointFormatError(f"{source.name} expected [experts, hidden, I], got {shape}")
        if shape[2] != self.old_i:
            raise CheckpointFormatError(f"{source.name} axis 2 is {shape[2]}, expected {self.old_i}")
        shape[2] = self.new_i
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new_shape = [old.shape[0], old.shape[1], self.new_i]
        new = np.zeros(new_shape, dtype=old.dtype)
        new[:, :, : self.old_i] = old
        output.write(new.tobytes(order="C"))

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[f"axis2[{self.old_i}:{self.new_i}] zero-start down columns"],
            notes=["new down columns are zero, preserving the function at step 0"],
        )


class ExpandDenseIn(TensorTransform):
    name = "expand_moe_intermediate:dense_in"

    def __init__(self, old_i: int, new_i: int, noise_std: float, seed: int) -> None:
        self.old_i = int(old_i)
        self.new_i = int(new_i)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if len(shape) != 2:
            raise CheckpointFormatError(f"{source.name} expected [I, hidden], got {shape}")
        if shape[0] != self.old_i:
            raise CheckpointFormatError(f"{source.name} axis 0 is {shape[0]}, expected {self.old_i}")
        shape[0] = self.new_i
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new = np.zeros([self.new_i, old.shape[1]], dtype=old.dtype)
        new[: self.old_i, :] = old
        if self.new_i > self.old_i:
            noise = deterministic_noise_bytes(
                payload.dtype,
                [self.new_i - self.old_i, old.shape[1]],
                self.noise_std,
                self.seed,
            )
            if payload.dtype == "BF16":
                new[self.old_i :, :] = np.frombuffer(noise, dtype=np.uint16).reshape(self.new_i - self.old_i, old.shape[1])
            else:
                new[self.old_i :, :] = np.frombuffer(noise, dtype=new.dtype).reshape(self.new_i - self.old_i, old.shape[1])
        output.write(new.tobytes(order="C"))

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[f"axis0[{self.old_i}:{self.new_i}] new dense rows"],
        )


class ExpandDenseOut(TensorTransform):
    name = "expand_moe_intermediate:dense_out"

    def __init__(self, old_i: int, new_i: int) -> None:
        self.old_i = int(old_i)
        self.new_i = int(new_i)

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if len(shape) != 2:
            raise CheckpointFormatError(f"{source.name} expected [hidden, I], got {shape}")
        if shape[1] != self.old_i:
            raise CheckpointFormatError(f"{source.name} axis 1 is {shape[1]}, expected {self.old_i}")
        shape[1] = self.new_i
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new = np.zeros([old.shape[0], self.new_i], dtype=old.dtype)
        new[:, : self.old_i] = old
        output.write(new.tobytes(order="C"))

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[f"axis1[{self.old_i}:{self.new_i}] zero-start dense columns"],
            notes=["new down columns are zero, preserving the function at step 0"],
        )


class ExpandDenseFusedGateUp(TensorTransform):
    name = "expand_moe_intermediate:dense_fused_gate_up"

    def __init__(self, old_i: int, new_i: int, noise_std: float, seed: int) -> None:
        self.impl = ExpandFusedGateUp(old_i, new_i, noise_std, seed)
        self.old_i = int(old_i)
        self.new_i = int(new_i)
        self.noise_std = float(noise_std)
        self.seed = int(seed)

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if len(shape) != 2:
            raise CheckpointFormatError(f"{source.name} expected [2I, hidden], got {shape}")
        if shape[0] != 2 * self.old_i:
            raise CheckpointFormatError(f"{source.name} axis 0 is {shape[0]}, expected {2 * self.old_i}")
        shape[0] = 2 * self.new_i
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new = np.zeros([2 * self.new_i, old.shape[1]], dtype=old.dtype)
        new[: self.old_i, :] = old[: self.old_i, :]
        new[self.new_i : self.new_i + self.old_i, :] = old[self.old_i : 2 * self.old_i, :]
        if self.new_i > self.old_i:
            gate_noise = deterministic_noise_bytes(source_info.dtype, [self.new_i - self.old_i, old.shape[1]], self.noise_std, self.seed)
            up_noise = deterministic_noise_bytes(source_info.dtype, [self.new_i - self.old_i, old.shape[1]], self.noise_std, self.seed + 1)
            dtype = np.uint16 if source_info.dtype == "BF16" else new.dtype
            new[self.old_i : self.new_i, :] = np.frombuffer(gate_noise, dtype=dtype).reshape(self.new_i - self.old_i, old.shape[1])
            new[self.new_i + self.old_i : 2 * self.new_i, :] = np.frombuffer(up_noise, dtype=dtype).reshape(self.new_i - self.old_i, old.shape[1])
        output.write(new.tobytes(order="C"))

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[
                f"axis0[{self.old_i}:{self.new_i}] gate expansion",
                f"axis0[{self.new_i + self.old_i}:{2 * self.new_i}] up expansion",
            ],
        )


class CloneExpertTensor(TensorTransform):
    name = "clone_experts:expert_tensor"

    def __init__(self, old_experts: int, new_experts: int, source_indices: Optional[List[int]] = None) -> None:
        self.old_experts = int(old_experts)
        self.new_experts = int(new_experts)
        self.source_indices = source_indices

    def _clone_sources(self) -> List[int]:
        count = self.new_experts - self.old_experts
        if self.source_indices:
            if len(self.source_indices) < count:
                raise RecipeError("clone_experts source_indices is shorter than new expert count")
            return [int(idx) % self.old_experts for idx in self.source_indices[:count]]
        return [idx % self.old_experts for idx in range(count)]

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if not shape or shape[0] != self.old_experts:
            raise CheckpointFormatError(f"{source.name} axis 0 is not old expert count {self.old_experts}: {shape}")
        shape[0] = self.new_experts
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new_shape = [self.new_experts, *old.shape[1:]]
        new = np.zeros(new_shape, dtype=old.dtype)
        new[: self.old_experts] = old
        for out_idx, src_idx in enumerate(self._clone_sources(), start=self.old_experts):
            new[out_idx] = old[src_idx]
        output.write(new.tobytes(order="C"))

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[f"axis0[{self.old_experts}:{self.new_experts}] cloned experts"],
            notes=["experimental gated transform; router wake-up should be scheduled during training"],
        )


class CloneRouter(TensorTransform):
    name = "clone_experts:router"

    def __init__(self, old_experts: int, new_experts: int, source_indices: Optional[List[int]] = None, router_init: str = "zero") -> None:
        self.old_experts = int(old_experts)
        self.new_experts = int(new_experts)
        self.source_indices = source_indices
        self.router_init = str(router_init)

    def _clone_sources(self) -> List[int]:
        count = self.new_experts - self.old_experts
        if self.source_indices:
            if len(self.source_indices) < count:
                raise RecipeError("clone_experts source_indices is shorter than new expert count")
            return [int(idx) % self.old_experts for idx in self.source_indices[:count]]
        return [idx % self.old_experts for idx in range(count)]

    def target_info(self, source: TensorInfo) -> TensorInfo:
        shape = list(source.shape)
        if not shape or shape[0] != self.old_experts:
            raise CheckpointFormatError(f"{source.name} axis 0 is not old expert count {self.old_experts}: {shape}")
        shape[0] = self.new_experts
        return TensorInfo(source.name, source.dtype, shape, [0, expected_nbytes(source.dtype, shape)], source.shard)

    def write(self, source_shard: Path, source_info: TensorInfo, output: BinaryIO) -> None:
        payload = read_tensor_payload(source_shard, source_info)
        old = tensor_to_numpy(payload)
        new_shape = [self.new_experts, *old.shape[1:]]
        new = np.zeros(new_shape, dtype=old.dtype)
        new[: self.old_experts] = old
        if self.router_init == "clone":
            for out_idx, src_idx in enumerate(self._clone_sources(), start=self.old_experts):
                new[out_idx] = old[src_idx]
        elif self.router_init != "zero":
            raise RecipeError("clone_experts router_init must be 'zero' or 'clone'")
        output.write(new.tobytes(order="C"))

    def mapping(self, source: TensorInfo, target: TensorInfo) -> TensorMapping:
        return TensorMapping(
            name=source.name,
            source_shape=list(source.shape),
            target_shape=list(target.shape),
            transform=self.name,
            trainable_slices=[f"axis0[{self.old_experts}:{self.new_experts}] router rows"],
            notes=[f"router_init={self.router_init}; top-k remains unchanged"],
        )


def upscale_checkpoint(source: str | Path, recipe: UpscaleRecipe, out: str | Path) -> ExpansionReport:
    source_dir = Path(source).expanduser().resolve()
    out_dir = Path(out).expanduser().resolve()
    if source_dir == out_dir:
        raise RecipeError("output directory must differ from source directory")
    spec = inspect_model(source_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    new_config = copy.deepcopy(spec.config)
    transforms = plan_transforms(spec, recipe, new_config)
    changed_names = set(transforms)
    skip_names = {"config.json"}
    _, index_path, index_data = load_weight_index(source_dir)
    if index_path:
        skip_names.add(index_path.name)
    copy_non_weight_files(source_dir, out_dir, skip_names=skip_names)

    changed_mappings: List[TensorMapping] = []
    copied_tensors = 0
    new_weight_map: Dict[str, str] = {}
    new_total_size = 0

    source_by_shard: Dict[str, List[TensorInfo]] = {}
    for tensor in spec.tensors:
        source_by_shard.setdefault(tensor.shard, []).append(tensor)

    for shard_path in iter_safetensor_files(source_dir):
        source_infos = source_by_shard.get(shard_path.name, [])
        planned_infos: List[PlannedTensor] = []
        for info in source_infos:
            transform = transforms.get(info.name)
            target = transform.target_info(info) if transform else info
            planned_infos.append(PlannedTensor(info=target, source_info=info, transform=transform))
            new_weight_map[target.name] = shard_path.name
            new_total_size += expected_nbytes(target.dtype, target.shape)
            if transform:
                changed_mappings.append(transform.mapping(info, target))
            else:
                copied_tensors += 1

        output_shard = out_dir / shard_path.name
        base_offset = tensor_data_start(shard_path)

        def write_payload(target_info: TensorInfo, handle: BinaryIO, planned=planned_infos, src_shard=shard_path, base=base_offset) -> None:
            plan = next(item for item in planned if item.info.name == target_info.name)
            if plan.transform is None:
                with src_shard.open("rb") as src_handle:
                    copy_tensor_bytes(src_handle, handle, base, plan.source_info)
            else:
                plan.transform.write(src_shard, plan.source_info, handle)

        write_streamed_safetensors(output_shard, [item.info for item in planned_infos], write_payload, metadata={"format": "pt"})

    (out_dir / "config.json").write_text(json.dumps(new_config, indent=2, sort_keys=False) + "\n", encoding="utf-8")
    if index_path:
        out_index = copy.deepcopy(index_data)
        out_index["weight_map"] = new_weight_map
        metadata = out_index.setdefault("metadata", {})
        if isinstance(metadata, dict):
            metadata["total_size"] = new_total_size
        (out_dir / index_path.name).write_text(json.dumps(out_index, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    report = ExpansionReport(
        source=str(source_dir),
        output=str(out_dir),
        recipe=recipe.to_dict(),
        changed_tensors=sorted(changed_mappings, key=lambda item: item.name),
        copied_tensors=copied_tensors,
        warnings=build_warnings(recipe, changed_mappings),
        slime_hints=build_slime_hints(recipe, changed_mappings),
    )
    (out_dir / "ablx_expansion_report.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def plan_transforms(spec: ModelSpec, recipe: UpscaleRecipe, config: Dict[str, object]) -> Dict[str, TensorTransform]:
    transforms: Dict[str, TensorTransform] = {}
    for op in recipe.transforms:
        if op.type == "expand_moe_intermediate":
            transforms.update(plan_expand_moe_intermediate(spec, op, config))
        elif op.type == "clone_experts":
            transforms.update(plan_clone_experts(spec, op, config))
        else:
            raise RecipeError(f"unsupported transform type {op.type!r}")
    return transforms


def active_text_config(config: Dict[str, object]) -> Dict[str, object]:
    text_config = config.get("text_config")
    if isinstance(text_config, dict):
        return text_config
    return config


def plan_expand_moe_intermediate(spec: ModelSpec, op: TransformOp, config: Dict[str, object]) -> Dict[str, TensorTransform]:
    params = op.params
    active = active_text_config(config)
    old_i = int(params.get("old_intermediate_size") or active.get("moe_intermediate_size") or 0)
    new_i = int(params.get("new_intermediate_size") or 0)
    if old_i <= 0 or new_i <= old_i:
        raise RecipeError("expand_moe_intermediate requires new_intermediate_size > old_intermediate_size > 0")
    noise_std = float(params.get("noise_std", 1e-6))
    seed = int(params.get("noise_seed", 1729))
    include_shared = bool(params.get("include_shared_expert", True))

    planned: Dict[str, TensorTransform] = {}
    for index, tensor in enumerate(spec.tensors):
        name = tensor.name
        seed_i = seed + index * 17
        if name.endswith(".mlp.experts.gate_up_proj"):
            planned[name] = ExpandFusedGateUp(old_i, new_i, noise_std, seed_i)
        elif name.endswith(".mlp.experts.down_proj"):
            planned[name] = ExpandExpertDown(old_i, new_i)
        elif include_shared and is_shared_gate_up(name):
            planned[name] = ExpandDenseFusedGateUp(old_i, new_i, noise_std, seed_i)
        elif include_shared and is_shared_dense_in(name):
            planned[name] = ExpandDenseIn(old_i, new_i, noise_std, seed_i)
        elif include_shared and is_shared_dense_out(name):
            planned[name] = ExpandDenseOut(old_i, new_i)

    if not planned:
        raise RecipeError("expand_moe_intermediate did not match any MoE tensors")

    active["moe_intermediate_size"] = new_i
    if "shared_expert_intermediate_size" in active:
        active["shared_expert_intermediate_size"] = new_i
    return planned


def is_shared_gate_up(name: str) -> bool:
    return (".shared_expert." in name or ".shared_experts." in name) and name.endswith("gate_up_proj")


def is_shared_dense_in(name: str) -> bool:
    return (".shared_expert." in name or ".shared_experts." in name) and (
        name.endswith("gate_proj") or name.endswith("up_proj")
    )


def is_shared_dense_out(name: str) -> bool:
    return (".shared_expert." in name or ".shared_experts." in name) and name.endswith("down_proj")


def plan_clone_experts(spec: ModelSpec, op: TransformOp, config: Dict[str, object]) -> Dict[str, TensorTransform]:
    params = op.params
    active = active_text_config(config)
    old_experts = int(params.get("old_num_experts") or active.get("num_experts") or active.get("n_routed_experts") or 0)
    new_experts = int(params.get("new_num_experts") or 0)
    if old_experts <= 0 or new_experts <= old_experts:
        raise RecipeError("clone_experts requires new_num_experts > old_num_experts > 0")
    source_indices = params.get("source_indices")
    if source_indices is not None and not isinstance(source_indices, list):
        raise RecipeError("clone_experts source_indices must be a list")
    router_init = str(params.get("router_init", "zero"))
    planned: Dict[str, TensorTransform] = {}
    for tensor in spec.tensors:
        name = tensor.name
        if is_routed_expert_tensor(name):
            planned[name] = CloneExpertTensor(old_experts, new_experts, source_indices)
        elif is_router_tensor(name):
            planned[name] = CloneRouter(old_experts, new_experts, source_indices, router_init=router_init)
    if not planned:
        raise RecipeError("clone_experts did not match any expert/router tensors")
    if "num_experts" in active:
        active["num_experts"] = new_experts
    if "n_routed_experts" in active:
        active["n_routed_experts"] = new_experts
    return planned


def is_routed_expert_tensor(name: str) -> bool:
    return ".mlp.experts." in name and not is_router_tensor(name)


def is_router_tensor(name: str) -> bool:
    return name.endswith(".mlp.gate.weight") or name.endswith(".mlp.router.weight") or name.endswith(".gate.weight")


def build_warnings(recipe: UpscaleRecipe, mappings: Iterable[TensorMapping]) -> List[str]:
    warnings: List[str] = []
    op_types = {op.type for op in recipe.transforms}
    if "clone_experts" in op_types:
        warnings.append(
            "clone_experts is experimental and not guaranteed function-preserving under sparse top-k routing."
        )
    if not list(mappings):
        warnings.append("no tensors were changed")
    return warnings


def build_slime_hints(recipe: UpscaleRecipe, mappings: Iterable[TensorMapping]) -> Dict[str, object]:
    trainable_patterns = sorted({mapping.name for mapping in mappings})
    return {
        "phase_0": "run ablx probe before training; require low KL/top-k drift",
        "phase_1_trainable_tensors": trainable_patterns,
        "phase_1_freeze": [
            "embeddings",
            "lm_head",
            "vision_tower",
            "attention",
            "norms",
            "old expert slices where supported by the training stack",
        ],
        "phase_2": "continue SFT/pretraining with source-logit KL plus CE",
        "phase_3": "use slime GRPO/OPD with SGLang rollouts for math/code/agent tasks",
    }
