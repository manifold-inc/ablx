from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from ablx.checkpoint.qwen36_map import tag_tensor


@dataclass(frozen=True)
class IntermediateExpansion:
    old_size: int
    new_size: int
    include_shared_expert: bool = True
    include_mtp: bool = True


def should_expand_tensor(name: str, cfg: IntermediateExpansion) -> bool:
    tag = tag_tensor(name)
    if tag.is_mtp and not cfg.include_mtp:
        return False
    if tag.is_shared and not cfg.include_shared_expert:
        return False
    return tag.role in {
        "expert_gate_up",
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate_up",
        "shared_gate",
        "shared_up",
        "shared_down",
    }


def expand_intermediate_tensor(
    name: str,
    tensor: torch.Tensor,
    cfg: IntermediateExpansion,
) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if not should_expand_tensor(name, cfg):
        return tensor, None

    tag = tag_tensor(name)
    old = cfg.old_size
    new = cfg.new_size
    if new <= old:
        raise ValueError("new intermediate size must exceed old intermediate size")

    role = tag.role
    original_shape = list(tensor.shape)
    if "gate_up" in role:
        expanded, mask = _expand_gate_up(tensor, old, new)
    elif "down" in role:
        expanded, mask = _expand_down(tensor, old, new)
    else:
        expanded, mask = _expand_gate_or_up(tensor, old, new)

    report = {
        "tensor": name,
        "role": role,
        "old_shape": original_shape,
        "new_shape": list(expanded.shape),
        "old_slices": mask["old_slices"],
        "new_slices": mask["new_slices"],
        "paper_sections": ["4", "7", "9", "11"],
    }
    return expanded, report


def _expand_gate_or_up(tensor: torch.Tensor, old: int, new: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if tensor.ndim < 2:
        return tensor, {"old_slices": [], "new_slices": []}
    if tensor.shape[0] == old:
        out = tensor.new_zeros((new, *tensor.shape[1:]))
        out[:old] = tensor
        return out, {"old_slices": [f"0:{old},..."], "new_slices": [f"{old}:{new},..."]}
    if tensor.shape[-1] == old:
        out = tensor.new_zeros((*tensor.shape[:-1], new))
        out[..., :old] = tensor
        return out, {"old_slices": [f"...,0:{old}"], "new_slices": [f"...,{old}:{new}"]}
    return tensor, {"old_slices": ["unchanged"], "new_slices": []}


def _expand_down(tensor: torch.Tensor, old: int, new: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if tensor.ndim < 2:
        return tensor, {"old_slices": [], "new_slices": []}
    # HF Linear down projection is typically [hidden, intermediate].
    if tensor.shape[-1] == old:
        out = tensor.new_zeros((*tensor.shape[:-1], new))
        out[..., :old] = tensor
        return out, {"old_slices": [f"...,0:{old}"], "new_slices": [f"...,{old}:{new}"]}
    if tensor.shape[0] == old:
        out = tensor.new_zeros((new, *tensor.shape[1:]))
        out[:old] = tensor
        return out, {"old_slices": [f"0:{old},..."], "new_slices": [f"{old}:{new},..."]}
    return tensor, {"old_slices": ["unchanged"], "new_slices": []}


def _expand_gate_up(tensor: torch.Tensor, old: int, new: int) -> tuple[torch.Tensor, dict[str, Any]]:
    if tensor.ndim < 2:
        return tensor, {"old_slices": [], "new_slices": []}
    if tensor.shape[0] == 2 * old:
        out = tensor.new_zeros((2 * new, *tensor.shape[1:]))
        out[:old] = tensor[:old]
        out[new : new + old] = tensor[old : 2 * old]
        return out, {
            "old_slices": [f"0:{old},...", f"{new}:{new + old},..."],
            "new_slices": [f"{old}:{new},...", f"{new + old}:{2 * new},..."],
        }
    if tensor.shape[-1] == 2 * old:
        out = tensor.new_zeros((*tensor.shape[:-1], 2 * new))
        out[..., :old] = tensor[..., :old]
        out[..., new : new + old] = tensor[..., old : 2 * old]
        return out, {
            "old_slices": [f"...,0:{old}", f"...,{new}:{new + old}"],
            "new_slices": [f"...,{old}:{new}", f"...,{new + old}:{2 * new}"],
        }
    return _expand_gate_or_up(tensor, old, new)
