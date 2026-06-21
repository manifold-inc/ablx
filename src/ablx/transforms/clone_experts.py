from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Iterable

import torch

from ablx.checkpoint.qwen36_map import EXPERT_RE, tag_tensor
from .router_expand import expand_router_tensor


@dataclass(frozen=True)
class ExpertCloneConfig:
    factor: int = 2
    zero_mean_deltas: bool = True
    include_shared_expert: bool = True
    include_mtp: bool = True


def clone_expert_tensor(
    name: str,
    tensor: torch.Tensor,
    cfg: ExpertCloneConfig,
) -> list[tuple[str, torch.Tensor, dict[str, Any] | None]]:
    tag = tag_tensor(name)
    if tag.is_mtp and not cfg.include_mtp:
        return [(name, tensor, None)]
    if tag.role == "router":
        expanded, report = expand_router_tensor(name, tensor, factor=cfg.factor)
        return [(name, expanded, report)]
    if tag.expert_index is None or cfg.factor <= 1:
        return [(name, tensor, None)]

    out: list[tuple[str, torch.Tensor, dict[str, Any] | None]] = []
    old_index = tag.expert_index
    for offset in range(cfg.factor):
        new_index = old_index * cfg.factor + offset
        new_name = replace_expert_index(name, old_index, new_index)
        report = {
            "tensor": name,
            "new_tensor": new_name,
            "role": tag.role,
            "old_expert": old_index,
            "child_expert": new_index,
            "clone_offset": offset,
            "zero_mean_delta_group": old_index,
            "old_shape": list(tensor.shape),
            "new_shape": list(tensor.shape),
            "paper_sections": ["11"],
        }
        out.append((new_name, tensor.clone(), report))
    return out


def replace_expert_index(name: str, old_index: int, new_index: int) -> str:
    pattern = re.compile(rf"((?:^|\.)experts?\. ){old_index}(?=\.|$)".replace(" ", ""))
    return pattern.sub(lambda match: match.group(1) + str(new_index), name, count=1)


def infer_num_experts(names: Iterable[str]) -> int | None:
    max_seen: int | None = None
    for name in names:
        match = EXPERT_RE.search(name)
        if match:
            value = int(match.group(2))
            max_seen = value if max_seen is None else max(max_seen, value)
    return None if max_seen is None else max_seen + 1
