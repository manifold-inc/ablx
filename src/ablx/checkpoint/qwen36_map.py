from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Literal

TensorRole = Literal[
    "expert_gate_up",
    "expert_gate",
    "expert_up",
    "expert_down",
    "router",
    "shared_gate_up",
    "shared_gate",
    "shared_up",
    "shared_down",
    "mtp",
    "vision",
    "passthrough",
]

EXPERT_RE = re.compile(r"(?:^|\.)(experts?)\.(\d+)(?:\.|$)")


@dataclass(frozen=True)
class TensorTag:
    name: str
    role: TensorRole
    expert_index: int | None = None
    is_mtp: bool = False
    is_shared: bool = False
    is_vision: bool = False


def tag_tensor(name: str) -> TensorTag:
    lowered = name.lower()
    is_mtp = "mtp" in lowered or "multi_token" in lowered
    is_vision = "vision" in lowered or "visual" in lowered
    is_shared = "shared_expert" in lowered or "sharedexpert" in lowered
    expert_match = EXPERT_RE.search(name)
    expert_index = int(expert_match.group(2)) if expert_match else None

    if is_vision:
        return TensorTag(name, "vision", is_mtp=is_mtp, is_vision=True)
    if "router" in lowered or "gate.weight" in lowered and "mlp" in lowered and expert_index is None:
        return TensorTag(name, "router", is_mtp=is_mtp, is_shared=is_shared)

    prefix = "shared_" if is_shared else "expert_"
    if "gate_up_proj" in lowered or "gateup" in lowered:
        return TensorTag(name, f"{prefix}gate_up", expert_index, is_mtp, is_shared)
    if "gate_proj" in lowered:
        return TensorTag(name, f"{prefix}gate", expert_index, is_mtp, is_shared)
    if "up_proj" in lowered:
        return TensorTag(name, f"{prefix}up", expert_index, is_mtp, is_shared)
    if "down_proj" in lowered:
        return TensorTag(name, f"{prefix}down", expert_index, is_mtp, is_shared)
    if is_mtp:
        return TensorTag(name, "mtp", expert_index, is_mtp, is_shared)
    return TensorTag(name, "passthrough", expert_index, is_mtp, is_shared)


def is_moe_weight(name: str) -> bool:
    return tag_tensor(name).role in {
        "expert_gate_up",
        "expert_gate",
        "expert_up",
        "expert_down",
        "shared_gate_up",
        "shared_gate",
        "shared_up",
        "shared_down",
    }


def cloned_expert_name(name: str, old_index: int, new_index: int) -> str:
    pattern = re.compile(rf"((?:^|\.)experts?\. ){old_index}(?=\.|$)".replace(" ", ""))
    return pattern.sub(lambda match: match.group(1) + str(new_index), name, count=1)
