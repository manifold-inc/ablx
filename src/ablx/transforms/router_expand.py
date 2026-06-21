from __future__ import annotations

from typing import Any

import torch


def expand_router_tensor(name: str, tensor: torch.Tensor, *, factor: int) -> tuple[torch.Tensor, dict[str, Any] | None]:
    if factor <= 1:
        return tensor, None
    if tensor.ndim != 2:
        return tensor, None

    # Qwen-style routers are [num_experts, hidden]. If a future checkpoint uses
    # the transposed layout this remains easy to detect from the tensor report.
    out = tensor.repeat_interleave(factor, dim=0)
    return out, _report(name, tensor, out, "rows")


def _report(name: str, old: torch.Tensor, new: torch.Tensor, axis: str) -> dict[str, Any]:
    return {
        "tensor": name,
        "role": "router",
        "old_shape": list(old.shape),
        "new_shape": list(new.shape),
        "mapping": f"repeat_interleave_{axis}",
        "paper_sections": ["11", "22.3"],
    }
