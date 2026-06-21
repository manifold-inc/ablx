from __future__ import annotations

from pathlib import Path
from typing import Any

from ablx.utils import read_json, write_json


def build_freeze_masks(expansion_report: str | Path, out_dir: str | Path) -> dict[str, Any]:
    report = read_json(expansion_report)
    trainable = []
    frozen = []
    for item in report.get("tensor_reports", []):
        tensor = item.get("new_tensor") or item.get("tensor")
        if item.get("new_slices") or item.get("child_expert") is not None:
            trainable.append({"tensor": tensor, "slices": item.get("new_slices", ["full_child_expert"])})
        else:
            frozen.append({"tensor": tensor, "reason": "passthrough_or_inherited"})
    masks = {
        "policy": "old_slices_frozen_new_slices_trainable",
        "trainable": trainable,
        "frozen": frozen,
        "source_report": str(expansion_report),
    }
    write_json(Path(out_dir) / "freeze_masks.json", masks)
    return masks
