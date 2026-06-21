from __future__ import annotations

from pathlib import Path
from typing import Any

from ablx.utils import read_json


def load_expansion_masks(report_path: str | Path) -> dict[str, Any]:
    report = read_json(report_path)
    by_tensor: dict[str, list[dict[str, Any]]] = {}
    clone_groups: dict[str, list[str]] = {}
    for item in report.get("tensor_reports", []):
        tensor_name = item.get("new_tensor") or item.get("tensor")
        if not tensor_name:
            continue
        by_tensor.setdefault(tensor_name, []).append(item)
        if "child_expert" in item:
            clone_groups.setdefault(str(item.get("old_expert")), []).append(str(tensor_name))
    return {"by_tensor": by_tensor, "clone_groups": clone_groups, "raw": report}
