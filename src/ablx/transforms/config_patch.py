from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def patch_qwen36_config(
    checkpoint_dir: str | Path,
    *,
    moe_intermediate_size: int | None = None,
    shared_expert_intermediate_size: int | None = None,
    num_experts: int | None = None,
) -> dict[str, Any]:
    root = Path(checkpoint_dir)
    path = root / "config.json"
    if not path.exists():
        return {"patched": False, "reason": "config.json not found"}
    config = json.loads(path.read_text(encoding="utf-8"))
    text_config = config.setdefault("text_config", {})

    if moe_intermediate_size is not None:
        text_config["moe_intermediate_size"] = int(moe_intermediate_size)
    if shared_expert_intermediate_size is not None:
        text_config["shared_expert_intermediate_size"] = int(shared_expert_intermediate_size)
    if num_experts is not None:
        text_config["num_experts"] = int(num_experts)

    path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return {
        "patched": True,
        "moe_intermediate_size": text_config.get("moe_intermediate_size"),
        "shared_expert_intermediate_size": text_config.get("shared_expert_intermediate_size"),
        "num_experts": text_config.get("num_experts"),
    }
