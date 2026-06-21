from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ablx.config import ExpandTransformConfig, PipelineConfig
from ablx.checkpoint.safetensors_io import rewrite_checkpoint
from ablx.fixtures import create_tiny_checkpoint
from ablx.transforms.clone_experts import ExpertCloneConfig, clone_expert_tensor
from ablx.transforms.config_patch import patch_qwen36_config
from ablx.transforms.moe_intermediate import IntermediateExpansion, expand_intermediate_tensor
from ablx.utils import ensure_dir, write_json


def expand_checkpoint(config: PipelineConfig) -> dict[str, Any]:
    source = Path(config.expand.source or config.parent.reference()).expanduser()
    if not source.exists() and config.name.startswith("tiny"):
        source = create_tiny_checkpoint(source)
    out = ensure_dir(config.out_dir / "expanded")
    transforms = config.expand.transforms
    reports: list[dict[str, Any]] = []
    config_updates: dict[str, Any] = {}

    def apply_all(name: str, tensor: torch.Tensor):
        candidates: list[tuple[str, torch.Tensor]] = [(name, tensor)]
        for transform in transforms:
            next_candidates: list[tuple[str, torch.Tensor]] = []
            if transform.type == "expand_moe_intermediate":
                stage_reports = []
                exp_cfg = IntermediateExpansion(
                    old_size=int(transform.old_intermediate_size or 512),
                    new_size=int(transform.new_intermediate_size or 768),
                    include_shared_expert=transform.include_shared_expert,
                    include_mtp=transform.include_mtp,
                )
                config_updates["moe_intermediate_size"] = exp_cfg.new_size
                if transform.include_shared_expert:
                    config_updates["shared_expert_intermediate_size"] = exp_cfg.new_size
                for cand_name, cand_tensor in candidates:
                    new_tensor, report = expand_intermediate_tensor(cand_name, cand_tensor, exp_cfg)
                    if report:
                        stage_reports.append(report)
                    next_candidates.append((cand_name, new_tensor))
                reports.extend(stage_reports)
            elif transform.type == "clone_experts":
                clone_cfg = ExpertCloneConfig(
                    factor=transform.factor,
                    zero_mean_deltas=transform.zero_mean_deltas,
                    include_shared_expert=transform.include_shared_expert,
                    include_mtp=transform.include_mtp,
                )
                for cand_name, cand_tensor in candidates:
                    for new_name, new_tensor, report in clone_expert_tensor(cand_name, cand_tensor, clone_cfg):
                        if report:
                            reports.append(report)
                        next_candidates.append((new_name, new_tensor))
                config_updates["num_experts_factor"] = transform.factor
            else:
                next_candidates = candidates
            candidates = next_candidates
        for cand_name, cand_tensor in candidates:
            yield cand_name, cand_tensor

    written = rewrite_checkpoint(source, out, apply_all)
    num_experts = _patched_num_experts(config, reports)
    patch_report = patch_qwen36_config(
        out,
        moe_intermediate_size=config_updates.get("moe_intermediate_size"),
        shared_expert_intermediate_size=config_updates.get("shared_expert_intermediate_size"),
        num_experts=num_experts,
    )
    report = {
        "stage": "expand",
        "source": str(source),
        "out": str(out),
        "written_shards": written,
        "tensor_reports": reports,
        "transforms": [transform.model_dump() for transform in transforms],
        "config_patch": patch_report,
        "new_tensor_count": sum(1 for item in reports if item.get("new_tensor")),
        "expanded_tensor_count": sum(1 for item in reports if item.get("new_shape") != item.get("old_shape")),
    }
    write_json(out / "expansion_report.json", report)
    return report


def _patched_num_experts(config: PipelineConfig, reports: list[dict[str, Any]]) -> int | None:
    factors = [t.factor for t in config.expand.transforms if t.type == "clone_experts" and t.factor > 1]
    if not factors:
        return None
    max_child = max((int(r["child_expert"]) for r in reports if "child_expert" in r), default=-1)
    return max_child + 1 if max_child >= 0 else None
