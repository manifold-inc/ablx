from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F

from ablx.config import ProbeGatesConfig


def next_token_metrics(parent_logits: torch.Tensor, child_logits: torch.Tensor) -> dict[str, float]:
    if parent_logits.shape != child_logits.shape:
        raise ValueError(f"logit shape mismatch: {tuple(parent_logits.shape)} != {tuple(child_logits.shape)}")
    p_log = F.log_softmax(parent_logits.float(), dim=-1)
    c_log = F.log_softmax(child_logits.float(), dim=-1)
    p = p_log.exp()
    kl = F.kl_div(c_log, p, reduction="batchmean", log_target=False)
    top1 = (parent_logits.argmax(dim=-1) == child_logits.argmax(dim=-1)).float().mean()
    top5_parent = parent_logits.topk(min(5, parent_logits.shape[-1]), dim=-1).indices
    top5_child = child_logits.topk(min(5, child_logits.shape[-1]), dim=-1).indices
    overlap = []
    for p_row, c_row in zip(top5_parent.reshape(-1, top5_parent.shape[-1]), top5_child.reshape(-1, top5_child.shape[-1])):
        overlap.append(len(set(p_row.tolist()) & set(c_row.tolist())) / float(p_row.numel()))
    return {
        "kl": float(kl.item()),
        "top1_agreement": float(top1.item()),
        "top5_overlap": float(sum(overlap) / max(len(overlap), 1)),
        "max_logit_abs_diff": float((parent_logits.float() - child_logits.float()).abs().max().item()),
    }


def router_load_l1(parent_load: torch.Tensor, child_load: torch.Tensor) -> float:
    if parent_load.numel() == 0 or child_load.numel() == 0:
        return 0.0
    parent = parent_load.float() / parent_load.float().sum().clamp_min(1e-9)
    child = child_load.float() / child_load.float().sum().clamp_min(1e-9)
    if child.numel() != parent.numel():
        factor = max(child.numel() // parent.numel(), 1)
        child = child.reshape(parent.numel(), factor).sum(dim=1)
    return float((parent - child).abs().sum().item())


def evaluate_gate(metrics: dict[str, float], thresholds: ProbeGatesConfig) -> dict[str, Any]:
    checks = {
        "kl": metrics.get("kl", 0.0) <= thresholds.max_kl,
        "top1_agreement": metrics.get("top1_agreement", 1.0) >= thresholds.top1_agreement,
        "top5_overlap": metrics.get("top5_overlap", 1.0) >= thresholds.top5_overlap,
        "max_logit_abs_diff": metrics.get("max_logit_abs_diff", 0.0) <= thresholds.max_logit_abs_diff,
        "router_load_l1": metrics.get("router_load_l1", 0.0) <= thresholds.max_router_load_l1,
    }
    return {"passed": all(checks.values()), "checks": checks, "metrics": metrics, "thresholds": thresholds.model_dump()}
