from __future__ import annotations

from typing import Any

from ablx.config import PipelineConfig
from ablx.utils import ensure_dir, write_json


def compute_plan(config: PipelineConfig) -> dict[str, Any]:
    text = {
        "base_total_params": 35_000_000_000,
        "base_active_params": 3_000_000_000,
        "old_moe_intermediate": 512,
        "new_moe_intermediate": 768,
        "old_experts": 256,
        "new_experts": 512,
        "num_experts_per_tok": 8,
    }
    widen_ratio = text["new_moe_intermediate"] / text["old_moe_intermediate"]
    expert_ratio = text["new_experts"] / text["old_experts"]
    # Expert-heavy approximation: passthrough params remain, routed expert params scale.
    expert_fraction = 0.78
    total_params = int(text["base_total_params"] * ((1 - expert_fraction) + expert_fraction * widen_ratio * expert_ratio))
    active_params = int(text["base_active_params"] * ((1 - expert_fraction) + expert_fraction * widen_ratio))
    flops_per_token = 6 * active_params
    seq_len = config.train.seq_len
    micro_batch = config.train.micro_batch
    gpus = config.train.num_gpus
    gpu_mem_gb = 288
    bytes_bf16 = 2
    bytes_adam = 8
    param_mem = total_params * bytes_bf16 / gpus
    grad_mem = total_params * bytes_bf16 / gpus
    opt_mem = active_params * bytes_adam / gpus
    activation_mem = micro_batch * seq_len * 2048 * 40 * bytes_bf16 * 4
    total_per_gpu_gb = (param_mem + grad_mem + opt_mem + activation_mem) / 1e9
    headroom_gb = gpu_mem_gb - total_per_gpu_gb
    report = {
        "stage": "compute-plan",
        "hardware": {"gpu": "B300", "num_gpus": gpus, "vram_per_gpu_gb": gpu_mem_gb},
        "model": {
            **text,
            "estimated_child_total_params": total_params,
            "estimated_child_active_params": active_params,
        },
        "training": {
            "seq_len": seq_len,
            "micro_batch": micro_batch,
            "grad_accum": config.train.grad_accum,
            "flops_per_token": flops_per_token,
            "flops_per_step": flops_per_token * seq_len * micro_batch * gpus,
        },
        "memory_per_gpu_gb": {
            "params_bf16_fsdp": param_mem / 1e9,
            "grads_bf16_fsdp": grad_mem / 1e9,
            "optimizer_active_adam": opt_mem / 1e9,
            "activations_rough": activation_mem / 1e9,
            "total_estimate": total_per_gpu_gb,
            "headroom": headroom_gb,
        },
        "recommendations": {
            "start_micro_batch": 1,
            "start_seq_len": min(seq_len, 4096),
            "start_grad_accum": max(config.train.grad_accum, 8),
            "activation_checkpointing": True,
            "fsdp": "FULL_SHARD",
            "teacher_strategy": "rank_subset_recompute_or_cpu_offload",
        },
        "safe_to_launch": headroom_gb > 24,
    }
    out = ensure_dir(config.out_dir / "compute")
    write_json(out / "compute_plan.json", report)
    return report
