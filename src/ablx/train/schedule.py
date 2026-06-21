from __future__ import annotations

from typing import Any

from ablx.config import TrainConfig


def build_stage_schedule(train: TrainConfig) -> dict[str, Any]:
    stages = [stage.model_dump() for stage in train.stages]
    if not stages:
        stages = [
            {"name": "preservation_warmup", "steps": 2000, "freeze": "old_slices", "lambda_kl": 1.0, "lambda_ce": 0.1},
            {"name": "capacity_wakeup", "steps": 8000, "freeze": "old_slices", "lambda_kl": 0.5, "lambda_ce": 1.0},
            {"name": "teacher_relax", "steps": 20000, "freeze": "partial_old", "lambda_kl": 0.1, "lambda_ce": 1.0},
        ]
    return {
        "stages": stages,
        "teacher_decay_rule": "decay lambda_kl only when G_preserve < delta",
        "soft_benchmark_monitoring": True,
    }


def should_decay_teacher(g_preserve: float, delta: float) -> bool:
    return g_preserve < delta
