from __future__ import annotations

from pathlib import Path
from typing import Any

from ablx.config import PipelineConfig
from ablx.checkpoint.safetensors_io import checkpoint_files
from ablx.errors import CheckpointFormatError, TrainingLaunchError


def trained_checkpoint_dir(config: PipelineConfig) -> Path:
    base = Path(config.train.checkpoint_dir).expanduser() if config.train.checkpoint_dir else config.out_dir / "train" / "checkpoint"
    return base / config.train.output_checkpoint


def is_valid_checkpoint(path: str | Path) -> bool:
    try:
        return bool(checkpoint_files(path))
    except CheckpointFormatError:
        return False


def resolve_benchmark_checkpoint(config: PipelineConfig, train_report: dict[str, Any]) -> Path | None:
    path = Path(str(train_report.get("checkpoint_path") or trained_checkpoint_dir(config))).expanduser()
    status = train_report.get("status")
    if status == "completed" and is_valid_checkpoint(path):
        return path
    if train_report.get("launch") is False or status == "launch_disabled_plan_emitted":
        return None
    raise TrainingLaunchError(
        f"training was requested but no completed checkpoint was found at {path}; "
        "not benchmarking the pre-training upsampled checkpoint as final"
    )
