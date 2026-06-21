from __future__ import annotations

from pathlib import Path
from typing import Any

from ablx.config import PipelineConfig
from ablx.train.data import data_manifest
from ablx.train.freeze import build_freeze_masks
from ablx.train.schedule import build_stage_schedule
from ablx.utils import ensure_dir, write_json


def train_model(config: PipelineConfig, *, dry_run: bool | None = None) -> dict[str, Any]:
    out = ensure_dir(config.out_dir / "train")
    launch = config.train.launch if dry_run is None else (config.train.launch and not dry_run)
    expansion_report = config.out_dir / "expanded" / "expansion_report.json"
    freeze_masks = {}
    if expansion_report.exists():
        freeze_masks = build_freeze_masks(expansion_report, out)
    schedule = build_stage_schedule(config.train)
    manifest = data_manifest(config.train.data)
    commands = build_training_commands(config)
    report = {
        "stage": "train",
        "backend": config.train.backend,
        "launch": launch,
        "num_gpus": config.train.num_gpus,
        "student": str(config.out_dir / "upsampled"),
        "teacher": config.parent.reference(),
        "freeze_masks": freeze_masks,
        "schedule": schedule,
        "data_manifest": manifest,
        "commands": commands,
        "status": "launch_disabled_plan_emitted" if not launch else "ready_to_launch",
    }
    write_json(out / "schedule.json", schedule)
    write_json(out / "data_manifest.json", manifest)
    write_json(out / "commands.json", commands)
    write_json(out / "train_report.json", report)
    return report


def build_training_commands(config: PipelineConfig) -> list[str]:
    cfg_path = "${CONFIG}"
    return [
        "torchrun "
        f"--nproc_per_node={config.train.num_gpus} "
        "-m ablx.cli train "
        f"--config {cfg_path} "
        "--launch"
    ]
