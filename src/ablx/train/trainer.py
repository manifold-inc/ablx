from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

from ablx.config import PipelineConfig
from ablx.errors import AblxError, ConfigError, TrainingLaunchError
from ablx.train.checkpoint import trained_checkpoint_dir
from ablx.train.data import data_manifest
from ablx.train.freeze import build_freeze_masks
from ablx.train.schedule import build_stage_schedule
from ablx.utils import ensure_dir, write_json


def train_model(
    config: PipelineConfig,
    *,
    config_path: str | Path | None = None,
    dry_run: bool | None = None,
) -> dict[str, Any]:
    out = ensure_dir(config.out_dir / "train")
    launch = config.train.launch if dry_run is None else (config.train.launch and not dry_run)
    expansion_report = config.out_dir / "expanded" / "expansion_report.json"
    freeze_masks = {}
    if expansion_report.exists():
        freeze_masks = build_freeze_masks(expansion_report, out)
    schedule = build_stage_schedule(config.train)
    manifest = data_manifest(config.train.data)
    commands = build_training_commands(config, config_path=config_path)
    checkpoint_path = trained_checkpoint_dir(config)
    report = {
        "stage": "train",
        "backend": config.train.backend,
        "launch": launch,
        "num_gpus": config.train.num_gpus,
        "student": str(config.out_dir / "upsampled"),
        "teacher": config.parent.reference(),
        "checkpoint_path": str(checkpoint_path),
        "benchmark_checkpoint": None,
        "freeze_masks": freeze_masks,
        "schedule": schedule,
        "data_manifest": manifest,
        "commands": commands,
        "status": "launch_disabled_plan_emitted" if not launch else "launch_requested",
    }
    write_json(out / "schedule.json", schedule)
    write_json(out / "data_manifest.json", manifest)
    write_json(out / "commands.json", commands)
    write_json(out / "train_report.json", report)
    if not launch:
        return report

    validate_training_launch(config, config_path=config_path)
    result = launch_training_worker(commands[0], cwd=Path.cwd())
    report["launch_result"] = result
    if int(result["returncode"]) != 0:
        report["status"] = "failed"
        write_json(out / "train_report.json", report)
        raise TrainingLaunchError(
            f"training worker failed with exit code {result['returncode']}; "
            f"stderr: {result.get('stderr', '').strip()[:1000]}"
        )
    report["status"] = "completed"
    report["benchmark_checkpoint"] = str(checkpoint_path)
    write_json(out / "train_report.json", report)
    return report


def build_training_commands(config: PipelineConfig, *, config_path: str | Path | None = None) -> list[list[str]]:
    if config_path is None:
        cfg_path = "<missing-config-path>"
    else:
        cfg_path = str(Path(config_path).expanduser().resolve())
    if config.train.backend in {"local", "local_smoke"} or config.name.startswith("tiny"):
        return [[sys.executable, "-m", "ablx.cli", "train-worker", "--config", cfg_path]]
    return [
        [
            "torchrun",
            f"--nproc_per_node={config.train.num_gpus}",
            "-m",
            "ablx.cli",
            "train-worker",
            "--config",
            cfg_path,
        ]
    ]


def validate_training_launch(config: PipelineConfig, *, config_path: str | Path | None) -> None:
    if config_path is None or str(config_path) == "<missing-config-path>":
        raise ConfigError("training launch requires a real --config path; refusing to run ${CONFIG} placeholder")
    if not (config.out_dir / "upsampled").exists():
        raise ConfigError(f"student checkpoint is missing: {config.out_dir / 'upsampled'}")
    if config.train.num_gpus < 1:
        raise ConfigError("train.num_gpus must be >= 1")
    if config.train.backend not in {"accelerate_fsdp", "local", "local_smoke"}:
        raise ConfigError(f"unsupported training backend: {config.train.backend}")
    if config.train.backend == "accelerate_fsdp":
        if shutil.which("torchrun") is None:
            raise ConfigError("torchrun was not found on PATH")
        text_items = config.train.data.get("text", []) if isinstance(config.train.data, dict) else []
        if not text_items:
            raise ConfigError(
                "train.launch is true but train.data.text is empty. "
                "Add local text files or inline text samples before launching training."
            )


def launch_training_worker(command: list[str], *, cwd: Path) -> dict[str, Any]:
    proc = subprocess.run(command, cwd=cwd, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }
