from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any

import torch

from ablx.config import PipelineConfig
from ablx.errors import ConfigError, OptionalDependencyError
from ablx.train.checkpoint import trained_checkpoint_dir
from ablx.utils import ensure_dir, write_json


def run_training_worker(config: PipelineConfig, *, config_path: str | Path | None = None) -> dict[str, Any]:
    out = ensure_dir(config.out_dir / "train")
    checkpoint = ensure_dir(trained_checkpoint_dir(config))
    student = config.out_dir / "upsampled"
    if not student.exists():
        raise ConfigError(f"student checkpoint does not exist: {student}")

    if config.train.backend in {"local", "local_smoke"} or config.name.startswith("tiny"):
        return run_local_smoke_worker(config, student, checkpoint, out, config_path=config_path)
    if config.train.backend != "accelerate_fsdp":
        raise ConfigError(f"unsupported training backend: {config.train.backend}")
    return run_hf_text_worker(config, student, checkpoint, out, config_path=config_path)


def run_local_smoke_worker(
    config: PipelineConfig,
    student: Path,
    checkpoint: Path,
    out: Path,
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    copy_checkpoint_tree(student, checkpoint)
    report = {
        "stage": "train-worker",
        "backend": config.train.backend,
        "status": "completed",
        "mode": "local_smoke_copy",
        "config_path": str(config_path) if config_path else None,
        "student": str(student),
        "checkpoint_path": str(checkpoint),
        "steps_completed": 0,
        "note": "local smoke worker copied the upsampled checkpoint to the trained-checkpoint contract path",
    }
    write_json(out / "worker_report.json", report)
    return report


def run_hf_text_worker(
    config: PipelineConfig,
    student: Path,
    checkpoint: Path,
    out: Path,
    *,
    config_path: str | Path | None,
) -> dict[str, Any]:
    text_samples = collect_text_samples(config.train.data)
    if not text_samples:
        raise ConfigError(
            "train.launch is true but train.data.text is empty. "
            "Add local text files or inline text samples before launching training."
        )
    try:
        from accelerate import Accelerator
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except Exception as exc:  # pragma: no cover - depends on optional deps
        raise OptionalDependencyError(
            "training dependencies are missing; install the project environment with: uv sync"
        ) from exc

    accelerator = Accelerator(mixed_precision="bf16")
    tokenizer_ref = config.parent.path or config.parent.hf_id or str(student)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_ref, trust_remote_code=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(student, torch_dtype=torch.bfloat16, trust_remote_code=False)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-6)
    model, optimizer = accelerator.prepare(model, optimizer)

    max_steps = config.train.max_worker_steps or max(sum(stage.steps for stage in config.train.stages), 1)
    max_steps = max(1, min(max_steps, 100_000))
    losses: list[float] = []
    for step in range(max_steps):
        text = text_samples[step % len(text_samples)]
        batch = tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=config.train.seq_len,
        )
        batch = {key: value.to(accelerator.device) for key, value in batch.items()}
        labels = batch["input_ids"].clone()
        outputs = model(**batch, labels=labels)
        loss = outputs.loss / max(config.train.grad_accum, 1)
        accelerator.backward(loss)
        if (step + 1) % max(config.train.grad_accum, 1) == 0:
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)
        losses.append(float(loss.detach().float().item()))

    accelerator.wait_for_everyone()
    if accelerator.is_main_process:
        unwrapped = accelerator.unwrap_model(model)
        unwrapped.save_pretrained(checkpoint, safe_serialization=True)
        tokenizer.save_pretrained(checkpoint)
        report = {
            "stage": "train-worker",
            "backend": config.train.backend,
            "status": "completed",
            "mode": "hf_text_ce",
            "config_path": str(config_path) if config_path else None,
            "student": str(student),
            "checkpoint_path": str(checkpoint),
            "steps_completed": max_steps,
            "losses": losses,
        }
        write_json(out / "worker_report.json", report)
    accelerator.wait_for_everyone()
    if not (out / "worker_report.json").exists():
        return {"stage": "train-worker", "status": "completed", "rank": "non_main"}
    return json.loads((out / "worker_report.json").read_text(encoding="utf-8"))


def collect_text_samples(data: dict[str, Any]) -> list[str]:
    samples: list[str] = []
    raw_items = data.get("text", []) if isinstance(data, dict) else []
    for item in raw_items:
        if isinstance(item, str):
            path = Path(item).expanduser()
            if path.exists():
                samples.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
            else:
                samples.append(item)
        elif isinstance(item, dict):
            if "text" in item:
                samples.append(str(item["text"]))
            elif "path" in item:
                path = Path(str(item["path"])).expanduser()
                if path.exists():
                    samples.extend(line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip())
    return samples


def copy_checkpoint_tree(source: Path, target: Path) -> None:
    for path in source.iterdir():
        dest = target / path.name
        if path.is_dir():
            if dest.exists():
                shutil.rmtree(dest)
            shutil.copytree(path, dest)
        elif path.is_file():
            shutil.copy2(path, dest)
