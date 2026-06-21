from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

_PROGRESS_FILE: Path | None = None


def ensure_dir(path: str | Path) -> Path:
    target = Path(path).expanduser().resolve()
    target.mkdir(parents=True, exist_ok=True)
    return target


def read_yaml(path: str | Path) -> dict[str, Any]:
    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        raise TypeError(f"YAML document must be a mapping: {path}")
    return data


def write_yaml(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
    return target


def write_json(path: str | Path, data: Any) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return target


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).expanduser().read_text(encoding="utf-8"))


def print_json(data: Any) -> None:
    print(json.dumps(data, indent=2, sort_keys=True))


def set_progress_file(path: str | Path | None) -> None:
    global _PROGRESS_FILE
    _PROGRESS_FILE = Path(path).expanduser().resolve() if path else None
    if _PROGRESS_FILE:
        _PROGRESS_FILE.parent.mkdir(parents=True, exist_ok=True)


def log_progress(message: str) -> None:
    line = f"[ablx] {message}"
    print(line, file=sys.stderr, flush=True)
    if _PROGRESS_FILE:
        timestamp = datetime.now(timezone.utc).isoformat()
        with _PROGRESS_FILE.open("a", encoding="utf-8") as handle:
            handle.write(f"{timestamp} {line}\n")
            handle.flush()


def compact_report(data: Any) -> Any:
    if not isinstance(data, dict):
        return data
    stage = data.get("stage")
    if stage == "pipeline":
        stages = data.get("stages", {})
        return {
            "stage": "pipeline",
            "dry_run": data.get("dry_run"),
            "output_dir": data.get("output_dir"),
            "benchmark_targets": data.get("benchmark_targets"),
            "stage_status": {
                name: _stage_status(value)
                for name, value in stages.items()
            },
            "report": str(Path(data.get("output_dir", ".")) / "pipeline_report.json"),
        }
    if stage == "expand":
        return {
            "stage": "expand",
            "source": data.get("source"),
            "out": data.get("out"),
            "written_shards": data.get("written_shards"),
            "expanded_tensor_count": data.get("expanded_tensor_count"),
            "new_tensor_count": data.get("new_tensor_count"),
            "report": str(Path(data.get("out", ".")) / "expansion_report.json"),
        }
    if stage == "upsample":
        return {
            "stage": "upsample",
            "source": data.get("source"),
            "out": data.get("out"),
            "written_shards": data.get("written_shards"),
            "affected_tensor_count": data.get("affected_tensor_count"),
            "report": str(Path(data.get("out", ".")) / "upsample_report.json"),
        }
    return data


def _stage_status(value: Any) -> Any:
    if not isinstance(value, dict):
        return "completed"
    if value.get("skipped"):
        return {"status": "skipped", "reason": value.get("reason")}
    if "status" in value:
        return value.get("status")
    if "gate" in value:
        return {"passed": value["gate"].get("passed")}
    if "written_shards" in value:
        return {"written_shards": len(value.get("written_shards") or [])}
    if "checkpoint" in value:
        return {"checkpoint": value.get("checkpoint")}
    return "completed"


def rel_artifact(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
