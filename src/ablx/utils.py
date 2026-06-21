from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


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


def rel_artifact(path: str | Path, root: str | Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(Path(root).resolve()))
    except ValueError:
        return str(path)
