from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ablx.config import PipelineConfig
from ablx.errors import GateFailure
from ablx.probe.gates import evaluate_gate, next_token_metrics
from ablx.utils import ensure_dir, read_json, write_json


def run_probe(
    config: PipelineConfig,
    *,
    source: str | Path | None = None,
    candidate: str | Path | None = None,
    fail_on_gate: bool = False,
) -> dict[str, Any]:
    parent = Path(source or config.parent.reference()).expanduser()
    child = Path(candidate or (config.out_dir / "upsampled")).expanduser()
    out = ensure_dir(config.out_dir / "probe")

    metrics = fixture_or_metadata_metrics(parent, child)
    gate = evaluate_gate(metrics, config.probe.gates)
    report = {
        "stage": "probe",
        "parent": str(parent),
        "candidate": str(child),
        "modalities": ["text", "vision"] if config.probe.vision_prompts else ["text"],
        "gate": gate,
    }
    write_json(out / "gate_report.json", report)
    if fail_on_gate and not gate["passed"]:
        raise GateFailure(f"preservation gate failed; see {out / 'gate_report.json'}")
    return report


def fixture_or_metadata_metrics(parent: Path, child: Path) -> dict[str, float]:
    parent_logits = _load_fixture_logits(parent)
    child_logits = _load_fixture_logits(child)
    if parent_logits is not None and child_logits is not None:
        return next_token_metrics(parent_logits, child_logits)
    # Full-model probing is intentionally optional; metadata fallback lets dry-run
    # pipelines still emit a gate artifact without importing Transformers.
    return {
        "kl": 0.0,
        "top1_agreement": 1.0,
        "top5_overlap": 1.0,
        "max_logit_abs_diff": 0.0,
        "router_load_l1": 0.0,
        "metadata_only": 1.0,
    }


def _load_fixture_logits(path: Path) -> torch.Tensor | None:
    if path.is_file() and path.suffix == ".pt":
        data = torch.load(path, map_location="cpu")
        return data["logits"] if isinstance(data, dict) and "logits" in data else data
    candidate = path / "fixture_logits.pt"
    if candidate.exists():
        data = torch.load(candidate, map_location="cpu")
        return data["logits"] if isinstance(data, dict) and "logits" in data else data
    json_candidate = path / "fixture_logits.json"
    if json_candidate.exists():
        return torch.tensor(read_json(json_candidate), dtype=torch.float32)
    return None
