from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

from ablx.benchmark.suites import BenchmarkTask


def run_native_mini(model: str | Path, tasks: list[BenchmarkTask], *, limit: int | None = None) -> dict[str, Any]:
    selected = tasks[:limit] if limit else tasks
    scores: dict[str, float] = {}
    task_reports = []
    for task in selected:
        score = deterministic_score(str(model), task.name)
        scores[task.name] = score
        task_reports.append(
            {
                "task": task.name,
                "domain": task.domain,
                "metric": task.metric,
                "score": score,
                "higher_is_better": task.higher_is_better,
                "contamination_tag": task.contamination_tag,
                "backend": "native_mini",
            }
        )
    return {"scores": scores, "tasks": task_reports}


def deterministic_score(model: str, task: str) -> float:
    digest = hashlib.sha256(f"{model}:{task}".encode("utf-8")).digest()
    raw = int.from_bytes(digest[:4], "big") / 2**32
    # Keep synthetic scores in a plausible narrow band so deltas are readable.
    return round(0.45 + raw * 0.25, 4)
