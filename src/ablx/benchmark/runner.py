from __future__ import annotations

import platform
from pathlib import Path
from typing import Any

from ablx.benchmark.adapters.lm_eval_adapter import run_lm_eval
from ablx.benchmark.adapters.native_mini import run_native_mini
from ablx.benchmark.suites import aggregate_domains, suite_tasks
from ablx.config import PipelineConfig
from ablx.errors import OptionalDependencyError
from ablx.utils import ensure_dir, write_json


def run_benchmark(
    model: str | Path,
    *,
    config: PipelineConfig | None = None,
    suite: str = "core",
    out: str | Path | None = None,
    limit: int | None = None,
    model_role: str | None = None,
    pipeline_stage: str | None = None,
) -> dict[str, Any]:
    out_dir = ensure_dir(out or ((config.out_dir / "benchmark") if config else "runs/ablx/benchmark"))
    limit = limit if limit is not None else (config.benchmark.limit if config else None)
    tasks = suite_tasks(suite)
    native = run_native_mini(model, tasks, limit=limit)
    scores = dict(native["scores"])
    task_reports = list(native["tasks"])
    optional_reports = []

    if config:
        for item in config.benchmark.suites:
            if isinstance(item, dict) and "lm_eval" in item:
                lm_cfg = item["lm_eval"] or {}
                try:
                    optional_reports.append(
                        run_lm_eval(
                            model,
                            tasks=list(lm_cfg.get("tasks", [])),
                            backend=str(lm_cfg.get("backend", "auto")),
                            num_fewshot=int(lm_cfg.get("num_fewshot", 0)),
                            limit=lm_cfg.get("limit"),
                        )
                    )
                except OptionalDependencyError as exc:
                    optional_reports.append({"backend": "lm_eval", "skipped": True, "reason": str(exc)})

    report = {
        "stage": "benchmark",
        "model": str(model),
        "checkpoint": {
            "path": str(model),
            "role": model_role,
            "pipeline_stage": pipeline_stage,
        },
        "suite": suite,
        "scores": scores,
        "aggregates": aggregate_domains(scores),
        "tasks": task_reports,
        "optional_reports": optional_reports,
        "metadata": {
            "python": platform.python_version(),
            "backend": "native_mini",
            "limit": limit,
        },
    }
    write_json(out_dir / "bench_report.json", report)
    return report
