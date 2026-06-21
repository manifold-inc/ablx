from __future__ import annotations

from pathlib import Path
from typing import Any

from ablx.benchmark.suites import aggregate_domains
from ablx.utils import write_json


def compare_reports(
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    thresholds: dict[str, float],
    out_dir: str | Path | None = None,
    label: str = "candidate",
) -> dict[str, Any]:
    base_scores = dict(baseline.get("scores", {}))
    cand_scores = dict(candidate.get("scores", {}))
    base_aggs = aggregate_domains(base_scores)
    cand_aggs = aggregate_domains(cand_scores)
    deltas = []

    for task, base_value in sorted(base_scores.items()):
        if task not in cand_scores:
            continue
        cand_value = cand_scores[task]
        delta = float(cand_value) - float(base_value)
        rel = delta / max(abs(float(base_value)), 1e-9)
        deltas.append(
            {
                "task": task,
                "parent": float(base_value),
                label: float(cand_value),
                "delta_parent": delta,
                "relative_change": rel,
                "is_regression": delta < 0,
            }
        )

    aggregate_deltas = {}
    warnings = []
    for domain, base_value in base_aggs.items():
        if domain not in cand_aggs:
            continue
        delta = cand_aggs[domain] - base_value
        rel = delta / max(abs(base_value), 1e-9)
        threshold = thresholds.get(domain, -0.02)
        regressed = rel < threshold
        aggregate_deltas[domain] = {
            "parent": base_value,
            label: cand_aggs[domain],
            "delta_parent": delta,
            "relative_change": rel,
            "regression_threshold": threshold,
            "is_regression": regressed,
        }
        if regressed:
            warnings.append(f"{domain} regressed by {rel:.2%} below threshold {threshold:.2%}")

    report = {
        "baseline_model": baseline.get("model"),
        "candidate_model": candidate.get("model"),
        "label": label,
        "task_deltas": deltas,
        "aggregate_deltas": aggregate_deltas,
        "warnings": warnings,
        "soft_gate_passed": not warnings,
    }
    if out_dir:
        out = Path(out_dir)
        write_json(out / f"delta_vs_parent_{label}.json", report)
        (out / "BENCH_SUMMARY.md").write_text(render_summary(report), encoding="utf-8")
    return report


def render_summary(report: dict[str, Any]) -> str:
    lines = [
        "# Benchmark Summary",
        "",
        f"Baseline: `{report.get('baseline_model')}`",
        f"Candidate: `{report.get('candidate_model')}`",
        "",
        "| Aggregate | Parent | Candidate | Relative Change | Status |",
        "| --- | ---: | ---: | ---: | --- |",
    ]
    for domain, values in report.get("aggregate_deltas", {}).items():
        status = "regression" if values["is_regression"] else "ok"
        lines.append(
            f"| {domain} | {values['parent']:.4f} | {values[report['label']]:.4f} | "
            f"{values['relative_change']:.2%} | {status} |"
        )
    if report.get("warnings"):
        lines.extend(["", "## Warnings", ""])
        lines.extend(f"- {warning}" for warning in report["warnings"])
    return "\n".join(lines) + "\n"
