from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional

import yaml

from .errors import AblxError
from .models import BenchmarkReport, ProbeReport
from .probe import metadata_probe


def run_benchmark_stage(
    *,
    stage: str,
    source: str | Path,
    candidate: str | Path,
    out: str | Path,
    config: Mapping[str, object],
    run_commands: bool = True,
    probe_report: Optional[ProbeReport] = None,
    final_model: str | Path | None = None,
) -> BenchmarkReport:
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    source_path = str(Path(source).expanduser().resolve())
    candidate_path = str(Path(candidate).expanduser().resolve())
    target_path = str(Path(final_model).expanduser().resolve()) if final_model else candidate_path
    suite = str(config.get("suite", "preserve"))
    backend = str(config.get("serve_backend") or config.get("backend") or "metadata")
    commands = build_benchmark_commands(config, source_path, candidate_path, target_path, stage)

    metrics = build_benchmark_metrics(stage, source_path, target_path, probe_report)
    gates = collect_stage_gates(config, stage)
    passed, gate_metrics, warnings = evaluate_gates(metrics, gates)
    metrics["gates"] = gate_metrics

    command_results: List[Dict[str, object]] = []
    if run_commands:
        for command in commands:
            command_results.append(run_command(command, cwd=out_dir))
        if any(result["returncode"] != 0 for result in command_results):
            passed = False
            warnings.append("one or more benchmark commands failed")

    plan = {
        "stage": stage,
        "suite": suite,
        "backend": backend,
        "source": source_path,
        "candidate": candidate_path,
        "target": target_path,
        "commands": commands,
        "gates": gates,
    }
    report = BenchmarkReport(
        stage=stage,
        source=source_path,
        candidate=target_path,
        out=str(out_dir),
        suite=suite,
        backend=backend,
        commands=commands,
        command_results=command_results,
        metrics=metrics,
        gates=gates,
        passed=passed,
        warnings=warnings,
    )

    (out_dir / "benchmark_plan.yaml").write_text(yaml.safe_dump(plan, sort_keys=False), encoding="utf-8")
    (out_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    (out_dir / "summary.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(render_benchmark_readme(stage, suite, backend), encoding="utf-8")
    return report


def build_benchmark_commands(
    config: Mapping[str, object],
    source: str,
    candidate: str,
    target: str,
    stage: str,
) -> List[str]:
    raw = config.get("commands")
    if isinstance(raw, list):
        return [render_command(str(command), source, candidate, target, stage) for command in raw]
    if isinstance(raw, str):
        return [render_command(raw, source, candidate, target, stage)]

    backend = str(config.get("serve_backend") or config.get("backend") or "metadata")
    if backend in {"metadata", "none"}:
        return []

    tasks = config.get("lm_eval_tasks") or config.get("tasks") or ["wikitext", "hellaswag", "arc_challenge", "gsm8k"]
    if isinstance(tasks, str):
        task_arg = tasks
    elif isinstance(tasks, list):
        task_arg = ",".join(str(task) for task in tasks)
    else:
        task_arg = "wikitext,hellaswag,arc_challenge,gsm8k"

    base_url_key = "candidate_base_url" if stage == "pre" else "tuned_base_url"
    base_url = str(config.get(base_url_key) or config.get("base_url") or "http://127.0.0.1:8000/v1/completions")
    model_name = str(config.get("served_model_name") or ("candidate" if stage == "pre" else "tuned"))
    return [
        "lm_eval "
        "--model local-completions "
        f"--model_args model={model_name},base_url={base_url} "
        f"--tasks {task_arg} "
        "--batch_size auto "
        "--output_path results.json "
        "--log_samples"
    ]


def render_command(command: str, source: str, candidate: str, target: str, stage: str) -> str:
    return command.format(source=source, candidate=candidate, target=target, tuned=target, stage=stage)


def build_benchmark_metrics(stage: str, source: str, target: str, probe_report: Optional[ProbeReport]) -> Dict[str, object]:
    metrics: Dict[str, object] = {
        "stage": stage,
        "source": source,
        "target": target,
    }
    if probe_report is not None:
        metrics["probe"] = probe_report.metrics
        logit_probe = probe_report.metrics.get("logit_probe") if isinstance(probe_report.metrics, dict) else None
        if isinstance(logit_probe, dict):
            metrics["mean_kl"] = logit_probe.get("mean_kl")
            metrics["top1_agreement"] = logit_probe.get("top1_agreement")
    else:
        try:
            metrics["metadata"] = metadata_probe(source, target)
        except AblxError as exc:
            metrics["metadata_error"] = str(exc)
    metadata = metrics.get("metadata")
    if not metadata and probe_report is not None:
        metadata = probe_report.metrics
    if isinstance(metadata, dict):
        metrics["missing_in_candidate_count"] = len(metadata.get("missing_in_candidate", []))
        metrics["changed_shape_count"] = metadata.get("changed_shape_count")
    return metrics


def collect_stage_gates(config: Mapping[str, object], stage: str) -> Dict[str, object]:
    gates = config.get("gates", {})
    if not isinstance(gates, Mapping):
        return {}
    specific = gates.get(stage)
    if isinstance(specific, Mapping):
        return {str(k): v for k, v in specific.items()}
    return {str(k): v for k, v in gates.items() if k not in {"pre", "post"}}


def evaluate_gates(metrics: Mapping[str, object], gates: Mapping[str, object]) -> tuple[bool, Dict[str, object], List[str]]:
    passed = True
    results: Dict[str, object] = {}
    warnings: List[str] = []
    for gate, expected in gates.items():
        metric_name, op = parse_gate(gate)
        actual = lookup_metric(metrics, metric_name)
        gate_passed = compare_gate(actual, expected, op)
        results[gate] = {"metric": metric_name, "actual": actual, "expected": expected, "passed": gate_passed}
        if not gate_passed:
            passed = False
            warnings.append(f"gate failed: {gate} actual={actual!r} expected={expected!r}")
    return passed, results, warnings


def parse_gate(gate: str) -> tuple[str, str]:
    for suffix, op in (("_max", "max"), ("_min", "min"), ("_equals", "equals")):
        if gate.endswith(suffix):
            return gate[: -len(suffix)], op
    return gate, "equals"


def lookup_metric(metrics: Mapping[str, object], name: str) -> object:
    if name in metrics:
        return metrics[name]
    parts = name.split(".")
    value: object = metrics
    for part in parts:
        if isinstance(value, Mapping) and part in value:
            value = value[part]
        else:
            return None
    return value


def compare_gate(actual: object, expected: object, op: str) -> bool:
    if actual is None:
        return False
    if op == "equals":
        return actual == expected
    try:
        actual_f = float(actual)
        expected_f = float(expected)
    except (TypeError, ValueError):
        return False
    if op == "max":
        return actual_f <= expected_f
    if op == "min":
        return actual_f >= expected_f
    return False


def run_command(command: str, cwd: Path) -> Dict[str, object]:
    proc = subprocess.run(
        command,
        cwd=str(cwd),
        shell=True,
        text=True,
        capture_output=True,
        check=False,
    )
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def render_benchmark_readme(stage: str, suite: str, backend: str) -> str:
    return f"""# ablx benchmark {stage}

Suite: `{suite}`
Backend: `{backend}`

This directory contains the benchmark plan, commands, and summary for one
pipeline benchmark stage. Pre-training benchmarks are preservation gates;
post-training benchmarks are preservation plus new-capacity usefulness gates.
"""
