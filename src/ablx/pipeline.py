from __future__ import annotations

from typing import Any
from pathlib import Path

from ablx.benchmark.compare import compare_reports
from ablx.benchmark.runner import run_benchmark
from ablx.compute.estimate import compute_plan
from ablx.config import PipelineConfig
from ablx.probe.runner import run_probe
from ablx.train.checkpoint import resolve_benchmark_checkpoint, trained_checkpoint_dir
from ablx.train.trainer import train_model
from ablx.transforms.expand import expand_checkpoint
from ablx.upsample.constrained_noise import upsample_checkpoint
from ablx.utils import ensure_dir, log_progress, write_json


def run_pipeline(
    config: PipelineConfig,
    *,
    config_path: str | Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    out = ensure_dir(config.out_dir)
    stages: dict[str, Any] = {}
    log_progress(f"pipeline: output_dir={out}")
    log_progress("pipeline: compute-plan")
    stages["compute"] = compute_plan(config)
    log_progress("pipeline: expand")
    stages["expand"] = expand_checkpoint(config)
    log_progress("pipeline: upsample")
    stages["upsample"] = upsample_checkpoint(config)
    log_progress("pipeline: probe")
    stages["probe"] = run_probe(config, fail_on_gate=not dry_run)

    benchmark_targets: dict[str, Any] = {
        "parent": {"path": config.parent.reference(), "role": "parent", "benchmarked": False},
        "lift": {"path": str(out / "upsampled"), "role": "upsampled", "benchmarked": False},
        "final": {"path": str(trained_checkpoint_dir(config)), "role": "trained", "benchmarked": False},
    }
    log_progress("pipeline: benchmark parent")
    bench_parent = run_benchmark(
        config.parent.reference(),
        config=config,
        suite="core",
        out=out / "bench_parent",
        model_role="parent",
        pipeline_stage="baseline",
    )
    benchmark_targets["parent"]["benchmarked"] = True
    log_progress("pipeline: benchmark lift")
    bench_lift = run_benchmark(
        out / "upsampled",
        config=config,
        suite="core",
        out=out / "bench_lift",
        model_role="upsampled",
        pipeline_stage="pre_train",
    )
    benchmark_targets["lift"]["benchmarked"] = True
    delta_lift = compare_reports(
        bench_parent,
        bench_lift,
        thresholds=config.benchmark.regression_thresholds,
        out_dir=out,
        label="lift",
        baseline_role="parent",
        candidate_role="upsampled",
    )
    stages["bench_parent"] = bench_parent
    stages["bench_lift"] = bench_lift
    stages["delta_lift"] = delta_lift
    log_progress("pipeline: train")
    stages["train"] = train_model(config, config_path=config_path, dry_run=dry_run)

    trained_path = resolve_benchmark_checkpoint(config, stages["train"])
    if trained_path is None:
        stages["bench_final"] = {
            "skipped": True,
            "reason": "training was not launched; no trained checkpoint exists",
            "expected_checkpoint": str(trained_checkpoint_dir(config)),
        }
        stages["delta_final"] = {"skipped": True, "reason": "bench_final skipped"}
    else:
        log_progress(f"pipeline: benchmark trained checkpoint {trained_path}")
        bench_final = run_benchmark(
            trained_path,
            config=config,
            suite="core",
            out=out / "bench_final",
            model_role="trained",
            pipeline_stage="post_train",
        )
        benchmark_targets["final"]["benchmarked"] = True
        delta_final = compare_reports(
            bench_parent,
            bench_final,
            thresholds=config.benchmark.regression_thresholds,
            out_dir=out,
            label="final",
            baseline_role="parent",
            candidate_role="trained",
        )
        stages["bench_final"] = bench_final
        stages["delta_final"] = delta_final
        write_json(out / "delta_vs_parent.json", delta_final)
    report = {
        "stage": "pipeline",
        "dry_run": dry_run,
        "output_dir": str(out),
        "benchmark_targets": benchmark_targets,
        "stages": stages,
    }
    write_json(out / "pipeline_report.json", report)
    log_progress(f"pipeline: wrote report {out / 'pipeline_report.json'}")
    return report
