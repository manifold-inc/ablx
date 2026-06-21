from __future__ import annotations

from typing import Any

from ablx.benchmark.compare import compare_reports
from ablx.benchmark.runner import run_benchmark
from ablx.compute.estimate import compute_plan
from ablx.config import PipelineConfig
from ablx.probe.runner import run_probe
from ablx.train.trainer import train_model
from ablx.transforms.expand import expand_checkpoint
from ablx.upsample.constrained_noise import upsample_checkpoint
from ablx.utils import ensure_dir, write_json


def run_pipeline(config: PipelineConfig, *, dry_run: bool = False) -> dict[str, Any]:
    out = ensure_dir(config.out_dir)
    stages: dict[str, Any] = {}
    stages["compute"] = compute_plan(config)
    stages["expand"] = expand_checkpoint(config)
    stages["upsample"] = upsample_checkpoint(config)
    stages["probe"] = run_probe(config, fail_on_gate=not dry_run)

    bench_parent = run_benchmark(config.parent.reference(), config=config, suite="core", out=out / "bench_parent")
    bench_lift = run_benchmark(out / "upsampled", config=config, suite="core", out=out / "bench_lift")
    delta_lift = compare_reports(
        bench_parent,
        bench_lift,
        thresholds=config.benchmark.regression_thresholds,
        out_dir=out,
        label="lift",
    )
    stages["bench_parent"] = bench_parent
    stages["bench_lift"] = bench_lift
    stages["delta_lift"] = delta_lift
    stages["train"] = train_model(config, dry_run=dry_run)

    bench_final = run_benchmark(out / "upsampled", config=config, suite="core", out=out / "bench_final")
    delta_final = compare_reports(
        bench_parent,
        bench_final,
        thresholds=config.benchmark.regression_thresholds,
        out_dir=out,
        label="final",
    )
    stages["bench_final"] = bench_final
    stages["delta_final"] = delta_final
    report = {"stage": "pipeline", "dry_run": dry_run, "output_dir": str(out), "stages": stages}
    write_json(out / "pipeline_report.json", report)
    write_json(out / "delta_vs_parent.json", delta_final)
    return report
