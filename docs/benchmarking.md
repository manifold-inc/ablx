# Benchmarking

Benchmarking answers whether the upscaled checkpoint is better, worse, or the
same on measured capabilities.

The pipeline writes:

- `bench_parent/bench_report.json`
- `bench_lift/bench_report.json`
- `bench_final/bench_report.json`
- `delta_vs_parent.json`
- `BENCH_SUMMARY.md`

The default `core` suite covers coding/agentic, reasoning/math, multimodal, and
long-context aggregates. It is small so it can run at every stage. Full
`lm-eval` tasks are optional and normalized into the same report schema.

Benchmark regressions are soft-gated by default: they produce warnings and
machine-readable deltas but do not stop training unless `hard_fail_on_regression`
is enabled in config.

For production runs, use held-out datasets that are disjoint from the training
mixture. Record dataset revisions and contamination notes in config.
