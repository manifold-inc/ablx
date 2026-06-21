# ablx

`ablx` is a research CLI for expanding, upsampling, probing, benchmarking, and
training Hugging Face MoE checkpoints. The first target is
`Qwen/Qwen3.6-35B-A3B`.

The pipeline is intentionally artifact-heavy: every stage writes JSON/YAML
reports so a run can be audited after the fact.

## Install

```bash
python -m pip install -e ".[dev]"
```

Optional training and benchmark extras:

```bash
python -m pip install -e ".[train,benchmark,dev]"
```

## CLI

```bash
ablx inspect --model /path/to/checkpoint
ablx expand --config configs/tiny.yaml
ablx upsample --config configs/tiny.yaml
ablx probe --config configs/tiny.yaml --fail-on-gate
ablx benchmark --model /path/to/checkpoint --suite core
ablx compute-plan --config configs/qwen36_a3b.yaml
ablx train --config configs/tiny.yaml
ablx pipeline --config configs/tiny.yaml --dry-run
```

All commands emit JSON summaries to stdout and write detailed artifacts under
`output_dir`.

## Scope

This repository implements the mechanics needed to study MoE upscaling:

- deterministic expert intermediate expansion,
- expert fission with router expansion,
- constrained new-slice noise,
- preservation probes,
- parent-vs-child benchmark deltas,
- staged training plans and a lightweight local trainer path.

The default tests use small CPU fixtures. Large Qwen3.6 runs require local model
weights, training extras, and suitable GPU hardware.
