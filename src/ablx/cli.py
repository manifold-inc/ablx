from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import AblxError
from .inspection import inspect_model
from .pipeline import load_pipeline_config, run_pipeline
from .probe import probe_checkpoints
from .recipes import default_qwen36_recipe, load_recipe
from .slime import emit_slime_plan
from .transforms import upscale_checkpoint


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        result = args.func(args)
    except AblxError as exc:
        print(f"ablx: error: {exc}", file=sys.stderr)
        return 2
    if result is not None:
        print(json.dumps(result, indent=2, sort_keys=True))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="ablx", description="Weight-space upscaling tools for MoE checkpoints.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    inspect_p = subparsers.add_parser("inspect", help="Inspect config and safetensors metadata.")
    inspect_p.add_argument("--model", required=True, help="Local checkpoint path or Hugging Face model ID.")
    inspect_p.add_argument("--no-tensors", action="store_true", help="Omit tensor list from output.")
    inspect_p.set_defaults(func=cmd_inspect)

    upscale_p = subparsers.add_parser("upscale", help="Apply an upscale recipe to a local checkpoint.")
    upscale_p.add_argument("--source", required=True, help="Local source checkpoint directory.")
    upscale_p.add_argument("--recipe", help="YAML/JSON recipe path. Defaults to Qwen3.6 512->768.")
    upscale_p.add_argument("--out", required=True, help="Output checkpoint directory.")
    upscale_p.add_argument("--new-intermediate-size", type=int, default=768, help="Default recipe target size.")
    upscale_p.set_defaults(func=cmd_upscale)

    probe_p = subparsers.add_parser("probe", help="Compare source/candidate metadata and optional logits.")
    probe_p.add_argument("--source", required=True, help="Source checkpoint directory.")
    probe_p.add_argument("--candidate", required=True, help="Candidate checkpoint directory.")
    probe_p.add_argument("--prompts", help="JSONL prompt file. Uses metadata-only probe if omitted.")
    probe_p.add_argument("--max-prompts", type=int, default=16, help="Maximum prompts for optional torch probe.")
    probe_p.set_defaults(func=cmd_probe)

    slime_p = subparsers.add_parser("slime-plan", help="Emit slime handoff templates for a candidate checkpoint.")
    slime_p.add_argument("--candidate", required=True, help="Candidate checkpoint directory.")
    slime_p.add_argument("--out", required=True, help="Output plan directory.")
    slime_p.set_defaults(func=cmd_slime_plan)

    add_pipeline_parser(subparsers, "pipeline")
    add_pipeline_parser(subparsers, "run-pipeline")

    return parser


def add_pipeline_parser(subparsers: argparse._SubParsersAction, name: str) -> None:
    pipeline_p = subparsers.add_parser(name, help="Execute an upscale/probe/benchmark/reverse-distill pipeline.")
    pipeline_p.add_argument("--config", required=True, help="Pipeline YAML config.")
    pipeline_p.add_argument("--source", help="Override source checkpoint path.")
    pipeline_p.add_argument("--out", help="Override run output directory.")
    pipeline_p.add_argument("--candidate", help="Override candidate checkpoint path.")
    pipeline_p.add_argument("--recipe", help="Override upscale recipe path.")
    pipeline_p.add_argument("--prompts", help="Override probe prompt JSONL.")
    pipeline_p.add_argument("--max-prompts", type=int, help="Override max probe prompts.")
    pipeline_p.add_argument("--bench-suite", help="Override benchmark suite.")
    pipeline_p.add_argument("--serve-backend", help="Override benchmark serving backend.")
    pipeline_p.add_argument("--train-backend", help="Override reverse-distill backend.")
    pipeline_p.add_argument("--dry-run", action="store_true", help="Preview stages without writing artifacts or launching jobs.")
    pipeline_p.add_argument("--skip-upscale", action="store_true", help="Use an existing candidate.")
    pipeline_p.add_argument("--skip-probe", action="store_true", help="Skip probe stage.")
    pipeline_p.add_argument("--skip-benchmark", action="store_true", help="Skip pre/post benchmark stages.")
    pipeline_p.add_argument("--skip-train", action="store_true", help="Skip reverse-distillation stage.")
    pipeline_p.add_argument("--stop-after", choices=["inspect", "upscale", "probe", "bench-pre", "train", "bench-post"])
    pipeline_p.add_argument("--continue-on-gate-fail", action="store_true", help="Continue later stages after gate failures.")
    pipeline_p.set_defaults(func=cmd_pipeline)


def cmd_inspect(args: argparse.Namespace) -> dict[str, Any]:
    spec = inspect_model(args.model)
    data = spec.to_dict()
    if args.no_tensors:
        data.pop("tensors", None)
    return data


def cmd_upscale(args: argparse.Namespace) -> dict[str, Any]:
    recipe = load_recipe(args.recipe) if args.recipe else default_qwen36_recipe(args.new_intermediate_size)
    report = upscale_checkpoint(args.source, recipe, args.out)
    return report.to_dict()


def cmd_probe(args: argparse.Namespace) -> dict[str, Any]:
    report = probe_checkpoints(args.source, args.candidate, prompts=args.prompts, max_prompts=args.max_prompts)
    return report.to_dict()


def cmd_slime_plan(args: argparse.Namespace) -> dict[str, Any]:
    return emit_slime_plan(args.candidate, args.out)


def cmd_pipeline(args: argparse.Namespace) -> dict[str, Any]:
    config = load_pipeline_config(args.config)
    overrides = {
        "source": args.source,
        "out": args.out,
        "candidate": args.candidate,
        "recipe": args.recipe,
        "prompts": args.prompts,
        "max_prompts": args.max_prompts,
        "bench_suite": args.bench_suite,
        "serve_backend": args.serve_backend,
        "train_backend": args.train_backend,
        "dry_run": True if args.dry_run else None,
        "skip_upscale": True if args.skip_upscale else None,
        "skip_probe": True if args.skip_probe else None,
        "skip_benchmark": True if args.skip_benchmark else None,
        "skip_train": True if args.skip_train else None,
        "stop_after": args.stop_after,
        "continue_on_gate_fail": True if args.continue_on_gate_fail else None,
    }
    return run_pipeline(config, overrides).to_dict()
