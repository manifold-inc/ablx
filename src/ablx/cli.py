from __future__ import annotations

import argparse
import json
import sys
from typing import Any

from .errors import AblxError
from .inspection import inspect_model
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

    return parser


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
