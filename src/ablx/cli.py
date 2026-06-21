from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from ablx.benchmark.runner import run_benchmark
from ablx.checkpoint.inspect import inspect_checkpoint
from ablx.compute.estimate import compute_plan
from ablx.config import load_config
from ablx.errors import AblxError, GateFailure
from ablx.pipeline import run_pipeline
from ablx.probe.runner import run_probe
from ablx.train.trainer import train_model
from ablx.transforms.expand import expand_checkpoint
from ablx.upsample.constrained_noise import upsample_checkpoint
from ablx.utils import print_json

app = typer.Typer(no_args_is_help=True, help="MoE upscaling pipeline for Hugging Face checkpoints.")


@app.command()
def inspect(model: Annotated[Path, typer.Option("--model", exists=True)]) -> None:
    print_json(inspect_checkpoint(model))


@app.command()
def expand(config: Annotated[Path, typer.Option("--config", exists=True)]) -> None:
    print_json(expand_checkpoint(load_config(config)))


@app.command()
def upsample(config: Annotated[Path, typer.Option("--config", exists=True)]) -> None:
    print_json(upsample_checkpoint(load_config(config)))


@app.command()
def probe(
    config: Annotated[Path, typer.Option("--config", exists=True)],
    fail_on_gate: Annotated[bool, typer.Option("--fail-on-gate")] = False,
) -> None:
    cfg = load_config(config)
    print_json(run_probe(cfg, fail_on_gate=fail_on_gate))


@app.command()
def benchmark(
    model: Annotated[Path, typer.Option("--model")],
    suite: Annotated[str, typer.Option("--suite")] = "core",
    out: Annotated[Path | None, typer.Option("--out")] = None,
    limit: Annotated[int | None, typer.Option("--limit")] = None,
) -> None:
    print_json(run_benchmark(model, suite=suite, out=out, limit=limit))


@app.command("compute-plan")
def compute_plan_cmd(config: Annotated[Path, typer.Option("--config", exists=True)]) -> None:
    print_json(compute_plan(load_config(config)))


@app.command()
def train(
    config: Annotated[Path, typer.Option("--config", exists=True)],
    launch: Annotated[bool, typer.Option("--launch")] = False,
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    cfg = load_config(config)
    if launch:
        cfg.train.launch = True
    print_json(train_model(cfg, dry_run=dry_run))


@app.command()
def pipeline(
    config: Annotated[Path, typer.Option("--config", exists=True)],
    dry_run: Annotated[bool, typer.Option("--dry-run")] = False,
) -> None:
    print_json(run_pipeline(load_config(config), dry_run=dry_run))


def main() -> None:
    try:
        app()
    except GateFailure as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=2) from exc
    except AblxError as exc:
        typer.echo(str(exc), err=True)
        raise typer.Exit(code=1) from exc


if __name__ == "__main__":
    main()
