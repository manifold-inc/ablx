from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Dict, List, Mapping, Optional

import yaml

from .benchmark import run_benchmark_stage
from .errors import AblxError, RecipeError
from .inspection import inspect_model
from .models import PipelineReport, PipelineStepReport
from .probe import probe_checkpoints
from .recipes import default_qwen36_recipe, load_recipe
from .training import run_reverse_distill_stage
from .transforms import plan_transforms, upscale_checkpoint


STOP_STAGES = ["inspect", "upscale", "probe", "bench-pre", "train", "bench-post"]


def load_pipeline_config(path: str | Path) -> Dict[str, object]:
    config_path = Path(path).expanduser()
    data = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise AblxError("pipeline config must be a YAML mapping")
    return data


def run_pipeline(config: Mapping[str, object], overrides: Optional[Mapping[str, object]] = None) -> PipelineReport:
    resolved = resolve_pipeline_config(config, overrides or {})
    name = str(resolved.get("name", "ablx_pipeline"))
    source_ref = str(required(resolved, "source"))
    out = Path(str(required(resolved, "out"))).expanduser().resolve()
    dry_run = bool(resolved.get("dry_run", False))
    stop_after = resolved.get("stop_after")
    if stop_after is not None and str(stop_after) not in STOP_STAGES:
        raise AblxError(f"stop_after must be one of {STOP_STAGES}")

    upscale_cfg = section(resolved, "upscale")
    probe_cfg = section(resolved, "probe")
    benchmark_cfg = section(resolved, "benchmark")
    train_cfg = section(resolved, "reverse_distill")

    skip_upscale = bool(resolved.get("skip_upscale", False)) or not bool(upscale_cfg.get("enabled", True))
    skip_probe = bool(resolved.get("skip_probe", False)) or not bool(probe_cfg.get("enabled", True))
    skip_benchmark = bool(resolved.get("skip_benchmark", False)) or not bool(benchmark_cfg.get("enabled", True))
    skip_train = bool(resolved.get("skip_train", False)) or not bool(train_cfg.get("enabled", True))
    continue_on_gate_fail = bool(resolved.get("continue_on_gate_fail", False))

    candidate = resolve_candidate_path(resolved, out, upscale_cfg)
    steps: List[PipelineStepReport] = []
    artifacts: List[str] = []
    warnings: List[str] = []
    accepted = True

    source_spec = inspect_model(source_ref)
    source = Path(source_spec.path).expanduser().resolve()
    steps.append(
        PipelineStepReport(
            name="inspect",
            status="planned" if dry_run else "completed",
            out=str(source),
            metrics=source_spec.to_dict()
            if dry_run
            else {"model_type": source_spec.model_type, "tensor_count": len(source_spec.tensors), "requested_source": source_ref},
        )
    )
    if should_stop(stop_after, "inspect"):
        return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)

    if dry_run:
        steps.extend(preview_pipeline_steps(source, candidate, out, resolved, skip_upscale, skip_probe, skip_benchmark, skip_train))
        return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)

    out.mkdir(parents=True, exist_ok=True)
    write_yaml(out / "pipeline_config.resolved.yaml", resolved)
    artifacts.append(str(out / "pipeline_config.resolved.yaml"))

    if skip_upscale:
        if not candidate.exists():
            raise AblxError(f"upscale is skipped but candidate does not exist: {candidate}")
        steps.append(PipelineStepReport(name="upscale", status="skipped", out=str(candidate), warnings=["using existing candidate"]))
    else:
        recipe = resolve_recipe(upscale_cfg)
        expansion = upscale_checkpoint(source, recipe, candidate)
        report_path = candidate / "ablx_expansion_report.json"
        artifacts.append(str(report_path))
        steps.append(
            PipelineStepReport(
                name="upscale",
                status="completed",
                out=str(candidate),
                artifacts=[str(report_path)],
                metrics={"changed_tensors": len(expansion.changed_tensors), "copied_tensors": expansion.copied_tensors},
                warnings=expansion.warnings,
            )
        )
    if should_stop(stop_after, "upscale"):
        return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)

    probe_report = None
    if skip_probe:
        steps.append(PipelineStepReport(name="probe", status="skipped", out=str(out / "probe")))
    else:
        probe_out = out / "probe"
        probe_out.mkdir(parents=True, exist_ok=True)
        probe_report = probe_checkpoints(
            source,
            candidate,
            prompts=probe_cfg.get("prompts"),
            max_prompts=int(probe_cfg.get("max_prompts", 16)),
        )
        probe_path = probe_out / "ablx_probe_report.json"
        probe_path.write_text(json.dumps(probe_report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        artifacts.append(str(probe_path))
        steps.append(
            PipelineStepReport(
                name="probe",
                status="completed",
                out=str(probe_out),
                artifacts=[str(probe_path)],
                metrics=probe_report.metrics,
                warnings=probe_report.warnings,
            )
        )
    if should_stop(stop_after, "probe"):
        return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)

    if skip_benchmark:
        steps.append(PipelineStepReport(name="bench-pre", status="skipped", out=str(out / "bench" / "pre")))
    else:
        pre_report = run_benchmark_stage(
            stage="pre",
            source=source,
            candidate=candidate,
            out=out / "bench" / "pre",
            config=benchmark_cfg,
            run_commands=bool(benchmark_cfg.get("run_commands", True)),
            probe_report=probe_report,
        )
        pre_artifacts = stage_files(out / "bench" / "pre")
        artifacts.extend(pre_artifacts)
        steps.append(
            PipelineStepReport(
                name="bench-pre",
                status="completed" if pre_report.passed else "failed",
                out=pre_report.out,
                artifacts=pre_artifacts,
                metrics=pre_report.metrics,
                warnings=pre_report.warnings,
            )
        )
        if not pre_report.passed:
            accepted = False
            failure = format_benchmark_failure("pre", pre_report)
            if continue_on_gate_fail:
                warnings.append(failure)
            else:
                warnings.append(failure)
                finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)
                raise AblxError(failure)
    if should_stop(stop_after, "bench-pre"):
        return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)

    final_model = candidate
    if skip_train:
        steps.append(PipelineStepReport(name="train", status="skipped", out=str(out / "train")))
    else:
        train_report = run_reverse_distill_stage(
            source=source,
            candidate=candidate,
            out=out / "train",
            config=train_cfg,
            launch_training=bool(train_cfg.get("launch_training", True)),
        )
        train_artifacts = stage_files(out / "train")
        artifacts.extend(train_artifacts)
        final_model = Path(str(train_cfg.get("final_model", candidate))).expanduser()
        if not final_model.is_absolute():
            final_model = (out / final_model).resolve()
        if not final_model.exists():
            final_model = candidate
        steps.append(
            PipelineStepReport(
                name="train",
                status="completed",
                out=train_report.out,
                artifacts=train_artifacts,
                metrics={"backend": train_report.backend, "mode": train_report.mode, "launch_training": train_report.launch_training},
                warnings=train_report.warnings,
            )
        )
    if should_stop(stop_after, "train"):
        return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)

    if skip_benchmark:
        steps.append(PipelineStepReport(name="bench-post", status="skipped", out=str(out / "bench" / "post")))
    else:
        post_report = run_benchmark_stage(
            stage="post",
            source=source,
            candidate=candidate,
            out=out / "bench" / "post",
            config=benchmark_cfg,
            run_commands=bool(benchmark_cfg.get("run_commands", True)),
            probe_report=None,
            final_model=final_model,
        )
        post_artifacts = stage_files(out / "bench" / "post")
        artifacts.extend(post_artifacts)
        steps.append(
            PipelineStepReport(
                name="bench-post",
                status="completed" if post_report.passed else "failed",
                out=post_report.out,
                artifacts=post_artifacts,
                metrics=post_report.metrics,
                warnings=post_report.warnings,
            )
        )
        if not post_report.passed:
            accepted = False
            failure = format_benchmark_failure("post", post_report)
            if continue_on_gate_fail:
                warnings.append(failure)
            else:
                warnings.append(failure)
                finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)
                raise AblxError(failure)

    return finalize_report(name, source, candidate, out, dry_run, accepted, steps, artifacts, warnings, resolved)


def resolve_pipeline_config(config: Mapping[str, object], overrides: Mapping[str, object]) -> Dict[str, object]:
    resolved = deep_merge(default_pipeline_config(), dict(config))
    for key, value in overrides.items():
        if value is None:
            continue
        if key in {"source", "out", "candidate", "dry_run", "stop_after", "continue_on_gate_fail", "skip_upscale", "skip_probe", "skip_benchmark", "skip_train"}:
            resolved[key] = value
        elif key == "recipe":
            section(resolved, "upscale")["recipe"] = value
        elif key == "prompts":
            section(resolved, "probe")["prompts"] = value
        elif key == "max_prompts":
            section(resolved, "probe")["max_prompts"] = value
        elif key == "bench_suite":
            section(resolved, "benchmark")["suite"] = value
        elif key == "serve_backend":
            section(resolved, "benchmark")["serve_backend"] = value
        elif key == "train_backend":
            section(resolved, "reverse_distill")["backend"] = value
    return resolved


def default_pipeline_config() -> Dict[str, object]:
    return {
        "name": "ablx_pipeline",
        "dry_run": False,
        "candidate": "candidate",
        "upscale": {"enabled": True, "out": "candidate", "new_intermediate_size": 768},
        "probe": {"enabled": True, "max_prompts": 16, "on_gate_fail": "stop"},
        "benchmark": {
            "enabled": True,
            "suite": "preserve",
            "serve_backend": "metadata",
            "run_commands": True,
            "gates": {
                "pre": {"missing_in_candidate_count_max": 0, "changed_shape_count_min": 1},
                "post": {"missing_in_candidate_count_max": 0},
            },
        },
        "reverse_distill": {
            "enabled": True,
            "backend": "slime",
            "mode": "light",
            "launch_training": True,
            "emit_slime_plan": True,
        },
    }


def preview_pipeline_steps(
    source: Path,
    candidate: Path,
    out: Path,
    resolved: Mapping[str, object],
    skip_upscale: bool,
    skip_probe: bool,
    skip_benchmark: bool,
    skip_train: bool,
) -> List[PipelineStepReport]:
    steps: List[PipelineStepReport] = []
    if skip_upscale:
        steps.append(PipelineStepReport(name="upscale", status="skipped", out=str(candidate)))
    else:
        metrics: Dict[str, object] = {}
        try:
            spec = inspect_model(source)
            recipe = resolve_recipe(section(resolved, "upscale"))
            config_copy = copy.deepcopy(spec.config)
            planned = plan_transforms(spec, recipe, config_copy)
            metrics = {
                "planned_changed_tensors": len(planned),
                "planned_tensor_names": sorted(planned)[:200],
                "recipe": recipe.to_dict(),
            }
        except Exception as exc:  # noqa: BLE001 - dry-run preview should report, not explode.
            metrics = {"preview_error": str(exc)}
        steps.append(PipelineStepReport(name="upscale", status="planned", out=str(candidate), metrics=metrics))
    steps.append(PipelineStepReport(name="probe", status="skipped" if skip_probe else "planned", out=str(out / "probe")))
    steps.append(PipelineStepReport(name="bench-pre", status="skipped" if skip_benchmark else "planned", out=str(out / "bench" / "pre")))
    steps.append(PipelineStepReport(name="train", status="skipped" if skip_train else "planned", out=str(out / "train")))
    steps.append(PipelineStepReport(name="bench-post", status="skipped" if skip_benchmark else "planned", out=str(out / "bench" / "post")))
    return steps


def resolve_recipe(upscale_cfg: Mapping[str, object]):
    recipe = upscale_cfg.get("recipe")
    if isinstance(recipe, str):
        return load_recipe(recipe)
    if isinstance(recipe, Mapping):
        transforms = recipe.get("transforms")
        if not isinstance(transforms, list):
            raise RecipeError("inline recipe requires transforms list")
        from .models import TransformOp, UpscaleRecipe

        ops = []
        for item in transforms:
            if not isinstance(item, Mapping) or not isinstance(item.get("type"), str):
                raise RecipeError("inline recipe transforms must be mappings with type")
            ops.append(TransformOp(str(item["type"]), {str(k): v for k, v in item.items() if k != "type"}))
        return UpscaleRecipe(name=str(recipe.get("name", "inline_recipe")), description=str(recipe.get("description", "")), transforms=ops)
    return default_qwen36_recipe(int(upscale_cfg.get("new_intermediate_size", 768)))


def resolve_candidate_path(resolved: Mapping[str, object], out: Path, upscale_cfg: Mapping[str, object]) -> Path:
    candidate_raw = upscale_cfg.get("out") or resolved.get("candidate") or "candidate"
    candidate = Path(str(candidate_raw)).expanduser()
    if not candidate.is_absolute():
        candidate = out / candidate
    return candidate.resolve()


def section(config: Mapping[str, object], name: str) -> Dict[str, object]:
    value = config.get(name, {})
    if not isinstance(value, dict):
        raise AblxError(f"pipeline section {name!r} must be a mapping")
    return value


def required(config: Mapping[str, object], name: str) -> object:
    if not config.get(name):
        raise AblxError(f"pipeline config requires {name!r}")
    return config[name]


def deep_merge(base: Dict[str, object], overlay: Dict[str, object]) -> Dict[str, object]:
    result = copy.deepcopy(base)
    for key, value in overlay.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)  # type: ignore[arg-type]
        else:
            result[key] = value
    return result


def should_stop(stop_after: object, stage: str) -> bool:
    return stop_after is not None and str(stop_after) == stage


def finalize_report(
    name: str,
    source: Path,
    candidate: Path,
    out: Path,
    dry_run: bool,
    accepted: bool,
    steps: List[PipelineStepReport],
    artifacts: List[str],
    warnings: List[str],
    resolved: Mapping[str, object],
) -> PipelineReport:
    report = PipelineReport(
        name=name,
        source=str(source),
        candidate=str(candidate),
        out=str(out),
        dry_run=dry_run,
        accepted=accepted,
        steps=steps,
        artifacts=artifacts,
        warnings=warnings,
    )
    if not dry_run and out.exists():
        report_path = out / "ablx_pipeline_report.json"
        report_path.write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if str(report_path) not in artifacts:
            artifacts.append(str(report_path))
    return report


def stage_files(path: Path) -> List[str]:
    if not path.exists():
        return []
    return [str(item) for item in sorted(path.iterdir()) if item.is_file() or item.is_dir()]


def format_benchmark_failure(stage: str, report) -> str:
    details = "; ".join(report.warnings[:4]) if report.warnings else "no detailed warning recorded"
    if len(details) > 1000:
        details = details[:997] + "..."
    return f"{stage}-benchmark failed: {details}. Summary: {Path(report.out) / 'summary.json'}"


def write_yaml(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")
