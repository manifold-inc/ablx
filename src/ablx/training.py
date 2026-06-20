from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path
from typing import Dict, List, Mapping

import yaml

from .errors import AblxError
from .models import TrainPlanReport
from .slime import emit_slime_plan


def run_reverse_distill_stage(
    *,
    source: str | Path,
    candidate: str | Path,
    out: str | Path,
    config: Mapping[str, object],
    launch_training: bool = True,
) -> TrainPlanReport:
    out_dir = Path(out).expanduser().resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    source_path = str(Path(source).expanduser().resolve())
    candidate_path = str(Path(candidate).expanduser().resolve())
    backend = str(config.get("backend", "slime"))
    mode = str(config.get("mode", "light"))
    commands = build_training_commands(config, source_path, candidate_path, out_dir)
    artifacts: List[str] = []
    warnings: List[str] = []

    reverse_distill_plan = build_reverse_distill_plan(config, source_path, candidate_path, backend, mode)
    freeze_masks = build_freeze_masks(config)
    loss_config = build_loss_config(config)
    data_manifest = {
        "mixture": config.get("data", {}),
        "note": "Hashes and licenses should be filled by the run owner before long training.",
    }

    write_yaml(out_dir / "reverse_distill_plan.yaml", reverse_distill_plan)
    write_yaml(out_dir / "freeze_masks.yaml", freeze_masks)
    write_yaml(out_dir / "loss_config.yaml", loss_config)
    (out_dir / "data_manifest.json").write_text(json.dumps(data_manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    (out_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n", encoding="utf-8")
    (out_dir / "README.md").write_text(render_training_readme(backend, mode), encoding="utf-8")
    artifacts.extend(["reverse_distill_plan.yaml", "freeze_masks.yaml", "loss_config.yaml", "data_manifest.json", "commands.json", "README.md"])

    if bool(config.get("emit_slime_plan", backend == "slime")):
        slime_out = out_dir / "slime"
        emit_slime_plan(candidate_path, slime_out)
        artifacts.append("slime/")

    command_results: List[Dict[str, object]] = []
    if launch_training:
        validate_training_backend(backend, config, commands)
        for command in commands:
            command_results.append(run_command(command, cwd=out_dir))
        if any(result["returncode"] != 0 for result in command_results):
            raise AblxError("reverse-distillation training command failed")
    else:
        warnings.append("training launch disabled; emitted reverse-distillation plan only")

    report = TrainPlanReport(
        source=source_path,
        candidate=candidate_path,
        out=str(out_dir),
        backend=backend,
        mode=mode,
        launch_training=launch_training,
        commands=commands,
        command_results=command_results,
        artifacts=artifacts,
        warnings=warnings,
    )
    (out_dir / "train_report.json").write_text(json.dumps(report.to_dict(), indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return report


def build_training_commands(config: Mapping[str, object], source: str, candidate: str, out_dir: Path) -> List[str]:
    raw = config.get("commands")
    if isinstance(raw, list):
        return [render_command(str(command), source, candidate, out_dir) for command in raw]
    if isinstance(raw, str):
        return [render_command(raw, source, candidate, out_dir)]

    backend = str(config.get("backend", "slime"))
    if backend == "local":
        return ["python3 -c \"print('ablx local reverse-distill placeholder')\""]
    if backend == "slime":
        slime_root = config.get("slime_root")
        if slime_root:
            return [
                "python "
                f"{Path(str(slime_root)).expanduser().resolve() / 'train.py'} "
                "<megatron args> <sglang args> <reverse distill data/reward args>"
            ]
        return []
    if backend == "generic":
        return []
    return []


def render_command(command: str, source: str, candidate: str, out_dir: Path) -> str:
    return command.format(source=source, candidate=candidate, student=candidate, teacher=source, out=str(out_dir))


def validate_training_backend(backend: str, config: Mapping[str, object], commands: List[str]) -> None:
    if not commands:
        raise AblxError(
            f"training backend {backend!r} has no launch command; set reverse_distill.commands, "
            "reverse_distill.slime_root, --skip-train, or --dry-run"
        )
    if backend == "slime":
        slime_root = config.get("slime_root")
        if slime_root and not (Path(str(slime_root)).expanduser().resolve() / "train.py").exists():
            raise AblxError(f"slime training entrypoint not found under {slime_root!r}")
        if not slime_root and not config.get("commands") and shutil.which("slime") is None:
            raise AblxError(
                "slime backend requested but no slime command or slime_root was found; "
                "set reverse_distill.commands or reverse_distill.slime_root"
            )


def build_reverse_distill_plan(config: Mapping[str, object], source: str, candidate: str, backend: str, mode: str) -> Dict[str, object]:
    return {
        "mode": mode,
        "backend": backend,
        "teacher": str(config.get("teacher", "source")),
        "teacher_path": source,
        "student": str(config.get("student", "candidate")),
        "student_path": candidate,
        "definition": "source-anchored weak-to-strong distillation into the expanded checkpoint",
        "stages": [
            {"name": "stage_0_no_train_validation", "goal": "probe, serve, benchmark, and inspect new-capacity silence"},
            {"name": "stage_1_preservation_warmup", "tokens": "10-25M", "trainable": "new MoE slices"},
            {"name": "stage_2_capability_topoff", "tokens": "25-75M", "trainable": "new slices plus optional router LoRA"},
            {"name": "stage_3_optional_verifier_rl", "prompts": "5k-20k", "trainable": "new capacity/adapters/router only"},
        ],
    }


def build_freeze_masks(config: Mapping[str, object]) -> Dict[str, object]:
    return {
        "trainable": config.get(
            "trainable_patterns",
            [
                ".*mlp\\.experts\\.(gate_up_proj|down_proj).* new slices",
                ".*mlp\\.shared_expert\\.(gate_proj|up_proj|down_proj|gate_up_proj).* new slices",
                "optional .*mlp\\.gate\\.weight.* LoRA/tiny LR",
            ],
        ),
        "freeze": config.get(
            "freeze_patterns",
            [
                ".*embed_tokens.*",
                ".*lm_head.*",
                ".*self_attn.*",
                ".*linear_attn.*",
                ".*norm.*",
                ".*vision.*",
                "old expert slices",
                "mtp unless explicitly enabled",
            ],
        ),
        "fallback": "if slice masks are unavailable, train MoE/router LoRA or adapters; do not full-finetune whole tensors silently",
    }


def build_loss_config(config: Mapping[str, object]) -> Dict[str, object]:
    return {
        "losses": config.get(
            "losses",
            [
                {"name": "CE_gold", "weight": 1.0},
                {"name": "KL_source_forward", "weight": 0.5, "temperature": 2.0},
                {"name": "symmetric_or_reverse_KL_drift", "weight": 0.05},
                {"name": "router_KL", "weight": 0.05},
                {"name": "old_slice_L2_anchor", "weight": 0.01},
                {"name": "router_balance_entropy", "weight": 0.01},
            ],
        )
    }


def write_yaml(path: Path, data: Mapping[str, object]) -> None:
    path.write_text(yaml.safe_dump(data, sort_keys=False), encoding="utf-8")


def run_command(command: str, cwd: Path) -> Dict[str, object]:
    proc = subprocess.run(command, cwd=str(cwd), shell=True, text=True, capture_output=True, check=False)
    return {
        "command": command,
        "returncode": proc.returncode,
        "stdout": proc.stdout[-4000:],
        "stderr": proc.stderr[-4000:],
    }


def render_training_readme(backend: str, mode: str) -> str:
    return f"""# ablx reverse distillation

Backend: `{backend}`
Mode: `{mode}`

Reverse distillation here means source-anchored weak-to-strong distillation into
the expanded checkpoint. The light recipe repairs and activates new MoE capacity
without full post-training.
"""
