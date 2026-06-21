from __future__ import annotations

from pathlib import Path
from typing import Any

import torch

from ablx.config import PipelineConfig
from ablx.errors import TransformError
from ablx.checkpoint.safetensors_io import rewrite_checkpoint
from ablx.upsample.masks import load_expansion_masks
from ablx.utils import ensure_dir, log_progress, write_json


def upsample_checkpoint(config: PipelineConfig, source: str | Path | None = None) -> dict[str, Any]:
    if config.upsample.generator != "constrained_noise":
        raise NotImplementedError("only generator='constrained_noise' is implemented in v1")
    src = Path(source or (config.out_dir / "expanded")).expanduser()
    log_progress(f"upsample: reading expanded checkpoint from {src}")
    report_path = src / "expansion_report.json"
    if not report_path.exists():
        raise TransformError(f"missing expansion report: {report_path}")
    masks = load_expansion_masks(report_path)
    out = ensure_dir(config.out_dir / "upsampled")
    log_progress(f"upsample: writing upsampled checkpoint to {out}")
    seed = config.upsample.noise_seed
    noise_std = config.upsample.noise_std
    touched: list[dict[str, Any]] = []

    def transform(name: str, tensor: torch.Tensor):
        reports = masks["by_tensor"].get(name, [])
        out_tensor = tensor.clone()
        if reports:
            before = out_tensor.clone()
            for item in reports:
                gen = torch.Generator(device="cpu")
                gen.manual_seed(seed + stable_name_seed(str(item.get("tensor", name))))
                out_tensor = apply_report_noise(out_tensor, item, noise_std, gen)
            delta = (out_tensor - before).float()
            touched.append(
                {
                    "tensor": name,
                    "reports": len(reports),
                    "max_abs_delta": float(delta.abs().max().item()) if delta.numel() else 0.0,
                    "l2_delta": float(torch.linalg.vector_norm(delta).item()) if delta.numel() else 0.0,
                }
            )
        yield name, out_tensor

    written = rewrite_checkpoint(src, out, transform, progress_label="upsample")
    report = {
        "stage": "upsample",
        "source": str(src),
        "out": str(out),
        "generator": config.upsample.generator,
        "noise_seed": seed,
        "noise_std": noise_std,
        "project_to_constraint_set": config.upsample.project_to_constraint_set,
        "written_shards": written,
        "affected_tensor_count": len(touched),
        "tensors": touched,
        "constraint": "new slices only; cloned expert deltas seeded by original expert group",
    }
    write_json(out / "upsample_report.json", report)
    log_progress(f"upsample: wrote report {out / 'upsample_report.json'}")
    return report


def stable_name_seed(name: str) -> int:
    value = 0
    for char in name:
        value = (value * 131 + ord(char)) % 1_000_000_007
    return value


def apply_report_noise(
    tensor: torch.Tensor,
    report: dict[str, Any],
    noise_std: float,
    gen: torch.Generator,
) -> torch.Tensor:
    if noise_std == 0:
        return tensor
    role = str(report.get("role", ""))
    old_shape = report.get("old_shape") or []
    new_shape = report.get("new_shape") or list(tensor.shape)
    if "child_expert" in report:
        noise = torch.randn(tensor.shape, generator=gen, dtype=torch.float32) * noise_std
        if int(report.get("clone_offset", 0)) == 0:
            noise = -noise
        return tensor + noise.to(tensor.dtype)
    if old_shape == new_shape:
        return tensor

    out = tensor.clone()
    if "gate_up" in role and len(old_shape) >= 1 and len(new_shape) >= 1:
        old = old_shape[0] // 2
        new = new_shape[0] // 2
        if old > 0 and new > old and out.shape[0] == 2 * new:
            add_noise(out[old:new], noise_std, gen)
            add_noise(out[new + old : 2 * new], noise_std, gen)
            return out
    if len(old_shape) >= 2 and len(new_shape) >= 2:
        if old_shape[-1] < new_shape[-1] and out.shape[-1] == new_shape[-1]:
            add_noise(out[..., old_shape[-1] : new_shape[-1]], noise_std, gen)
        elif old_shape[0] < new_shape[0] and out.shape[0] == new_shape[0]:
            add_noise(out[old_shape[0] : new_shape[0]], noise_std, gen)
    return out


def add_noise(view: torch.Tensor, noise_std: float, gen: torch.Generator) -> None:
    if view.numel() == 0:
        return
    noise = torch.randn(view.shape, generator=gen, dtype=torch.float32) * noise_std
    view.add_(noise.to(view.dtype))
