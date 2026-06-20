"""Deterministic weight-space upscaling tools for MoE LLM checkpoints."""

from .models import (
    ExpansionReport,
    ModelSpec,
    ProbeReport,
    TensorInfo,
    TensorMapping,
    TransformOp,
    UpscaleRecipe,
)

__all__ = [
    "ExpansionReport",
    "ModelSpec",
    "ProbeReport",
    "TensorInfo",
    "TensorMapping",
    "TransformOp",
    "UpscaleRecipe",
]

__version__ = "0.1.0"
