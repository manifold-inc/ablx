from __future__ import annotations

import json
from pathlib import Path
from typing import Mapping

import yaml

from .errors import RecipeError
from .models import TransformOp, UpscaleRecipe


def load_recipe(path: str | Path) -> UpscaleRecipe:
    recipe_path = Path(path).expanduser()
    try:
        raw = recipe_path.read_text(encoding="utf-8")
    except OSError as exc:
        raise RecipeError(f"could not read recipe {recipe_path}: {exc}") from exc

    try:
        if recipe_path.suffix.lower() == ".json":
            data = json.loads(raw)
        else:
            data = yaml.safe_load(raw)
    except Exception as exc:  # noqa: BLE001 - keep parser context for CLI users.
        raise RecipeError(f"could not parse recipe {recipe_path}: {exc}") from exc

    if not isinstance(data, Mapping):
        raise RecipeError("recipe must be a mapping")
    transforms = data.get("transforms", [])
    if not isinstance(transforms, list) or not transforms:
        raise RecipeError("recipe must contain a non-empty transforms list")

    ops = []
    for index, item in enumerate(transforms):
        if not isinstance(item, Mapping):
            raise RecipeError(f"transform {index} must be a mapping")
        op_type = item.get("type")
        if not isinstance(op_type, str) or not op_type:
            raise RecipeError(f"transform {index} is missing a type")
        params = {str(k): v for k, v in item.items() if k != "type"}
        ops.append(TransformOp(type=op_type, params=params))

    name = str(data.get("name") or recipe_path.stem)
    description = str(data.get("description") or "")
    return UpscaleRecipe(name=name, description=description, transforms=ops)


def default_qwen36_recipe(new_intermediate_size: int = 768) -> UpscaleRecipe:
    return UpscaleRecipe(
        name=f"qwen36_expand_moe_intermediate_{new_intermediate_size}",
        description="Default deterministic Qwen3.6 MoE intermediate expansion.",
        transforms=[
            TransformOp(
                type="expand_moe_intermediate",
                params={
                    "old_intermediate_size": 512,
                    "new_intermediate_size": int(new_intermediate_size),
                    "noise_std": 1e-6,
                    "noise_seed": 1729,
                    "include_mtp": True,
                    "include_shared_expert": True,
                },
            )
        ],
    )
