from __future__ import annotations

from pathlib import Path

from .errors import DependencyMissingError


def resolve_model_path(model: str) -> Path:
    path = Path(model).expanduser()
    if path.exists():
        return path.resolve()

    try:
        from huggingface_hub import snapshot_download
    except ImportError as exc:
        raise DependencyMissingError(
            f"{model!r} is not a local path. Install ablx[hf] or huggingface_hub "
            "to resolve Hugging Face model IDs."
        ) from exc

    return Path(snapshot_download(repo_id=model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py"])).resolve()
