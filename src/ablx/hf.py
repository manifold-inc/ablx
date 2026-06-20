from __future__ import annotations

from pathlib import Path

from .errors import AblxError, DependencyMissingError


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

    try:
        return Path(snapshot_download(repo_id=model, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.py"])).resolve()
    except Exception as exc:  # noqa: BLE001 - normalize HF/client errors for the CLI.
        raise AblxError(f"could not resolve model {model!r} as a local path or Hugging Face repo ID: {exc}") from exc
