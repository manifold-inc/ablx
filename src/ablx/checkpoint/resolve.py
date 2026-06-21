from __future__ import annotations

from pathlib import Path

from ablx.config import ParentConfig
from ablx.errors import CheckpointFormatError, OptionalDependencyError

DEFAULT_ALLOW_PATTERNS = [
    "*.json",
    "*.safetensors",
    "*.safetensors.index.json",
    "tokenizer*",
    "*.model",
    "*.txt",
    "*.py",
    "preprocessor_config.json",
    "processor_config.json",
    "chat_template*",
]


def resolve_checkpoint_ref(ref: str | Path, *, parent: ParentConfig | None = None) -> Path:
    """Resolve a local checkpoint path or download/cache a Hugging Face repo ID."""

    raw = str(ref)
    path = Path(raw).expanduser()
    if path.exists():
        return path.resolve()
    if is_probable_hf_id(raw):
        return download_hf_checkpoint(raw, parent=parent)
    raise CheckpointFormatError(
        f"checkpoint path does not exist: {path}. "
        "If this is a Hugging Face repo, use a repo id like Qwen/Qwen3.6-35B-A3B "
        "in parent.hf_id, or download it first and set parent.path / expand.source."
    )


def is_probable_hf_id(value: str) -> bool:
    if value.startswith(("/", "./", "../", "~")):
        return False
    parts = value.split("/")
    return len(parts) == 2 and all(parts) and not any(part in {".", ".."} for part in parts)


def download_hf_checkpoint(repo_id: str, *, parent: ParentConfig | None = None) -> Path:
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:  # pragma: no cover - dependency error path
        raise OptionalDependencyError(
            f"{repo_id!r} looks like a Hugging Face repo id, but huggingface_hub is not installed. "
            "Run `python -m pip install -e '.[train]'` or `python -m pip install huggingface-hub`, "
            "or set parent.path / expand.source to a local checkpoint directory."
        ) from exc

    kwargs = {}
    if parent:
        if parent.revision:
            kwargs["revision"] = parent.revision
        if parent.cache_dir:
            kwargs["cache_dir"] = parent.cache_dir
        if parent.local_dir:
            kwargs["local_dir"] = parent.local_dir
        if parent.allow_patterns:
            kwargs["allow_patterns"] = parent.allow_patterns
    kwargs.setdefault("allow_patterns", DEFAULT_ALLOW_PATTERNS)
    try:
        return Path(snapshot_download(repo_id=repo_id, **kwargs)).resolve()
    except Exception as exc:  # pragma: no cover - depends on network/auth
        raise CheckpointFormatError(
            f"failed to resolve Hugging Face checkpoint {repo_id!r}: {exc}. "
            "If the model is gated or private, authenticate with `huggingface-cli login` "
            "or set HF_TOKEN before rerunning."
        ) from exc
