from __future__ import annotations


class AblxError(RuntimeError):
    """Base error for user-facing ablx failures."""


class ConfigError(AblxError):
    """Invalid or incomplete pipeline configuration."""


class CheckpointFormatError(AblxError):
    """Checkpoint files or metadata do not match supported layouts."""


class TransformError(AblxError):
    """Checkpoint expansion or upsample operation failed."""


class GateFailure(AblxError):
    """A preservation or benchmark gate failed."""


class OptionalDependencyError(AblxError):
    """An optional runtime integration was requested but is unavailable."""
