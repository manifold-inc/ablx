class AblxError(RuntimeError):
    """Base exception for expected ablx failures."""


class DependencyMissingError(AblxError):
    """Raised when an optional runtime dependency is required."""


class CheckpointFormatError(AblxError):
    """Raised when checkpoint metadata or tensor layout is unsupported."""


class RecipeError(AblxError):
    """Raised when an upscale recipe is invalid."""
