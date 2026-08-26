"""Typed project exceptions with actionable error messages."""


class RL4XSError(RuntimeError):
    """Base exception for an expected RL4XS WSI failure."""


class ConfigurationError(RL4XSError):
    """Configuration is invalid or cannot be resolved."""


class DatasetValidationError(RL4XSError):
    """Dataset or embedding validation failed."""


class PreprocessingError(RL4XSError):
    """A preprocessing stage failed."""


class ArtifactCompatibilityError(RL4XSError):
    """An existing artifact is incomplete or incompatible."""


class TrainingError(RL4XSError):
    """Model training cannot safely continue."""


class TestSplitAccessError(RL4XSError):
    """The guarded official test split was requested without explicit consent."""

    __test__ = False
