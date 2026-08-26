"""Typed project exceptions with actionable error messages."""


class WSILabError(RuntimeError):
    """Base exception for an expected WSI Lab failure."""


class ConfigurationError(WSILabError):
    """Configuration is invalid or cannot be resolved."""


class DatasetValidationError(WSILabError):
    """Dataset or embedding validation failed."""


class PreprocessingError(WSILabError):
    """A preprocessing stage failed."""


class ArtifactCompatibilityError(WSILabError):
    """An existing artifact is incomplete or incompatible."""


class TrainingError(WSILabError):
    """Model training cannot safely continue."""


class TestSplitAccessError(WSILabError):
    """A guarded labeled test split was requested without explicit consent."""

    __test__ = False
