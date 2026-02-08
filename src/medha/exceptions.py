"""Custom exception hierarchy for Medha."""


class MedhaError(Exception):
    """Base exception for all Medha errors."""


class ConfigurationError(MedhaError):
    """Raised when configuration is invalid or required environment variables are missing."""


class EmbeddingError(MedhaError):
    """Raised when embedding generation fails."""


class StorageError(MedhaError):
    """Raised when the vector storage backend encounters an error."""


class StorageInitializationError(StorageError):
    """Raised when collection creation or storage initialization fails."""


class TemplateError(MedhaError):
    """Raised when template loading, matching, or rendering fails."""


class ParameterExtractionError(MedhaError):
    """Raised when required parameters cannot be extracted from the input."""
