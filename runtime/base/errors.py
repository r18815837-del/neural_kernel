from __future__ import annotations


class RuntimeErrorBase(Exception):
    """Base class for runtime-layer errors."""
    pass


class InvalidRequestError(RuntimeErrorBase):
    """Raised when an incoming request is invalid."""
    pass


class ContextOverflowError(RuntimeErrorBase):
    """Raised when prompt/history exceeds model context window."""
    pass


class GenerationError(RuntimeErrorBase):
    """Raised when model generation fails."""
    pass


class TokenizationError(RuntimeErrorBase):
    """Raised when tokenization or detokenization fails."""
    pass


class SessionError(RuntimeErrorBase):
    """Raised for session-related failures."""
    pass


class SpecError(RuntimeErrorBase):
    """Raised when project/client spec parsing fails."""
    pass


class ToolExecutionError(RuntimeErrorBase):
    """Raised when a tool fails during execution."""
    pass


class ArtifactError(RuntimeErrorBase):
    """Raised when artifact generation/validation/packaging fails."""
    pass