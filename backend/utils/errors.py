"""
Custom exceptions for the SENSE backend.

These let the API layer translate failures into clean HTTP responses
without leaking stack traces to the mobile client.
"""


class SenseBackendError(Exception):
    """Base class for all backend errors."""
    error_code = "backend_error"
    http_status = 500

    def __init__(self, message: str, detail=None):
        super().__init__(message)
        self.message = message
        self.detail = detail


class SessionNotFoundError(SenseBackendError):
    error_code = "session_not_found"
    http_status = 404


class InvalidAudioError(SenseBackendError):
    error_code = "invalid_audio"
    http_status = 400


class PipelineError(SenseBackendError):
    """Raised when the AI pipeline fails or times out."""
    error_code = "pipeline_error"
    http_status = 502


class PipelineTimeoutError(PipelineError):
    error_code = "pipeline_timeout"
    http_status = 504
