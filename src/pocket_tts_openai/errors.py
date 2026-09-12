"""OpenAI-shaped API errors.

Every error response matches the OpenAI convention::

    {"error": {"message": ..., "type": ..., "code": ...}}
"""

from __future__ import annotations


class OpenAIError(Exception):
    def __init__(self, message: str, *, status: int = 400, type_: str = "invalid_request_error", code: str | None = None):
        super().__init__(message)
        self.message = message
        self.status = status
        self.type = type_
        self.code = code

    def to_payload(self) -> dict:
        return {"error": {"message": self.message, "type": self.type, "code": self.code}}


def invalid_request(message: str, code: str | None = None) -> OpenAIError:
    return OpenAIError(message, status=400, type_="invalid_request_error", code=code)


def api_error(message: str) -> OpenAIError:
    return OpenAIError(message, status=500, type_="api_error", code="internal_error")


def unavailable(message: str) -> OpenAIError:
    return OpenAIError(message, status=503, type_="api_error", code="unavailable")


def invalid_api_key(message: str) -> OpenAIError:
    return OpenAIError(message, status=401, type_="invalid_request_error", code="invalid_api_key")
