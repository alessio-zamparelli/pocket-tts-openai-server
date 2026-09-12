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


def bad_gateway(message: str) -> OpenAIError:
    return OpenAIError(message, status=502, type_="api_error", code="bad_gateway")


def invalid_api_key(message: str) -> OpenAIError:
    return OpenAIError(message, status=401, type_="invalid_request_error", code="invalid_api_key")


def conflict(message: str) -> OpenAIError:
    return OpenAIError(message, status=409, type_="invalid_request_error", code="conflict")


def not_found(message: str, code: str | None = "not_found") -> OpenAIError:
    return OpenAIError(message, status=404, type_="invalid_request_error", code=code)


def payload_too_large(message: str) -> OpenAIError:
    return OpenAIError(message, status=413, type_="invalid_request_error", code="entity_too_large")


def method_not_allowed(message: str) -> OpenAIError:
    return OpenAIError(message, status=405, type_="invalid_request_error", code="method_not_allowed")
