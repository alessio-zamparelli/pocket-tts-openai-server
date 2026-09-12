"""OpenAI-compatible STT endpoints: ``/v1/audio/transcriptions`` +
``/v1/audio/translations`` (PLAN-STT.md / M6).

The engine is a native ``whisper-server`` sidecar (:mod:`.stt`). This module
implements the OpenAI contract on top: multipart ``file`` upload validation,
the ``model``/``language``/``prompt``/``response_format``/``temperature``/
``timestamp_granularities[]`` field mapping to whisper-server's own multipart
``/inference``, and response shaping back to the OpenAI shapes (json /
verbose_json / text / srt / vtt).
"""

from __future__ import annotations

from typing import TYPE_CHECKING
from pathlib import Path

from fastapi import APIRouter, Request, Response
from fastapi.responses import JSONResponse
from starlette.datastructures import UploadFile

from .errors import (
    bad_gateway,
    invalid_request,
    payload_too_large,
    unavailable,
)
from .stt import WhisperSidecar

if TYPE_CHECKING:
    from starlette.datastructures import FormData

logger = __import__("logging").getLogger(__name__)

router = APIRouter()

# All OpenAI STT model aliases map to the single whisper.cpp model.
STT_MODEL_ALIASES: tuple[str, ...] = ("whisper-1",)

_VALID_RESPONSE_FORMATS = ("json", "text", "srt", "vtt", "verbose_json")
_VALID_GRANULARITIES = ("segment", "word")

# How whisper-server's /inference is asked to produce each OpenAI format.
# ``json`` (the default) maps to whisper's bare default, which already returns
# ``{"text": ...}`` — so we send no ``response_format`` field at all.
_UPSTREAM_FORMAT: dict[str, str | None] = {
    "json": None,
    "text": "text",
    "srt": "srt",
    "vtt": "vtt",
    "verbose_json": "verbose_json",
}

# Response content-type + attachment filename per OpenAI format.
_RESPONSE_MEDIA = {
    "json": "application/json",
    "verbose_json": "application/json",
    "text": "text/plain",
    "srt": "application/x-subrip",
    "vtt": "text/vtt",
}
_ATTACHMENT_FILENAME = {
    "json": None,
    "verbose_json": None,
    "text": "transcription.txt",
    "srt": "transcription.srt",
    "vtt": "transcription.vtt",
}


def _sidecar(request: Request) -> WhisperSidecar | None:
    return getattr(request.app.state, "stt", None)


@router.post("/v1/audio/transcriptions", response_model=None)
async def transcribe(request: Request) -> Response:
    """Transcribe an uploaded audio file (multipart/form-data), OpenAI-shaped."""
    return await _handle_stt(request, translate=False)


@router.post("/v1/audio/translations", response_model=None)
async def translation(request: Request) -> Response:
    """Translate an uploaded audio file into English (multipart/form-data)."""
    return await _handle_stt(request, translate=True)


async def _handle_stt(request: Request, *, translate: bool) -> Response:
    sidecar = _sidecar(request)
    cfg = request.app.state.config
    if sidecar is None:
        raise unavailable(
            "STT is not enabled (POCKET_TTS_STT_ENABLED=true) or the sidecar is "
            "still initializing."
        )

    form = await request.form()

    model = _form_str(form, "model")
    if not model:
        raise invalid_request("Missing required form field 'model'.")
    if model not in STT_MODEL_ALIASES:
        raise invalid_request(
            f"Unknown model '{model}'. Available STT models: {', '.join(STT_MODEL_ALIASES)}."
        )

    file = _form_file(form)
    if file is None:
        raise invalid_request("Missing required file field. Attach audio as 'file'.")
    if not (file.filename or "").strip():
        raise invalid_request("Audio filename must not be empty.")

    response_format = _form_str(form, "response_format", "json") or "json"
    if response_format not in _VALID_RESPONSE_FORMATS:
        raise invalid_request(
            f"Unsupported response_format '{response_format}'. "
            f"Supported: {', '.join(_VALID_RESPONSE_FORMATS)}."
        )

    granularities = _form_list(form, "timestamp_granularities")
    unknown = [g for g in granularities if g not in _VALID_GRANULARITIES]
    if unknown:
        raise invalid_request(
            f"Unsupported timestamp_granularities value(s): {', '.join(unknown)}. "
            f"Supported: {', '.join(_VALID_GRANULARITIES)}."
        )

    limit = cfg.max_upload_mb * 1024 * 1024
    audio = await _read_upload(file, limit)
    if not audio:
        raise invalid_request("Uploaded audio file is empty.")
    if len(audio) > limit:
        raise payload_too_large(f"Audio exceeds the {cfg.max_upload_mb} MB limit.")

    filename = Path(file.filename or "audio").name
    content_type = file.content_type or "application/octet-stream"
    await file.close()

    # Build the upstream whisper-server /inference field set.
    data: dict[str, str] = {}
    upstream = _UPSTREAM_FORMAT[response_format]
    if upstream is not None:
        data["response_format"] = upstream
    if not translate:
        language = _form_str(form, "language")
        if language:
            data["language"] = language
    prompt = _form_str(form, "prompt")
    if prompt:
        data["prompt"] = prompt
    temperature = _form_str(form, "temperature")
    if temperature:
        data["temperature"] = temperature
    if "word" in granularities:
        # word granularity -> whisper token timestamps (best-effort in verbose_json).
        data["token_timestamps"] = "true"
    if translate:
        data["translate"] = "true"

    # Wake path: block until the sidecar is ready (single-flight re-spawn after
    # an idle eviction), then proxy. No 503 in the wake path; only a genuine
    # failure to (re)start surfaces as unavailable.
    if not sidecar.ensure_started():
        raise unavailable(f"STT sidecar is not ready: {sidecar.last_error_hint()}.")

    # The request counts as STT activity (resets the idle eviction window).
    sidecar.touch()

    resp = sidecar.transcribe_raw(audio, filename, content_type, data)
    return _render_upstream(resp, response_format)


def _form_str(form, name: str, default: str = "") -> str:
    """Extract a string form field, tolerating an UploadFile (never a string)."""
    value = form.get(name)
    if value is None:
        return default
    if isinstance(value, UploadFile):
        return default
    return str(value).strip()


def _form_file(form) -> UploadFile | None:
    """Return the ``file`` upload, or None (also for a non-file string field)."""
    value = form.get("file")
    return value if isinstance(value, UploadFile) else None


def _form_list(form, name: str) -> list[str]:
    """Read a repeated form field, tolerating either bare or ``name[]`` spellings
    (some OpenAI clients send ``timestamp_granularities[]``)."""
    try:
        vals = form.getlist(name) or form.getlist(f"{name}[]")
    except AttributeError:  # pragma: no cover - plain dict in some test harnesses
        vals = form.get(name)
        if isinstance(vals, str):
            vals = [vals]
        elif isinstance(vals, UploadFile):
            vals = []
        else:
            vals = []
    return [str(v).strip() for v in vals if str(v).strip()]


async def _read_upload(file: UploadFile, limit: int) -> bytes:
    """Read the uploaded file up to limit+1 bytes so we can reject oversize."""
    chunks: list[bytes] = []
    total = 0
    while True:
        chunk = await file.read(1 << 20)
        if not chunk:
            break
        total += len(chunk)
        if total > limit:
            raise payload_too_large("Audio exceeds the upload size limit.")
        chunks.append(chunk)
    return b"".join(chunks)


def _render_upstream(resp, response_format: str) -> Response:
    """Shape whisper-server's /inference response into the OpenAI format."""
    if resp.status_code >= 500:
        logger.error("whisper-server upstream error: %s", resp.text[:500])
        raise bad_gateway("The STT backend returned an internal error; retry shortly.")
    if resp.status_code >= 400:
        try:
            detail = resp.json().get("error") or resp.text
        except Exception:  # pragma: no cover - non-JSON error body
            detail = resp.text
        raise invalid_request(f"STT backend rejected the request: {detail}")

    if response_format == "json":
        # whisper's bare default returns {"text": ...}; fall back to a text wrap
        # if the upstream body is not JSON (defensive).
        try:
            body = resp.json()
        except Exception:
            body = {"text": resp.text.strip()}
        return JSONResponse(content={"text": body.get("text", "")})

    if response_format == "verbose_json":
        try:
            body = resp.json()
        except Exception:  # pragma: no cover - should be JSON from upstream
            body = {"text": resp.text.strip(), "segments": []}
        return JSONResponse(content=body)

    # text / srt / vtt -> raw body with the matching content type + attachment.
    filename = _ATTACHMENT_FILENAME[response_format]
    headers: dict[str, str] = {}
    if filename:
        headers["Content-Disposition"] = f'attachment; filename="{filename}"'
    return Response(
        content=resp.content,
        media_type=_RESPONSE_MEDIA[response_format],
        headers=headers,
    )
