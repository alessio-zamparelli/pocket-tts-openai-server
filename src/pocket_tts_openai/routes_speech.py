"""OpenAI-compatible endpoint handlers: /v1/audio/speech, /v1/models, /health."""

from __future__ import annotations

from typing import Literal

from fastapi import Request, Response
from pydantic import BaseModel, Field

from .config import Config
from .engine import TTSEngine, pcm_to_wav
from .errors import OpenAIError, invalid_request, unavailable
from .voices import resolve_voice

AudioFormat = Literal["wav", "pcm", "mp3", "opus", "aac", "flac"]

# Formats shippable in M1/M2. mp3/opus/aac/flac need ffmpeg (M2): 400 until then.
SUPPORTED_FORMATS = ("wav", "pcm")
PLANNED_FORMATS = ("mp3", "opus", "aac", "flac")

# All OpenAI TTS model aliases map to the single pocket-tts model.
MODEL_ALIASES: tuple[str, ...] = ("tts-1", "tts-1-hd", "gpt-4o-mini-tts")

_CREATED = 1700000000  # fixed timestamp so responses are deterministic


class SpeechRequest(BaseModel):
    model: str = Field(default="tts-1", description="Model alias; all aliases map to pocket-tts.")
    input: str = Field(description="Text to synthesize.")
    voice: str = Field(default="alloy", description="OpenAI voice alias or Kyutai voice name.")
    response_format: AudioFormat = Field(default="wav")
    speed: float | None = Field(
        default=None, description="Accepted for compatibility but ignored (see README).")
    instructions: str | None = Field(
        default=None, description="Accepted for compatibility but ignored (see README).")
    language: str | None = Field(
        default=None, description="Server-configured language always wins; logged and ignored if different.")


def _engine_or_503(request: Request) -> TTSEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise unavailable("Model is still loading; retry shortly.")
    return engine


def speech(req: SpeechRequest, request: Request) -> Response:
    """Synthesize ``req.input`` with the requested voice; return wav or pcm bytes."""
    if req.model not in MODEL_ALIASES:
        raise invalid_request(
            f"Unknown model '{req.model}'. Available models: {', '.join(MODEL_ALIASES)}.")
    if not req.input.strip():
        raise invalid_request("input must be non-empty text.")
    if req.response_format not in SUPPORTED_FORMATS:
        raise invalid_request(
            f"response_format '{req.response_format}' is not supported yet; "
            f"supported formats: {', '.join(SUPPORTED_FORMATS)}.")
    config: Config = request.app.state.config
    try:
        resolved_voice = resolve_voice(req.voice, config.voice_map)
    except ValueError as exc:
        raise invalid_request(str(exc)) from exc

    engine = _engine_or_503(request)
    pcm = engine.generate_pcm(req.input, resolved_voice)
    if req.response_format == "pcm":
        body, filename, media_type = pcm, "speech.pcm", "audio/pcm"
    else:
        body, filename, media_type = pcm_to_wav(pcm, engine.sample_rate), "speech.wav", "audio/wav"
    return Response(
        content=body,
        media_type=media_type,
        headers={"Content-Disposition": f"attachment; filename={filename}"},
    )


def models() -> dict:
    """List available models: every alias maps to the single pocket-tts model."""
    return {
        "object": "list",
        "data": [
            {"id": alias, "object": "model", "created": _CREATED, "owned_by": "pocket-tts"}
            for alias in MODEL_ALIASES
        ],
    }


def health(request: Request) -> dict:
    """Liveness + engine stats. Status is 'loading' until the model is ready."""
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {"status": "loading", "model": "pocket-tts"}
    stats = engine.stats
    return {
        "status": "ok",
        "model": "pocket-tts",
        "language": engine.language,
        "requests": stats.requests,
        "avg_rtf": round(stats.avg_rtf or 0.0, 4),
        "queue_depth": stats.waiting,
    }
