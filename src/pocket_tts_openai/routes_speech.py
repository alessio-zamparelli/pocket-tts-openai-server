"""OpenAI-compatible endpoint handlers: /v1/audio/speech, /v1/models, /health."""

from __future__ import annotations

import time
from typing import Literal

from fastapi import Request, Response
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from .audio_codecs import MEDIA_TYPES, encode_pcm
from .config import Config
from .engine import RateLimited, TTSEngine, pcm_to_wav, streaming_wav_header
from .errors import OpenAIError, invalid_request, rate_limited, unavailable

AudioFormat = Literal["wav", "pcm", "mp3", "opus", "aac", "flac"]

# Supported response formats. wav/pcm are stdlib; mp3/opus/aac/flac are
# encoded through ffmpeg (audio_codecs.py) and 400 with a clear message when
# ffmpeg is missing (the Docker image ships it, so they just work there).
SUPPORTED_FORMATS = ("wav", "pcm", "mp3", "opus", "aac", "flac")

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
    stream: bool = Field(
        default=False,
        description="Private extension: chunked transfer for wav/pcm. Ignored (buffered) for compressed formats.")


def _engine_or_503(request: Request) -> TTSEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise unavailable("Model is still loading; retry shortly.")
    return engine


def _stream_speech_pcm(engine: TTSEngine, text: str, voice: str):
    """Yield raw s16le PCM chunks from the engine's streaming generator."""
    for chunk in engine.generate_pcm_stream(text, voice):
        yield chunk


def _stream_speech_wav(engine: TTSEngine, text: str, voice: str):
    """Yield a size-less WAV header then s16le PCM chunks."""
    yield streaming_wav_header(engine.sample_rate)
    for chunk in engine.generate_pcm_stream(text, voice):
        yield chunk


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

    engine = _engine_or_503(request)
    # Voice resolution (aliases, catalog, custom registry) happens inside the
    # engine; propagate unknown-voice as a 400.

    def _speech_error() -> Exception:
        return invalid_request(
            f"Unknown voice {req.voice!r}. Use an OpenAI alias, a catalog voice, "
            "a custom voice, an https:// URL or an hf:// reference."
        )

    # streaming path (wav/pcm only -- lossy formats are buffered whole-file)
    if req.stream and req.response_format in ("wav", "pcm"):
        # Admit-or-429 before headers so overload doesn't kill a stream midway.
        try:
            engine.check_capacity()
        except RateLimited as exc:
            raise rate_limited(str(exc)) from None
        filename = "speech.pcm" if req.response_format == "pcm" else "speech.wav"
        media_type = "audio/pcm" if req.response_format == "pcm" else "audio/wav"
        gen = _stream_speech_pcm if req.response_format == "pcm" else _stream_speech_wav
        try:
            return StreamingResponse(
                gen(engine, req.input, req.voice),
                media_type=media_type,
                headers={"Content-Disposition": f"attachment; filename={filename}"},
            )
        except ValueError:
            raise _speech_error() from None

    try:
        pcm = engine.generate_pcm(req.input, req.voice)
    except ValueError:
        raise _speech_error() from None
    except RateLimited as exc:
        raise rate_limited(str(exc)) from None
    if req.response_format == "pcm":
        body, filename, media_type = pcm, "speech.pcm", "audio/pcm"
    elif req.response_format == "wav":
        body, filename, media_type = pcm_to_wav(pcm, engine.sample_rate), "speech.wav", "audio/wav"
    else:
        try:
            body = encode_pcm(pcm, engine.sample_rate, req.response_format)
        except RuntimeError as exc:
            raise invalid_request(str(exc)) from exc
        filename, media_type = f"speech.{req.response_format}", MEDIA_TYPES[req.response_format]
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
    """Liveness + engine stats. Status is 'loading' until the model is ready.
    After an idle eviction ``loaded`` is False (model waking on next request).
    Health probes deliberately do NOT touch the engine's idle timer.
    """
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        return {"status": "loading", "model": "pocket-tts"}
    stats = engine.stats
    return {
        "status": "ok",
        "model": "pocket-tts",
        "language": engine.language,
        "loaded": engine.loaded,
        "idle_unload_s": engine._config.idle_unload_s,
        "last_request_age_s": round(time.monotonic() - engine._last_activity, 1),
        "unloads": stats.unloads,
        "reloads": stats.reloads,
        "requests": stats.requests,
        "avg_rtf": round(stats.avg_rtf or 0.0, 4),
        "queue_depth": stats.waiting,
    }
