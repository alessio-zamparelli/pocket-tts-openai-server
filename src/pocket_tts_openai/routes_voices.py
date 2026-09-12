"""Voice catalog + voice cloning: ``GET/POST/DELETE /v1/voices``.

Private extensions (not part of the OpenAI Audio API). Let clients discover
valid ``voice`` values and clone new voices from an uploaded audio prompt.
"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, File, Form, Request, UploadFile
from fastapi import Response

from .config import Config
from .engine import TTSEngine
from .errors import (
    OpenAIError,
    conflict,
    invalid_request,
    method_not_allowed,
    not_found,
    payload_too_large,
    unavailable,
)
from .voice_registry import validate_voice_name
from .voices import DEFAULT_VOICE_ALIASES, KYUTAI_CATALOG, VOICE_LANGUAGE

# Extensions that pocket-tts can decode as an audio prompt.
_AUDIO_EXTS = (".wav", ".mp3", ".flac")

# HF repo that hosts the Kyutai voice licenses / origins.
VOICES_REPO = "https://huggingface.co/kyutai/tts-voices"

# Content-type header used by multipart client hints.
router = APIRouter()


def _engine_or_503(request: Request) -> TTSEngine:
    engine = getattr(request.app.state, "engine", None)
    if engine is None:
        raise unavailable("Model is still loading; retry shortly.")
    return engine


def _builtin_voice_entry(alias: str, target: str) -> dict:
    return {
        "id": alias,
        "aliases": [alias, target],
        "source": "builtin",
        "language": VOICE_LANGUAGE.get(target, "en"),
        "license": VOICES_REPO,
        "cached": False,
    }


def _custom_voice_entry(request: Request, name: str, voice) -> dict:
    engine = getattr(request.app.state, "engine", None)
    return {
        "id": name,
        "aliases": [name],
        "source": "custom",
        "language": voice.language,
        "license": None,
        "cached": bool(engine and name in engine.cached_voices()),
        "safetensors": True,
    }


@router.get("")
def list_voices(request: Request) -> dict:
    """Return the full voice catalog: OpenAI aliases, extra catalog names, customs."""
    cfg: Config = request.app.state.config
    engine = getattr(request.app.state, "engine", None)
    cached = engine.cached_voices() if engine is not None else frozenset()

    data: list[dict] = []
    seen: set[str] = set()
    # OpenAI aliases first (stable, declared order).
    for alias, target in cfg.voice_map.items():
        entry = _builtin_voice_entry(alias, target)
        entry["cached"] = target in cached
        data.append(entry)
        seen.add(target)
    # Extra catalog voices not already covered by an alias.
    for name in sorted(KYUTAI_CATALOG):
        if name in seen:
            continue
        entry = _builtin_voice_entry(name, name)
        entry["cached"] = name in cached
        data.append(entry)
    # Custom voices from the registry.
    for voice in (engine.registry.all() if engine and engine.registry else []):
        data.append(_custom_voice_entry(request, voice.name, voice))

    return {"object": "list", "data": data}


def _collision(name: str, cfg: Config, engine: TTSEngine) -> str | None:
    """Return a human-readable conflict source if ``name`` is taken, else None."""
    if name in cfg.voice_map:
        return f"'{name}' is a built-in OpenAI alias"
    if name in KYUTAI_CATALOG:
        return f"'{name}' is a built-in Kyutai catalog voice"
    if engine.registry and name in engine.registry.names():
        return f"'{name}' already exists as a custom voice"
    return None


@router.post("", status_code=201)
def create_voice(
    request: Request,
    name: str = Form(...),
    file: UploadFile = File(...),
    language: str | None = Form(default=None),
) -> dict:
    """Clone a voice from an uploaded audio prompt and persist it."""
    cfg: Config = request.app.state.config
    engine = _engine_or_503(request)
    if engine.registry is None:
        raise invalid_request("voice registry is not configured")

    error = validate_voice_name(name)
    if error:
        raise invalid_request(error)

    source = _collision(name, cfg, engine)
    if source:
        raise conflict(f"Cannot create voice: {source}.")

    # Extension decides the audio format (pocket-tts decodes by extension).
    suffix = Path(file.filename or "").suffix.lower()
    if suffix not in _AUDIO_EXTS:
        raise invalid_request(
            f"Unsupported file type {suffix or '(none)'!r}. "
            f"Supported: {', '.join(_AUDIO_EXTS)}."
        )

    limit = cfg.max_upload_mb * 1024 * 1024
    # Write the upload up to limit+1 bytes so we can reject oversize cleanly.
    with tempfile.NamedTemporaryFile(
        prefix=f"clone-{name}-", suffix=suffix, delete=False, dir=engine.registry.directory
    ) as tmp:
        tmp_path = Path(tmp.name)
        try:
            total = 0
            while True:
                chunk = file.file.read(1024 * 1024)
                if not chunk:
                    break
                total += len(chunk)
                if total > limit:
                    raise payload_too_large(
                        f"Upload exceeds the {cfg.max_upload_mb} MB limit."
                    )
                tmp.write(chunk)
        finally:
            file.file.close()

    try:
        engine.clone_voice(name, tmp_path)
    except OpenAIError:
        tmp_path.unlink(missing_ok=True)
        raise
    except Exception as exc:
        tmp_path.unlink(missing_ok=True)
        raise invalid_request(f"Failed to encode voice prompt: {exc}") from exc
    finally:
        tmp_path.unlink(missing_ok=True)

    voice = engine.registry.add(name, language=language)
    return _custom_voice_entry(request, voice.name, voice)


@router.delete("/{name}")
def delete_voice(name: str, request: Request) -> Response:
    """Remove a custom voice (file + registry entry). Builtin voices are 405."""
    cfg: Config = request.app.state.config
    engine = _engine_or_503(request)
    if engine.registry is None:
        raise invalid_request("voice registry is not configured")

    if name in cfg.voice_map or name in KYUTAI_CATALOG:
        raise method_not_allowed(f"'{name}' is a built-in voice and cannot be deleted.")

    voice = engine.registry.get(name)
    if voice is None:
        raise not_found(f"Unknown custom voice '{name}'.")

    engine.registry.remove(name)
    engine.evict_voice(name)
    engine.registry.path_for(name).unlink(missing_ok=True)
    return Response(status_code=204)