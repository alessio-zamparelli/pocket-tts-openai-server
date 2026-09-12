"""Server configuration from environment variables.

All settings have sane defaults; every knob is optional.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

from .voices import DEFAULT_VOICE_ALIASES


def _split_voice_map(raw: str) -> dict[str, str]:
    """Parse ``alloy=alba,coral=giovanni`` into a dict."""
    out: dict[str, str] = {}
    for pair in raw.split(","):
        pair = pair.strip()
        if not pair:
            continue
        if "=" not in pair:
            raise ValueError(f"POCKET_TTS_VOICE_MAP entry must be `alias=voice`, got: {pair!r}")
        alias, voice = pair.split("=", 1)
        out[alias.strip()] = voice.strip()
    return out


def _env_int(env: dict[str, str], key: str, default: int) -> int:
    raw = env.get(key, "")
    if not raw:
        return default
    try:
        return int(raw)
    except ValueError:
        raise ValueError(f"{key} must be an integer, got: {raw!r}") from None


@dataclass(frozen=True)
class Config:
    host: str = "127.0.0.1"
    port: int = 8000
    language: str = "english"
    default_voice: str = "alloy"
    voice_map: dict[str, str] = field(default_factory=lambda: dict(DEFAULT_VOICE_ALIASES))
    api_key: str | None = None
    warmup_voices: tuple[str, ...] = ()
    quantize: bool = False
    max_cached_voices: int = 32

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        env = dict(os.environ if env is None else env)
        voice_map = dict(DEFAULT_VOICE_ALIASES)
        if raw := env.get("POCKET_TTS_VOICE_MAP", ""):
            voice_map.update(_split_voice_map(raw))
        warmup = tuple(
            v.strip() for v in env.get("POCKET_TTS_WARMUP_VOICES", "").split(",") if v.strip()
        )
        return cls(
            host=env.get("POCKET_TTS_HOST", "127.0.0.1"),
            port=_env_int(env, "POCKET_TTS_PORT", 8000),
            language=env.get("POCKET_TTS_LANGUAGE", "english"),
            default_voice=env.get("POCKET_TTS_DEFAULT_VOICE", "alloy"),
            voice_map=voice_map,
            api_key=env.get("POCKET_TTS_API_KEY") or None,
            warmup_voices=warmup,
            quantize=env.get("POCKET_TTS_QUANTIZE", "").lower() in ("1", "true", "yes"),
            max_cached_voices=_env_int(env, "POCKET_TTS_MAX_CACHED_VOICES", 32),
        )
