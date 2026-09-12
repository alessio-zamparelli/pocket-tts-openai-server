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
            raise ValueError(f"STTS_VOICE_MAP entry must be `alias=voice`, got: {pair!r}")
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


def _env_float(env: dict[str, str], key: str, default: float) -> float:
    raw = env.get(key, "")
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        raise ValueError(f"{key} must be a number, got: {raw!r}") from None


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
    cache_dir: str = ""  # empty = pocket-tts default (~/.cache/pocket_tts)
    max_upload_mb: int = 25
    idle_unload_s: int = 300  # evict the model to free RAM after this much idle; 0 = off
    idle_poll_s: int = 30  # watchdog cadence (only when idle_unload_s > 0)
    # STT (whisper.cpp sidecar). Master switch defaults OFF so TTS-only deploys
    # are unaffected; set STTS_STT_ENABLED=true to serve /v1/audio/transcriptions.
    stt_enabled: bool = False
    stt_model: str = "small"  # ggml-{model}.bin (multilingual); a quantized variant
    # (e.g. "small.q5_0") is selectable by full filename.
    stt_model_repo: str = "ggerganov/whisper.cpp"  # HF repo hosting ggml-*.bin
    stt_model_dir: str = ""  # empty = {cache_dir}/stt-models
    stt_bin: str = "whisper-server"  # sidecar binary path (tests/advanced override)
    stt_host: str = "127.0.0.1"  # internal loopback bind, never exposed
    stt_port: int = 8787  # internal HTTP port proxied by the app
    stt_threads: int = 1  # whisper -t compute threads; 1 leaves low-power/
    # SBC cores for the TTS engine (e.g. Intel N150 = 4 C-cores no HT) — raise
    # on beefier hosts via STTS_STT_THREADS.
    stt_language: str = ""  # optional default whisper language; empty = auto-detect
    stt_idle_unload_s: int = 300  # kill the sidecar after this long without an STT
    # request (reclaims its RAM); 0 = off, independent of idle_unload_s.
    stt_idle_poll_s: int = 30  # dedicated STT watchdog cadence
    stt_min_duration_s: float = 1.2  # whisper.cpp drops audio under ~1.0-1.2 s;
    # short uploads are time-stretched (slowed) to at least this many seconds
    # before transcribing so sub-second clips are not silently lost; 0 = off.

    @classmethod
    def from_env(cls, env: dict[str, str] | None = None) -> "Config":
        env = dict(os.environ if env is None else env)
        voice_map = dict(DEFAULT_VOICE_ALIASES)
        if raw := env.get("STTS_VOICE_MAP", ""):
            voice_map.update(_split_voice_map(raw))
        warmup = tuple(
            v.strip() for v in env.get("STTS_WARMUP_VOICES", "").split(",") if v.strip()
        )
        return cls(
            host=env.get("STTS_HOST", "127.0.0.1"),
            port=_env_int(env, "STTS_PORT", 8000),
            language=env.get("STTS_LANGUAGE", "english"),
            default_voice=env.get("STTS_DEFAULT_VOICE", "alloy"),
            voice_map=voice_map,
            api_key=env.get("STTS_API_KEY") or None,
            warmup_voices=warmup,
            quantize=env.get("STTS_QUANTIZE", "").lower() in ("1", "true", "yes"),
            max_cached_voices=_env_int(env, "STTS_MAX_CACHED_VOICES", 32),
            cache_dir=env.get("STTS_CACHE_DIR", ""),
            max_upload_mb=_env_int(env, "STTS_MAX_UPLOAD_MB", 25),
            idle_unload_s=_env_int(env, "STTS_IDLE_UNLOAD_S", 300),
            idle_poll_s=_env_int(env, "STTS_IDLE_POLL_S", 30),
            stt_enabled=env.get("STTS_STT_ENABLED", "").lower() in ("1", "true", "yes"),
            stt_model=env.get("STTS_STT_MODEL", "small").strip() or "small",
            stt_model_repo=env.get("STTS_STT_MODEL_REPO", "ggerganov/whisper.cpp").strip() or "ggerganov/whisper.cpp",
            stt_model_dir=env.get("STTS_STT_MODEL_DIR", "").strip(),
            stt_bin=env.get("STTS_STT_BIN", "whisper-server").strip() or "whisper-server",
            stt_host=env.get("STTS_STT_HOST", "127.0.0.1").strip() or "127.0.0.1",
            stt_port=_env_int(env, "STTS_STT_PORT", 8787),
            stt_threads=_env_int(env, "STTS_STT_THREADS", 1),
            stt_language=env.get("STTS_STT_LANGUAGE", "").strip(),
            stt_idle_unload_s=_env_int(env, "STTS_STT_IDLE_UNLOAD_S", 300),
            stt_idle_poll_s=_env_int(env, "STTS_STT_IDLE_POLL_S", 30),
            stt_min_duration_s=_env_float(env, "STTS_STT_MIN_DURATION_S", 1.2),
        )
