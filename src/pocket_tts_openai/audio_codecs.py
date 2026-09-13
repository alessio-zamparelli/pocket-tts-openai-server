"""ffmpeg-backed encoding for the compressed /v1/audio/speech formats.

The engine produces raw mono s16le PCM at 24 kHz; ``wav``/``pcm`` are handled
by the stdlib in :mod:`pocket_tts_openai.engine`. The lossy/lossless compressed
formats (mp3/opus/aac/flac) are encoded here by piping that PCM through ffmpeg.

ffmpeg is optional at the app level: without it the compressed formats return a
clear client error, but the Docker image ships ffmpeg so they just work there.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
from functools import lru_cache

logger = logging.getLogger(__name__)

# Compressed response formats handled by this module (OpenAI audio API).
COMPRESSED_FORMATS = ("mp3", "opus", "aac", "flac")

# Content types matching the OpenAI audio API for each compressed format.
MEDIA_TYPES: dict[str, str] = {
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",  # Opus-in-Ogg
    "aac": "audio/aac",  # ADTS AAC
    "flac": "audio/x-flac",
}

# Per-format ffmpeg output flags (fed PCM s16le mono on stdin, bytes on stdout).
_FFMPEG_OUT: dict[str, list[str]] = {
    # VBR mp3 at roughly ~128 kbps.
    "mp3": ["-f", "mp3", "-codec:a", "libmp3lame", "-q:a", "4"],
    # Low-latency, network-friendly 96 kbps Opus in an Ogg container.
    "opus": ["-codec:a", "libopus", "-b:a", "96k", "-f", "ogg"],
    # ADTS-streamable AAC.
    "aac": ["-f", "adts", "-codec:a", "aac", "-b:a", "128k"],
    # Lossless FLAC.
    "flac": ["-f", "flac", "-codec:a", "flac"],
}


@lru_cache(maxsize=1)
def ffmpeg_binary() -> str | None:
    """Absolute path to an ``ffmpeg`` on PATH (cached), or ``None``."""
    return shutil.which("ffmpeg")


def encode_pcm(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    """Encode raw mono s16le PCM into ``fmt`` via ffmpeg.

    Raises ``RuntimeError`` when ffmpeg is absent or the encode fails; routes
    map that to a client-facing 400.
    """
    ffmpeg = ffmpeg_binary()
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg is not installed; response_format={fmt!r} requires it. "
            "Install ffmpeg, or use wav/pcm. (The Docker image already ships ffmpeg.)"
        )
    args = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-f",
        "s16le",
        "-ar",
        str(sample_rate),
        "-ac",
        "1",
        "-i",
        "pipe:0",
        *_FFMPEG_OUT[fmt],
        "pipe:1",
    ]
    proc = subprocess.run(args, input=pcm, capture_output=True)
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        logger.error("ffmpeg failed encoding %s: %s", fmt, stderr)
        raise RuntimeError(f"ffmpeg failed encoding {fmt} (sample_rate={sample_rate}).")
    return proc.stdout
