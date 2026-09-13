"""ffmpeg-backed encoding for the compressed /v1/audio/speech formats.

The engine produces raw mono s16le PCM at 24 kHz; ``wav``/``pcm`` are handled
by the stdlib in :mod:`pocket_tts_openai.engine`. The lossy/lossless compressed
formats (mp3/opus/aac/flac) are encoded here by piping that PCM through ffmpeg --
whole-file via :func:`encode_pcm` (``stream:false``) or live chunk-by-chunk via
:func:`encode_pcm_stream` (``stream:true``).

ffmpeg is optional at the app level: without it the compressed formats return a
clear client error, but the Docker image ships ffmpeg so they just work there.
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import threading
from collections.abc import Iterator
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


def _ffmpeg_command(
    ffmpeg: str, sample_rate: int, fmt: str, *, flush_packets: bool = False
) -> list[str]:
    """Build the ffmpeg argv: read mono s16le PCM on stdin, write ``fmt`` on stdout.

    ``flush_packets`` adds ``-flush_packets 1`` to the output options, which
    pushes each encoded packet out immediately (otherwise ffmpeg may buffer several
    seconds for some muxers) -- used by the streaming path so encoded bytes flow as
    they are produced. The buffered :func:`encode_pcm` leaves it off and therefore
    produces byte-identical output to the pre-streaming implementation.
    """
    if fmt not in _FFMPEG_OUT:
        raise KeyError(f"no ffmpeg mapping for format {fmt!r}")
    cmd = [
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
    ]
    if flush_packets:
        cmd.append("-flush_packets")
        cmd.append("1")
    cmd.append("pipe:1")
    return cmd


def encode_pcm(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
    """Encode raw mono s16le PCM into ``fmt`` via ffmpeg (whole-file, buffered).

    Raises ``RuntimeError`` when ffmpeg is absent or the encode fails; routes
    map that to a client-facing 400.
    """
    ffmpeg = ffmpeg_binary()
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg is not installed; response_format={fmt!r} requires it. "
            "Install ffmpeg, or use wav/pcm. (The Docker image already ships ffmpeg.)"
        )
    proc = subprocess.run(
        _ffmpeg_command(ffmpeg, sample_rate, fmt), input=pcm, capture_output=True
    )
    if proc.returncode != 0:
        stderr = proc.stderr.decode(errors="replace").strip()
        logger.error("ffmpeg failed encoding %s: %s", fmt, stderr)
        raise RuntimeError(f"ffmpeg failed encoding {fmt} (sample_rate={sample_rate}).")
    return proc.stdout


def encode_pcm_stream(
    pcm_iter: "Iterator[bytes]",
    sample_rate: int,
    fmt: str,
) -> "Iterator[bytes]":
    """Encode raw mono s16le PCM chunks *as they arrive* via ffmpeg, yielding the
    encoded bytes live -- the streaming counterpart of :func:`encode_pcm`.

    A pump thread feeds each PCM chunk into an ffmpeg subprocess (two Popen pipes)
    while this generator's body runs on the ASGI worker thread and reads ffmpeg's
    stdout with ``os.read``, so the first encoded bytes flow before PCM generation
    has finished. A second daemon thread drains ffmpeg's stderr continuously so a
    verbose encoder can never fill its ~64 KB error pipe and deadlock.

    Lifetime: the input iterator is always closed -- on both normal completion
    (it is exhausted by the pump) and abandonment. On abandonment (consumer
    ``GeneratorExit``) ffmpeg is killed and the pump joined *before*
    ``pcm_iter.close()`` is called, so releasing any lock the iterator gates
    (the engine's generation lock) can never race code that is still reading it.

    Raises ``RuntimeError`` when ffmpeg is absent or the encode fails, and
    re-raises any exception raised by ``pcm_iter`` (e.g. an unknown voice).
    """
    ffmpeg = ffmpeg_binary()
    if ffmpeg is None:
        raise RuntimeError(
            f"ffmpeg is not installed; response_format={fmt!r} requires it. "
            "Install ffmpeg, or use wav/pcm. (The Docker image already ships ffmpeg.)"
        )
    proc = subprocess.Popen(
        _ffmpeg_command(ffmpeg, sample_rate, fmt, flush_packets=True),
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    pump_error: list[BaseException] = []
    stderr_chunks: list[bytes] = []

    def _pump() -> None:
        try:
            for chunk in pcm_iter:
                try:
                    proc.stdin.write(chunk)  # type: ignore[union-attr]
                    proc.stdin.flush()  # type: ignore[union-attr]
                except (BrokenPipeError, OSError):
                    return  # ffmpeg exited; the reader surfaces it via EOF/rc
        except BaseException as exc:  # forward a source error (e.g. bad voice)
            pump_error.append(exc)
        finally:
            try:
                proc.stdin.close()  # type: ignore[union-attr]
            except OSError:
                pass

    def _drain_stderr() -> None:
        try:
            while True:
                data = proc.stderr.read(4096)  # type: ignore[union-attr]
                if not data:
                    break
                stderr_chunks.append(data)
        except (OSError, ValueError):  # pragma: no cover - fd teardown race
            pass

    pump_t = threading.Thread(target=_pump, name="ffmpeg-pump", daemon=True)
    stderr_t = threading.Thread(target=_drain_stderr, name="ffmpeg-stderr", daemon=True)
    pump_t.start()
    stderr_t.start()

    # The kill/join invariants (identical on the abort paths):
    #   1. kill ffmpeg first  (unblocks the pump's stdin write with EPIPE)
    #   2. join the pump -- now NO thread is mid-iteration on pcm_iter
    #   3. only then close pcm_iter (safe: releases the gate it holds)
    def _teardown(*, kill: bool) -> None:
        if kill:
            try:
                proc.kill()
            except ProcessLookupError:  # pragma: no cover
                pass
        pump_t.join(timeout=10)
        proc.wait()
        stderr_t.join(timeout=10)
        try:
            proc.stdout.close()  # type: ignore[union-attr]
        except OSError:  # pragma: no cover
            pass
        if kill and not pump_t.is_alive():
            close = getattr(pcm_iter, "close", None)
            if close is not None:
                close()

    try:
        while True:
            data = os.read(proc.stdout.fileno(), 65536)  # type: ignore[union-attr]
            if not data:
                break
            yield data
    except GeneratorExit:
        _teardown(kill=True)
        raise
    except BaseException:
        _teardown(kill=True)
        raise
    else:
        _teardown(kill=False)
        if pump_error:
            raise pump_error[0]
        if proc.returncode != 0:
            err = b"".join(stderr_chunks).decode(errors="replace").strip()
            logger.error("ffmpeg failed encoding %s: %s", fmt, err)
            raise RuntimeError(
                f"ffmpeg failed encoding {fmt} (sample_rate={sample_rate})."
            )
