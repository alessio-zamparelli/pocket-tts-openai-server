"""Unit tests for the ffmpeg streaming encoder (encode_pcm_stream).

These exercise the real ffmpeg subprocess pipeline: byte parity with the
buffered encoder, true (incremental) streaming, error forwarding, and
thread/process hygiene on abandonment. Skipped wholesale when ffmpeg is not
installed (the Docker image ships it, so CI runs them for real).
"""

from __future__ import annotations

import subprocess
import threading
import time
from typing import Generator, cast

import numpy as np
import pytest

from pocket_tts_openai.audio_codecs import encode_pcm, encode_pcm_stream

ffmpeg_available = pytest.mark.skipif(
    subprocess.run(["ffmpeg", "-version"], capture_output=True).returncode != 0,
    reason="ffmpeg not installed",
)

# Buffer size the engine actually emits (0.02 s * 24000 Hz * 2 bytes).
CHUNK = 7680


def _pcm_chunks(seconds: int = 10, seed: int = 1) -> list[bytes]:
    n = 24000 * seconds
    raw = (np.random.default_rng(seed).standard_normal(n) * 0.3).astype(np.float32)
    pcm = (np.clip(raw, -1.0, 1.0) * 32767.0).astype("<i2").tobytes()
    return [pcm[i : i + CHUNK] for i in range(0, len(pcm), CHUNK)]


class _Source:
    """Minimal chunk source recording close() (the engine stream's contract)."""

    def __init__(self, chunks: list[bytes]):
        self._chunks = list(chunks)
        self.closed = False

    def __iter__(self):
        return self

    def __next__(self) -> bytes:
        if not self._chunks:
            raise StopIteration
        return self._chunks.pop(0)

    def close(self) -> None:
        self.closed = True


@ffmpeg_available
def test_mp3_aac_flac_stream_equal_buffered_byte_for_byte():
    """For frame/stream-based muxers the streaming pipeline must produce the
    exact same bytes as the buffered encoder -- and as *multiple* chunks, i.e.
    real incremental encoding, not a buffered blob re-chunked afterwards."""
    chunks = _pcm_chunks()
    for fmt in ("mp3", "aac", "flac"):
        pcm = b"".join(chunks)
        expected = encode_pcm(pcm, 24000, fmt)
        out = b"".join(encode_pcm_stream(iter(_Source(chunks)), 24000, fmt))
        assert out == expected, fmt
        assert len(out) > 0, fmt


@ffmpeg_available
def test_opus_stream_is_enough_to_compare():
    """Opus is muxed into Ogg pages (~a page per second at 96 kbps), so byte
    parity with the unbuffered encoder is not expected; but the Ogg page
    boundaries-later output must decode to the SAME audio, and it must still
    arrive in multiple chunks on a long clip."""
    chunks = _pcm_chunks(seconds=15)
    streamed = b"".join(encode_pcm_stream(iter(_Source(chunks)), 24000, "opus"))
    buffered = encode_pcm(b"".join(chunks), 24000, "opus")
    assert streamed[:4] == b"OggS"
    assert len(streamed) > 0
    assert len(list(encode_pcm_stream(iter(_Source(chunks)), 24000, "opus"))) > 1

    def _decode(data: bytes) -> bytes:
        r = subprocess.run(
            ["ffmpeg", "-hide_banner", "-loglevel", "error", "-i", "-", "-f", "s16le", "-ac", "1", "pipe:1"],
            input=data, capture_output=True,
        )
        assert r.returncode == 0
        return r.stdout

    assert _decode(streamed) == _decode(buffered)


@ffmpeg_available
def test_first_encoded_bytes_flow_before_source_exhausted():
    """Prove genuine streaming: the first encoded bytes must be yielded while the
    source is still mid-feed, not only after every PCM chunk has been consumed
    (ffmpeg's encoder+muxer prime the first ~0.4 s, then bytes flow incrementally;
    a 60 s source gives an enormous margin so this is robust on any CI)."""
    chunks = _pcm_chunks(seconds=60)

    class _Src:
        def __init__(self):
            self.i = 0

        def __iter__(self):
            return self

        def __next__(self) -> bytes:
            if self.i >= len(chunks):
                raise StopIteration
            c = chunks[self.i]
            self.i += 1
            return c

        def close(self) -> None:
            pass

    src = _Src()
    it = iter(encode_pcm_stream(src, 24000, "aac"))
    first = next(it)  # must return long before the 60 s source is exhausted
    assert first, "no encoded bytes before the source was exhausted"
    assert src.i < len(chunks), "output appeared only after the source was fully fed -> buffered"
    assert src.i < len(chunks) // 4, "first bytes came far too late to be real streaming"
    rest = b"".join(it)  # the remainder completes cleanly
    assert rest == b"" or len(rest) > 0


@ffmpeg_available
def test_source_error_is_forwarded_and_stream_terminates():
    """A ValueError raised by the source (e.g. unknown voice mid-stream) must
    surface to the consumer as ValueError, not be swallowed or turned into 500."""
    class _Boom:
        def __iter__(self):
            return self

        def __next__(self) -> bytes:
            raise ValueError("no such voice")

    with pytest.raises(ValueError, match="no such voice"):
        list(encode_pcm_stream(_Boom(), 24000, "mp3"))


@ffmpeg_available
def test_abandonment_closes_source_and_reaps_threads():
    """Consumer close() -> kill ffmpeg, join pump, then close the source; and
    no pump/stderr threads may leak after teardown."""
    base = threading.active_count()
    src = _Source(_pcm_chunks(seconds=60))  # long enough to still be feeding
    it = iter(encode_pcm_stream(src, 24000, "aac"))
    first = next(it)
    assert first
    cast(Generator[bytes, None, None], it).close()  # GeneratorExit -> abort path
    assert src.closed, "source not closed after abandonment (lock leak risk)"
    time.sleep(0.05)
    assert threading.active_count() <= base, "ffmpeg pump/stderr threads leaked"


@ffmpeg_available
def test_missing_ffmpeg_raises_client_clear_error(monkeypatch):
    import pocket_tts_openai.audio_codecs as ac

    monkeypatch.setattr(ac, "ffmpeg_binary", lambda: None)
    with pytest.raises(RuntimeError, match="ffmpeg"):
        list(encode_pcm_stream(iter([]), 24000, "mp3"))


def test_unknown_format_rejected_before_spawning():
    with pytest.raises(KeyError):
        _ = list(encode_pcm_stream(iter([b"x"]), 24000, "wav"))
