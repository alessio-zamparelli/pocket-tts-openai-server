"""Tests for M4 streaming: /v1/audio/speech with ``stream: true``."""

from __future__ import annotations

import io
import struct
from threading import Thread

from fastapi.testclient import TestClient

from pocket_tts_openai.server import create_app

S16LE = struct.Struct("<h")


def _assert_valid_pcm(pcm: bytes) -> None:
    assert len(pcm) == 24000 * 2  # 1 s of silence
    # frame boundary aligned and silent
    assert len(pcm) % 2 == 0
    for i in range(0, len(pcm), 2):
        assert S16LE.unpack_from(pcm, i)[0] == 0


def test_stream_pcm_chunks(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "hello world", "voice": "alloy", "response_format": "pcm", "stream": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    _assert_valid_pcm(r.content)


def test_stream_wav_header_and_chunks(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "hello", "voice": "alloy", "response_format": "wav", "stream": True},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    body = r.content
    # 44-byte RIFF header + 1 s of audio
    assert len(body) == 44 + 24000 * 2
    # streaming header: RIFF + data sizes = 0xFFFFFFFF
    assert body[:4] == b"RIFF"
    assert struct.unpack_from("<I", body, 4)[0] == 0xFFFFFFFF
    assert body[36:40] == b"data"
    assert struct.unpack_from("<I", body, 40)[0] == 0xFFFFFFFF
    assert struct.unpack_from("<HH", body, 22) == (1, 24000)  # mono, 24 kHz
    # remaining bytes are valid silence
    _assert_valid_pcm(body[44:])


def test_stream_wav_is_playable_by_wave_module(client):
    """The 0xFFFFFFFF sizes are recognized leniently by Python's wave reader."""
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "voice": "alloy", "response_format": "wav", "stream": True},
    )
    body = r.content
    with io.BytesIO(body) as buf:
        with __import__("wave").open(buf, "rb") as w:
            assert w.getnchannels() == 1
            assert w.getsampwidth() == 2
            assert w.getframerate() == 24000


def test_stream_false_is_identical_to_static_response(client):
    """stream:false must byte-match the non-streaming path (regression guard)."""
    r_static = client.post(
        "/v1/audio/speech", json={"input": "hello", "voice": "echo", "response_format": "pcm"}
    )
    r_stream_false = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "echo", "response_format": "pcm", "stream": False},
    )
    assert r_static.status_code == 200
    assert r_stream_false.status_code == 200
    assert r_stream_false.content == r_static.content


def test_stream_wav_false_identical(client):
    r_static = client.post(
        "/v1/audio/speech", json={"input": "hello", "voice": "alloy", "response_format": "wav"}
    )
    r_stream_false = client.post(
        "/v1/audio/speech",
        json={"input": "hello", "voice": "alloy", "response_format": "wav", "stream": False},
    )
    assert r_static.content == r_stream_false.content


def test_compressed_format_ignores_stream(client, monkeypatch):
    """stream is ignored (buffered whole-file) for compressed formats: ffmpeg
    needs the complete PCM before it can encode, so there is no chunked path."""
    import pocket_tts_openai.routes_speech as rs

    monkeypatch.setattr(rs, "encode_pcm", lambda pcm, sr, fmt: b"ENCODED:" + b"mp3")
    r_static = client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "mp3"}
    )
    r_stream = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "response_format": "mp3", "stream": True},
    )
    assert r_static.status_code == 200
    assert r_stream.status_code == 200
    assert r_stream.headers["content-type"].startswith("audio/mpeg")
    assert r_stream.content == r_static.content  # stream ignored -> identical


def test_concurrent_streams_serialize_but_do_not_overlap(fake_model, config, engine):
    """Two concurrent streams must not run the model simultaneously (gen lock)."""
    results: dict[str, bytes] = {}
    errors: list[Exception] = []

    def run(kind: str, body: dict):
        app = create_app(config, engine)
        with TestClient(app) as c:
            try:
                r = c.post("/v1/audio/speech", json={**body, "stream": True})
                results[kind] = r.content
            except Exception as exc:  # pragma: no cover
                errors.append(exc)

    t1 = Thread(target=run, args=("a", {"input": "one", "response_format": "pcm"}))
    t2 = Thread(target=run, args=("b", {"input": "two", "voice": "echo", "response_format": "pcm"}))
    t1.start(); t2.start(); t1.join(); t2.join()
    assert not errors
    assert fake_model.overlap_seen is False, "model ran concurrently -> gen lock leaked"
    _assert_valid_pcm(results["a"])
    _assert_valid_pcm(results["b"])


def test_disconnect_releases_lock(fake_model, config, engine):
    """A client that abandons the stream mid-way must still release gen lock."""
    app = create_app(config, engine)
    with TestClient(app) as c:
        with c.stream(
            "POST",
            "/v1/audio/speech",
            json={"input": "hello", "response_format": "pcm", "stream": True},
        ) as r:
            assert r.status_code == 200
            # read only the first chunk, then abandon the generator (close).
            first = next(iter(r.iter_bytes()))
            assert first
        # After the client context closes, the gen lock must be free for a
        # subsequent request to complete promptly.
    with TestClient(app) as c:
        r = c.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
        assert r.status_code == 200
    _assert_valid_pcm(r.content)


def test_stream_stats_recorded(fake_model, config, engine):
    app = create_app(config, engine)
    with TestClient(app) as c:
        c.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm", "stream": True})
    assert engine.stats.requests == 1
    assert engine.stats.audio_seconds == 1.0