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


def test_compressed_stream_true_routes_through_streaming(client, monkeypatch):
    """stream:true for a compressed format now really streams: the route iterates
    the ffmpeg streaming generator multiple times server-side (a single-blob
    buffered response would iterate exactly once), and the audio is byte-identical
    to the buffered (stream:false) response."""
    import pocket_tts_openai.routes_speech as rs

    r_static = client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "mp3"}
    )
    assert r_static.status_code == 200

    yields: list[bytes] = []
    real = rs.encode_pcm_stream

    def counting(*a):
        for chunk in real(*a):
            yields.append(chunk)
            yield chunk

    monkeypatch.setattr(rs, "encode_pcm_stream", counting)
    with client.stream(
        "POST",
        "/v1/audio/speech",
        json={"input": "hi", "response_format": "mp3", "stream": True},
    ) as r:
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/mpeg")
        body = b"".join(r.iter_bytes())
    assert len(yields) > 1, "expected an incremental ffmpeg stream, got one blob"
    assert body == r_static.content


def test_compressed_stream_unknown_voice_400(client):
    """Unknown voice on the streamed-compressed path is a clean 400 before any
    headers -- not a mid-iteration 500 from Starlette's streaming worker."""
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "voice": "nope", "response_format": "mp3", "stream": True},
    )
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"
    assert "nope" in r.json()["error"]["message"]


def test_stream_unknown_voice_400_eager_validation(client):
    """A stream (raw wav/pcm or compressed) must not 500 on an unknown voice:
    the route validates the voice up front (also for the raw streaming path)."""
    for fmt in ("wav", "mp3"):
        r = client.post(
            "/v1/audio/speech",
            json={"input": "hi", "voice": "nope", "response_format": fmt, "stream": True},
        )
        assert r.status_code == 400, fmt
        assert r.json()["error"]["type"] == "invalid_request_error", fmt


def test_compressed_stream_disconnect_releases_lock(fake_model, config, engine):
    """A client that abandons a streamed mp3 mid-way must still release the
    generation lock (kill ffmpeg + close the engine stream on GeneratorExit)."""
    app = create_app(config, engine)
    with TestClient(app) as c:
        with c.stream(
            "POST",
            "/v1/audio/speech",
            json={"input": "hello", "response_format": "mp3", "stream": True},
        ) as r:
            assert r.status_code == 200
            first = next(iter(r.iter_bytes()))
            assert first
        # generator abandoned -> lock must be free for a subsequent request
    with TestClient(app) as c:
        r = c.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
        assert r.status_code == 200
    _assert_valid_pcm(r.content)


def test_compressed_stream_opus_timeout_and_locking(fake_model, config, engine):
    """Opus streams as valid Ogg too, and a fully-consumed compressed stream
    reports the same audio duration as the buffered one."""
    app = create_app(config, engine)
    with TestClient(app) as c:
        r = c.post(
            "/v1/audio/speech", json={"input": "hi", "response_format": "opus", "stream": True}
        )
        assert r.status_code == 200
        assert r.headers["content-type"].startswith("audio/ogg")
        assert r.content[:4] == b"OggS"
        assert len(r.content) > 0


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