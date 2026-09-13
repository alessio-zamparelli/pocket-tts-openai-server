"""Contract tests for the OpenAI-compatible API (mocked engine, no pocket-tts)."""

from __future__ import annotations

WAV_BYTES_FOR_1S = 44 + 24000 * 2  # 44-byte RIFF header + 1 s of 24 kHz mono s16le


def test_health_ok(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok"
    assert body["model"] == "pocket-tts"
    assert body["language"] == "english"
    assert body["requests"] == 0
    assert body["avg_rtf"] == 0
    assert body["queue_depth"] == 0


def test_models_shape(client):
    r = client.get("/v1/models")
    assert r.status_code == 200
    body = r.json()
    assert body["object"] == "list"
    ids = {m["id"] for m in body["data"]}
    assert ids == {"tts-1", "tts-1-hd", "gpt-4o-mini-tts"}
    for m in body["data"]:
        assert m["object"] == "model"
        assert m["owned_by"] == "pocket-tts"


def test_speech_wav(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "ciao mondo", "voice": "alloy", "response_format": "wav"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/wav")
    assert r.headers["content-disposition"] == "attachment; filename=speech.wav"
    body = r.content
    assert body[:4] == b"RIFF"
    assert len(body) == WAV_BYTES_FOR_1S


def test_speech_pcm_raw(client):
    r = client.post(
        "/v1/audio/speech",
        json={"model": "tts-1", "input": "hello", "voice": "echo", "response_format": "pcm"},
    )
    assert r.status_code == 200
    assert r.headers["content-type"].startswith("audio/pcm")
    assert r.headers["content-disposition"] == "attachment; filename=speech.pcm"
    assert len(r.content) == 24000 * 2  # 1 s raw s16le


def test_default_voice_is_alloy(client, fake_model):
    r = client.post("/v1/audio/speech", json={"input": "hi", "response_format": "pcm"})
    assert r.status_code == 200
    assert fake_model.generate_calls == ["hi"]
    # alloy -> alba per the voice map; state is encoded for the resolved voice
    assert fake_model.encode_calls == ["alba"]


def test_unknown_model_400(client):
    r = client.post("/v1/audio/speech", json={"input": "hi", "model": "whisper-1"})
    assert r.status_code == 400
    err = r.json()["error"]
    assert err["type"] == "invalid_request_error"


def test_empty_input_400(client):
    r = client.post("/v1/audio/speech", json={"input": "   "})
    assert r.status_code == 400
    assert r.json()["error"]["type"] == "invalid_request_error"


def test_invalid_format_422(client):
    """'ogg' is not in the response_format Literal -> FastAPI 422 validation."""
    r = client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "ogg"}
    )
    assert r.status_code == 422


def _stub_encode(monkeypatch) -> None:
    """Route tests never depend on ffmpeg: the codec is stubbed here (the real
    ffmpeg path is verified end-to-end against the live API)."""
    import pocket_tts_openai.routes_speech as rs

    def fake_encode(pcm: bytes, sample_rate: int, fmt: str) -> bytes:
        return b"ENCODED:" + fmt.encode()

    monkeypatch.setattr(rs, "encode_pcm", fake_encode)


def test_speech_compressed_formats_via_ffmpeg(client, monkeypatch):
    """mp3/opus/aac/flac routes return whole-file encoded audio with the right
    content-type and filename."""
    _stub_encode(monkeypatch)
    for fmt, media in (
        ("mp3", "audio/mpeg"),
        ("opus", "audio/ogg"),
        ("aac", "audio/aac"),
        ("flac", "audio/x-flac"),
    ):
        r = client.post(
            "/v1/audio/speech",
            json={"model": "tts-1", "input": "hi", "voice": "alloy", "response_format": fmt},
        )
        assert r.status_code == 200, fmt
        assert r.headers["content-type"].startswith(media), fmt
        assert r.headers["content-disposition"] == f"attachment; filename=speech.{fmt}", fmt
        assert r.content == b"ENCODED:" + fmt.encode(), fmt


def test_compressed_format_missing_ffmpeg_400(client, monkeypatch):
    """Without ffmpeg the compressed formats surface a clear 400, not a 500."""
    import pocket_tts_openai.audio_codecs as ac

    monkeypatch.setattr(ac, "ffmpeg_binary", lambda: None)
    r = client.post(
        "/v1/audio/speech", json={"input": "hi", "response_format": "mp3"}
    )
    assert r.status_code == 400
    assert "ffmpeg" in r.json()["error"]["message"]


def test_unknown_voice_400(client):
    r = client.post("/v1/audio/speech", json={"input": "hi", "voice": "not-a-voice"})
    assert r.status_code == 400
    assert "not-a-voice" in r.json()["error"]["message"]


def test_speed_and_instructions_accepted_but_ignored(client, fake_model):
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi", "voice": "alloy", "speed": 1.5, "instructions": "shout"},
    )
    assert r.status_code == 200
    # ignored params must not change the single call made
    assert fake_model.generate_calls == ["hi"]


def test_auth_required_when_configured():
    from fastapi.testclient import TestClient

    from pocket_tts_openai.config import Config
    from pocket_tts_openai.server import create_app

    config = Config(api_key="sk-secret")
    client = TestClient(create_app(config, engine=None))
    # /health is public
    assert client.get("/health").status_code == 200
    # /v1 without a key -> 401 with OpenAI error shape
    r = client.post("/v1/audio/speech", json={"input": "hi"})
    assert r.status_code == 401
    assert r.json()["error"]["code"] == "invalid_api_key"
    # wrong key -> 401
    r = client.post(
        "/v1/audio/speech",
        json={"input": "hi"},
        headers={"Authorization": "Bearer nope"},
    )
    assert r.status_code == 401


def test_503_while_engine_loading():
    """engine=None and pocket-tts absent: background load fails, speech stays 503."""
    from fastapi.testclient import TestClient

    from pocket_tts_openai.config import Config
    from pocket_tts_openai.server import create_app

    app = create_app(Config(), engine=None)
    client = TestClient(app)
    # give the (failing) background loader a moment; engine stays None either way
    r = client.post("/v1/audio/speech", json={"input": "hi"})
    assert r.status_code == 503
    assert r.json()["error"]["code"] == "unavailable"
