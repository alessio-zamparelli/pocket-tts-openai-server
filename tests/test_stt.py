"""STT endpoint + whisper-server sidecar tests (PLAN-STT.md / M6).

The sidecar is stubbed: ``WhisperSidecar`` accepts an injectable ``http``
(httpx.MockTransport) and ``spawn`` (FakePopen) plus a fake ``now`` clock, so
every behavior — routing, format mapping, idle eviction, wake, crash restart —
is tested without a real whisper.cpp binary.
"""

from __future__ import annotations

import httpx
import pytest
from fastapi.testclient import TestClient

from pocket_tts_openai.config import Config
from pocket_tts_openai.server import create_app
from pocket_tts_openai.stt import WhisperSidecar, model_filename

AUDIO = b"\x00" * 4096  # pretend a 4 KB wav


class FakePopen:
    """Duck-typed subprocess.Popen: alive until told otherwise."""

    def __init__(self, pid: int = 4242):
        self.pid = pid
        self._exit_code: int | None = None
        self.terminated = False

    def poll(self) -> int | None:
        return self._exit_code

    def terminate(self) -> None:
        self.terminated = True
        self._exit_code = 0

    def kill(self) -> None:
        self._exit_code = 9

    def wait(self, timeout: float | None = None) -> int | None:  # noqa: ARG002
        return self._exit_code


class FakeClock:
    def __init__(self, start: float = 1000.0):
        self.t = start

    def __call__(self) -> float:
        return self.t

    def advance(self, seconds: float) -> None:
        self.t += seconds


def _multipart_fields(raw: bytes) -> dict[str, list[str]]:
    """Crude multipart parser: returns text-form field name -> values."""
    fields: dict[str, list[str]] = {}
    for part in raw.split(b"--"):
        head, sep, value = part.partition(b"\r\n\r\n")
        if not sep or b'name="' not in head or b'filename="' in head:
            continue
        name = head.split(b'name="', 1)[1].split(b'"', 1)[0].decode()
        text = value.rstrip(b"\r\n").decode(errors="replace")
        fields.setdefault(name, []).append(text)
    return fields


def _inference_response(fields: dict[str, list[str]]) -> httpx.Response:
    fmt = fields.get("response_format", ["json"])[0]
    if fmt == "text":
        return httpx.Response(200, text="Hello world.", headers={"content-type": "text/html"})
    if fmt == "srt":
        return httpx.Response(
            200,
            text="1\n00:00:00,000 --> 00:00:01,000\nHello world.\n",
            headers={"content-type": "application/x-subrip"},
        )
    if fmt == "vtt":
        return httpx.Response(
            200,
            text="WEBVTT\n\n00:00:00.000 --> 00:00:01.000\nHello world.\n",
            headers={"content-type": "text/vtt"},
        )
    if fmt == "verbose_json":
        return httpx.Response(
            200,
            json={"text": "Hello world.", "segments": [{"id": 0, "start": 0.0, "text": "Hello world."}]},
        )
    return httpx.Response(200, json={"text": "Hello world."})


def _stub_sidecar(
    config: Config,
    captured: list,
    procs: list,
    clock: FakeClock,
    *,
    health: int = 200,
    spawn_error: Exception | None = None,
) -> WhisperSidecar:
    def spawn():
        if spawn_error is not None:
            raise spawn_error
        p = FakePopen(pid=1000 + len(procs))
        procs.append(p)
        return p

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(health, json={"status": "ok" if health == 200 else "down"})
        if request.url.path == "/inference":
            fields = _multipart_fields(request.content)
            captured.append(fields)
            return _inference_response(fields)
        return httpx.Response(404)

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=f"http://{config.stt_host}:{config.stt_port}",
    )
    return WhisperSidecar(
        config=config,
        model_path="ggml-small.bin",
        http=http,
        spawn=spawn,
        now=clock,
        startup_poll=0.0,
    )


@pytest.fixture
def stt_config() -> Config:
    return Config(stt_enabled=True, stt_model="small", stt_idle_unload_s=0)


@pytest.fixture
def stt_app(stt_config: Config, engine):
    """TestClient with a healthy stubbed whisper-server sidecar injected."""
    captured: list = []
    procs: list = []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(stt_config, captured, procs, clock)
    app = create_app(stt_config, engine, stt_sidecar=sidecar)
    with TestClient(app) as c:
        c.state_labels = {"captured": captured, "procs": procs, "clock": clock, "sidecar": sidecar}
        yield c


# -- request field mapping --------------------------------------------------


def test_transcribe_default_json(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.json() == {"text": "Hello world."}


def test_transcribe_response_format_text(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "text"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/plain")
    assert resp.text.strip() == "Hello world."


def test_transcribe_response_format_srt(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "srt"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("application/x-subrip")
    assert 'filename="transcription.srt"' in resp.headers["content-disposition"]
    assert "Hello world." in resp.text


def test_transcribe_response_format_vtt(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "vtt"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 200
    assert resp.headers["content-type"].startswith("text/vtt")
    assert resp.text.startswith("WEBVTT")


def test_transcribe_response_format_verbose_json(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "verbose_json"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "Hello world."
    assert body["segments"][0]["id"] == 0


def test_transcription_forwards_language_prompt_temperature_granularity(stt_app):
    stt_app.post(
        "/v1/audio/transcriptions",
        data={
            "model": "whisper-1",
            "language": "ja",
            "prompt": "subtitles",
            "temperature": "0.2",
            "timestamp_granularities[]": ["segment", "word"],
        },
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    captured = stt_app.state_labels["captured"]
    fields = captured[0]
    assert fields["language"] == ["ja"]
    assert fields["prompt"] == ["subtitles"]
    assert fields["temperature"] == ["0.2"]
    # word granularity -> token timestamps
    assert fields["token_timestamps"] == ["true"]
    # plain json must NOT send an explicit response_format (whisper default)
    assert "response_format" not in fields
    assert "translate" not in fields


def test_translation_forward_translate_no_language(stt_app):
    stt_app.post(
        "/v1/audio/translations",
        data={"model": "whisper-1", "language": "de", "response_format": "text"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    fields = stt_app.state_labels["captured"][0]
    assert fields["translate"] == ["true"]
    assert fields["response_format"] == ["text"]
    # language must be dropped for translations (always -> English)
    assert "language" not in fields


# -- validation errors --------------------------------------------------------


def test_transcribe_unknown_model_400(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "gpt-4o-transcribe"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "whisper-1" in resp.json()["error"]["message"]


def test_transcribe_missing_model_400(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "model" in resp.json()["error"]["message"].lower()


def test_transcribe_missing_file_400(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"},
    )
    assert resp.status_code == 400
    assert "file" in resp.json()["error"]["message"].lower()


def test_transcribe_empty_file_400(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1"},
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_transcribe_unsupported_response_format_400(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "response_format": "xml"},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "response_format" in resp.json()["error"]["message"]


def test_transcribe_unsupported_granularity_400(stt_app):
    resp = stt_app.post(
        "/v1/audio/transcriptions",
        data={"model": "whisper-1", "timestamp_granularities[]": ["frame"]},
        files={"file": ("speech.wav", AUDIO, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "granularit" in resp.json()["error"]["message"]


def test_transcribe_oversized_file_413(stt_config, engine):
    small = Config(stt_enabled=True, max_upload_mb=1)
    captured, procs = [], []
    sidecar = _stub_sidecar(small, captured, procs, FakeClock(1000.0))
    app = create_app(small, engine, stt_sidecar=sidecar)
    with TestClient(app) as c:
        resp = c.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files={"file": ("big.wav", b"\x00" * (3 * 1024 * 1024), "audio/wav")},
        )
    assert resp.status_code == 413


def test_transcribe_503_when_sidecar_cannot_start(engine):
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    captured, procs = [], []
    sidecar = _stub_sidecar(
        cfg, captured, procs, FakeClock(1000.0), spawn_error=RuntimeError("no binary")
    )
    app = create_app(cfg, engine, stt_sidecar=sidecar)
    with TestClient(app) as c:
        resp = c.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files={"file": ("speech.wav", AUDIO, "audio/wav")},
        )
    assert resp.status_code == 503


def test_transcribe_upstream_error_502(stt_app, engine):
    """An upstream /inference 5xx surfaces as 502 bad_gateway, not a client 400."""
    captured, procs = [], []
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    clock = FakeClock(1000.0)

    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/health":
            return httpx.Response(200, json={"status": "ok"})
        return httpx.Response(500, text="whisper backend blew up")

    http = httpx.Client(
        transport=httpx.MockTransport(handler),
        base_url=f"http://{cfg.stt_host}:{cfg.stt_port}",
    )
    sidecar = WhisperSidecar(
        config=cfg, model_path="ggml-small.bin", http=http,
        spawn=lambda: FakePopen(1000), now=clock, startup_poll=0.0,
    )
    app = create_app(cfg, engine, stt_sidecar=sidecar)
    with TestClient(app) as c:
        resp = c.post(
            "/v1/audio/transcriptions",
            data={"model": "whisper-1"},
            files={"file": ("speech.wav", AUDIO, "audio/wav")},
        )
    assert resp.status_code == 502
    error = resp.json()["error"]
    assert "backend" in error["message"]


# -- /v1/models + /health ------------------------------------------------------


def test_models_includes_whisper1_when_enabled(stt_app):
    ids = {m["id"] for m in stt_app.get("/v1/models").json()["data"]}
    assert "whisper-1" in ids
    assert "tts-1" in ids


def test_models_excludes_whisper1_when_disabled(config, engine):
    app = create_app(config, engine)  # config fixture: stt disabled
    with TestClient(app) as c:
        ids = {m["id"] for m in c.get("/v1/models").json()["data"]}
    assert "whisper-1" not in ids


def test_health_stt_block(stt_app):
    body = stt_app.get("/health").json()
    stt = body["stt"]
    assert stt["enabled"] is True
    assert stt["model"] == "small"
    assert stt["ready"] is False or stt["ready"] is True
    assert "idle_unload_s" in stt
    assert "idle_stops" in stt
    assert "restarts" in stt


# -- sidecar lifecycle (unit, direct) ------------------------------------------


def test_start_spawns_and_becomes_ready():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start() is True
    assert sidecar.ready is True
    assert len(procs) == 1
    assert procs[0].pid == 1000
    # start() is idempotent
    assert sidecar.start() is True
    assert len(procs) == 1


def test_start_fails_when_binary_missing():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    captured, procs = [], []
    sidecar = _stub_sidecar(
        cfg, captured, procs, FakeClock(1000.0), spawn_error=RuntimeError("no binary")
    )
    assert sidecar.start() is False
    assert sidecar.last_error_hint() != "unknown startup failure"
    assert sidecar.ready is False


def test_idle_no_eviction_before_window():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=100)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()  # simulate an STT request
    clock.advance(50)
    assert sidecar.stop_if_idle() is False
    assert sidecar.ready is True
    assert sidecar.stats.idle_stops == 0


def test_idle_eviction_after_window():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=100)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()
    clock.advance(101)
    assert sidecar.stop_if_idle() is True
    assert sidecar.ready is False
    assert sidecar.stats.idle_stops == 1
    assert any(p.terminated for p in procs)


def test_idle_eviction_disabled_when_zero():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()
    clock.advance(10_000)
    assert sidecar.stop_if_idle() is False
    assert sidecar.ready is True


def test_health_probe_does_not_reset_idle_timer():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=100)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()
    # many health polls...
    for _ in range(5):
        clock.advance(10)
        sidecar.health_info()
    clock.advance(51)  # total 101 s of STT silence
    assert sidecar.stop_if_idle() is True


def test_wake_after_idle_eviction_respawns_and_works():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=100, stt_port=8790)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()
    clock.advance(200)
    assert sidecar.stop_if_idle() is True
    assert sidecar.stats.idle_stops == 1

    # wake path: block until re-spawned + healthy
    assert sidecar.ensure_started() is True
    assert sidecar.ready is True
    assert sidecar.stats.idle_starts == 1
    assert len(procs) == 2  # initial + wake

    # and a proxied request works again
    resp = sidecar.transcribe_raw(AUDIO, "speech.wav", "audio/wav", {})
    assert resp.status_code == 200
    assert sidecar.stats.requests == 1


def test_single_flight_wake_only_one_respawn(stt_config):
    cfg = Config(stt_enabled=True, stt_idle_unload_s=100)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()
    clock.advance(200)
    assert sidecar.stop_if_idle() is True
    # concurrent wake calls share the single re-spawn (lock-serialised)
    import threading

    results = {}
    threads = [
        threading.Thread(target=lambda: results.setdefault(id(t), sidecar.ensure_started()))
        for _ in range(4)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=5)
    assert all(results.values())  # every waiter got ready
    assert sidecar.stats.idle_starts == 1
    assert len(procs) == 2  # initial + exactly one wake spawn


def test_crash_watch_restarts_once():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    procs[0]._exit_code = 7  # simulate an unexpected death -> then watch()
    sidecar.watch()
    assert sidecar.ready is True
    assert sidecar.stats.restarts == 1
    assert len(procs) == 2


def test_idle_stop_not_seen_as_crash():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=100)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.touch()
    clock.advance(200)
    assert sidecar.stop_if_idle() is True
    sidecar.watch()  # idle stop is intended, never a crash restart
    assert sidecar.stats.restarts == 0
    assert len(procs) == 1


def test_shutdown_terminates_proc():
    cfg = Config(stt_enabled=True, stt_idle_unload_s=0)
    captured, procs = [], []
    clock = FakeClock(1000.0)
    sidecar = _stub_sidecar(cfg, captured, procs, clock)
    assert sidecar.start()
    sidecar.shutdown()
    assert any(p.terminated for p in procs)
    assert sidecar.ready is False


# -- model filename mapping ------------------------------------------------------


def test_model_filename_mapping():
    assert model_filename("small") == "ggml-small.bin"
    assert model_filename("small.q5_0") == "ggml-small.q5_0.bin"
    assert model_filename("ggml-base") == "ggml-base.bin"
    assert model_filename("ggml-base.bin") == "ggml-base.bin"
