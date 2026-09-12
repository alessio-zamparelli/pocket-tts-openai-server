"""Tests for the idle-eviction feature (PLAN-idle-unload.md / M5.5).

A real TTSEngine around the fake model, with a fake *loader* so the engine can
rebuild itself after an eviction. All timers are explicit (no sleeps): the
engine accepts ``maybe_unload(now=...)`` and tests reach into ``_last_activity``
to place the idle window deterministically.
"""

from __future__ import annotations

import threading

from fastapi.testclient import TestClient

from pocket_tts_openai.config import Config
from pocket_tts_openai.engine import TTSModelLike, TTSEngine
from pocket_tts_openai.server import create_app
from tests.conftest import FakeTTSModel, fake_export_state

DEFAULT_IDLE = Config().idle_unload_s


def _make_engine(
    config: Config | None = None, *, loader_calls: list[int] | None = None
) -> TTSEngine:
    """Production-shaped engine: initial model + a reload loader.

    ``loader_calls`` is mutated on every reload (records the new model index),
    so tests can assert single-flight / reload counts.
    """
    cfg = config or Config(idle_unload_s=DEFAULT_IDLE)
    calls: list[int] = loader_calls if loader_calls is not None else []
    models: list[TTSModelLike] = [FakeTTSModel()]

    def loader() -> TTSModelLike:
        calls.append(len(models))
        m: TTSModelLike = FakeTTSModel()
        models.append(m)
        return m

    return TTSEngine(models[0], config=cfg, loader=loader)


def _make_client(engine: TTSEngine, config: Config | None = None):
    app = create_app(config or Config(idle_unload_s=DEFAULT_IDLE), engine)
    return TestClient(app)


# -- config ------------------------------------------------------------------


def test_idle_config_defaults():
    cfg = Config()
    assert cfg.idle_unload_s == 300  # 5 min
    assert cfg.idle_poll_s == 30


def test_idle_config_zero_disables():
    assert Config(idle_unload_s=0).idle_unload_s == 0


def test_idle_config_from_env():
    cfg = Config.from_env({"STTS_IDLE_UNLOAD_S": "120", "STTS_IDLE_POLL_S": "10"})
    assert cfg.idle_unload_s == 120
    assert cfg.idle_poll_s == 10


def test_idle_config_from_env_disable():
    assert Config.from_env({"STTS_IDLE_UNLOAD_S": "0"}).idle_unload_s == 0


# -- engine lifecycle --------------------------------------------------------


def test_eviction_disabled_without_loader():
    """Fakes/tests that construct engines without a loader never evict."""
    model: TTSModelLike = FakeTTSModel()
    engine = TTSEngine(model, config=Config(idle_unload_s=DEFAULT_IDLE))
    assert not engine.eviction_enabled
    assert engine.loaded is True
    engine._last_activity -= 10_000  # stale timer
    assert engine.maybe_unload(now=100_000) is False
    assert engine._model is model


def test_eviction_disabled_when_timeout_zero():
    engine = _make_engine(Config(idle_unload_s=0))
    assert not engine.eviction_enabled
    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is False
    assert engine._model is not None


def test_maybe_unload_evicts_when_idle():
    calls: list[int] = []
    engine = _make_engine(loader_calls=calls)
    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is True
    assert engine._model is None
    assert engine.loaded is False
    assert calls == []  # unload alone never calls the loader
    assert engine.stats.unloads == 1
    assert engine.stats.reloads == 0


def test_no_unload_when_recently_active():
    engine = _make_engine()
    engine.touch()
    assert engine.maybe_unload(now=engine._last_activity + 1) is False
    assert engine._model is not None


def test_no_unload_when_generation_in_flight():
    """A running generation holds _gen_lock; eviction must skip that pass."""
    engine = _make_engine()
    engine._last_activity -= 10_000
    with engine._gen_lock:  # simulate an in-flight generation/stream
        assert engine.maybe_unload(now=100_000) is False
    assert engine._model is not None


def test_reload_restores_model_after_eviction():
    calls: list[int] = []
    engine = _make_engine(loader_calls=calls)
    old = engine._model
    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is True

    fresh = engine.ensure_loaded()
    assert fresh is not None
    assert fresh is not old
    assert fresh is engine._model
    assert engine.loaded is True
    assert calls == [1]  # one reload
    assert engine.stats.reloads == 1
    assert engine.sample_rate == 24000


def test_generate_wakes_engine_after_eviction():
    """The block-until-reloaded wake: a generate after eviction reloads, then
    synthesizes on the fresh model."""
    calls: list[int] = []
    engine = _make_engine(loader_calls=calls)
    pcm = engine.generate_pcm("wake call", "alloy")
    assert len(pcm) > 0
    assert engine.stats.reloads == 0

    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is True
    assert engine._model is None

    pcm2 = engine.generate_pcm("second call", "alloy")
    assert len(pcm2) > 0
    assert engine._model is not None
    assert calls == [1]
    assert engine.stats.reloads == 1


def test_generate_uses_resident_model_without_reloading():
    engine = _make_engine()
    resident = engine._model
    engine.generate_pcm("hello", "alloy")
    assert engine._model is resident
    assert engine.stats.reloads == 0


def test_single_flight_reload():
    """Many concurrent waiters after an eviction share ONE reload (loader called
    once), then all generate successfully."""
    import time

    calls: list[int] = []
    engine = _make_engine(loader_calls=calls)

    def slow_loader():
        calls.append(len(calls))
        time.sleep(0.05)
        return FakeTTSModel()

    engine._loader = slow_loader
    engine._model = None  # simulate prior eviction

    failures: list[Exception] = []
    lock = threading.Lock()

    def worker():
        try:
            engine.generate_pcm("x", "alloy")
        except Exception as exc:  # pragma: no cover - failure path
            with lock:
                failures.append(exc)

    threads = [threading.Thread(target=worker) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert failures == []
    assert len(calls) == 1  # single-flight: exactly one reload
    assert engine.stats.reloads == 1


def test_eviction_clears_voice_cache():
    engine = _make_engine()
    engine.voice_state("alloy")
    engine.voice_state("echo")
    resolved = {engine._resolved_name(v) for v in ("alloy", "echo")}
    assert resolved <= engine.cached_voices()

    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is True
    assert engine.cached_voices() == frozenset()


# -- health observability ----------------------------------------------------


def test_health_reports_idle_fields():
    engine = TTSEngine(
        FakeTTSModel(), config=Config(idle_unload_s=DEFAULT_IDLE), export_state=fake_export_state
    )
    with _make_client(engine) as client:
        body = client.get("/health").json()
    assert body["loaded"] is True
    assert body["idle_unload_s"] == DEFAULT_IDLE
    assert body["unloads"] == 0
    assert body["reloads"] == 0
    assert body["last_request_age_s"] >= 0


def test_health_reflects_eviction():
    engine = TTSEngine(
        FakeTTSModel(), config=Config(idle_unload_s=DEFAULT_IDLE), loader=lambda: FakeTTSModel()
    )
    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is True
    with _make_client(engine) as client:
        body = client.get("/health").json()
    assert body["status"] == "ok"  # still serving; next request reloads
    assert body["loaded"] is False
    assert body["unloads"] == 1


# -- route-level wake path ---------------------------------------------------


def test_speech_after_eviction_serves_200_with_reload():
    """End-to-end: POST /v1/audio/speech wakes an evicted engine (block until
    reloaded) and returns 200 — not 503."""
    calls: list[int] = []
    engine = _make_engine(loader_calls=calls)
    with _make_client(engine) as client:
        r1 = client.post("/v1/audio/speech", json={"input": "first", "voice": "alloy"})
        assert r1.status_code == 200

        engine._last_activity -= 10_000
        assert engine.maybe_unload(now=100_000) is True
        assert engine._model is None

        r2 = client.post("/v1/audio/speech", json={"input": "wake me up", "voice": "alloy"})
        assert r2.status_code == 200
        assert r2.headers["content-type"].startswith("audio/wav")
        assert engine._model is not None
        assert calls == [1]
        assert engine.stats.reloads == 1


def test_streaming_after_eviction_serves_200():
    engine = _make_engine()
    engine._last_activity -= 10_000
    assert engine.maybe_unload(now=100_000) is True
    with _make_client(engine) as client:
        r = client.post(
            "/v1/audio/speech",
            json={"input": "wake me up", "voice": "alloy", "response_format": "pcm"},
        )
    assert r.status_code == 200
    assert engine._model is not None
    assert engine.stats.reloads == 1
