"""Logging for the model load/unload lifecycle (plan: load/unload logs).

Asserts the uniform ``model unloaded`` / ``model (re)loaded`` log lines carry
the reason, timing and RSS, that every unload path (idle eviction, ``unload``
primitive, server shutdown) produces them, and that the release is idempotent.
"""

from __future__ import annotations

import logging

from fastapi.testclient import TestClient

from pocket_tts_openai.config import Config
from pocket_tts_openai.engine import TTSEngine, _rss_mb
from pocket_tts_openai.server import create_app
from tests.conftest import FakeTTSModel

ENGINE_LOGGER = "pocket_tts_openai.engine"


def _messages(caplog, needle: str) -> list[str]:
    return [r.getMessage() for r in caplog.records if needle in r.getMessage()]


def _engine(**kwargs) -> TTSEngine:
    cfg = Config(idle_unload_s=60)
    return TTSEngine(FakeTTSModel(), config=cfg, **kwargs)


def test_rss_helper_sanity():
    mb = _rss_mb()
    assert mb is None or (isinstance(mb, int) and mb > 0)


def test_unload_logs_reason_and_releases(caplog):
    engine = _engine()
    engine.voice_state("alloy")  # populate the LRU
    with caplog.at_level(logging.INFO, logger=ENGINE_LOGGER):
        assert engine.unload(reason="test") is True

    assert engine.loaded is False
    assert engine._model is None
    assert engine.stats.unloads == 1
    assert engine.cached_voices() == frozenset()

    msgs = _messages(caplog, "model unloaded")
    assert len(msgs) == 1
    assert "reason='test'" in msgs[0]
    assert "RSS" in msgs[0] and "MB" in msgs[0]


def test_unload_is_idempotent(caplog):
    engine = _engine()
    with caplog.at_level(logging.INFO, logger=ENGINE_LOGGER):
        assert engine.unload(reason="first") is True
        assert engine.unload(reason="second") is False
    assert engine.stats.unloads == 1
    assert len(_messages(caplog, "model unloaded")) == 1


def test_maybe_unload_delegates_with_idle_reason(caplog):
    engine = _engine(loader=FakeTTSModel)
    engine._last_activity -= 10_000
    with caplog.at_level(logging.INFO, logger=ENGINE_LOGGER):
        assert engine.maybe_unload(now=100_000) is True

    assert engine.stats.unloads == 1
    msgs = _messages(caplog, "model unloaded")
    assert len(msgs) == 1
    assert "idle" in msgs[0]


def test_reload_logs_timing_and_rss(caplog):
    engine = _engine(loader=FakeTTSModel)
    engine._model = None  # simulate a prior eviction
    with caplog.at_level(logging.INFO, logger=ENGINE_LOGGER):
        engine.ensure_loaded()

    assert engine.loaded is True
    assert engine.stats.reloads == 1
    reloaded = _messages(caplog, "model reloaded in")
    assert len(reloaded) == 1
    assert "RSS" in reloaded[0]


def test_shutdown_unloads_owned_model(caplog):
    """Lifespan teardown releases an auto-loaded model with reason='server shutdown'."""
    engine = _engine()
    engine._auto_loaded = True  # simulate a production (background-loaded) engine
    app = create_app(Config(idle_unload_s=0), engine)
    with caplog.at_level(logging.INFO, logger="pocket_tts_openai"):
        with TestClient(app) as client:
            assert engine.loaded is True
            assert client.get("/health").status_code == 200
        # lifespan shutdown finally-block has run
        assert engine._model is None
        assert engine.loaded is False

    msgs = _messages(caplog, "model unloaded")
    assert len(msgs) == 1
    assert "server shutdown" in msgs[0]


def test_shutdown_leaves_injected_engine_resident(caplog):
    """Tests/preloaded engines the server did not build are NOT unloaded by the
    lifespan teardown (they may be reused in another lifespan, as the streaming
    disconnect test does), but the transition is still logged."""
    engine = _engine()
    app = create_app(Config(idle_unload_s=0), engine)
    with caplog.at_level(logging.INFO, logger="pocket_tts_openai"):
        with TestClient(app):
            assert engine.loaded is True
        assert engine._model is not None  # left resident
        assert engine.loaded is True

    assert "model unloaded" not in caplog.text
    assert "engine left resident" in caplog.text
    assert engine.stats.unloads == 0


def test_shutdown_idempotent_when_already_unloaded(caplog):
    engine = _engine()
    engine._auto_loaded = True
    engine.unload(reason="premature")  # logged before the capture window
    app = create_app(Config(idle_unload_s=0), engine)
    with caplog.at_level(logging.INFO, logger="pocket_tts_openai"):
        with TestClient(app):
            assert engine.loaded is False
        # shutdown calls unload() but it no-ops on an absent model: the log shows
        # only the shutdown transition, never a second 'model unloaded' line.
        assert "model unloaded" not in caplog.text
