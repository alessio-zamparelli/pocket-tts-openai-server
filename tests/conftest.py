"""Shared fixtures: a fake pocket-tts model + a real TTSEngine around it."""

from __future__ import annotations

import threading
import time

import numpy as np
import pytest

from pocket_tts_openai.config import Config
from pocket_tts_openai.engine import TTSEngine
from pocket_tts_openai.server import create_app


class FakeTTSModel:
    """Duck-typed pocket-tts model: 24 kHz, 1 s of silence per request.

    Simulates real cost: ``get_state_for_audio_prompt`` is slow (voice
    encoding) and ``generate_audio`` holds ``gen_lock`` for a configurable
    time so the engine's serialization behavior is observable.
    """

    sample_rate = 24000

    def __init__(self, *, encode_seconds: float = 0.02, gen_seconds: float = 0.05):
        self._encode_seconds = encode_seconds
        self._gen_seconds = gen_seconds
        self.encode_calls: list[str] = []
        self.generate_calls: list[str] = []
        self._inside_generate = 0
        self.overlap_seen = False
        self._overlap_lock = threading.Lock()

    def get_state_for_audio_prompt(self, voice: str) -> tuple[str, str]:
        time.sleep(self._encode_seconds)
        self.encode_calls.append(voice)
        return ("state", voice)

    def generate_audio(self, model_state: object, text_to_generate: str) -> np.ndarray:
        with self._overlap_lock:
            self._inside_generate += 1
            if self._inside_generate > 1:
                self.overlap_seen = True
        try:
            time.sleep(self._gen_seconds)
        finally:
            with self._overlap_lock:
                self._inside_generate -= 1
        self.generate_calls.append(text_to_generate)
        return np.zeros(self.sample_rate, dtype=np.float32)


@pytest.fixture
def fake_model() -> FakeTTSModel:
    return FakeTTSModel()


@pytest.fixture
def config() -> Config:
    return Config()


@pytest.fixture
def engine(fake_model: FakeTTSModel, config: Config) -> TTSEngine:
    return TTSEngine(fake_model, config=config)


@pytest.fixture
def client(engine: TTSEngine, config: Config):
    """TestClient with lifespan active so app.state.engine is set on startup."""
    from fastapi.testclient import TestClient

    app = create_app(config, engine)
    with TestClient(app) as c:
        yield c
