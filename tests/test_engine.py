"""Engine tests: lock serialization, voice-state LRU cache, PCM conversion."""

from __future__ import annotations

import concurrent.futures
import threading
from pathlib import Path

import numpy as np
import pytest

from pocket_tts_openai.config import Config
from pocket_tts_openai.engine import RateLimited, TTSEngine, pcm_to_wav, to_pcm16
from pocket_tts_openai.voices import DEFAULT_VOICE_ALIASES, resolve_voice


def test_generation_is_serialized(engine, fake_model):
    import concurrent.futures
    import time

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        list(pool.map(lambda i: engine.generate_pcm(f"t{i}", "alloy"), range(4)))
    elapsed = time.perf_counter() - t0
    # 4 requests x 50 ms generation, fully serialized (plus tiny voice encode)
    assert elapsed >= 4 * fake_model._gen_seconds
    assert not fake_model.overlap_seen
    assert engine.stats.requests == 4


def test_voice_state_cached_across_requests(engine, fake_model):
    engine.generate_pcm("one", "alloy")
    engine.generate_pcm("two", "alloy")
    # state encoded once for the resolved voice, generation ran twice
    assert fake_model.encode_calls == ["alba"]
    assert fake_model.generate_calls == ["one", "two"]


def test_voice_lru_eviction(fake_model):
    from pocket_tts_openai.config import Config

    engine = TTSEngine(fake_model, config=Config(max_cached_voices=2))
    engine.generate_pcm("a", "alloy")  # alba
    engine.generate_pcm("b", "echo")  # charles
    engine.generate_pcm("c", "fable")  # eponine -> evicts alba
    assert engine._voice_states.keys() == {"charles", "eponine"}


def test_voice_alias_resolution():
    vm = dict(DEFAULT_VOICE_ALIASES)
    assert resolve_voice("alloy", vm) == "alba"
    assert resolve_voice("coral", vm) == "giovanni"


def test_voice_passthrough_catalog_and_urls():
    vm = dict(DEFAULT_VOICE_ALIASES)
    # Kyutai catalog names pass through untouched
    for name in ("alba", "charles", "bill_boerst"):
        assert resolve_voice(name, vm) == name
    # hf:// URIs, http(s) URLs and local paths pass through
    for raw in (
        "hf://kyutai/tts-voice-alba",
        "https://example.com/voice.wav",
        "/path/to/voice.wav",
        "./voices/custom.wav",
    ):
        assert resolve_voice(raw, vm) == raw


def test_unknown_voice_raises():
    with pytest.raises(ValueError, match="not-a-voice"):
        resolve_voice("not-a-voice", dict(DEFAULT_VOICE_ALIASES))


def test_to_pcm16_converts_float_to_s16le():
    arr = np.array([0.0, 1.0, -1.0, 0.5], dtype=np.float32)
    pcm = to_pcm16(arr)
    assert pcm == np.array([0, 32767, -32767, 16383], dtype="<i2").tobytes()


def test_to_pcm16_clips_and_accepts_tensor_like():
    class FakeTensor:
        def __init__(self, arr):
            self._arr = arr

        def detach(self):
            return self

        def cpu(self):
            return self._arr

    pcm = to_pcm16(FakeTensor(np.array([2.0, -2.0], dtype=np.float32)))
    assert pcm == np.array([32767, -32767], dtype="<i2").tobytes()


def test_pcm_to_wav_header():
    pcm = b"\x00\x00" * 24000  # 1 s silence at 24 kHz
    wav = pcm_to_wav(pcm, 24000)
    assert wav[:4] == b"RIFF"
    assert wav[8:12] == b"WAVE"
    assert len(wav) == 44 + len(pcm)
    # parse the fmt chunk rate
    import struct

    (sample_rate,) = struct.unpack("<I", wav[24:28])
    assert sample_rate == 24000


def test_stats_track_audio_seconds(engine):
    engine.generate_pcm("hi", "alloy")  # fake returns 1 s of audio
    assert engine.stats.audio_seconds == pytest.approx(1.0)
    assert engine.stats.generate_seconds > 0
    assert engine.stats.avg_rtf > 0


# --- P1.1 single-flight voice encode ----------------------------------------

def test_voice_state_encode_is_single_flight(engine, fake_model):
    """Concurrent misses for the same voice share ONE slow-path encode instead
    of hammering the stateful model in parallel."""
    fake_model._encode_seconds = 0.05  # make the slow path observable
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda i: engine.generate_pcm(f"t{i}", "alloy"), range(2)))
    assert fake_model.encode_calls == ["alba"]  # encoded exactly once
    assert engine.stats.requests == 2


def test_single_flight_reuses_result_across_voices(engine, fake_model):
    """Mirrors of the same encode complete and land in the LRU cache."""
    fake_model._encode_seconds = 0.03
    with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
        list(pool.map(lambda _: engine.generate_pcm("x", "echo"), range(2)))
    assert fake_model.encode_calls == ["charles"]


# --- P1.2 atomic clone write -------------------------------------------------

def test_clone_voice_writes_atomically_no_tmp_leftover(engine, registry):
    engine.clone_voice("mario", "s.wav")
    dest = registry.path_for("mario")
    assert dest.exists()
    assert dest.read_bytes() == b"fake-safetensors"
    assert list(registry.directory.glob("*.safetensors.tmp")) == []


def test_clone_voice_export_failure_leaves_no_dest(engine, registry, fake_model):
    """A crash mid-export must never leave a truncated .safetensors behind."""
    def boom(state: object, path: str):  # noqa: ARG001 - duck-typed exporter
        del state
        Path(path).write_bytes(b"partial")
        raise ValueError("disk full")

    fake_model.get_state_for_audio_prompt = lambda voice: ("state", voice)  # type: ignore[method-assign]
    engine._export_state = boom  # type: ignore[method-assign]
    with pytest.raises(ValueError):
        engine.clone_voice("mario", "s.wav")
    # never a truncated/missing dest, and the temp file is cleaned up
    assert not registry.path_for("mario").exists()
    assert list(registry.directory.glob("*.safetensors.tmp")) == []


# --- P1.3 load shedding + queue timeout --------------------------------------

def test_load_shedding_rejects_overflow(fake_model):
    """With max_waiting=1 the second concurrent request is rejected (429),
    while the first (which holds the generation lock) completes."""
    engine = TTSEngine(fake_model, config=Config(max_waiting=1))
    engine._gen_lock.acquire()  # simulate a long generation already holding the lock
    outcomes: list[str] = []

    def first():
        try:
            engine.generate_pcm("a", "alloy")
            outcomes.append("ok")
        except RateLimited:
            outcomes.append("429")

    t = threading.Thread(target=first)
    t.start()
    assert t.join(timeout=1.0) is None  # returned from join within timeout
    assert engine.stats.waiting == 1  # first request is queued on the lock

    try:
        # second request -> already over the limit -> rejected, never touches the model
        with pytest.raises(RateLimited, match="Too many concurrent requests"):
            engine.generate_pcm("b", "echo")
    finally:
        engine._gen_lock.release()
    t.join(timeout=2.0)

    assert fake_model.generate_calls == ["a"]  # only the first one ran
    assert engine.stats.waiting == 0
    assert outcomes == ["ok"]


def test_queue_timeout_raises_and_recovers(fake_model):
    """queue_timeout_s caps how long a request waits for the generation lock;
    once the lock frees, the same request succeeds."""
    engine = TTSEngine(fake_model, config=Config(queue_timeout_s=0.05))
    engine._gen_lock.acquire()  # a long generation holds the lock
    try:
        with pytest.raises(RateLimited, match="Timed out"):
            engine.generate_pcm("hi", "alloy")
    finally:
        engine._gen_lock.release()
    engine.generate_pcm("hi2", "alloy")
    assert fake_model.generate_calls == ["hi2"]


def test_shedding_and_timeout_default_off(engine):
    """Default config has both knobs off, so behavior is unchanged."""
    assert engine._config.max_waiting == 0
    assert engine._config.queue_timeout_s == 0.0
    engine.generate_pcm("hi", "alloy")  # still works normally
