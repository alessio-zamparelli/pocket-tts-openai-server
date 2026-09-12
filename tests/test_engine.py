"""Engine tests: lock serialization, voice-state LRU cache, PCM conversion."""

from __future__ import annotations

import numpy as np
import pytest

from pocket_tts_openai.engine import TTSEngine, pcm_to_wav, to_pcm16
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
