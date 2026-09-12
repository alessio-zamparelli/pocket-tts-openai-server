"""TTS engine: single model instance, generation lock, voice-state cache.

v1 concurrency model (per PLAN.md §2/§8): one ``TTSModel`` instance, one global
lock, requests serialize FIFO. Voice-state encoding (the slow part) happens
*outside* the generation lock and is cached per voice name with an LRU policy.
"""

from __future__ import annotations

import importlib
import io
import logging
import threading
import time
import wave
from collections import OrderedDict
from dataclasses import dataclass, field

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from .config import Config
from .voices import resolve_voice

logger = logging.getLogger(__name__)

PCM_DTYPE = np.dtype("<i2")


@runtime_checkable
class AudioBuffer(Protocol):
    """1-D audio buffer: a torch tensor (device-local, [-1..1] float) or ndarray."""

    def detach(self) -> "AudioBuffer": ...

    def cpu(self) -> "AudioBuffer | np.ndarray": ...


def to_pcm16(audio: "AudioBuffer | np.ndarray") -> bytes:
    """Convert a 1-D audio buffer (torch tensor or ndarray, [-1..1] float) to s16le bytes."""
    if not isinstance(audio, np.ndarray):
        audio = audio.detach().cpu()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(PCM_DTYPE).tobytes()


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono s16le PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


@dataclass
class EngineStats:
    requests: int = 0
    audio_seconds: float = 0.0
    generate_seconds: float = 0.0
    waiting: int = 0
    max_waiting: int = 0

    @property
    def avg_rtf(self) -> float | None:
        """Real-time factor: generation wall time / produced audio time."""
        return self.generate_seconds / self.audio_seconds if self.audio_seconds > 0 else None


@dataclass
class _QueueGuard:
    """Tracks how many requests are waiting for the generation lock."""

    engine: "TTSEngine"
    inner_lock: threading.Lock = field(default_factory=threading.Lock)

    def __enter__(self) -> "_QueueGuard":
        with self.inner_lock:
            self.engine.stats.waiting += 1
            self.engine.stats.max_waiting = max(
                self.engine.stats.max_waiting, self.engine.stats.waiting
            )
        return self

    def __exit__(self, *exc) -> None:
        with self.inner_lock:
            self.engine.stats.waiting -= 1


@runtime_checkable
class TTSModelLike(Protocol):
    """Duck-typed subset of ``pocket_tts.TTSModel`` used by the engine."""

    sample_rate: int

    def get_state_for_audio_prompt(self, voice: str) -> object: ...

    def generate_audio(self, model_state: object, text_to_generate: str) -> "AudioBuffer | np.ndarray": ...


class TTSEngine:
    """Wraps a pocket-tts model behind a generation lock.

    Any duck-typed model works; tests inject fakes and production loads
    pocket-tts lazily via :func:`load_engine`.
    """

    def __init__(self, model: TTSModelLike, *, config: Config):
        self._model = model
        self._config = config
        self.sample_rate: int = model.sample_rate
        self.language: str = config.language
        self.stats = EngineStats()
        self._gen_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._voice_states: OrderedDict[str, object] = OrderedDict()

    # -- voice states -------------------------------------------------------

    def voice_state(self, requested_voice: str) -> tuple[object, str]:
        """Return (ModelState, resolved_voice_name), encoding/caching on miss."""
        resolved = resolve_voice(requested_voice, self._config.voice_map)
        with self._state_lock:
            if resolved in self._voice_states:
                self._voice_states.move_to_end(resolved)
                return self._voice_states[resolved], resolved
        # Slow path (download + prompt encoding) happens outside all locks.
        t0 = time.perf_counter()
        state = self._model.get_state_for_audio_prompt(resolved)
        logger.info("voice %r encoded in %.2fs", resolved, time.perf_counter() - t0)
        with self._state_lock:
            self._voice_states[resolved] = state
            self._voice_states.move_to_end(resolved)
            while len(self._voice_states) > self._config.max_cached_voices:
                self._voice_states.popitem(last=False)
        return state, resolved

    # -- generation ---------------------------------------------------------

    def generate_pcm(self, text: str, voice: str) -> bytes:
        """Serialize on the global lock; return raw mono s16le PCM at 24 kHz."""
        guard = _QueueGuard(self)
        state, resolved = self.voice_state(voice)
        with guard, self._gen_lock:
            t0 = time.perf_counter()
            audio = self._model.generate_audio(state, text)
            elapsed = time.perf_counter() - t0
        pcm = to_pcm16(audio)
        seconds = len(pcm) / (2 * self.sample_rate)
        self.stats.requests += 1
        self.stats.audio_seconds += seconds
        self.stats.generate_seconds += elapsed
        logger.info(
            "generated %.2fs audio for voice=%r in %.2fs (rtf %.2f)",
            seconds,
            resolved,
            elapsed,
            elapsed / max(seconds, 1e-9),
        )
        return pcm


def load_engine(config: Config) -> TTSEngine:
    """Production engine factory. Imports pocket-tts lazily so tests/dev can run
    without it, then loads the model (first call downloads weights to disk cache).
    """
    try:
        # Optional dependency: only installed with `uv sync --extra engine`.
        # Dynamic import keeps optional deps out of static analysis paths.
        TTSModel = importlib.import_module("pocket_tts").TTSModel
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "The pocket-tts engine is not installed. "
            "Install it with: uv sync --extra engine "
            "(on Linux add --index https://download.pytorch.org/whl/cpu for CPU-only torch)"
        ) from exc

    kwargs: dict[str, object] = {"language": config.language}
    if config.quantize:
        kwargs["quantize"] = True
    logger.info("loading pocket-tts model (language=%s)…", config.language)
    model = TTSModel.load_model(**kwargs)
    return TTSEngine(model, config=config)
