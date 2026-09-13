"""TTS engine: single model instance, generation lock, voice-state cache.

v1 concurrency model (per PLAN.md §2/§8): one ``TTSModel`` instance, one global
lock, requests serialize FIFO. Voice-state encoding (the slow part) happens
*outside* the generation lock and is cached per voice name with an LRU policy.

Custom (cloned) voices live in a :class:`VoiceRegistry`; their names resolve
to on-disk ``.safetensors`` files that ``get_state_for_audio_prompt`` reloads
cheaply. Streaming wraps ``generate_audio_stream`` and holds the generation
lock for the whole stream.
"""

from __future__ import annotations

import ctypes
import gc
import importlib
import io
import logging
import threading
import time
import wave
from collections import OrderedDict
from collections.abc import Iterator
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, cast
from uuid import uuid4

import numpy as np
from typing_extensions import Protocol, runtime_checkable

from .config import Config
from .voice_registry import VoiceRegistry
from .voices import resolve_voice

logger = logging.getLogger(__name__)

PCM_DTYPE = np.dtype("<i2")


def _rss_mb() -> int | None:
    """Resident set size in MB, best-effort (``None`` off-Linux/unavailable).

    Reads ``VmRSS`` from ``/proc/self/status`` — the same figure the idle-unload
    plan (PLAN-idle-unload.md) uses to reason about RAM reclamation, surfaced in
    load/unload lifecycle logs.
    """
    try:
        with open("/proc/self/status", encoding="utf-8") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError):  # pragma: no cover - non-Linux / parse quirks
        return None
    return None


class RateLimited(Exception):
    """Load-shedding signal: the generation queue is full (``max_waiting``) or a
    request waited past ``queue_timeout_s`` for the generation lock. Routes map
    this to an OpenAI-shaped HTTP 429."""


@runtime_checkable
class AudioBuffer(Protocol):
    """1-D audio buffer: a torch tensor (device-local, [-1..1] float) or ndarray."""

    def detach(self) -> "AudioBuffer": ...

    def cpu(self) -> "AudioBuffer | np.ndarray": ...


def to_pcm16(audio: "AudioBuffer | np.ndarray") -> bytes:
    """Convert a 1-D audio buffer (torch tensor or ndarray, [-1..1] float) to s16le bytes.

    Allocates a single f32 working buffer when the input is already float32:
    ``asarray`` is a view, and the clip + scale write in place (``out=arr``), so
    only the int16 ``astype`` and the ``tobytes`` copy allocate fresh memory (M4's
    streaming loop calls this once per chunk — every saved allocation multiplies
    across a long stream).

    Contract: the input buffer may be **mutated in place** (when it's a writable
    float32 view). Callers must not reuse ``audio`` after conversion — true at
    every call site (each chunk/tensor from :meth:`TTSEngine.generate_pcm` /
    ``generate_audio_stream`` is consumed once).
    """
    if not isinstance(audio, np.ndarray):
        audio = audio.detach().cpu()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    np.clip(arr, -1.0, 1.0, out=arr)
    np.multiply(arr, 32767.0, out=arr)
    return arr.astype(PCM_DTYPE).tobytes()


def pcm_to_wav(pcm: bytes, sample_rate: int) -> bytes:
    """Wrap raw mono s16le PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        w.writeframes(pcm)
    return buf.getvalue()


def streaming_wav_header(sample_rate: int) -> bytes:
    """44-byte RIFF/WAVE header with unknown sizes (``0xFFFFFFFF``).

    Used for chunked WAV streaming where the total audio length isn't known
    up front. The ``RIFF`` size and ``data`` subchunk size are set to the
    maximum so players that read non-seekable streams (ffplay, VLC, most
    browsers) still render audio incrementally.

    Layout matches the 44-byte canonical PCM WAV header produced by
    :mod:`wave` (mono, 16-bit).
    """
    import struct

    return struct.pack(
        "<4sI4s4sIHHIIHH4sI",
        b"RIFF",
        0xFFFFFFFF,
        b"WAVE",
        b"fmt ",
        16,                   # subchunk1 size
        1,                    # PCM
        1,                    # mono
        sample_rate,
        sample_rate * 2,      # byte rate (mono, 16-bit)
        2,                    # block align
        16,                   # bits per sample
        b"data",
        0xFFFFFFFF,
    )


@dataclass
class EngineStats:
    requests: int = 0
    audio_seconds: float = 0.0
    generate_seconds: float = 0.0
    waiting: int = 0
    max_waiting: int = 0
    unloads: int = 0  # idle-eviction events (model dropped)
    reloads: int = 0  # model rebuilds after eviction

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

    def generate_audio_stream(
        self, model_state: object, text_to_generate: str
    ) -> "Iterator[AudioBuffer | np.ndarray]": ...


class TTSEngine:
    """Wraps a pocket-tts model behind a generation lock.

    Any duck-typed model works; tests inject fakes and production loads
    pocket-tts lazily via :func:`load_engine`.
    """

    def __init__(
        self,
        model: TTSModelLike | None,
        *,
        config: Config,
        registry: VoiceRegistry | None = None,
        export_state: "Callable[[object, str | Path], None] | None" = None,
        loader: "Callable[[], TTSModelLike] | None" = None,
    ):
        """Wrap a pocket-tts model behind a generation lock.

        ``model`` may be ``None`` (pre-load); production passes a real model and a
        ``loader`` closure so the engine can rebuild it after an idle eviction.
        Tests inject fakes and leave ``loader`` unset — eviction is then disabled.
        """
        self._model = model
        self._loader = loader
        self._config = config
        # True when the model was built by this server's own background loader
        # (``load_engine``); injected engines (tests, preloaded) leave it False,
        # so server shutdown releases only engines it owns.
        self._auto_loaded = False
        self.registry = registry
        # Serialization hook for cloning. Defaults (lazily) to pocket-tts'
        # module-level export_model_state; tests inject a no-op/fake.
        self._export_state = export_state
        self.sample_rate: int = model.sample_rate if model is not None else 0
        self.language: str = config.language
        self.stats = EngineStats()
        self._gen_lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._voice_states: OrderedDict[str, object] = OrderedDict()
        # In-flight slow-path encodes keyed by resolved voice (single-flight).
        self._encode_pending: dict[str, threading.Event] = {}
        # Idle-eviction bookkeeping.
        self._last_activity = time.monotonic()
        self._reload_lock = threading.Lock()
        self._reload_in_flight = False

    # -- idle eviction / reload --------------------------------------------

    @property
    def loaded(self) -> bool:
        """True while a model is resident in RAM (eviction set it to None)."""
        return self._model is not None

    @property
    def eviction_enabled(self) -> bool:
        """Eviction is only possible when a loader (production) is attached
        and ``idle_unload_s > 0``. Fakes without a loader never evict."""
        return self._loader is not None and self._config.idle_unload_s > 0

    def touch(self) -> None:
        """Record an API request at ``now`` (resets the idle window)."""
        self._last_activity = time.monotonic()

    def last_request_age(self) -> float:
        """Seconds since the last API request touched the engine."""
        return time.monotonic() - self._last_activity

    def ensure_loaded(self) -> TTSModelLike:
        """Return the resident model, (re)building it single-flight if evicted.

        Concurrent callers block on ``_reload_lock`` and share the first rebuild
        (no thundering herd), then all use the same model object.
        """
        model = self._model
        if model is not None:
            return model
        if self._loader is None:  # pragma: no cover - tests keep models present
            raise RuntimeError("engine has no model loader; cannot reload after eviction")
        with self._reload_lock:
            model = self._model
            if model is not None:
                return model
            if self._reload_in_flight:
                raise RuntimeError("reload already in progress")  # unreachable (lock serializes)
            self._reload_in_flight = True
            try:
                t0 = time.perf_counter()
                logger.info("reloading pocket-tts model…")
                model = self._loader()
                self._model = model
                self.sample_rate = model.sample_rate
                self.stats.reloads += 1
                logger.info(
                    "model reloaded in %.2fs (RSS %s MB)", time.perf_counter() - t0, _rss_mb()
                )
            finally:
                self._reload_in_flight = False
        self.warmup()  # re-encode configured warmup voices after a rebuild
        return model

    def unload(self, reason: str = "idle") -> bool:
        """Drop the resident model + voice-state LRU; log and free heap.

        The single unload primitive backing both idle eviction and server
        shutdown, so every model release produces one uniform log line carrying
        the reason, the RSS before/after and the RAM reclaimed. Idempotent: a
        second call while already unloaded is a no-op. The caller is responsible
        for holding ``_gen_lock`` when it must not race an in-flight generation
        (see :meth:`maybe_unload`).
        """
        if not self.loaded:
            return False
        before = _rss_mb()
        self._model = None
        self.stats.unloads += 1
        self._voice_states.clear()
        gc.collect()
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except Exception:  # pragma: no cover - non-glibc
            logger.debug("malloc_trim unavailable; skipping heap release")
        after = _rss_mb()
        released = f" (released {before - after} MB)" if before and after else ""
        logger.info(
            "model unloaded: reason=%r RSS %s->%s MB%s", reason, before, after, released
        )
        return True

    def maybe_unload(self, now: float | None = None) -> bool:
        """Evict the resident model if idle past ``idle_unload_s``.

        Returns True if the model was dropped. Non-blocking on ``_gen_lock`` so
        an in-flight generation/stream is never evicted under it. Clears the
        voice-state LRU and best-effort returns freed heap to the OS.
        """
        if not self.eviction_enabled or self._model is None:
            return False
        if now is None:
            now = time.monotonic()
        if now - self._last_activity < self._config.idle_unload_s:
            return False
        # Don't race a running generation: if the lock is busy, skip this pass
        # (the watchdog retries next tick).
        if not self._gen_lock.acquire(blocking=False):
            return False
        try:
            # Re-check under the lock; a request may have landed since we looked.
            if now - self._last_activity >= self._config.idle_unload_s and self._model is not None:
                return self.unload(reason=f"idle {self._config.idle_unload_s}s")
            return False
        finally:
            self._gen_lock.release()

    # -- voice states -------------------------------------------------------

    def _resolved_name(self, requested_voice: str) -> str:
        """Resolve an alias/custom name/catalog name to the encode target.

        Custom registry names map to their on-disk ``.safetensors`` path so
        ``get_state_for_audio_prompt`` reloads them cheaply; the LRU cache key
        stays the pleasant custom name (so GET /v1/voices can report it as
        ``cached``).
        """
        v = requested_voice.strip()
        registry = self.registry
        if registry is not None and v in registry.names():
            path = registry.path_for(v)
            if path.exists():
                return str(path)
        return resolve_voice(requested_voice, self._config.voice_map)

    def voice_state(self, requested_voice: str) -> tuple[object, str]:
        """Return (ModelState, resolved_voice_name), encoding/caching on miss.

        Captures the model via :meth:`ensure_loaded` so a concurrent eviction
        can't null the object under an in-flight encode.

        The slow path (download + prompt encoding) runs outside all locks but is
        single-flight per resolved voice: concurrent misses for the same voice
        wait on the first encoder instead of hammering the stateful batch=1
        model in parallel or duplicating a network download.
        """
        model = self.ensure_loaded()
        self.touch()
        resolved = self._resolved_name(requested_voice)
        with self._state_lock:
            if resolved in self._voice_states:
                self._voice_states.move_to_end(resolved)
                return self._voice_states[resolved], resolved

        # Single-flight: claim the in-flight encode for this key, or wait for
        # the current encoder to finish and reuse its cached result.
        while True:
            with self._state_lock:
                pending = self._encode_pending.get(resolved)
                if pending is None:
                    event = threading.Event()
                    self._encode_pending[resolved] = event
                    break
            event = pending
            event.wait()  # another thread is encoding this voice right now
            with self._state_lock:
                if resolved in self._voice_states:
                    self._voice_states.move_to_end(resolved)
                    return self._voice_states[resolved], resolved
            # The encoder's result was already pruned (only reachable when
            # max_cached_voices forced it out); loop to become the next
            # encoder. Practically unreachable for real cache sizes.
        try:
            t0 = time.perf_counter()
            state = model.get_state_for_audio_prompt(resolved)
            logger.info("voice %r encoded in %.2fs", resolved, time.perf_counter() - t0)
            with self._state_lock:
                self._voice_states[resolved] = state
                self._voice_states.move_to_end(resolved)
                while len(self._voice_states) > self._config.max_cached_voices:
                    self._voice_states.popitem(last=False)
            return state, resolved
        finally:
            # Wake waiters, then drop the memo so a later miss re-encodes. Set
            # before pop: an arrival racing the pop either finds the fresh entry
            # above or (rare) starts its own encode.
            with self._state_lock:
                event.set()
                self._encode_pending.pop(resolved, None)

    def cached_voices(self) -> frozenset[str]:
        """Names/keys currently resident in the LRU voice-state cache."""
        with self._state_lock:
            return frozenset(self._voice_states)

    # -- custom voice cloning ------------------------------------------------

    def _exporter(self) -> "Callable[[object, str | Path], None]":
        if self._export_state is not None:
            return self._export_state
        try:
            from pocket_tts import export_model_state
        except ImportError:  # pragma: no cover
            raise ImportError(
                "pocket-tts is not installed; cannot clone voices. "
                "Run: uv sync --extra engine"
            ) from None
        # importlib-cached function; the model state is always a weights dict at
        # runtime, but the hook type is intentionally the widest (``object``).
        return cast("Callable[[object, str | Path], None]", export_model_state)

    def clone_voice(self, name: str, audio_path: str | Path) -> str:
        """Encode an audio prompt, persist it as ``<registry>/<name>.safetensors``,
        cache it live, and return the resolved cache key.

        Only the encode is serialized with generation (``get_state_for_audio_prompt``
        touches the stateful batch=1 model). The ``.safetensors`` export + atomic
        rename are pure I/O on the returned ``ModelState`` *snapshot* and run
        outside the generation lock, so a slow disk write during a clone never
        stalls in-flight synthesis requests.
        """
        registry = self.registry
        if registry is None:
            raise RuntimeError("voice registry is not configured")
        dest = registry.path_for(name)
        self.touch()
        model = self.ensure_loaded()  # capture; safe across a concurrent eviction
        t0 = time.perf_counter()
        # Serialize with generation: encoding touches the stateful batch=1 model.
        with self._gen_lock:
            state = model.get_state_for_audio_prompt(str(audio_path))
        # Export to a temp name then atomically move into place so a crash
        # mid-export can't leave a truncated/corrupt .safetensors. Stale temps
        # are swept on the next registry load. `state` is a strong local ref, so
        # a concurrent idle unload() during this I/O is harmless (the export
        # serializes the in-memory snapshot, not the live model).
        tmp = registry.directory / f".{name}.{uuid4().hex}.safetensors.tmp"
        try:
            self._exporter()(state, tmp)
            tmp.replace(dest)
        finally:
            tmp.unlink(missing_ok=True)
        logger.info(
            "cloned voice %r -> %s in %.2fs", name, dest, time.perf_counter() - t0
        )
        with self._state_lock:
            self._voice_states[name] = state
            self._voice_states.move_to_end(name)
        return name

    def evict_voice(self, name: str) -> None:
        """Drop a cached voice state (used by DELETE /v1/voices)."""
        with self._state_lock:
            self._voice_states.pop(name, None)

    # -- warmup --------------------------------------------------------------

    def warmup(self) -> None:
        """Pre-encode every configured warmup voice (startup). Never fatal."""
        for voice in self._config.warmup_voices:
            try:
                self.voice_state(voice)
            except Exception:
                logger.exception("warmup voice %r failed; skipping", voice)

    # -- generation ---------------------------------------------------------

    def check_capacity(self) -> None:
        """Pre-flight admission check for streaming: raise :class:`RateLimited`
        when the queue is already at ``max_waiting``. Called before response
        headers are sent so overload surfaces as a clean HTTP 429 instead of a
        chunked stream that dies midway.
        """
        self._reject_if_full()

    def _reject_if_full(self) -> None:
        """Reject a new request when more than ``max_waiting`` requests are
        already inside the critical path (encode + wait + generate)."""
        limit = self._config.max_waiting
        if limit > 0 and self.stats.waiting > limit:
            raise RateLimited(
                f"Too many concurrent requests (queue depth {self.stats.waiting}, "
                f"limit {limit}); retry later."
            )

    def _acquire_gen_lock(self) -> None:
        """Acquire the generation lock, honoring the optional queue timeout."""
        timeout = self._config.queue_timeout_s
        if timeout > 0:
            if not self._gen_lock.acquire(timeout=timeout):
                raise RateLimited(
                    f"Timed out after {timeout:g}s waiting for the generation queue."
                )
        else:
            self._gen_lock.acquire()

    def generate_pcm(self, text: str, voice: str) -> bytes:
        """Serialize on the global lock; return raw mono s16le PCM at 24 kHz.

        Captures the (possibly just-reloaded) model up front so a wake-up runs
        on the same instance that encoded the voice state. The queue guard
        covers encode + wait + generate so load shedding (``max_waiting``)
        counts the whole in-flight window, not just time on the lock.
        """
        self.touch()
        guard = _QueueGuard(self)
        with guard:
            self._reject_if_full()
            model = self.ensure_loaded()
            state, resolved = self.voice_state(voice)
            self._acquire_gen_lock()
            try:
                t0 = time.perf_counter()
                audio = model.generate_audio(state, text)
                elapsed = time.perf_counter() - t0
            finally:
                self._gen_lock.release()
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

    def generate_pcm_stream(self, text: str, voice: str) -> "Iterator[bytes]":
        """Yield mono s16le PCM chunks while holding the global generation lock.

        The lock is held for the lifetime of the generator (StreamingResponse
        iterates it in a worker thread). If the consumer abandons the generator
        (client disconnect), the pocket-tts iterator stops being pulled and the
        lock is released at the generator's close. The queue guard likewise
        spans the whole stream, so capacity is accounted start to finish.
        """
        guard = _QueueGuard(self)
        with guard:
            self._reject_if_full()
            model = self.ensure_loaded()
            self.touch()
            state, resolved = self.voice_state(voice)
            self._acquire_gen_lock()
            try:
                t0 = time.perf_counter()
                audio_seconds = 0.0
                try:
                    for chunk in model.generate_audio_stream(state, text):
                        pcm = to_pcm16(chunk)
                        audio_seconds += len(pcm) / (2 * self.sample_rate)
                        yield pcm
                finally:
                    elapsed = time.perf_counter() - t0
            finally:
                self._gen_lock.release()
        self.stats.requests += 1
        self.stats.audio_seconds += audio_seconds
        self.stats.generate_seconds += elapsed
        logger.info(
            "streamed %.2fs audio for voice=%r in %.2fs (rtf %.2f)",
            audio_seconds,
            resolved,
            elapsed,
            elapsed / max(audio_seconds, 1e-9),
        )


def load_engine(config: Config) -> TTSEngine:
    """Production engine factory. Imports pocket-tts lazily so tests/dev can run
    without it, then loads the model (first call downloads weights to disk cache)
    and attaches a ``loader`` so the engine can rebuild itself after an idle
    eviction (PLAN-idle-unload.md).
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

    def build() -> "TTSModelLike":
        kwargs: dict[str, object] = {"language": config.language}
        if config.quantize:
            kwargs["quantize"] = True
        logger.info("loading pocket-tts model (language=%s)…", config.language)
        return cast("TTSModelLike", TTSModel.load_model(**kwargs))

    t0 = time.perf_counter()
    model = build()
    logger.info(
        "model loaded: language=%s quantize=%s sample_rate=%d in %.2fs (RSS %s MB)",
        config.language,
        config.quantize,
        model.sample_rate,
        time.perf_counter() - t0,
        _rss_mb(),
    )
    registry = VoiceRegistry.from_config_dir(Path(config.cache_dir) if config.cache_dir else None)
    registry.load()
    model = TTSEngine(model, config=config, registry=registry, loader=build)
    model._auto_loaded = True  # owned by this server: shutdown will unload it
    return model