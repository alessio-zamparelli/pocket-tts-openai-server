"""STT engine: a native ``whisper-server`` sidecar we spawn, health-check, and
proxy OpenAI-formatted transcriptions to (see PLAN-STT.md / M6).

whisper.cpp ships a native HTTP server binary (``examples/server``) that speaks
multipart and returns OpenAI-shaped JSON, ``srt``/``vtt``/``text``, plus a
``translate`` flag. We manage that process:

- **start()** spawns it and waits until it is ready (bounded). Readiness is
  probed with a version-agnostic signal: ``GET /health`` (200 ready, 503 still
  loading) exists on whisper.cpp master, while the v1.7.4 the Dockerfile pins
  only serves ``GET /`` — so a 404 from ``/health`` falls back to ``GET /``.
- **ensure_started()** is the wake path: single-flight re-spawn + readiness
  poll, so a request after an idle eviction blocks briefly then succeeds
  (no 503 in the wake path).
- **stop_if_idle()** implements idle RAM eviction the *process-level* way:
  whisper-server has no ``/unload`` endpoint, so killing the process reclaims
  ~100% of its RSS (the next request wakes it per §6 of the plan).
- **watch()** is a crash watcher: if the process dies unexpectedly it attempts
  one supervised restart.

The sidecar's combined stdout/stderr is drained into a bounded tail so a
startup failure (missing model, port in use, missing shared lib) surfaces
whisper's own error instead of a bare "Connection refused".

The route layer (``routes_stt.py``) talks to a thin injectable HTTP client so
tests can point at a stub ``whisper-server`` without a real binary.
"""

from __future__ import annotations

import logging
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import httpx

from .config import Config
from .voice_registry import DEFAULT_CACHE_DIR

logger = logging.getLogger(__name__)

# How long to wait for the sidecar subprocess to reach responsiveness and how
# often to poll /health while doing so.
STARTUP_TIMEOUT_S = 30.0
STARTUP_POLL_S = 0.2
# How long to wait for a graceful SIGTERM before SIGKILL during shutdown/eviction.
TERMINATE_TIMEOUT_S = 5.0


def model_filename(model: str) -> str:
    """Map ``STTS_STT_MODEL`` to the GGUF filename in the HF repo.

    - ``small`` -> ``ggml-small.bin``
    - ``small.q5_0`` (quantized variant) -> ``ggml-small.q5_0.bin``
    - a full filename (``ggml-base.bin``) passes through unchanged.
    """
    m = model.strip()
    if m.endswith(".bin"):
        return m
    if m.startswith("ggml-"):
        return f"{m}.bin"
    return f"ggml-{m}.bin"


def stt_model_dir(config: Config) -> Path:
    """Where the GGUF weights are cached (persist under /data in the container)."""
    if config.stt_model_dir:
        return Path(config.stt_model_dir)
    base = Path(config.cache_dir) if config.cache_dir else DEFAULT_CACHE_DIR
    return base / "stt-models"


def ensure_model(config: Config, *, http: httpx.Client | None = None) -> Path:
    """Download the GGUF model on first use into ``{model_dir}``; return its path.

    The download is streamed to ``<name>.part`` then atomically renamed, so a
    crash never leaves a half-written model as a valid file. ``http`` is
    injectable for tests (httpx.MockTransport); production uses the HF resolve
    URL.
    """
    dest = stt_model_dir(config) / model_filename(config.stt_model)
    if dest.exists() and dest.stat().st_size > 0:
        return dest
    dest.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://huggingface.co/{config.stt_model_repo}/resolve/main/{dest.name}"
    logger.info("downloading STT model %s -> %s", url, dest)
    client = http or httpx.Client(
        follow_redirects=True, timeout=httpx.Timeout(120.0, connect=10.0)
    )
    part = dest.with_suffix(dest.suffix + ".part")
    try:
        with client.stream("GET", url) as resp:
            resp.raise_for_status()
            with open(part, "wb") as fh:
                for chunk in resp.iter_bytes(chunk_size=1 << 20):
                    fh.write(chunk)
        part.replace(dest)
    except httpx.HTTPError as exc:
        part.unlink(missing_ok=True)
        raise RuntimeError(
            f"failed to download STT model {dest.name} from {url}: {exc}"
        ) from exc
    finally:
        if http is None:
            client.close()
    return dest


@dataclass
class SttStats:
    requests: int = 0
    idle_stops: int = 0  # idle-eviction events (process killed)
    idle_starts: int = 0  # re-spawns after an idle eviction (wake)
    restarts: int = 0  # supervised crash restarts
    spawn_errors: int = 0


@dataclass
class WhisperSidecar:
    """Owns a ``whisper-server`` subprocess and proxies to it.

    ``http`` is injectable: tests pass an ``httpx.Client`` pointed at a stub
    ``whisper-server`` (or ``httpx.MockTransport``), so the whole proxy + idle
    + wake path is exercisable without a real binary.
    """

    config: Config
    model_path: str | Path
    http: httpx.Client | None = None
    spawn: "Callable[[], Any] | None" = None
    now: "Callable[[], float]" = time.monotonic
    startup_timeout: float = STARTUP_TIMEOUT_S
    startup_poll: float = STARTUP_POLL_S
    terminate_timeout: float = TERMINATE_TIMEOUT_S

    stats: SttStats = field(default_factory=SttStats, init=False)
    _proc: "Any" = field(default=None, init=False, repr=False)
    _ready: bool = field(default=False, init=False)
    _deflated: bool = field(default=False, init=False)  # evicted by idle policy
    _last_error: str | None = field(default=None, init=False)
    _last_activity: float = field(init=False, repr=False)
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False)
    _spawn_fn: "Callable[[], Any]" = field(init=False, repr=False)
    # Bounded tail of the sidecar's combined stdout/stderr, so a startup
    # failure is diagnosable (whisper-server prints the real reason to stderr).
    _diag: deque[str] = field(default_factory=lambda: deque(maxlen=40), init=False, repr=False)
    _diag_thread: threading.Thread | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.model_path = Path(self.model_path)
        # Anchor the idle timer on the (possibly injected) clock so eviction
        # logic is fully deterministic under a fake ``now``.
        self._last_activity = self.now()
        if self.http is None:
            self.http = httpx.Client(
                base_url=f"http://{self.config.stt_host}:{self.config.stt_port}",
                timeout=httpx.Timeout(600.0, connect=3.0),  # whisper is slow on CPU
            )
        self._spawn_fn = self.spawn if self.spawn is not None else self._real_spawn

    # -- configuration ------------------------------------------------------

    @property
    def eviction_enabled(self) -> bool:
        return self.config.stt_idle_unload_s > 0

    @property
    def ready(self) -> bool:
        """Healthy and the process is still alive (not just ``True`` after a crash)."""
        if not self._ready or self._proc is None:
            return False
        if self._proc.poll() is not None:  # process died under us
            self._ready = False
            return False
        return True

    @property
    def pid(self) -> int | None:
        if self._proc is not None and self._proc.poll() is None:
            return self._proc.pid
        return None

    def touch(self) -> None:
        """Record an STT API request (resets the idle window). Health probes and
        TTS traffic must NOT call this (per-subsystem timer)."""
        self._last_activity = self.now()

    def last_request_age(self) -> float:
        return self.now() - self._last_activity

    # -- process lifecycle --------------------------------------------------

    def _real_spawn(self) -> "Any":
        """Production command: whisper-server bound to loopback with our model.

        stdout/stderr are piped and drained into ``_diag`` (a bounded tail) so a
        startup failure is diagnosable: instead of a bare "Connection refused"
        the user sees whisper's own ``error: failed to initialize whisper
        context`` / ``couldn't bind to server socket`` on stderr.
        """
        cmd = [
            self.config.stt_bin,
            "--host", self.config.stt_host,
            "--port", str(self.config.stt_port),
            "-m", str(self.model_path),
            "-t", str(self.config.stt_threads),
            # whisper.cpp's wav reader hard-requires 16 kHz / 16-bit uploads;
            # --convert shells out to ffmpeg (already installed in the image) to
            # resample/transcode any upload (24 kHz TTS output, mp3, m4a, ...)
            # to the 16 kHz mono PCM whisper wants. Without it every non-16 kHz
            # upload fails with "failed to read WAV file".
            "--convert",
        ]
        if self.config.stt_language:
            cmd += ["-l", self.config.stt_language]
        # OpenVINO is a documented follow-up: when wired, add
        #   ["-oved", self.config.stt_openvino_device]
        # here (native ggml CPU is the unconditional default).
        logger.info("starting whisper-server: %s", " ".join(cmd))
        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(
                f"cannot start whisper-server ({self.config.stt_bin!r}): {exc}"
            ) from exc
        self._diag.clear()
        self._diag_thread = threading.Thread(
            target=self._drain_output, args=(proc,), name="whisper-server-stderr", daemon=True
        )
        self._diag_thread.start()
        return proc

    def _drain_output(self, proc: "Any") -> None:
        """Drain the sidecar's combined stdout/stderr into ``_diag`` (bounded).

        Prevents the pipe from filling up (blocking the child) and keeps the
        last lines around so a crash/startup failure is explainable.
        """
        if proc.stdout is None:  # pragma: no cover - real Popen always has one
            return
        try:
            for raw in proc.stdout:
                try:
                    line = raw.decode("utf-8", errors="replace").rstrip()
                except (AttributeError, ValueError):  # pragma: no cover - defensive
                    continue
                if line:
                    self._diag.append(line)
        except (OSError, ValueError):  # pipe closed on terminate
            pass

    def _diag_tail(self, n: int = 8) -> str:
        """Last ``n`` captured lines of the sidecar output, joined for display."""
        if not self._diag:
            return ""
        return "\n".join(list(self._diag)[-n:])

    def start(self) -> bool:
        """Spawn the sidecar and wait until it's ready. Idempotent.

        Holds ``_lock`` for the whole spawn+readiness so ``ensure_started`` is
        automatically single-flight: concurrent waiters serialize and look at
        ``ready`` again after the first finishes (no thundering herd).
        """
        with self._lock:
            if self.ready:
                return True
            was_deflated = self._deflated
            if self._proc is not None and self._proc.poll() is None:
                self._terminate()  # stale process from a previous life
            self._ready = False
            self._deflated = False
            self._last_error = None
            try:
                self._proc = self._spawn_fn()
            except Exception as exc:
                self._last_error = str(exc)
                self.stats.spawn_errors += 1
                logger.error("whisper-server start failed: %s", exc)
                return False
            if self._wait_ready():
                self._ready = True
                if was_deflated:
                    self.stats.idle_starts += 1  # wake after an idle eviction
                logger.info("whisper-server ready on port %d", self.config.stt_port)
                return True
            self._last_error = self._last_error or "whisper-server did not become ready"
            self.stats.spawn_errors += 1
            return False

    def ensure_started(self) -> bool:
        """Wake path: single-flight re-spawn + readiness poll (blocks until ready).

        Concurrent waiters share one re-spawn because they serialize on
        ``_lock`` inside ``start()``; a request that raced an idle eviction
        simply blocks briefly then succeeds (no 503 in the wake path).
        """
        return self.start()

    def _wait_ready(self) -> bool:
        """Poll for a healthy sidecar until ``startup_timeout`` elapses.

        Readiness signals: whisper.cpp ``/health`` (200 ready, 503 still loading)
        exists on master but NOT on the v1.7.4 the Dockerfile pins (it returns
        404) — so on a 404 we fall back to ``GET /`` (200 on both once the model
        is loaded and bound). Fails fast with the child's exit code + captured
        stderr if the process died before listening (no 30 s of dead polling).
        """
        deadline = time.monotonic() + self.startup_timeout
        last: str | None = None
        http = self.http
        if http is None:  # pragma: no cover - __post_init__ always provides one
            return False
        while time.monotonic() < deadline:
            proc = self._proc
            if proc is not None and proc.poll() is not None:
                # Fail fast: the child exited before serving — report why.
                code = proc.poll()
                diag = self._diag_tail()
                self._last_error = (
                    f"whisper-server exited during startup (code {code})"
                    + (f": {diag}" if diag else "")
                )
                return False
            try:
                resp = http.get("/health", timeout=1.0)
            except Exception as exc:  # pragma: no cover - connection refused etc.
                last = str(exc)
                time.sleep(self.startup_poll)
                continue
            if resp.status_code == 200:
                return True
            if resp.status_code == 404:
                # v1.7.4 has no /health route; GET / is 200 on both versions.
                try:
                    root = http.get("/", timeout=1.0)
                except Exception as exc:  # pragma: no cover - defensive
                    last = str(exc)
                else:
                    if root.status_code == 200:
                        return True
                    last = f"/={root.status_code}"
            elif resp.status_code in (409, 503):
                last = f"health={resp.status_code} (still loading)"
            else:
                last = f"health={resp.status_code}"
            time.sleep(self.startup_poll)
        diag = self._diag_tail()
        self._last_error = (
            f"whisper-server not ready: {last}" + (f"\n{diag}" if diag else "")
        )
        return False

    def _terminate(self) -> None:
        proc = self._proc
        if proc is None or proc.poll() is not None:
            if proc is not None:
                self._proc = None
            return
        logger.info("terminating whisper-server (pid %d)", proc.pid)
        try:
            proc.terminate()
            proc.wait(timeout=self.terminate_timeout)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=self.terminate_timeout)
        except OSError:  # pragma: no cover - already gone
            pass
        self._proc = None

    def shutdown(self) -> None:
        """Lifespan shutdown hook: stop the sidecar process."""
        with self._lock:
            self._terminate()
            self._ready = False

    # -- idle eviction + crash watch ----------------------------------------

    def stop_if_idle(self, now: float | None = None) -> bool:
        """Kill the sidecar (reclaim ~100% of its RSS) if it has been idle past
        ``stt_idle_unload_s``. Returns True if it was stopped.

        Unlike the TTS in-process eviction this is a plain process kill — there
        is no in-object state to drop. The timer is per-subsystem: only STT API
        requests (``touch``) reset it, never /health or TTS traffic.
        """
        if not self.eviction_enabled:
            return False
        now = self.now() if now is None else now
        if now - self._last_activity < self.config.stt_idle_unload_s:
            return False
        # Don't race a request in flight: if the lock is busy, skip this pass
        # (the watchdog retries next tick).
        if not self._lock.acquire(blocking=False):
            return False
        try:
            if self._deflated or self._proc is None:
                return False
            if now - self._last_activity < self.config.stt_idle_unload_s:
                return False
            self.stats.idle_stops += 1
            self._deflated = True
            self._terminate()
            logger.info(
                "stt idle %.0fs: terminated whisper-server to reclaim RAM",
                self.config.stt_idle_unload_s,
            )
            return True
        finally:
            self._lock.release()

    def watch(self) -> None:
        """Crash watcher: if the process died unexpectedly, attempt one supervised
        restart. Intended idle stops are marked ``_deflated`` so they are never
        mistaken for a crash. Runs in the dedicated STT watchdog thread."""
        proc = self._proc
        if proc is None or self._deflated or not self._ready:
            return
        if proc.poll() is not None:
            with self._lock:
                if self._deflated or not self._ready:
                    return
                self._ready = False
                self._last_error = (
                    f"whisper-server exited unexpectedly (code {proc.poll()})"
                )
                logger.warning("whisper-server crashed; supervised restart…")
                try:
                    self._proc = self._spawn_fn()
                    if self._wait_ready():
                        self._ready = True
                        self.stats.restarts += 1
                        self._last_error = None
                        logger.info("whisper-server restarted")
                except Exception as exc:  # pragma: no cover - defensive
                    logger.exception("whisper-server restart failed")
                    self._last_error = str(exc)

    def last_error_hint(self) -> str:
        """Short, user-safe description of why the sidecar is down (for 503s)."""
        return self._last_error or "unknown startup failure"

    def transcribe_raw(self, audio: bytes, filename: str, content_type: str, data: dict[str, str]) -> httpx.Response:
        """Proxy with a caller-controlled field set (used when the route needs to
        forward response_format / language / token_timestamps alongside)."""
        return self._post_inference(audio, filename, content_type, data)

    def _post_inference(
        self, audio: bytes, filename: str, content_type: str, data: dict[str, str]
    ) -> httpx.Response:
        self.stats.requests += 1
        if self.http is None:  # pragma: no cover - constructor always provides one
            raise RuntimeError("whisper-server HTTP client not configured")
        resp = self.http.post(
            "/inference",
            files={"file": (filename, audio, content_type)},
            data=data,
        )
        if not resp.is_success:
            logger.warning(
                "whisper-server /inference -> %s %.200s", resp.status_code, resp.text
            )
        return resp

    def health_info(self) -> dict:
        return {
            "enabled": True,
            "model": self.config.stt_model,
            "ready": self.ready,
            "pid": self.pid,
            "last_error": self._last_error,
            "idle_unload_s": self.config.stt_idle_unload_s,
            "last_request_age_s": round(self.last_request_age(), 1),
            "idle_stops": self.stats.idle_stops,
            "idle_starts": self.stats.idle_starts,
            "restarts": self.stats.restarts,
        }


def load_sidecar(config: Config) -> WhisperSidecar:
    """Production factory: ensure the model file exists, then return a sidecar
    configured to proxy to it (spawn happens on first request / startup)."""
    model = ensure_model(config)
    return WhisperSidecar(config=config, model_path=model)
