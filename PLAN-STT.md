# PLAN — OpenAI-compatible STT endpoint via whisper.cpp (+ OpenVINO)

> **Goal:** expose OpenAI-compatible Speech-To-Text on the existing server —
> `POST /v1/audio/transcriptions` and `POST /v1/audio/translations` — backed by
> [whisper.cpp](https://github.com/ggml-org/whisper.cpp) (ggml-org), with a
> **persistent native `whisper-server` sidecar** as the engine. Default model:
> **`small` (multilingual, `ggml-small.bin`)**. OpenVINO is accepted as a
> **documented compile-time follow-up**, not the v1 default (see below).

## Why a persistent native sidecar (and not the Python bindings)

Confirmed during research:

- whisper.cpp's OpenVINO path is **compile-time** (`-DWHISPER_OPENVINO=1` CMake
  flag + a converted encoder IR via `convert-whisper-to-openvino.py`). It is not
  something a pip wheel cleanly toggles, and it needs the OpenVINO runtime at
  build/run time.
- The PyPI binding (`whisper-cpp-python`) is **sdist-only**, compiles from C++
  at install time (`gcc` required), has **no cp314 wheel**, and its OpenVINO
  support is unverified. That is fragile against our pinned `uv.lock`,
  Python 3.14 image, and CPU-only policy.
- whisper.cpp ships a native HTTP server binary `whisper-server`
  (`examples/server`) that already speaks **multipart** and emits
  OpenAI-shaped JSON (`{"text": ...}`, `verbose_json` with `segments`),
  plus `srt`/`vtt`/`text`, a `translate` flag, `-oved` OpenVINO device flag,
  and `GET /health`. Wiring a proxy to it is low-risk and battle-tested.

**So the engine is a native `whisper-server` process managed by our FastAPI app:**
one model resident in RAM (no per-request spawn → RAM stays flat, matching the
M5.5 RAM-frugality theme), crash-isolated from Python, and the request path is
a thin OpenAI-contract translation layer we fully own and can unit-test.

Idle RAM reclamation extends M5.5's policy to the STT sidecar (see
**[Idle eviction of the sidecar](#6-idle-eviction-of-the-sidecar-mirrors-m55-process-level)**): because
whisper-server has **no `/unload` endpoint** (only `POST /load` + `GET /health`),
the eviction mechanism is **process kill + lazy restart** — which reclaims
~100% of the sidecar's RSS and is structurally simpler than the in-process
eviction TTS needs.

## Endpoint surface (OpenAI-compatible)

| Method/Path | Behavior |
| --- | --- |
| `POST /v1/audio/transcriptions` | multipart `file` (audio), `model` (required), optional `language`, `prompt`, `response_format` (`json` default \| `text` \| `srt` \| `vtt` \| `verbose_json`), `temperature`, `timestamp_granularities[]` (`segment` \| `word`). Returns transcription; default `{"text": "..."}` as `application/json`. |
| `POST /v1/audio/translations` | same contract, but `whisper-server` runs with `translate=true` → non-English source becomes **English** text (requires a multilingual model — `small` is). |
| `GET /v1/models` | adds the STT model `whisper-1` (`owned_by: whisper.cpp`) alongside the TTS aliases; TTS entries unchanged. |
| `GET /health` | adds an `stt` block (model, loaded/ready, language, sidecar pid/status, last error) without changing existing fields. |

Multipart parsing is already available (`python-multipart` + `UploadFile` used
by `POST /v1/voices`); the same `max_upload_mb` limit applies to the audio file.

## Design

### 1. Native build (`deploy/Dockerfile`, multi-stage)

- **Build stage (new)** `FROM ubuntu:24.04` (or `debian:bookworm`) with
  `gcc g++ cmake git python3`:
  - `git clone --branch <pinned tag/commit> --depth 1 https://github.com/ggml-org/whisper.cpp`,
  - `cmake -B build -DWHISPER_BUILD_SERVER=ON` (CPU build; no CUDA — respects the
    no-NVIDIA policy),
  - copy `build/bin/whisper-server` into the runtime stage.
- **Runtime stage:** copy the binary (chmod +x, owned by `tts`); add `ffmpeg`
  for non-WAV input decoding (whisper-server `--convert` shells out to ffmpeg;
  OpenAI clients routinely upload `mp3`/`m4a`/`webm`). Note in README that WAV
  needs no ffmpeg; image grows by ~100 MB — acceptable, documented tradeoff.
- **Model:** download `ggml-small.bin` (~466 MB) from
  `ggerganov/whisper.cpp` (HF) **on first start** into
  `POCKET_TTS_STT_MODEL_DIR` (default `{cache_dir}/stt-models`) — same
  first-run-cache pattern as pocket-tts → models live on the persistent
  `/data` volume, owned by `tts` (bind-mount hosts must pre-chown
  `10001:10001`, see README container note). No image blow-up, easy model swap.
- **No Docker in the sandbox** ⇒ the build/run contract above is **manual
  validation** (YAML/Dockerfile review + clean-dir repro of the uv layers); the
  Python side is fully testable with a stub `whisper-server` (below).

### 2. Sidecar lifecycle (`server.py`, new `stt/` module)

- New module `src/pocket_tts_openai/stt.py`:
  - `ensure_model(config)` — download GGUF if missing (reuse HF/httpx download
    pattern from engine/voices), return the path.
  - `WhisperSidecar` (or `SttEngine`) — owns a `subprocess.Popen` of
    `whisper-server --host 127.0.0.1 --port <port> -m <model>
    [-l <language>] [-t <threads>] [--ov-e-device CPU]`, plus:
    - **readiness**: poll `GET /health` until 200 (bounded),
    - **crash/watch**: watchdog thread asserts the pid is alive; if it dies,
      mark `down` → routes return 503 and log; v1 attempts **one** supervised
      restart (simple), then stays down until a manual/health reset request,
    - **idle state**: `last_activity` (monotonic), `touch()` bumped by STT API
      requests only, `stop_if_idle(now)` = `terminate()` + `wait()` after the
      per-subsystem idle window (see §6), and `ensure_started()` doing a
      single-flight re-spawn + readiness poll on wake,
    - **shutdown**: `terminate()` in `lifespan` finally.
  - When `stt_enabled=false` (or the binary/model is unavailable) the sidecar
    is not spawned and STT routes return 503 with a clear message; TTS is
    unaffected.
- **Port:** fixed internal `POCKET_TTS_STT_PORT` (default `8787`) bound to
  `127.0.0.1` only (never exposed; the app proxies). The binary path is
  overridable (`POCKET_TTS_STT_BIN`) so tests can point at a stub and the
  compose/Docker runtime always finds `/usr/local/bin/whisper-server`.
- **Testability:** the route layer talks to an injectable HTTP client
  (httpx against `http://127.0.0.1:<port>`); tests start a tiny stub
  HTTP server with canned responses for each `response_format`, mirroring the
  fake-engine pattern used for TTS. No real binary needed in CI.

### 3. Routes (`routes_stt.py`)

- `_stt_or_503(request)` — sidecar present + healthy else `unavailable(...)`
  (match `_engine_or_503` convention).
- Request: `model` must be in `STT_MODEL_ALIASES = ("whisper-1",)` (else
  `invalid_request`); enforce upload size (reuse `max_upload_mb`); require a
  non-empty audio `file`; validate `response_format` and
  `timestamp_granularities` values up front.
- Forward to `/inference` as multipart, mapping:
  - `language` → `language`, `prompt` → `prompt`,
    `temperature` → `temperature`,
  - `response_format` → `response_format` (pass-through; skip the field for the
    default `json` since whisper-server's bare default already returns
    `{"text": ...}`),
  - `timestamp_granularities=word` → enable whisper token timestamps
    (`token_timestamps=true`; word support in `verbose_json` is best-effort —
    whisper.cpp token timestamps, not full DTW; document),
  - translations → `translate=true` + drop client `language`.
- Response:
  - `json` → return `{"text": ...}` as `application/json`,
  - `verbose_json` → sanitize/pass through whisper-server's OpenAI-shaped
    object (it already emits `task/language/duration/text/segments[...]` with
    `id/start/end/text/tokens/words/temperature/avg_logprob/no_speech_prob`),
  - `text` → plain text body, `srt` → `application/x-subrip`,
    `vtt` → `text/vtt` (Content-Type mirrors OpenAI), with
    `Content-Disposition: attachment`.
- Errors: upstream 500 → `invalid_request` (or 502) mapping; missing file →
  400; oversized → `payload_too_large` (reuse `errors.py`).

### 4. Config (`config.py`)

| Env | Default | Meaning |
| --- | --- | --- |
| `POCKET_TTS_STT_ENABLED` | `false` | Master switch; default **off** until a sidecar+binary is present (safe for existing TTS-only deploys). Set `true` to enable STT. |
| `POCKET_TTS_STT_MODEL` | `small` | GGUF model name → `ggml-{model}.bin` (multilingual default). Quantized variants (`small.q5_0`) selectable by full filename. |
| `POCKET_TTS_STT_MODEL_REPO` | `ggerganov/whisper.cpp` | HF repo hosting `ggml-*.bin`. |
| `POCKET_TTS_STT_MODEL_DIR` | `{cache_dir}/stt-models` | Where the GGUF is cached (persisted under `/data`). |
| `POCKET_TTS_STT_BIN` | `whisper-server` | Sidecar binary path (tests/advanced override). |
| `POCKET_TTS_STT_HOST` | `127.0.0.1` | Internal bind (loopback only). |
| `POCKET_TTS_STT_PORT` | `8787` | Internal HTTP port proxied by the app. |
| `POCKET_TTS_STT_THREADS` | `4` | `-t` compute threads for the sidecar. |
| `POCKET_TTS_STT_LANGUAGE` | `""` | Optional default language; empty = auto-detect. |
| `POCKET_TTS_STT_IDLE_UNLOAD_S` | `300` | Kill the sidecar (reclaim its RAM) after this long without an STT request; `0` disables (independent of `POCKET_TTS_IDLE_UNLOAD_S`). |
| `POCKET_TTS_STT_IDLE_POLL_S` | `30` | Dedicated STT watchdog cadence (only when idle-unload `> 0`). |

### 5. `/v1/models` + `/health`

- `/v1/models` appends `{"id": "whisper-1", "object": "model", "created": …,
  "owned_by": "whisper.cpp"}`; TTS aliases untouched (backward compatible).
- `/health` adds `"stt": {"enabled", "model", "ready", "pid", "last_error",
  "idle_unload_s", "last_request_age_s"}` (or omits when disabled). Health
  probes do NOT reset the STT idle timer — only real STT API requests bump
  `last_activity` (same rule as M5.5's TTS watchdog).

### 6. Idle eviction of the sidecar (mirrors M5.5, process-level)

**Mechanism: process kill + lazy restart** (whisper-server has no `/unload`;
only `/load` + `/health`). Killing the sidecar reclaims ~100% of its RSS
(whisper `small` loads to roughly ~1 GB RSS on CPU); there is no in-process
object to drop, so none of M5.5's `gc.collect()` / `malloc_trim` machinery
applies — the C process just dies and its pages return to the OS.

- **Config (separate knob, independent of TTS):**
  - `POCKET_TTS_STT_IDLE_UNLOAD_S` (default `300`, `0` disables)
  - `POCKET_TTS_STT_IDLE_POLL_S` (default `30`)
- **Per-subsystem timer:** `WhisperSidecar.last_activity` is bumped only by STT
  API requests (`transcriptions` / `translations` routes call `touch()`);
  `/health` probes and TTS traffic never touch it — so STT evicts only on STT
  silence, independent of what the TTS engine is doing.
- **Stop path:** a dedicated daemon thread in `lifespan` (new, not shared with
  the TTS `_idle_watchdog`) loops `wait(poll_s)` → `sidecar.stop_if_idle(now)`:
  `terminate()` + `wait(timeout)` (graceful, not `SIGKILL`), mark `ready=False`,
  log a stats counter `stt_idle_stops`. Only spawned when
  `stt_enabled` and `stt_idle_unload_s > 0`.
- **Wake path (block until ready, no 503):** the first STT request after an
  idle stop calls `sidecar.ensure_started()` — a **single-flight** re-spawn
  guarded by a lock (concurrent waiters share one restart, like M5.5's
  `_reload_lock`), then poll `/health` until 200. Warm restart ≈
  `whisper-server` exec (~50 ms) + `small` load from `/data` (~0.5–2 s CPU) →
  the request blocks briefly then succeeds; no 503 in the wake path. If the
  re-spawn fails (missing binary/model, port taken), return 503.
- **Interplay with crash-watch:** the watchdog must not race `stop_if_idle`
  against an unexpected crash restart — `stop_if_idle` and the crash handler
  coordinate on the same restart lock; an idle stop is marked as intended so
  the crash-watcher doesn't immediately respawn it.
- **Stats/observability:** `stt_idle_stops` / `stt_idle_starts` counters on the
  sidecar; surfaced in `/health`'s `stt` block.

### 7. Tests (keep 88 green + new)

- Config parse tests for every `POCKET_TTS_STT_*` knob (+ `ENABLED` off/on,
  `IDLE_UNLOAD_S`/`IDLE_POLL_S` parse and `0` disables).
- Routes tests (stub HTTP server as fake sidecar): happy-path transcription for
  each `response_format`, translations (`translate=true` observed on the stub),
  unknown-model → 400, missing/oversized file → 400 (`payload_too_large`),
  sidecar down → 503, `/v1/models` includes `whisper-1`.
- Sidecar lifecycle: `Popen` spawn with stub binary + fake `/health`; readiness
  wait; crash detected → 503; terminate on shutdown (all synchronous, no real
  sleeps — use short timeouts/small poll intervals).
- Idle-eviction tests (injectable `now`, stub sidecar): `stop_if_idle` evicts
  after the window; no eviction before the window or when `IDLE_UNLOAD_S=0`;
  first request after idle `ensure_started()` respawns and returns 200 (assert
  single-flight: concurrent waiters trigger one re-spawn); health probes do not
  bump the STT timer.
- Target: ~14–18 new tests, existing 88 stay green; pyright clean.

### 8. README + PLAN.md

- Config table: document the `POCKET_TTS_STT_*` knobs (incl. idle eviction).
- New "Speech-To-Text" section: quickstart curl for `transcriptions`
  (`response_format=json`) and `translations`; note default `small` model,
  ffmpeg for non-WAV, `/data` volume holds GGUF weights, and that the sidecar
  is evicted after `POCKET_TTS_STT_IDLE_UNLOAD_S` of STT inactivity (next
  request restarts it).
- Container/mount note: host folders bind-mounted at `/data` must be owned
  `10001:10001` so the sidecar/user can write model weights.
- PLAN.md milestone table: add M6 STT row (status: planned).

## OpenVINO: documented follow-up (out of v1 default)

Accepted decision: **skip OpenVINO by default; ship native ggml CPU first.** The
architecture already leaves the door open — everything is in the sidecar's
launch args (`-oved`/`--ov-e-device`). To enable OpenVINO later, without code
changes beyond wiring:

1. Build stage: install the OpenVINO toolkit (pin to the whisper.cpp README's
   recommended release), `source setupvars.sh`, build with
   `-DWHISPER_OPENVINO=1`.
2. Convert the encoder IR once: `python convert-whisper-to-openvino.py --model
   small` → `ggml-small-encoder-openvino.xml/.bin` beside the GGUF.
3. Launch: `whisper-server … --ov-e-device CPU` (x86 CPU; Intel iGPU/dGPU also
   supported). First run compiles/caches the IR blob (slow), subsequent runs
   reuse it.
4. Wire `POCKET_TTS_STT_OPENVINO_DEVICE` config → `--ov-e-device`; keep native
   ggml as the unconditional fallback when the encoder IR is absent.

`small` on CPU/ggml is faster-than-realtime on modern x86; OpenVINO mainly buys
encoder speed-up at larger models (medium/large-v3) or on Intel dGPU.

## Considerations / rejected alternatives

- **Per-request `whisper-cli` subprocess:** simplest, but model re-load per
  request (~hundreds of ms + RSS spike) and temp-WAV I/O; rejected in favor of
  the resident sidecar (flat RAM, warm path, matches M5.5 frugality).
- **In-process Python binding (`whisper-cpp-python` / ctypes):** no cp314 wheel,
  sdist-only build (needs `gcc`), OpenVINO-in-binding unverified, no crash
  isolation from Python; rejected.
- **Running `whisper-server` on a second public port:** would bypass our
  OpenAI contract, auth (API-key middleware), errors, `/v1/models`; rejected —
  the app owns `/v1/audio/*` and proxies internally.
- **Idle-eviction of the sidecar (formerly out-of-scope, now §6):** only
  possible as process kill + lazy restart (no in-process eviction like M5.5),
  because whisper-server has no `/unload`. Adopted in v1 with a
  dedicated watchdog + separate `POCKET_TTS_STT_IDLE_UNLOAD_S` knob and a
  block-until-ready wake path (~0.5–2 s warm) — reclaims ~1 GB of sidecar RAM
  on STT silence with no traffic cost beyond a brief warm-up on the next call.
- **ffmpeg-less runtime (WAV-only STT):** cuts image size but OpenAI clients
  upload `mp3`/`m4a`/`webm` by default; include ffmpeg for real-world compat.

## Milestone

M6 · STT endpoint. Files: `deploy/Dockerfile`, `config.py`, `server.py`,
`src/pocket_tts_openai/stt.py`, `src/pocket_tts_openai/routes_stt.py`,
`routes_speech.py` (`/v1/models`, `/health`), `README.md`, `PLAN.md`, tests.

> **Status: PLANNED** — decisions locked: persistent `whisper-server` sidecar;
> default `small` (multilingual); transcriptions + translations + `/v1/models`;
> OpenVINO deferred as a documented build+env follow-up (native ggml CPU v1);
> **idle RAM eviction for the sidecar (process kill + lazy restart)** with a
> separate `POCKET_TTS_STT_IDLE_UNLOAD_S`, per-subsystem timer, dedicated
> watchdog, and block-until-ready wake.
