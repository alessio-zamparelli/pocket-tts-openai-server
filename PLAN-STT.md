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

### 5. `/v1/models` + `/health`

- `/v1/models` appends `{"id": "whisper-1", "object": "model", "created": …,
  "owned_by": "whisper.cpp"}`; TTS aliases untouched (backward compatible).
- `/health` adds `"stt": {"enabled", "model", "ready", "pid", "last_error"}`
  (or omits when disabled). Health probes must NOT reset any STT activity timer
  (none in v1 — sidecar keeps the model resident; see rejected alternatives).

### 6. Tests (keep 88 green + new)

- Config parse tests for every `POCKET_TTS_STT_*` knob (+ `ENABLED` off/on).
- Routes tests (stub HTTP server as fake sidecar): happy-path transcription for
  each `response_format`, translations (`translate=true` observed on the stub),
  unknown-model → 400, missing/oversized file → 400 (`payload_too_large`),
  sidecar down → 503, `/v1/models` includes `whisper-1`.
- Sidecar lifecycle: `Popen` spawn with stub binary + fake `/health`; readiness
  wait; crash detected → 503; terminate on shutdown (all synchronous, no real
  sleeps — use short timeouts/small poll intervals).
- Target: ~10–14 new tests, existing 88 stay green; pyright clean.

### 7. README + PLAN.md

- Config table: document the `POCKET_TTS_STT_*` knobs.
- New "Speech-To-Text" section: quickstart curl for `transcriptions`
  (`response_format=json`) and `translations`; note default `small` model,
  ffmpeg for non-WAV, `/data` volume holds GGUF weights.
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
- **Idle-eviction of the sidecar:** would require kill+restart of the process
  (no in-process eviction like M5.5). Out of v1 scope; noted as a possible M6.1
  (stop sidecar when idle, warm-start on next request). Kept resident for now
  (predictable latency; one model ~1 GB).
- **ffmpeg-less runtime (WAV-only STT):** cuts image size but OpenAI clients
  upload `mp3`/`m4a`/`webm` by default; include ffmpeg for real-world compat.

## Milestone

M6 · STT endpoint. Files: `deploy/Dockerfile`, `config.py`, `server.py`,
`src/pocket_tts_openai/stt.py`, `src/pocket_tts_openai/routes_stt.py`,
`routes_speech.py` (`/v1/models`, `/health`), `README.md`, `PLAN.md`, tests.

> **Status: PLANNED** — decisions locked: persistent `whisper-server` sidecar;
> default `small` (multilingual); transcriptions + translations + `/v1/models`;
> OpenVINO deferred as a documented build+env follow-up (native ggml CPU v1).
