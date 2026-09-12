# pocket-tts-openai

OpenAI-compatible TTS server (`POST /v1/audio/speech`, `stream` chunked PCM/WAV)
powered by [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) —
100M-param speech synthesis on CPU.

> **Implemented milestones**: M1 server skeleton, M2 audio formats, M3 voice
> catalog + cloning (`GET/POST/DELETE /v1/voices`), M4 streamed `wav`/`pcm`,
> M5.5 idle RAM eviction, M6 STT (whisper.cpp sidecar). Design decisions live
> in `./plan/PLAN.md`; per-milestone plans in `./plan/PLAN-M3-M4.md`, `./plan/PLAN-idle-unload.md`, `./plan/PLAN-STT.md`.

## Dev

```sh
uv sync                                # deps without the engine (tests use a fake model)
uv run pytest                          # 119 tests
uv sync --extra engine                 # + pocket-tts (CPU-only torch on Linux, see pyproject)
uv run pocket-tts-openai               # serve on :8000
```

To run on all interfaces / another port (as used by the live server):
`STTS_HOST=0.0.0.0 STTS_PORT=8080 uv run pocket-tts-openai`.

## Endpoints

### `POST /v1/audio/speech`

Body (JSON):

| field | type | default | notes |
| --- | --- | --- | --- |
| `model` | string | `tts-1` | any of `tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` |
| `input` | string | — | text to synthesize (required, non-empty) |
| `voice` | string | `alloy` | OpenAI alias, Kyutai catalog voice, custom voice, `https://`/`hf://`/path |
| `response_format` | string | `wav` | `wav` \| `pcm` (compressed formats pending M2) |
| `stream` | bool | `false` | **private extension** — chunked transfer for `wav`/`pcm`. Ignored (buffered) for compressed formats |
| `speed` | float | — | accepted, ignored |
| `instructions` | string | — | accepted, ignored |
| `language` | string | — | server-configured language always wins; logged and ignored |

`stream: true` returns a `StreamingResponse`:

- `response_format=pcm` → `audio/pcm`, raw mono s16le chunks.
- `response_format=wav` → `audio/wav` with a **streaming WAV header**
  (`RIFF`/`data` sizes = `0xFFFFFFFF`) followed by PCM chunks — renders
  incrementally in ffplay/VLC and most browsers.

`Content-Disposition` is `attachment; filename=speech.wav` (or `.pcm`).

### `GET /v1/models` · `GET /health`

`/v1/models` lists the model aliases. When STT is enabled it also advertises
`whisper-1` (the whisper.cpp sidecar). `/health` is liveness + engine stats
(`status`, `language`, `requests`, `avg_rtf`, `queue_depth`); it reports
`loading` until the model is ready, plus an `stt` block when STT is enabled.

### `POST /v1/audio/transcriptions` · `POST /v1/audio/translations` (STT)

Whisper.cpp STT (requires `STTS_STT_ENABLED=true`). Multipart/form-data
matching the OpenAI audio API: `file` (wav/mp3/m4a/webm — ffmpeg-decoded in
the container), `model` (`whisper-1`), plus optional `language`, `prompt`,
`response_format`, `temperature`, `timestamp_granularities[]`.

- `response_format`: `json` (default) · `text` · `srt` · `vtt` · `verbose_json`.
- `timestamp_granularities[]`: `segment` (default) · `word` (maps to whisper's
  token timestamps — best-effort, only in `verbose_json`).
- `translations` always translates into **English** (`language` is dropped).

Sub-second audio: whisper.cpp silently returns an empty transcript for clips
under ~1.0–1.2 s (a short TTS round-trip like „hello“ otherwise comes back as
`{"text": ""}`). Uploads under `STTS_STT_MIN_DURATION_S` are automatically
time-stretched (slowed with ffmpeg `atempo`) to at least that floor before
transcribing, so single-word clips transcribe instead of vanishing. Only
sub-floor uploads pay the ffmpeg round-trip; normal audio is forwarded
untouched. `STTS_STT_MIN_DURATION_S=0` disables the workaround.

Powered by a persistent native `whisper-server` subprocess sidecar (default
model `small`, ggml-small.bin ~466 MB multilingual, overridable via
`STTS_STT_MODEL`). STT routes return **503** when STT is disabled or the
sidecar cannot start, **400** on validation errors, **413** over the upload
limit.

### Voice catalog & cloning (`/v1/voices`)

**`GET /v1/voices`** — list every valid voice:

```json
{"object": "list", "data": [
  {"id": "alloy", "aliases": ["alloy", "alba"], "source": "builtin",
   "language": "en", "license": "https://huggingface.co/kyutai/tts-voices", "cached": true},
  {"id": "mario", "aliases": ["mario"], "source": "custom",
   "language": "it", "license": null, "cached": false, "safetensors": true}
]}
```

**`POST /v1/voices`** — `multipart/form-data` with:

- `name` (required): `^[a-z0-9][a-z0-9_-]{0,63}$`. **409** on collision with a
  built-in alias/catalog voice or an existing custom voice.
- `file` (required): `.wav` / `.mp3` / `.flac`. **413** over
  `STTS_MAX_UPLOAD_MB` (default 25). **400** on unknown extension.
- `language` (optional): free-form tag, stored and echoed.

Returns **201** with the custom-voice payload after the audio prompt is encoded
and exported to `<cache_dir>/voices/<name>.safetensors`. The encode runs under
the global generation lock — a clone blocks concurrent synthesis requests.

**`DELETE /v1/voices/{name}`** — **405** for built-in voices; **404** for
unknown; **204** on success (removes the registry entry, the `.safetensors`
file, and its LRU cache slot).

Custom voices are immediately usable as `voice` in `/v1/audio/speech`.

### Idle RAM reclamation

After `STTS_IDLE_UNLOAD_S` (default 300 s = 5 min) with **no API
requests**, the model is dropped from RAM (~60% of the resident footprint) to
keep an idle API cheap. The next request **blocks while it reloads** (~1 s warm
from the HF cache) instead of returning 503. Set `STTS_IDLE_UNLOAD_S=0`
to keep the model resident always. `/health` exposes `loaded`, `unloads`,
`reloads` and `last_request_age_s`.

**STT sidecar eviction** mirrors this but at the process level: after
`STTS_STT_IDLE_UNLOAD_S` (default 300 s) with **no STT request**, the
`whisper-server` process is killed to reclaim ~100% of its RSS (~1 GB for
`small`). TTS traffic and `/health` probes never reset the STT timer — a
request after the eviction just blocks a moment while the sidecar re-spawns
(from `/data`, no re-model-download) instead of returning 503. Set
`STTS_STT_IDLE_UNLOAD_S=0` to always keep the sidecar resident.
`/health` `stt` block exposes `idle_stops`, `idle_starts`, `restarts` and
`last_request_age_s`.

## Configuration (env)

| var | default | notes |
| --- | --- | --- |
| `STTS_HOST` | `127.0.0.1` | bind host |
| `STTS_PORT` | `8000` | bind port |
| `STTS_LANGUAGE` | `english` | model language |
| `STTS_QUANTIZE` | `false` | quantize the model |
| `STTS_MAX_CACHED_VOICES` | `32` | LRU voice-state cache size |
| `STTS_WARMUP_VOICES` | — | comma-separated voices to pre-encode at boot |
| `STTS_IDLE_UNLOAD_S` | `300` | evict the model from RAM after this many idle seconds; `0` disables |
| `STTS_IDLE_POLL_S` | `30` | idle-eviction watchdog poll interval (seconds) |
| `STTS_API_KEY` | — | if set, `Bearer <key>` required on `/v1/*` |
| `STTS_CACHE_DIR` | `~/.cache/pocket_tts` | base for cloned voice registry |
| `STTS_MAX_UPLOAD_MB` | `25` | max cloned-voice / STT-audio upload size |
| `STTS_STT_ENABLED` | `false` | serve `/v1/audio/transcriptions` + `/v1/audio/translations` via a whisper.cpp sidecar |
| `STTS_STT_MODEL` | `small` | ggml model (`small`, `base`, `small.q5_0`, …) |
| `STTS_STT_MODEL_REPO` | `ggerganov/whisper.cpp` | HF repo hosting `ggml-*.bin` |
| `STTS_STT_MODEL_DIR` | `{cache_dir}/stt-models` | where the GGUF is stored (first run downloads) |
| `STTS_STT_BIN` | `whisper-server` | sidecar binary on PATH |
| `STTS_STT_HOST` | `127.0.0.1` | sidecar bind host (loopback only) |
| `STTS_STT_PORT` | `8787` | sidecar bind port |
| `STTS_STT_THREADS` | `1` | whisper compute threads (default 1 so the sidecar doesn't starve the TTS engine on low-core/low-power hosts; raise on beefier hardware) |
| `STTS_STT_LANGUAGE` | — | force transcription language (empty = auto-detect) |
| `STTS_STT_IDLE_UNLOAD_S` | `300` | kill the sidecar after this many STT-idle seconds; `0` disables |
| `STTS_STT_IDLE_POLL_S` | `30` | STT watchdog poll interval (seconds) |
| `STTS_STT_MIN_DURATION_S` | `1.2` | whisper drops audio under ~1.0-1.2 s; sub-floor uploads are time-stretched to at least this many seconds (ffmpeg `atempo`) before transcribing; `0` disables |

## Container (Docker)

CPU-only multi-stage image following the official
[uv Docker guide](https://docs.astral.sh/uv/guides/integration/docker/) —
`deploy/Dockerfile` (`python:3.14-slim` base, pinned `uv`, Python 3.14):

```sh
docker build -f deploy/Dockerfile -t pocket-tts-openai .
docker run --rm -p 8080:8000 \
  -v tts-data:/data \
  pocket-tts-openai
# -> GET  http://localhost:8080/health
#    POST http://localhost:8080/v1/audio/speech
```

Or `docker compose -f deploy/docker-compose.yml up -d --build` (see `deploy/docker-compose.yml`).

- **Runtime user**: non-root (`tts`, uid 10001), exposes `8000`.
- **Persistent volume** `/data`: model weights (`HF_HOME=/data/hf`) +
  cloned voices (`STTS_CACHE_DIR=/data/cache` → `/data/cache/voices`)
  - STT GGUFs (`/data/cache/stt-models`). Mount it so weights are downloaded
  once and custom voices survive restarts. A host bind-mount at `/data` must
  be owned `10001:10001`, else HF-cache/STT-model writes fail (named volumes
  are fine).
- **STT (optional)**: set `STTS_STT_ENABLED=true`. A native
  `whisper-server` binary (built in a dedicated `whispercpp` stage, CPU-only,
  no CUDA) is copied into the runtime image; `ffmpeg` is installed so mp3/
  m4a/webm uploads are decoded. The `small` GGUF (~466 MB) downloads on first
  STT use into `/data/cache/stt-models` (the GGUF stays on disk). The resident
  `whisper-server` (~1 GB RSS for `small`) is auto-killed after
  `STTS_STT_IDLE_UNLOAD_S` of STT silence and lazily re-spawned (no
  re-download) on the next STT request.
- **No CUDA**: `torch` resolves from the PyTorch **CPU** wheel index via
  `uv.lock` (`[tool.uv.sources]`); the whisper.cpp stage builds CPU-only
  (`-DWHISPER_BUILD_SERVER=ON`, no CUDA flag). Nothing NVIDIA layers in.
- The bind host defaults to `0.0.0.0` inside the container (port `8000`);
  override with `STTS_HOST`/`STTS_PORT`.
- Model weights download on first boot (~430 MB); warm the cache beforehand
  with `STTS_WARMUP_VOICES` if you want them pre-encoded at startup.

Image is published to **GHCR** (`ghcr.io/<repo>`) by the
`.github/workflows/docker-publish.yml` pipeline: tests gate the build, then
Buildx publishes `latest` on `main` pushes + semver tags (`v*.*.*`), tagged
`latest`/`<major>.<minor>`/`<version>`, with GHA build-cache and SLSA
attestations. Build locally to verify (no GitHub account needed):

```sh
git tag v0.1.0 && git push origin main --tags  # triggers the workflow
```

## Performance (benchmarked, i7-9750H, 12 threads, 8 GiB, CPU)

| | short (28 ch) | medium (290 ch) |
| --- | --- | --- |
| wall | 1.32 s | 11.5 s |
| audio | 4.48 s | ~33 s |
| RTF | 0.29× | 0.35× |
| RSS floor | ~1.0 GiB (peak 1105 MiB) | — |

- Cold model load: ~22 s; warm voice encode: ~0.01 s (cached). TTFT to first
  streamed chunk: **~0.9 s** on a warm model (not the ~200 ms some docs cite).
- Streaming RTF on this hardware: **0.58×** (24.6 s audio in 14.4 s).

## Known limitations / risks

- One model, one global generation lock — requests serialize (queue-depth is
  surfaced in `/health`). Long streams hold the lock for their whole duration.
- First-chunk latency is ~0.9 s, not ~200 ms (see perf table).
- The streaming WAV header uses the `0xFFFFFFFF` "streaming WAV" convention;
  verified with Python's `wave` module and the chunk math is exact.
- `speed`/`instructions` are accepted for API compatibility but ignored.
- Custom voices are CLI/HTTP-created only; no web UI yet.
- **STT**: CPU-only whisper (real-time-factor on a modest laptop); word-level
  timestamps are best-effort; OpenVINO acceleration is a documented follow-up
  (manual compile flag) — STT ships on native ggml CPU.
- STT is disabled by default; if `STTS_STT_ENABLED=true` and the
  `whisper-server` binary is missing, STT routes return **503** — TTS is
  unaffected.

## Layout

```
src/pocket_tts_openai/
  config.py         Config (env-driven)
  engine.py         TTSEngine: gen lock, voice-state LRU, clone, generate_pcm_stream
  voice_registry.py persistent custom-voice index (<cache>/voices/registry.json)
  voices.py         aliases, catalog, resolve_voice, language tags
  routes_speech.py  /v1/audio/speech, /v1/models, /health
  routes_voices.py  GET/POST/DELETE /v1/voices
  routes_stt.py     POST /v1/audio/transcriptions + /v1/audio/translations
  stt.py            WhisperSidecar: spawn/health check, idle eviction, crash watch, GGUF download, /inference proxy
  server.py         create_app, API-key middleware, background model load + warmup, TTS + STT watchdog
  errors.py         OpenAI-shaped error helpers
tests/              contract + engine + voices + streaming + idle-unload + STT
                    (fake model; STT uses a stubbed sidecar — no network)
deploy/Dockerfile      multi-stage CPU-only image (python:3.14-slim, non-root, whisper.cpp stage)
deploy/docker-compose.yml  commented env reference + local build/run convenience
.github/workflows/  docker-publish.yml -> GHCR (tests gate, buildx, attestations)
.dockerignore       keep .venv/.git out of the build context
```
