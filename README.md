# pocket-tts-openai

OpenAI-compatible TTS server (`POST /v1/audio/speech`, `stream` chunked PCM/WAV)
powered by [Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) —
100M-param speech synthesis on CPU.

> Work in progress — see `./PLAN.md`. **M1** (server skeleton), **M2**
> (audio formats), **M3** (voice catalog + cloning `GET/POST/DELETE /v1/voices`)
> and **M4** (streamed `wav`/`pcm`) are implemented.

## Dev

```sh
uv sync                                # deps without the engine (tests use a fake model)
uv run pytest                          # 70 tests
uv sync --extra engine                 # + pocket-tts (CPU-only torch on Linux, see pyproject)
uv run pocket-tts-openai               # serve on :8000
```

To run on all interfaces / another port (as used by the live server):
`POCKET_TTS_HOST=0.0.0.0 POCKET_TTS_PORT=8080 uv run pocket-tts-openai`.

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

`/v1/models` lists the model aliases. `/health` is liveness + engine stats
(`status`, `language`, `requests`, `avg_rtf`, `queue_depth`); it reports
`loading` until the model is ready.

### `GET/POST/DELETE /v1/voices` (private extension)

Catalog + voice cloning.

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
  `POCKET_TTS_MAX_UPLOAD_MB` (default 25). **400** on unknown extension.
- `language` (optional): free-form tag, stored and echoed.

Returns **201** with the custom-voice payload after the audio prompt is encoded
and exported to `<cache_dir>/voices/<name>.safetensors`. The encode runs under
the global generation lock — a clone blocks concurrent synthesis requests.

**`DELETE /v1/voices/{name}`** — **405** for built-in voices; **404** for
unknown; **204** on success (removes the registry entry, the `.safetensors`
file, and its LRU cache slot).

Custom voices are immediately usable as `voice` in `/v1/audio/speech`.

## Configuration (env)

| var | default | notes |
| --- | --- | --- |
| `POCKET_TTS_HOST` | `127.0.0.1` | bind host |
| `POCKET_TTS_PORT` | `8000` | bind port |
| `POCKET_TTS_LANGUAGE` | `english` | model language |
| `POCKET_TTS_QUANTIZE` | `false` | quantize the model |
| `POCKET_TTS_MAX_CACHED_VOICES` | `32` | LRU voice-state cache size |
| `POCKET_TTS_WARMUP_VOICES` | — | comma-separated voices to pre-encode at boot |
| `POCKET_TTS_API_KEY` | — | if set, `Bearer <key>` required on `/v1/*` |
| `POCKET_TTS_CACHE_DIR` | `~/.cache/pocket_tts` | base for cloned voice registry |
| `POCKET_TTS_MAX_UPLOAD_MB` | `25` | max cloned-voice upload size |

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

Or `docker compose up -d --build` (see `docker-compose.yml`).

- **Runtime user**: non-root (`tts`, uid 10001), exposes `8000`.
- **Persistent volume** `/data`: model weights (`HF_HOME=/data/hf`) + the
  voice registry (`POCKET_TTS_CACHE_DIR=/data/cache` → `/data/cache/voices`).
  Mount it so weights are downloaded once and cloned voices survive restarts.
- **No CUDA**: `torch` resolves from the PyTorch **CPU** wheel index via
  `uv.lock` (`[tool.uv.sources]`); nothing NVIDIA layers into the image.
- The bind host defaults to `0.0.0.0` inside the container (port `8000`);
  override with `POCKET_TTS_HOST`/`POCKET_TTS_PORT`.
- Model weights download on first boot (~430 MB); warm the cache beforehand
  with `POCKET_TTS_WARMUP_VOICES` if you want them pre-encoded at startup.

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

## Layout

```
src/pocket_tts_openai/
  config.py         Config (env-driven)
  engine.py         TTSEngine: gen lock, voice-state LRU, clone, generate_pcm_stream
  voice_registry.py persistent custom-voice index (<cache>/voices/registry.json)
  voices.py         aliases, catalog, resolve_voice, language tags
  routes_speech.py  /v1/audio/speech, /v1/models, /health
  routes_voices.py  GET/POST/DELETE /v1/voices
  server.py         create_app, API-key middleware, background model load + warmup
  errors.py         OpenAI-shaped error helpers
tests/              contract + engine + registry + voices + streaming (fake model)
deploy/Dockerfile   multi-stage CPU-only image (uv, python:3.14-slim, non-root)
docker-compose.yml  local build/run convenience
.github/workflows/  docker-publish.yml -> GHCR (tests gate, buildx, attestations)
.dockerignore       keep .venv/.git out of the build context
```
