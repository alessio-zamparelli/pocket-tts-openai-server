# pocket-tts-openai

OpenAI-compatible TTS server (`POST /v1/audio/speech`, `stream` chunked PCM/WAV
or live ffmpeg-encoded mp3/opus/aac/flac) powered by
[Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) — 100M-param
speech synthesis on CPU.

> Work in progress — see `./PLAN.md`. **M1** (server skeleton), **M2**
> (audio formats), **M3** (voice catalog + cloning `GET/POST/DELETE /v1/voices`),
> **M4** (streamed output incl. live ffmpeg encoding for compressed formats) and
> **M5.5** (idle RAM eviction) are implemented.

## Dev

```sh
uv sync                                # deps without the engine (tests use a fake model)
uv run pytest                          # 127 tests
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
| `response_format` | string | `wav` | `wav` \| `pcm` \| `mp3` \| `opus` \| `aac` \| `flac` (compressed need ffmpeg; see below) |
| `stream` | bool | `false` | **private extension** — chunked transfer for `wav`/`pcm`; compressed formats are encoded through ffmpeg live (no whole-file buffering) |
| `speed` | float | — | accepted, ignored |
| `instructions` | string | — | accepted, ignored |
| `language` | string | — | server-configured language always wins; logged and ignored |

`stream: true` returns a `StreamingResponse` for every format:

- `response_format=pcm` → `audio/pcm`, raw mono s16le chunks.
- `response_format=wav` → `audio/wav` with a **streaming WAV header**
  (`RIFF`/`data` sizes = `0xFFFFFFFF`) followed by PCM chunks — renders
  incrementally in ffplay/VLC and most browsers.
- `response_format=mp3|aac|flac` → the compressed format, but encoded **live**:
  PCM chunks are piped into an ffmpeg subprocess as they are generated
  (`encode_pcm_stream`) and the encoded bytes flow incrementally, so the first
  audio arrives before synthesis finishes — with byte-identical output to the
  buffered (non-streaming) response.
- `response_format=opus` → also streamed live, but in **Ogg pages** (~a page
  per second at 96 kbps), so the chunk granularity is coarser than the other
  formats — inherent to the Ogg/Opus muxing.

`Content-Disposition` is `attachment; filename=speech.{format}`.

Compressed formats (`mp3`/`opus`/`aac`/`flac`) are encoded through **ffmpeg**
(`audio_codecs.py`) with the OpenAI content types (`audio/mpeg`, `audio/ogg`,
`audio/aac`, `audio/x-flac`). `stream: false` (the default) buffers the whole
file in memory first; `stream: true` encodes live as described above. Without
`ffmpeg` on PATH they return a clear 400; **the Docker image ships ffmpeg**, so
they just work there — locally run `apt-get install ffmpeg` (or your package
manager).

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
and exported to `<cache_dir>/voices/<name>.safetensors`. The audio-prompt
*encode* runs under the global generation lock; the `.safetensors` export runs
outside it, so a slow disk delays only the clone, never a concurrent synthesis
request.

**`DELETE /v1/voices/{name}`** — **405** for built-in voices; **404** for
unknown; **204** on success (removes the registry entry, the `.safetensors`
file, and its LRU cache slot).

Custom voices are immediately usable as `voice` in `/v1/audio/speech`.

**Persisted custom voices.** Cloned voices are stored in `<cache_dir>/voices/`
(`POCKET_TTS_CACHE_DIR`, default `~/.cache/pocket_tts`) as `<name>.safetensors`
plus a `registry.json` index, and are re-loaded from disk at every boot — they
survive restarts without re-cloning. The voice is served by its id like any
other (a custom voice name is just another valid `voice` value).

Example — serve the previously cloned Italian voice `mtc` (saved from the
`audio files/` corpus):

```sh
POCKET_TTS_LANGUAGE=italian_24l uv run python -m pocket_tts_openai.server
```

```sh
curl -s localhost:8000/v1/audio/speech -H 'content-type: application/json' \
  -d '{"model":"tts-1","voice":"mtc","input":"Questa è la voce salvata."}' -o mtc.wav
```

The cloned voice only makes sense with the matching model language it was encoded
from (`italian_24l` here). Pre-encode it at boot with `POCKET_TTS_WARMUP_VOICES=mtc`.
Uploads accepted as `.mp3`/`.flac` need `soundfile` at runtime (not installed);
the zero-dependency workaround is to upload a 16-bit PCM WAV.

### Idle RAM reclamation

After `POCKET_TTS_IDLE_UNLOAD_S` (default 300 s = 5 min) with **no API
requests**, the model is dropped from RAM (~60% of the resident footprint) to
keep an idle API cheap. The next request **blocks while it reloads** (~1 s warm
from the HF cache) instead of returning 503. Set `POCKET_TTS_IDLE_UNLOAD_S=0`
to keep the model resident always. `/health` exposes `loaded`, `unloads`,
`reloads` and `last_request_age_s`.

### Model size & RAM

pocket-tts ships **no whisper-style model family** — there is no smaller
`tiny`/`nano` to pick. Each language has a single **base model** (6-layer,
~430 MB download), and that's already the smallest available. Some languages
also offer a `_24l` **quality-upgrade** variant (24-layer, larger + slower):
`english_2026-04_24l`, `french_24l`, `german_24l`, `spanish_24l`. The model
tier is chosen by `POCKET_TTS_LANGUAGE`; there is no size knob below "base".

The sharpest RAM lever is `POCKET_TTS_QUANTIZE=1`, which loads the
transformer's attention/FFN weights as **int8** (dynamic quantization,
FBGEMM on x86; requires AVX2) instead of fp32. Measured at rest with the
model resident (`/health` → `loaded:true`):

| `POCKET_TTS_QUANTIZE` | resident RSS |
| --- | --- |
| `false` (default) | ~1,042 MB |
| `1` | ~395 MB (≈ 62% less) |

Quantization also speeds up inference ~27% on x86 with no measurable quality
change (WER unchanged). For even lower counts, `pip install
pocket-tts[quantize]` swaps in the `torchao` backend. Paired with idle
eviction (previous section) an idle process holds ~0 MB of model.

## Configuration (env)

| var | default | notes |
| --- | --- | --- |
| `POCKET_TTS_HOST` | `127.0.0.1` | bind host |
| `POCKET_TTS_PORT` | `8000` | bind port |
| `POCKET_TTS_LANGUAGE` | `english` | model language — base 6-layer model; `*_24l` variants are larger quality upgrades |
| `POCKET_TTS_QUANTIZE` | `false` | int8 weights: ~62% less resident RAM + ~27% faster on x86 (FBGEMM, needs AVX2) |
| `POCKET_TTS_MAX_CACHED_VOICES` | `32` | LRU voice-state cache size |
| `POCKET_TTS_WARMUP_VOICES` | — | comma-separated voices to pre-encode at boot |
| `POCKET_TTS_IDLE_UNLOAD_S` | `300` | evict the model from RAM after this many idle seconds; `0` disables |
| `POCKET_TTS_IDLE_POLL_S` | `30` | idle-eviction watchdog poll interval (seconds) |
| `POCKET_TTS_MAX_WAITING` | `0` | load shedding: HTTP 429 when more than this many requests are in-flight (encode+wait+generate); `0` = unlimited |
| `POCKET_TTS_QUEUE_TIMEOUT_S` | `0` | max seconds a request waits for the generation lock before a 429; `0` = wait forever |
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
- **Ships the pre-cloned Italian voice `mtc`**: the image seeds the registry
  (`/data/cache/voices`) from the repo `voices/` dir, so a fresh container
  already lists `mtc` in `GET /v1/voices` — no runtime clone needed. To serve
  it, run with the matching model language:

  ```sh
  docker run --rm -p 8080:8000 -e POCKET_TTS_LANGUAGE=italian_24l \
    -v tts-data:/data pocket-tts-openai
  # GET  http://localhost:8080/v1/voices     -> lists "mtc" (custom, it)
  # POST http://localhost:8080/v1/audio/speech \
  #      -d '{"model":"tts-1","voice":"mtc","input":"Ciao"}'
  ```

  `italian_24l` weights come from the **gated** `kyutai/pocket-tts` repo: on a
  cold `/data` volume the container downloads them at first boot, so pass
  `-e HF_TOKEN=...` (or mount a volume that already has them cached under
  `/data/hf`). The voice file itself is our own cloned state (10 MB) and is
  always in the image — no token needed for the voice, only for the model
  weights.
- **No CUDA**: `torch` resolves from the PyTorch **CPU** wheel index via
  `uv.lock` (`[tool.uv.sources]`); nothing NVIDIA layers into the image.
- **ffmpeg included**: the runtime image installs `ffmpeg`, so the compressed
  `/v1/audio/speech` formats (`mp3`/`opus`/`aac`/`flac`) work out of the box
  (Debian's build ships the libmp3lame + libopus encoders).
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

```text
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
