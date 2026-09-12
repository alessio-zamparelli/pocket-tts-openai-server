# M3 + M4 Implementation Plan — Voices & Streaming

> Detailed implementation plan for the two selected milestones of `./PLAN.md`,
> grounded in the current codebase (post-M1) and the **installed** pocket-tts
> 3.1.0 API. Plan only — no code written yet.

## 0. Corrections to PLAN.md (verified against installed pocket-tts 3.1.0)

1. **`export_model_state` is module-level, not a method.**
   `TTSModel.export_model_state` does not exist. Use:
   `from pocket_tts import export_model_state; export_model_state(model_state, dest)`.
   It flattens `dict[str, dict[str, Tensor]]` into `"<module>/<key>"` safetensors keys.
2. **Reload is free:** `TTSModel.get_state_for_audio_prompt()` accepts a
   `.safetensors` path directly ("much faster than extracting from an audio file").
   So persistence = `export_model_state(state, path)`; reload = `voice_state(path)`.
   No custom load code needed.
3. **Streaming generator confirmed:**
   `generate_audio_stream(model_state, text, max_tokens=50, frames_after_eos=None,
   copy_state=True) -> Iterator[Tensor]`. It is a *sync* generator → abandoning it
   (breaking the consumer loop) stops frame generation at the next pull, which is
   our cancellation mechanism for M4.
4. **`warmup_voices` config exists but is unused** — M3 wires it up.

---

## 1. M3 — Voices (`GET/POST/DELETE /v1/voices`)

### 1.1 New file `src/pocket_tts_openai/routes_voices.py`

Three route functions registered in `server.py::create_app`:

```python
app.get("/v1/voices")(list_voices)
app.post("/v1/voices")(create_voice)          # multipart/form-data
app.delete("/v1/voices/{name}")(delete_voice)
```

### 1.2 Custom-voice registry (new file `src/pocket_tts_openai/voice_registry.py`)

- Directory: `<cache_dir>/voices/` where `cache_dir` defaults to
  `~/.cache/pocket_tts` (mirrors pocket-tts). Make it overridable via
  `POCKET_TTS_CACHE_DIR` so tests use tmp_path.
- `registry.json` next to the safetensors files:
  `{"voices": [{"name": "mario", "language": "it", "created": "…iso8601…"}]}`
- API: `load() -> list[CustomVoice]`, `add(name, language)`, `remove(name)`,
  `path_for(name) -> Path` (`<dir>/<name>.safetensors`).
- Atomic writes (write temp + `os.replace`) so a crash can't corrupt it.
- On startup, registry names are injected into the *effective* voice map as
  `name -> "<cache_dir>/voices/<name>.safetensors"` (fast `.safetensors`
  reload path). Not written back to `Config.voice_map` — a separate
  `engine.custom_names: set[str]`, consulted by `voice_state()` **before**
  `resolve_voice`, avoids polluting config and surviving DELETE correctly.

**Resolution order in `TTSEngine.voice_state()` becomes:**

1. custom registry name → safetensors path
2. `Config.voice_map` (aliases)
3. Kyutai catalog / passthrough (`resolve_voice` as today)

### 1.3 `GET /v1/voices`

Response shape per PLAN.md. Builtin entries = OpenAI aliases (one entry per
alias, `aliases: [alias, kyutai_name]`) **plus** one entry per extra catalog
voice not already aliased. Custom entries from the registry.

```json
{"object": "list", "data": [
  {"id": "alloy", "aliases": ["alloy", "alba"], "source": "builtin",
   "language": "en", "license": "<hf url>", "cached": true},
  {"id": "mario", "aliases": ["mario"], "source": "custom",
   "language": "it", "license": null, "cached": false, "safetensors": true}
]}
```

- `cached`: new `TTSEngine.cached_voices() -> frozenset[str]` (read under
  `_state_lock`).
- `license`: per-voice HF blob URL. **Needs verification** — the exact
  per-voice paths in `kyutai/tts-voices` must be checked; fallback to the
  repo root URL (`https://huggingface.co/kyutai/tts-voices`) for every builtin
  voice if per-voice paths can't be derived. Keep a `VOICE_LICENSES` map in
  `voices.py` so it's one place to fix.
- Sorting: builtin first (stable order), then custom by name.

### 1.4 `POST /v1/voices` (multipart/form-data)

| Field | Rules |
| --- | --- |
| `name` | required; `^[a-z0-9][a-z0-9_-]{0,63}$`; **409** if it collides with an OpenAI alias, a Kyutai catalog name, an env-mapped alias, or an existing custom voice (PLAN.md said 400; 409 communicates the conflict better — use 409) |
| `file` | required; extension decides format: `.wav`/`.mp3`/`.flac`; **413** if > `POCKET_TTS_MAX_UPLOAD_MB` (default 25); **400** on missing/unknown extension |
| `language` | optional; free-form tag, stored and echoed only |

Flow (new `TTSEngine.clone_voice(name, tmp_path, language)`):

1. Validate + write upload to a temp file inside the cache dir (same fs).
2. **Acquire the global generation lock** while encoding
   (`get_state_for_audio_prompt` touches the stateful batch=1 model — the
   current `voice_state()` slow path runs encoding *outside* all locks, which
   is fine for read-mostly startup traffic but cloning must not race an
   in-flight generation). Document that a clone blocks other requests.
3. `export_model_state(state, path_for(name))`.
4. Insert into LRU cache, append to registry (atomic), delete temp file.
5. Respond `201` with the custom-voice payload.

Errors: OpenAI error format (existing `errors.py` helpers; add
`conflict()`, `payload_too_large()`).

Client-facing note: the encode step is the slow part (seconds); respond only
after it completes (simplest correct v1). No background jobs.

### 1.5 `DELETE /v1/voices/{name}`

- Builtin alias or catalog name → **405** `builtin voices cannot be deleted`.
- Unknown → **404**.
- Custom: evict from LRU (`_state_lock`), remove safetensors file, remove
  registry entry. If a request is mid-generation with that state, the LRU
  eviction just drops the reference — in-flight generation is unaffected.
- Respond `204`.

### 1.6 Warmup wiring (closes the gap in §0.4)

In `server.py::_load_engine_background`, after `load_engine(config)`:
for each voice in `config.warmup_voices` → `engine.voice_state(v)`,
catching and logging per-voice failures (startup must not die from one bad
voice). Same loop serves both prefetch-at-boot and the future
`pocket-tts-openai warmup` CLI (M5).

### 1.7 Tests (`tests/test_routes_voices.py`, fake engine)

- GET shape: builtin entries + empty custom list; `cached` flips after a
  speech request with that voice.
- POST: 201 happy path (fake engine's `clone_voice` returns canned state),
  400 empty/missing name, 409 collision with `alloy`, 413 oversized
  (monkeypatch limit), 400 unknown extension.
- DELETE: 204 removes entry + file (tmp_path), 405 builtin, 404 unknown.
- Registry persistence: add → reload registry from disk in a fresh app →
  custom voice resolvable in `voice_state()`.
- `_load_engine_background` warmup: one bad voice in env doesn't crash startup.

---

## 2. M4 — Streaming (`generate_audio_stream`, chunked wav/pcm)

### 2.1 Engine: `TTSEngine.generate_pcm_stream(text, voice) -> Iterator[bytes]`

- Resolve voice + acquire `_gen_lock` **for the whole stream duration**
  (serializes with everything else, as decided in PLAN.md §2). `_QueueGuard`
  wraps the lock acquisition so queue-depth stats stay honest.
- Yield `to_pcm16(chunk)` per pocket-tts frame (24 kHz mono s16le;
  ~50 tokens/frame → a few hundred KB per chunk at most; measure real chunk
  size in implementation).
- Cancellation: the route's generator will be closed by Starlette on client
  disconnect (`GeneratorExit`) → `break`/close the pocket-tts iterator →
  frame loop stops at next pull. Residual in-flight frame completes but the
  lock is released immediately. Document this in README (§9 risk list).
- Stats: `requests += 1`, `generate_seconds += (t_end - t_start)` after the
  generator is exhausted or closed; `audio_seconds +=` accumulated per chunk.
  RTF is only final once the stream ends — health reflects it after the fact.

### 2.2 Route: `stream` field (private extension, default `false`)

Add to `SpeechRequest`:

```python
stream: bool = Field(default=False,
    description="Chunked transfer for wav/pcm. Ignored (buffered) for compressed formats.")
```

Behavior matrix in `speech()`:

| format | stream=false | stream=true |
| --- | --- | --- |
| `pcm` | buffered bytes (today) | `StreamingResponse(pcm chunks, media_type="audio/pcm")` |
| `wav` | buffered bytes (today) | `StreamingResponse(wav header + chunks)` |
| `mp3/opus/aac/flac` | (M2) buffered | buffered, log warning "stream not available for X" |

### 2.3 WAV streaming header trick

WAV needs byte counts in `RIFF`/`data` headers that are unknown up front.
Standard workaround (Plan): emit a header with
`riff_size = data_size = 0xFFFFFFFF` ("streaming WAV", tolerated by ffplay,
VLC, browsers, and the `wave` module only when it uses seekable-less reads —
verify each). Alternatives if verification fails:
(a) header only when format=`pcm` (already raw), (b) header-patch by
re-generating? No — accept the 0xFFFFFFFF convention and document the
supported players. **Action item during implementation: verify with
`ffprobe`, Chrome `<audio>`, and Python `wave` on a non-seekable stream.**

Content-Disposition + `Transfer-Encoding: chunked` come free from
Starlette; do **not** set `Content-Length`.

### 2.4 First-token latency target

PLAN.md cites ~200 ms to first chunk. During implementation, measure
TTFT (time to first chunk) with the benchmark harness from
`/tmp/tts_bench.py` extended with a streaming client; record in README
(perf table). This is a measurement task, not a hard gate.

### 2.5 Tests (`tests/test_streaming.py`, fake engine)

- Fake engine: `generate_pcm_stream` yields 3 known chunks.
- `stream=true, pcm`: chunked body == concatenation of chunks; media type.
- `stream=true, wav`: RIFF header present, `data` size == 0xFFFFFFFF,
  decodable when sizes patched.
- `stream=false`: unchanged byte-for-byte vs today (regression).
- `stream=true` with compressed format: buffered response + warning log.
- Disconnect: close the response early; assert lock released
  (fake engine tracks `GeneratorExit`/close) and next request still works.
- Concurrency: two concurrent streams serialize (second completes after
  first), consistent with the global lock decision.

---

## 3. Suggested execution order

1. **M3.2 registry** (pure module + tests, no HTTP) →
2. **M3.1+M3.3 routes GET** →
3. **M3.4 POST + engine.clone_voice** →
4. **M3.5 DELETE** →
5. **M3.6 warmup wiring** →
6. **M4.1 engine stream** (can be developed in parallel with 2–5; touches
   `engine.py` only) →
7. **M4.2 route + M4.3 wav header** →
8. Docs: README endpoint reference + §9 risk notes; update `./PLAN.md`
   milestone checkboxes.

Touch list: new `routes_voices.py`, `voice_registry.py`;
modified `engine.py`, `server.py`, `routes_speech.py`, `voices.py`,
`config.py` (`POCKET_TTS_CACHE_DIR`, `POCKET_TTS_MAX_UPLOAD_MB`);
new tests as above. No new dependencies (multipart via existing
`python-multipart` from pocket-tts; verify it's a direct dep or add it).

Estimate: M3 ≈ 2–3 focused sessions, M4 ≈ 1–2 (the WAV-header verification
is the only genuinely unknown bit).
