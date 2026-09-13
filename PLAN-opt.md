# PLAN — Performance micro-optimizations (3 items)

> **Scope:** three localized, low-risk changes touching hot paths / lock scope in the
> engine plus the compressed-format encode path. No API or dependency changes.
> Plan only — no code written yet.

| # | Site | Problem | Fix | Impact |
| --- | --- | --- | --- | --- |
| 1 | `engine.py` `to_pcm16` | 2 extra chunk-sized allocations per call | in-place `clip`/`multiply` (`out=`) | ~halves allocs on the streaming hot path (savings/chunk tiny, but runs every chunk) |
| 2 | `audio_codecs.py` / `routes_speech.py` | compressed formats buffer TTS, then run a full synchronous ffmpeg round-trip before any byte is sent | `Popen` + stdin/stdout pipes, feed PCM as it streams, surface ffmpeg stdout via `StreamingResponse` | first byte before generation ends; TTS ∥ ffmpeg (wall time ≈ max, not sum) |
| 3 | `engine.py` `clone_voice` | `.safetensors` export + atomic rename run inside `_gen_lock` | release the lock right after `get_state_for_audio_prompt`; export unlocked | a slow disk write during a clone no longer stalls in-flight synthesis |

---

## 1. `to_pcm16` — fewer allocations (`engine.py:73-80`)

Current:

```python
def to_pcm16(audio: "AudioBuffer | np.ndarray") -> bytes:
    if not isinstance(audio, np.ndarray):
        audio = audio.detach().cpu()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    arr = np.clip(arr, -1.0, 1.0)
    return (arr * 32767.0).astype(PCM_DTYPE).tobytes()
```

Per-call allocations today: `asarray` (view when already f32 → usually free),
`clip` (**new buffer**), `* 32767.0` (**new buffer**), `astype` (int16 buffer,
unavoidable), `tobytes` (bytes copy, unavoidable). `detach().cpu()` copies only
when the tensor isn't CPU-f32. So clip + multiply are the two removable
allocations.

**Fix** (write into the existing f32 buffer):

```python
def to_pcm16(audio: "AudioBuffer | np.ndarray") -> bytes:
    if not isinstance(audio, np.ndarray):
        audio = audio.detach().cpu()
    arr = np.asarray(audio, dtype=np.float32).reshape(-1)
    np.clip(arr, -1.0, 1.0, out=arr)
    np.multiply(arr, 32767.0, out=arr)
    return arr.astype(PCM_DTYPE).tobytes()
```

**Contract to document + test:** `to_pcm16` now mutates its input buffer in place
when `arr` is a writable view of it (always the case for the current call sites):

- `generate_pcm`: `audio` is the one-shot result of `model.generate_audio(...)` — consumed once.
- `generate_pcm_stream`: each yielded chunk tensor is converted once and never reused — verified in M4's live measurement (uniform 7680-byte chunks, consumed once).
- Call-site audit shows **no buffer is reused after conversion**, so the in-place
  write is safe. This is the same trade-off the user flagged ("free"); do *not*
  reintroduce a copy to be defensive — that would defeat the point.

**Tests** (`tests/test_engine.py`):

- need `to_pcm16` byte-identical to the old reference implementation on: a f32
  ndarray test vector, values outside `[-1,1]` that must clip, an int16/float64
  input (exercises the dtype-copy branch), and a torch CPU tensor when torch is
  importable (skip otherwise — the default suite has no torch).
- assert the return dtype/length invariants `len == n*2`, little-endian s16.

**Perf footnote (optional, not a gate):** a `timeit` compare (e.g. 5k iterations of
each version) is expected to show ~25–40% fewer allocations/CPU; not needed to
justify landing.

---

## 2. Compressed formats: true streaming through piped ffmpeg

### 2.1 Today

- `routes_speech.py:88-89` — the stream branch is gated `if req.stream and
  req.response_format in ("wav", "pcm")`, so `stream: true` is silently ignored
  for mp3/opus/aac/flac (the `SpeechRequest.stream` Field doc even says
  "Ignored (buffered) for compressed formats").
- Compressed path (`routes_speech.py` ≈ line 118): `pcm = engine.generate_pcm(...)`
  waits for **full** TTS, then `encode_pcm(pcm, ...)` (`audio_codecs.py:79`)
  does a **synchronous** `subprocess.run(input=pcm, capture_output=True)` —
  full TTS wall-time **plus** full ffmpeg wall-time before the response body ends;
  nothing reaches the client early.

### 2.2 Design

**New** in `audio_codecs.py`:

```python
def _ffmpeg_args(sample_rate: int, fmt: str) -> list[str]: ...   # factored from today's builder

def encode_pcm_stream(pcm_iter: Iterator[bytes], sample_rate: int, fmt: str) -> Iterator[bytes]:
    """Encode PCM chunks *as they arrive* via ffmpeg; yield encoded bytes live.
    Pipeline: pump thread writes each PCM chunk to ffmpeg stdin; this generator
    (ASGI worker thread) reads ffmpeg stdout and yields it chunk by chunk."""
```

- `Popen([*_ffmpeg_args(sample_rate, fmt), "-flush_packets", "1"], stdin=PIPE, stdout=PIPE, stderr=PIPE)`.
  `-flush_packets 1` encourages the muxer to flush per-packet instead of buffering
  a big blob — **verify empirically** per format (see 2.4) and keep only if it lowers
  first-byte latency without breaking the byte stream.
- **Pump thread** (daemon): `for chunk in pcm_iter: proc.stdin.write(chunk); proc.stdin.flush()`
  (flush per chunk to keep ffmpeg fed promptly); on `BrokenPipeError`/`OSError` (ffmpeg
  died) exit the loop. **Captures the first non-OSError exception** from `next(pcm_iter)`
  (e.g. unknown voice `ValueError`) into a slot; the reader re-raises it → the route's
  `_speech_error()` / 400 path works for streamed compressed formats too (see 2.5).
- **stderr drain thread** (daemon): `proc.stderr.read(4096)` loop into a `list[bytes]`
  so ffmpeg can never deadlock on a full stderr pipe; join before reading the buffer.
- **Reader loop** (the generator body): `data = os.read(proc.stdout.fileno(), 65536)`;
  `os.read` (not `BufferedReader.read`) returns as soon as *any* bytes are available —
  this is what makes first byte flow early; yield each chunk.
- **Success path:** EOF on stdout → join pump (`timeout=5`), `proc.wait()`, join stderr;
  if `returncode != 0` raise `RuntimeError(stderr)` (route → 400, same as today).
- **Abort path (`GeneratorExit`, client disconnect):** `proc.kill()` → `t_pump.join(5)`
  → **then `pcm_iter.close()`** → `proc.wait()`/close fds → `raise`. Ordering is an
  invariant: after the kill, the pump's blocked `write` gets EPIPE and exits, so by the
  time we join, **no thread is executing `pcm_iter`** — closing it is safe and it is what
  triggers `generate_pcm_stream`'s `finally` to release `_gen_lock`. Never call
  `pcm_iter.close()` while the pump may be inside the generator (would raise
  `ValueError: generator already executing`).

Keep the existing buffered `encode_pcm(pcm, ...)` (subprocess.run) unchanged — it
serves `stream: false` and all current tests; both share `_ffmpeg_args`. Do **not**
reimplement `encode_pcm` over the stream core (that would pay a thread per call for
no benefit).

### 2.3 Route wiring (`routes_speech.py`)

```python
# compressed branch, stream: true  -> true streaming
if req.response_format in COMPRESSED_FORMATS and req.stream:
    engine.check_capacity()                       # 429 pre-flight, like wav/pcm
    pcm_iter = engine.generate_pcm_stream(req.input, req.voice)
    try:
        enc = encode_pcm_stream(pcm_iter, engine.sample_rate, req.response_format)
        return StreamingResponse(
            enc,
            media_type=MEDIA_TYPES[req.response_format],
            headers={"Content-Disposition": f"attachment; filename=speech.{req.response_format}"},
        )
    except ValueError:
        pcm_iter.close() ; raise _speech_error() from None
# stream: false or wav/pcm -> unchanged
pcm = engine.generate_pcm(req.input, req.voice)
...
```

No container-size header trick needed for compressed (ADTS/Ogg/mp3/FLAC don't need a
leading size — unlike the WAV `RIFF` size); just stream ffmpeg stdout, and Starlette
supplies `Transfer-Encoding: chunked` + `Content-Disposition`. `stream: false` keeps
its exact current behavior (buffered `Response` with `Content-Length`).

**Docs to update:** `SpeechRequest.stream` Field description ("…now honored for all
formats; wav/pcm stream raw, compressed stream ffmpeg-encoded"), the module comment
on `audio_codecs.py`, README's `stream` + compressed-format sections, and the
`// streaming path (wav/pcm only…)` comment. (There is **no** runtime warning today —
the guard at `:89` just excludes compressed — so nothing to remove, only the guard to
extend.)

### 2.4 Latency model + measurement

- First encoded byte flows while generation is *still running*: client-perceived time
  to first byte roughly `min(TTS_first_chunk, ffmpeg_buffering) + encode_latency`
  instead of `TTS_total + ffmpeg_total`.
- mp3 (`libmp3lame`) and FLAC buffer more before emitting; AAC (ADTS) and Opus (Ogg)
  are per-frame and stream at low latency. Empirical-first-byte check during
  implementation (real ffmpeg + a few seconds of PCM): record TTFT per format with and
  without `-flush_packets 1`; tune or drop the flag per findings; keep the field
  meaningful regardless (correctness > granularity — a chunked mp3 that flushes every
  ~64KB still removes ffmpeg tail from the *perceived* path only partially; document).
- Record per-format first-byte numbers in the README perf note (feeds M6 benchmark).

### 2.5 Bonus correctness fix this unlocks

Today, **unknown voice on the streamed path surfaces as a 500, not a 400**: the route
catches `ValueError` only around `StreamingResponse(...)` *construction*, but the
generator body (voice resolution) runs at first pull, inside Starlette's stream worker
→ 500. The pump's "capture first exception and re-raise in the reader" makes the
compressed stream path raise the `ValueError` through `encode_pcm_stream` → route 400.
Verify whether the wav/pcm stream path has the same latent issue and, if so, prefer a
single shared fix (surface engine errors) rather than per-format special-casing.

### 2.6 Tests

**Unit — `tests/test_api_contract.py` / a new `tests/test_audio_codecs.py`** (ffmpeg
present → run; absent → skip, mirroring existing practice):

- `encode_pcm_stream` over an in-memory iterator **byte-for-byte equal** to
  `encode_pcm(b"".join(chunks), ...)` on the same binary (mp3 + aac).
- Abandonment: pull 1 chunk of the encoder, stop (close the generator) → no hang, the
  underlying fake source's `finally` ran (`closed` flag → proves the lock-release
  analogue), and `threading.active_count()` returns to baseline (no thread leak).
- Error: source raises mid-way → the generator raises that error; ffmpeg-kill path owns
  cleanup (no deadlock, source closed).

**Route — `tests/test_streaming.py`**:

- **Replace** `test_compressed_format_ignores_stream` with `test_compressed_format_streams`:
  `stream: true` + mp3 → chunked StreamingResponse, `content-type: audio/mpeg`,
  body starts with an mp3 magic (`ID3` or `0xFFFB`), length > 0.
- `stream: false` + compressed → unchanged buffered `Response` (regression; the
  existing `test_speech_compressed_formats_via_ffmpeg` stays green as-is).
- Disconnect mid-compressed-stream releases the engine lock (mirror
  `test_disconnect_releases_lock`).
- **Also `/v1/audio/speech` with `stream: true` + compressed + *unknown voice* → 400**
  (the 2.5 fix) — fake engine raises `ValueError` from `generate_pcm_stream`.

**E2E — `tests/test_integration.py`** (real model, `POCKET_TTS_E2E=1`): add one
`stream: true` + `aac` (low-latency muxer) request → chunked, non-empty, first chunk
arrives before generation completes (elapsed < full buffer time), and a
`stream: false` + `mp3` sanity. Skip if ffmpeg absent.

---

## 3. `clone_voice`: export outside `_gen_lock` (`engine.py` ≈ 431-458)

Current (lock held through encode **and** export + atomic rename):

```python
with self._gen_lock:
    state = model.get_state_for_audio_prompt(str(audio_path))
    tmp = registry.directory / f".{name}.{uuid4().hex}.safetensors.tmp"
    try:
        self._exporter()(state, tmp)      # pure I/O: reads the returned snapshot
        tmp.replace(dest)
    finally:
        tmp.unlink(missing_ok=True)
```

Why encode *must* stay locked: `get_state_for_audio_prompt` touches the stateful
batch=1 model (`voice_state()` slow path / generation must not race it).
Why export may leave the lock: `export_model_state(state, dest)` flattens the
**returned** `ModelState` dict (memory-only serialization, verified against the
installed pocket-tts API) — it does not touch the live model. The tmp-write +
`os.replace` are plain disk I/O.

**Fix:**

```python
with self._gen_lock:
    state = model.get_state_for_audio_prompt(str(audio_path))
tmp = registry.directory / f".{name}.{uuid4().hex}.safetensors.tmp"
try:
    self._exporter()(state, tmp)
    tmp.replace(dest)
finally:
    tmp.unlink(missing_ok=True)
```

**Safety re-check:**

- `state` is a strong local ref → a concurrent `unload()` (idle eviction) during the
  export writes the snapshot from memory and is harmless; the final LRU insert lands in
  whatever LRU exists. (Eviction is additionally gated on `idle_unload_s` idle, and a
  clone just called `touch()`.)
- The LRU insert stays where it is (after success, outside both locks) → preserves
  "persisted **and** served" semantics; a miss racing the same custom name may re-encode
  in a tiny window (rare, negligible — clone requests are one-off).
- **Stale docstring:** `clone_voice` says "the caller (POST /v1/voices) acquires it via
  the shared `generate` path" — inaccurate (lock is acquired in-body at `:439`); update
  while here.

**Tests** (`tests/test_engine.py` / `tests/test_routes_voices.py`), fake engine:

1. Fake `export_state` records that `engine._gen_lock.acquire(blocking=False)` succeeds
   *inside the exporter* on a free engine → proves the lock is released before export.
2. Concurrency: clone runs with a slow exporter (blocks on a `threading.Event`); a
   `generate_pcm` started after the encode finishes **completes before** the exporter
   unblocks → no cross-cost coupling.
3. Existing clone happy-path/400/413/hygiene tests stay green (export/rename still
   atomic + temp cleanup on failure).

---

## 4. Files + test inventory

| File | Change |
| --- | --- |
| `src/pocket_tts_openai/engine.py` | `to_pcm16` in-place ops (+contract docstring); `clone_voice` lock scope (+docstring) |
| `src/pocket_tts_openai/audio_codecs.py` | add `_ffmpeg_args`, `encode_pcm_stream` (pump + stderr threads, `-flush_packets 1`); keep `encode_pcm` |
| `src/pocket_tts_openai/routes_speech.py` | compressed `stream:true` → `StreamingResponse(encode_pcm_stream(...))`; update `stream` Field doc; extend stream-branch guard |
| `tests/test_engine.py` | `to_pcm16` parity tests; clone lock-scope + concurrency tests |
| `tests/test_api_contract.py` (or new `test_audio_codecs.py`) | `encode_pcm_stream` ≡ `encode_pcm`; abandonment/thread hygiene; error path |
| `tests/test_streaming.py` | replace `test_compressed_format_ignores_stream`; add streamed-compressed 400 / disconnect / regression cases |
| `tests/test_integration.py` | real-model compressed streaming (aac) + buffered mp3 sanity |
| `README.md` | `stream` field + compressed-format streaming note; per-format first-byte note |
| `PLAN-opt.md` | this plan; flip to IMPLEMENTED with deviations at the end |

---

## 5. Execution order

1. **Item 1** (`to_pcm16`) — trivial, isolated → suite + pyright.
2. **Item 3** (`clone_voice`) — trivial, isolated → suite + pyright.
3. **Item 2** (ffmpeg streaming) — the substantive change, own test slice → suite +
   pyright, then `POCKET_TTS_E2E=1 uv run pytest -m e2e` (real-model validation, incl.
   the new compressed-streaming e2e). Commit each item separately (or squash 1+3 into a
   "perf/locks" commit and keep 2 separate) so the ffmpeg mechanics are reviewable in
   isolation.

Suite baseline: **110 mocked + 6 e2e pass, pyright 0/0/21 files** — must stay non-flaky
(compressed e2e skips when ffmpeg missing).

---

## 6. Risks / open questions

- **ffmpeg flush variance:** `-flush_packets 1` may not shrink mp3/FLAC first-chunk
  granularity much (muxer/buffering is format-dependent). Correctness holds regardless;
  first-byte benefit is format-dependent → document per-format TTFT instead of a hard
  latency guarantee (aligns with M6 benchmark).
- **Thread hygiene:** pump + stderr daemon threads must be joined on *both* success and
  abort (bounded timeouts) or a long-lived server leaks a thread per stream. Tested via
  `threading.active_count()` baseline check.
- **Cross-thread generator close:** only ever `pcm_iter.close()` after the pump thread
  is joined (kill → EPIPE → pump exits); do not close while the pump could be executing
  the generator.
- **`to_pcm16` input mutation:** safe at today's call sites; documented as a consumed-once
  contract. If a future caller reuses a buffer, it must copy first (noted in docstring).
- **Latent 500-vs-400 for unknown voice on streamed formats** — fixed for compressed via
  pump exception forwarding; audit wav/pcm stream path (test) and unify if affected.
- **`stream: false` compressed overlap** (start ffmpeg during generation but return
  buffered bytes, `Content-Length` preserved) is a drop-in later win via
  `encode_pcm_stream` + `b"".join` — explicitly **out of scope** here to keep the
  buffered path byte-identical to today.

---

## 7. STATUS — IMPLEMENTED (all 3 items + tests + e2e + README + commit)

Landing notes + deviations from the plan above, so the record matches the code:

1. **Item 1 (`to_pcm16` in-place)** — done with contract docstring. New tests:
   `test_to_pcm16_*` in `tests/test_engine.py` (parity vs the pre-optimization oracle,
   dtype-copy branch NOT mutating the caller's f64/int16 buffer, length/endianness,
   torch path via `importorskip`).

2. **Item 3 (`clone_voice` lock scope)** — done. Only the encode holds
   `_gen_lock`; export + atomic rename run outside. New tests: exporter asserts
   `engine._gen_lock.acquire(blocking=False)` succeeds *inside the exporter*, and a
   slow (event-blocked) exporter does **not** block a concurrent `generate_pcm`.
   README's clone paragraph updated to match.

3. **Item 2 (compressed true streaming)** — done with two deviations from the draft:
   - **§2.3 wiring deviated:** catching `ValueError` around `StreamingResponse(...)`
     construction cannot work (generator bodies run lazily). Instead the route
     pre-validates with **eager `engine.voice_state(req.voice)`** for the *entire*
     streaming branch (wav/pcm **and** compressed) → clean 400 before headers on
     unknown voice, resolving §2.5 for wav/pcm too (single shared fix). The pump's
     first-exception forwarding (the plan's mechanism) is kept as defense-in-depth
     for any mid-stream source error.
   - **Measured first-byte timing:** ffmpeg's encoder+muxer priming means the *first*
     encoded packet flows after ~0.4 s of audio (measured 0.37 s for aac & mp3), then
     streams incrementally. So "true streaming" = bytes flow long before generation
     ends, not zero-latency; tests assert "first output while the source is still
     mid-feed" with a 60 s margin instead of a wall-clock gate. mp3/aac/flac are
     **byte-identical** to the buffered output and finely chunked (371/228/101 chunks
     for 10 s); opus streams at **Ogg-page** granularity (~11 chunks/10 s, bytes differ
     from buffered due to page boundaries — semantically identical, verified by
     decoding both to raw PCM) — documented in README.
   - One extra guard: on abandonment the pump is joined (kill → EPIPE) **before**
     `pcm_iter.close()`, per the §3 invariant; thread hygiene asserted via
     `threading.active_count()` back to baseline.

   New tests: `tests/test_audio_codecs.py` (7: byte parity mp3/aac/flac, opus
   decode-parity, first-bytes-before-exhausted, source-error forwarding, abandonment
   + thread reaping, missing-ffmpeg, unknown format); `tests/test_streaming.py`
   (routes through the streaming generator multiple times + byte-identical to
   buffered, streamed unknown-voice 400 for wav *and* mp3, disconnect releases lock,
   opus OggS valid); `test_integration.py` gains `test_e2e_streaming_aac_chunked`
   (real model, chunked + ADTS sync; both wav-chunked and aac e2e kept).

4. **Environment:** `ffmpeg` was not installed locally — `apt-get install -y ffmpeg
   (5.1.9)` added so the compressed tests exercise the real subprocess pipeline
   (the Docker image ships ffmpeg anyway).

5. **Gates:** full mocked suite `127 passed, 6 skipped (e2e opt-in)`; real-model e2e
   `7 passed`; `pyright 0 errors / 0 warnings / 0 informations`. Committed as one
   `feat(perf)` commit: in-place `to_pcm16`, `clone_voice` lock scope, and piped-ffmpeg
   streaming for compressed formats.

**Perf footnote (measured, not a gate):** mp3/aac/flac now start streaming before
synthesis completes (previously full TTS + full ffmpeg before any byte). Client TTFT ≈
`max(TTS_streambegin, 0.4 s encoder priming)`, i.e.
`TTS_total + ffmpeg_total` → `≈ TTS_total`.
