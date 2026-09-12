# PLAN — RAM footprint: idle model eviction

> **Goal:** when the API has no requests for a configurable window (default 5 min),
> drop the in-RAM pocket-tts model so RSS drops from ~1.0 GB to ~0.4 GB. The next
> request **blocks until the model is reloaded** (warm disk cache ≈ 1 s; cold/network
> path can hold for many seconds).

## Measured baseline (this machine, CPU torch, /proc VmRSS)

| State | RSS |
| --- | --- |
| empty python | ~32 MB |
| torch + sentencepiece imported (no model) | ~229 MB |
| engine loaded (model + voice LRU) | ~1030 MB |
| after dropping model + `gc.collect()` | ~430 MB |
| after `gc.collect()` + `malloc_trim(0)` | ~405 MB |
| warm reload (model from HF cache) | ~0.9 s, back to ~1030 MB |

**Takeaway:** ~600 MB (≈60%) is reclaimable by evicting the in-process model.
The hard floor is ~230 MB (loaded libraries) → ~400 MB with allocator slack;
below that requires a full process restart (≈30 MB), which we consider out of
scope (kills in-flight lines, cold-start 20 s+, needs a supervisor).

## Design: in-process eviction + lazy reload (recommended)

Keep the process alive; evict only the heavy object.

### 1. Split model creation from the engine (`engine.py`, `load_engine`)

- `build_model(config) -> TTSModelLike` — the current `TTSModel.load_model(...)` block.
- `load_engine(config)` keeps today's signature but constructs the engine with a
  `loader: Callable[[], TTSModelLike]` (the build closure) so it can rebuild later.

### 2. `TTSEngine` changes (`engine.py`)

- Field `_model: TTSModelLike | None` (currently always set).
- `loader: Callable[[], TTSModelLike] | None` — present only in production
  (tests inject fakes; with no loader, eviction is disabled).
- `last_activity: float` (monotonic), `idle_unload_s: int` from config.
- `touch()` — bump `last_activity`; called at the top of every model-touching
  entry point (`generate_pcm`, `generate_pcm_stream`, `voice_state`, `clone_voice`).
- `maybe_unload(now)` — if `idle_unload_s>0`, loaded, enabled, and
  `now - last_activity >= idle_unload_s`, then:
  - acquire `_gen_lock` **non-blocking** (skip this pass if a stream is running),
  - clear the LRU `_voice_states` + set `_model = None`,
  - `gc.collect()` then `ctypes libc.malloc_trim(0)` (best-effort, ~25 MB extra).
- `ensure_loaded()` — if `_model is None`:
  - with a single-flight lock, call `loader()` (blocks only the *reloading*
    thread; the request side just needs the engine to report `ready`),
  - set `_model`, re-`warmup()` the configured warmup voices, restore `sample_rate`.
- `loaded -> bool` property; `last_activity`/`unloads` exposed for `/health`.
- Stats: add `unloads`/`reloads` counters (observability).

### 3. Watchdog (`server.py`)

- In `lifespan`, when `config.idle_unload_s > 0`, start a daemon thread
  (mirrors `_load_engine_background`) with a `threading.Event` stop signal:
  loop `wait(poll_s)` → `engine.maybe_unload(time.monotonic())`.
  `poll_s = min(30, idle_unload_s // 2)` (5 min ⇒ poll 30 s).
- On shutdown, set the event so the daemon exits. Guard: fake engines in tests
  pass `loader=None`, which disables eviction (no thread spawned via a
  `can_unload` check).

### 4. Request path: block until reloaded (`engine.py`; routes unchanged)

- `generate_pcm` / `generate_pcm_stream` / `voice_state` / `clone_voice` each start
  with `self._ensure_loaded()`: if `_model is None`, do a **single-flight** reload
  (guarded by a lock so concurrent requests wait, not duplicate) then proceed.
  The requesting call holds the generation lock / queue slot while the model loads,
  then serves the response normally — no 503 in the wake path.
- **Captured-model rule:** `_ensure_loaded()` returns the model and every caller
  threads that reference through (`model = self._ensure_loaded()` then
  `model.get_state_for_audio_prompt(...)`). Encoding runs *outside* `_gen_lock`, so
  a concurrent eviction must not null the object under it — holding a strong local
  ref keeps the old model alive for that encode; `maybe_unload`'s `_gen_lock`
  non-blocking guard covers only in-lock generation. Same rule at the top of
  `generate_pcm*` before accessing the model.
- `sync reload`: warm ~1 s (disk cache) is what the request observes; the cold
  network path (HF re-download, ~20 s+) blocks accordingly. To avoid a thundering
  herd, other waiters block on the same single-flight lock rather than each reloading.
- Routes' `_engine_or_503` keeps its existing `app.state.engine is None` check
  (only true pre-first-load); post-eviction the engine object stays present and
  `loaded=False`, so no route change is needed for wake — the engine itself blocks.

### 5. Config (`config.py`)

- `POCKET_TTS_IDLE_UNLOAD_S` (int, default **300** = 5 min; `0` disables).
- `POCKET_TTS_IDLE_POLL_S` (int, default 30) — watchdog cadence (only when unload enabled).

### 6. `/health` observability (`routes_speech.py`)

- Add `"loaded": bool`, `"idle_unload_s": int`, `"unloads"/"reloads"`, and
  `"last_request_age_s": float | null` so the footprint drop is visible to
  monitoring. 503 only appears pre-first-load (engine absent), as today.

### 7. Tests (keep 70 green + new)

- `tests/test_engine.py`: fake model + fake `loader`, injectable clock via
  `maybe_unload(now=...)`; assert eviction clears model+LRU, asserts no eviction
  when idle window not reached, asserts `ensure_loaded` rebuilds + rewarmups.
- `tests/test_server.py` / routes: with `loader` returning a fake and a big
  `idle_unload_s` simulated by calling `maybe_unload`, first request after eviction
  blocks until `ensure_loaded` rebuilds, then returns 200 (assert the response is
  correct after wake, and that no extra `loader` calls happen for concurrent
  waiters — single-flight).
- Config tests for `POCKET_TTS_IDLE_UNLOAD_S` parse + `0` disables.
- Keep tests deterministic (no real sleeps; call `maybe_unload(now)` directly).
- Full suite target: 70 + ~8 new, all green; `git diff --stat` for the commit.

### 8. Docker / README

- No Dockerfile change required (feature is runtime-config via env).
- README Configuration table: document `POCKET_TTS_IDLE_UNLOAD_S` / `_POLL_S`;
  Container section: note `/data` volume keeps the HF cache warm so a reload is ~1 s
  and eviction+relaad needs no re-download.
- PLAN.md: mark this section.

## Considerations / rejected alternatives

- **Process restart on idle:** recovers to ~30 MB but kills open connections,
  forces a 20 s+ cold start and an orchestrator/supervisor. Not worth it here.
- **Lowering the floor (torch import):** lazy-import torch would complicate the
  whole codebase; ~230 MB floor is acceptable for a TTS box.
- **Manual `/health`-driven unload:** health checks ping continuously, so they
  would *reset* the idle timer and defeat eviction — the watchdog must use
  **API request timestamps**, not health probes. (Keep health from touching.)
- **Memory-mapping / `torch.no_grad`:** orthogonal; doesn't reclaim the floor.
- Quantization (`POCKET_TTS_QUANTIZE=true`) already exists and reduces the
  *resident* weights further; idle-eviction composes with it (both on ⇒ bigger drop).

## Milestone

M5.5 · idle unload. Files: `config.py`, `engine.py`, `server.py`,
`routes_speech.py`, `routes_voices.py`, `errors.py`, `README.md`, `PLAN.md`, tests.

> **Status: IMPLEMENTED (M5.5)** — default 300 s; **block-until-reloaded**
> wake policy; in-process eviction + single-flight lazy reload; watchdog daemon
> in `lifespan`; `/health` adds `loaded`/`unloads`/`reloads`/`last_request_age_s`;
> 18 new tests (70→88 green); pyright clean. Config: `POCKET_TTS_IDLE_UNLOAD_S`,
> `POCKET_TTS_IDLE_POLL_S`. (See commit for the diff; this artifact remains a
> review record.)
