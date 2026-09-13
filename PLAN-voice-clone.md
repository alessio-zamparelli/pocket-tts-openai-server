# PLAN — Clone the voice in `audio files/` (Italian male) + produce Italian samples

> **Goal:** take the voice(s) in `audio files/` (480 Italian recordings), clone the
> target speaker with pocket-tts `italian_24l` (the *voice-cloning-capable* weights),
> and synthesize a handful of Italian sample lines with that voice. Fully in-process,
> no new deps, `uv run`; HF token only as an env read.
>
> Status of this file flips to **IMPLEMENTED** at the end with measured results.

## 0. TL;DR

- `audio files/` = **12 distinct male-speaker groups** (mFB, mFF, mTC, mMB, mMC, mRTS,
  mWC, mGM, mFS, mNB, mIM, mSI), ~40–65 utt. each, mono 48 kHz MP3, ~2–4 s each.
  There is **not a single speaker** — "clone *this* voice" must target **one** of them.
- pocket-tts real clone = `model.get_state_for_audio_prompt(<audio path>)` on the
  **gated** `kyutai/pocket-tts` weights (`italian_24l.yaml` has
  `weights_path` (cloning) vs `weights_path_without_voice_cloning` (fallback,
  `has_voice_cloning=False` → `VOICE_CLONING_UNSUPPORTED`)).
- Gated repo is `gated='auto'`: a logged-in HF token authorizes automatically.
  Local HF cache held only the open (non-cloning) **english** model — the **blocker.**
- **Now unblocked:** user added `HF_TOKEN` to `.env` (gitignored; loaded into the env
  when running, never printed/committed). We can fetch the gated `italian_24l`
  weights (~2 GB) and clone for real.
- Deliverables: cloned voice → `.safetensors` + live cache, plus 3–5 Italian WAV/MP3
  samples, verified (duration, RMS, no silence). Outputs under gitignored `artifacts/`.

> **Measured (see §7):** the acoustic check found the 12 prefixes are **one male voice**
> split into recording batches (median F0 151–167 Hz in every group), so "this voice" is
> unambiguous; the clone targeted the largest pool (`mTC`, 65 files).

## 1. Environment facts (verified this session)

| Item | Finding |
| --- | --- |
| `audio files/` | 480 MP3, mono 48 kHz, 2.4–4.2 s; names `m<ID>_S<n>_<m>_Italian.wav.mp3`; **12 speaker prefixes** (see §0); no ID3 tags |
| pocket-tts version | installed in `.venv` (engine extra), `models/tts_model.py` |
| Clone API | `get_state_for_audio_prompt(audio_conditioning, truncate=False)` — accepts local file path, `.safetensors`, URL, or predefined voice name; uses Mimi + flow latent states (no separate adapter model) |
| Config `italian_24l.yaml` | `weights_path: hf://kyutai/pocket-tts/languages/italian_24l/model.safetensors@39592ff…` (cloning, **gated `auto`**) — fallback `weights_path_without_voice_cloning` (open) |
| Fallback logic | `tts_model.has_voice_cloning=True` init; on any download failure of `weights_path` → `=False` + open weights → `clone_voice` raises `VOICE_CLONING_UNSUPPORTED` (message points to huggingface.co/kyutai/pocket-tts + `uvx hf auth login`) |
| Local HF cache (before) | only `models--kyutai--pocket-tts-without-voice-cloning`, **english** only (209 MB model + tokenizer + english embeddings) — the e2e suite runs on the default `english` model |
| Server default | `config.py` `language: str = "english"`; env `POCKET_TTS_LANGUAGE`; **no dotenv loader** in `config.py`/`server.py` (env must be exported when launching) |
| python-dotenv | **not installed** (we keep zero new deps; load `.env` via shell `set -a; . ./.env; set +a`) |
| RAM/disk | 8 GB total / 6.2 GB available at check; 44 GB free disk — `italian_24l` fp32 ~2 GB is fine |
| Git | `.env` untracked ✓ and now **gitignored** (this change) |

## 2. Blocker (was) and how the token resolves it

`load_model(language="italian_24l")` tries `download_if_necessary(config.weights_path)`
→ gated repo → 401 without token → catches → open weights + `has_voice_cloning=False`.
With `HF_TOKEN` in the process env, `huggingface_hub` authenticates the request;
`gated='auto'` grants access on login (no manual approval wait). `has_voice_cloning`
stays `True` → `get_state_for_audio_prompt(<local mp3>)` works.

Security rules for this whole task:

- Token read **only** from env (`HF_TOKEN`); never printed, never committed, never in
  logs (download script asserts presence + prints only file names/sizes; logs are
  sed-redacted defensively).
- `.env` + `.env.*` gitignored (`!.env.example` reserved).
- Gated weights land in `~/.cache/huggingface` (home, outside the repo).

## 3. Steps

1. **Secure** — `.env` gitignored (done); verify `git status` shows no `.env`.
2. **Verify token** — `HfApi(model_info)` on gated repo with token → HTTP 200, git the
   `italian_24l/model.safetensors` size (~2 GB) — done in the download script.
3. **Download gated weights** — `artifacts/download_gated_italian.py` (background →
   `artifacts/download.log`); also caches the open `italian_24l/tokenizer.model`.
4. **Select the target speaker** — 12 prefixes; before cloning, confirm/collapse. We
   do an acoustic sanity check (pitch/spectral centroid per prefix) to *confirm* the
   prefixes are separate speakers, then **the plan uses the speaker with the largest,
   cleanest sample pool** (currently `mTC` 65 files) as the default, and exposes the
   `--speaker` knob to re-run any of the 12. *(If the user actually wants all speakers,
   we can clone each → 12 voices.)*
5. **Clone** — script `artifacts/clone_and_sample.py`:
   - load `TTSModel.load_model(language="italian_24l")` (voice-cloning weights; CPU).
   - pick 1 reference file for the target speaker (first, we verify 1 file is enough;
     `truncate=True` only if >30 s — ours are 2–4 s so prompt = whole file).
   - `state = model.get_state_for_audio_prompt(ref, truncate=True)` → persist via
     `export_model_state(state, artifacts/voices/<speaker>.safetensors)` (mirrors the
     server `clone_voice` path).
   - synthesize 4–6 Italian demo lines (including the kit's own text if known, else
     neutral sentences) → `artifacts/samples/<speaker>-<n>.wav` via
     `model.generate_audio(state, text)` ; also mp3 via our `audio_codecs.encode_pcm`.
6. **Verify outputs** — wav: valid header, duration in range, RMS/peak > silence
   threshold with the same checks the e2e suite uses; mp3: ffprobe parseable, duration
   ≈ expected; log pitch/RMS so a "generic male vs mTC" contrast is available.
7. **Server proof (bonus)** — boot the real app with `POCKET_TTS_LANGUAGE=italian_24l`
   - token, `POST /v1/voices` clone for the speaker, `POST /v1/audio/speech` with the
   cloned voice → wav; proves the clone path through our own stack (same code the
   server already ships). Speed: first load ~1–2 min (download done, CPU load of a
   24-layer model ~15–40 s); synthesis on CPU ~1–4 s/sentence.

## 4. Files produced (this task)

| Path | Purpose | Git |
| --- | --- | --- |
| `.gitignore` | add `.env`, `.env.*` | committed |
| `PLAN-voice-clone.md` | this plan | committed |
| `artifacts/download_gated_italian.py` | one-shot token-authorized download | scratch (gitignored) |
| `artifacts/clone_and_sample.py` | clone + sample generation (reusable, `--speaker`) | scratch (gitignored) |
| `artifacts/voices/<speaker>.safetensors` | cloned voice state | scratch |
| `artifacts/samples/…` | Italian WAV/MP3 samples | scratch |

Scratch lives under `artifacts/` (gitignored) so the repo stays clean; the scripts are
small and can graduate to `examples/` later if useful.

## 5. Risks / open questions

- **Multi-speaker corpus** — "the voice" is ambiguous. Default = `mTC` (largest clean
  pool). Re-run per speaker is trivial. Will confirm with the user only if the
  acoustic check contradicts the prefixes.
- **Clone quality from 2–4 s prompts** — pocket-tts is prompt-conditional; a couple of
  seconds is on the low side but workable (the predefined `giovanni` sample is a
  single utterance too). We'll verify with a 2–3 sentence sample and compare
  `giovanni` (open, available offline) vs our cloned `mTC` voice in the report.
- **CPU cost:** `italian_24l` 24-layer on CPU — load ~1–2 min, each sentence ~1–4 s
  synth. Fine for one-off; not for server-latency benchmarking (M6 untouched).
- **Gated repo `region:us`** — metadata tag; token access confirmed by HTTP 200 in
  step 2 (if a region block appears at download time it shows up in `download.log`
  and we report it).

## 6. Definition of done

- [x] `.env` gitignored, not in `git status`, not in any commit.
- [x] Gated `italian_24l` weights cached (`download.log` has `ALL_DOWNLOADS_OK`).
- [x] One real clone: `<speaker>.safetensors` persisted **and** synthesize ≥4 Italian
  WAV + MP3 samples; all pass duration/RMS/silence checks.
- [x] Server proof: `/v1/voices` clone + `/v1/audio/speech` with the cloned voice
  returned a non-empty wav (`SERVER_PROOF_OK`).
- [x] Report written here (§7) and this file marked IMPLEMENTED.

## 7. Results (measured)

**Status: IMPLEMENTED.** All gates green; commits are the security/plan/docs pieces
only (no tokens, no audio, no weights).

**Acoustic speaker check** (one-shot diagnostic, run once, not retained): median F0 per prefix is
151–167 Hz and spectral centroid ~1450–1610 Hz across **all 12 groups** — a single
male voice split into recording batches, not 12 speakers. So "clone this voice" had
one target; default speaker `mTC` (largest pool, 65 files). Reference file chosen by
length (3–4.5 s preferred) → `mTC_S10_1_Italian.wav.mp3` (2.4 s).

**Weight download** (`artifacts/download_gated_italian.py`): token-authorized fetch of
the **gated** `kyutai/pocket-tts` `italian_24l` — `model.safetensors` is **672 MB**
(not ~2 GB) + `tokenizer.model`, plus the open repo's italian tokenizer. Bonus: the
gated repo also ships 28 precomputed `embeddings/*.safetensors` (incl.
`giovanni.safetensors`). `ALL_DOWNLOADS_OK` in `artifacts/download.log`. HF cache
`863M` (was 222M).

**Clone + samples** (`artifacts/clone_and_sample.py`, mirrors the server code path):

- `TTSModel.load_model(language="italian_24l")` → **`has_voice_cloning = True`** ✓
- `get_state_for_audio_prompt(<ref wav>)` → exported
  `artifacts/voices/mTC.safetensors` (loads back in **0.0 s** and re-synthesizes,
  round-trip proven).
- 5 Italian lines synthesized on CPU (~6 s/sentence): durations **3.20–4.16 s**, RMS
  **0.121–0.138**, peak **0.658–0.788** — all valid 24 kHz mono 16-bit WAV, MP3
  variants ffprobe-parseable. Voice similarity: clone F0 **121–159 Hz** vs reference
  **133 Hz** (|Δ| 1–26 Hz); centroids 1458–1713 vs 1477 Hz — same band.
- Baseline vs the built-in Italian male `giovanni` (synthesized on the same model,
  saved as `artifacts/samples/giovanni-baseline.wav/.mp3` for A/B listening):
  giovanni F0 **106 Hz** / centroid **1579 Hz** vs mTC ref 133 Hz / 1477 Hz and
  clones 121–159 Hz / 1458–1713 Hz. Pitch (not timbre — both Italian males) places
  the clone on the corpus voice (mTC) and off the default low `giovanni`, consistent
  with the clone conditioning on the actual reference audio.

**Server proof** (`artifacts/server_proof.py`): boots the real app with
`language="italian_24l"` → `POST /v1/voices` (multipart, ref WAV) → **201** custom
voice `mtc` (`cached: True`, `safetensors: true`) → `POST /v1/audio/speech`
`voice="mtc"` → **200 audio/wav 6.96 s** (RMS 0.116, peak 0.798).
`SERVER_PROOF_OK`; `/v1/voices` registry lists `mtc`.

**New findings worth carrying forward**

- **mp3/flac clone uploads need `soundfile` at runtime**: pocket-tts `audio_read`
  decodes 16-bit WAV with the stdlib `wave` module but every other format needs
  `soundfile` (not installed). Our dep-free fix: transcode the reference to 16-bit
  PCM WAV with ffmpeg before `get_state_for_audio_prompt` (done in the script and in
  the server proof). The advertised `_AUDIO_EXTS = (`.wav`, `.mp3`, `.flac`)` in
  `routes_voices.py` therefore only *truly* works for WAV today — a future
  server-side transcoder would honor mp3/flac uploads with zero new deps.
- **Voice-name validator rejects uppercase**: `mTC` → 400 (`[a-z0-9_-]` only); the
  server proof used `mtc`. Keep this in mind if the corpus prefixes are re-used as
  voice ids.

**Security post-conditions**

- `.env` / `.env.*` gitignored; `HF_TOKEN` only ever read from the env; scripts print
  no token; logs defensively sed-redacted. `git status` shows no `.env`.
- Scratch (`artifacts/`) and the raw corpus (`audio files/`) are gitignored.
- Gated weights live only in the home HF cache, outside the repo.
