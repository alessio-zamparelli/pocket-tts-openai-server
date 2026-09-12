# Piano — Server TTS compatibile OpenAI con motore Pocket TTS

> Versione finale delle decisioni. Concorrenza: **opzione 1 (serializzazione con lock
> globale)** per la v1; pool di worker e scale-out documentati solo come evoluzione futura.

## 1. Obiettivo

Esporre un server HTTP **compatibile con l'OpenAI Audio API** (endpoint `POST /v1/audio/speech`)
che usi [pocket-tts](https://github.com/kyutai-labs/pocket-tts) (Kyutai) come motore di
sintesi, così che qualsiasi client già scritto per OpenAI (SDK `openai`, LangChain, ecc.)
funzioni puntandolo a questo server via `base_url`.

Vincoli chiave di pocket-tts emersi dall'analisi del codice sorgente:

- 100M parametri, gira **solo su CPU** (batch=1, ~6x real-time su MacBook Air M4, ~200ms al primo chunk)
- Python 3.10–3.14, PyTorch 2.5+ (su Linux serve `--extra-index-url https://download.pytorch.org/whl/cpu`)
- API Python: `TTSModel.load_model()`, `get_state_for_audio_prompt(voice)`,
  `generate_audio()`, `generate_audio_stream()`, `export_model_state()`; sample rate 24 kHz mono
- Lingue: en, fr, de, pt, it, es (+ varianti `_24l` più lente ma migliori, es. `italian_24l`)
- Output nativo: PCM → WAV streamato via `stream_audio_chunks`
- **Download on-demand con cache su disco**: pesi del modello e file voce scaricati al primo
  uso in `~/.cache/pocket_tts` (o cache HF per path `hf://`); nessun download ripetuto alle
  partenze successive
- **Codifica voce lenta**: trasformare un file audio in `ModelState` è costoso (il download
  no) → va fatto una volta per voce e mantenuto in memoria / esportato in `.safetensors`
  (ricaricamento quasi istantaneo: solo lettura kv-cache da disco)

## 2. Decisioni architetturali (finali)

| Decisione | Scelta v1 | Evoluzione futura |
| --- | --- | --- |
| Integrazione | **In-process**: import diretto di `pocket_tts` in un'app FastAPI | Proxy verso `pocket-tts serve` scartato (doppio processo, multipart, no controllo cache) |
| Concorrenza | **1 istanza modello, lock globale di generazione, coda FIFO** — le richieste si serializzano; correttezza garantita, latenza della n-esima richiesta = somma delle precedenti | Pool di N worker con *voice-affinity routing* (§8); scale-out N container dietro LB |
| Formati | `wav` e `pcm` **nativi**; `mp3`, `opus`, `aac`, `flac` via `ffmpeg` subprocess se presente, altrimenti 400 con messaggio chiaro | Codifiche pure-Python (`lameenc`, `opuslib`) senza ffmpeg |
| `speed` | Ignorato (pocket-tts non ha controllo di velocità); log warning | Time-stretch pitch-preserving (`ffmpeg atempo` / rubberband) |
| `instructions` | Ignorato con log warning | Preset voce/lingua via config |
| Mappa voci | Alias OpenAI → voci Kyutai (§5); nomi voce pocket-tts accettati direttamente (passthrough) | — |
| Autenticazione | Bearer token opzionale via env `POCKET_TTS_API_KEY` (non impostata = open) | — |
| Backpressure | v1: coda senza limiti, documentato il comportamento serializzato | Limite coda + 429 `Too Many Requests` con `Retry-After` |

## 3. Superficie API

### `POST /v1/audio/speech` (compatibile OpenAI)

Request JSON:

```json
{
  "model": "tts-1",
  "input": "Buongiorno, come stai?",
  "voice": "alloy",
  "response_format": "mp3",
  "speed": 1.0,
  "instructions": "…",
  "language": "italian"
}
```

- `model`: `tts-1` | `tts-1-hd` | `gpt-4o-mini-tts` — alias accettati, tutti la stessa
  gabella pocket-tts
- `voice`: alias OpenAI, nome voce Kyutai, alias configurato, o `hf://`/URL/file locale
- `response_format`: `mp3|opus|aac|flac|wav|pcm` (default `mp3`, come OpenAI)
- `language`: **estensione** — seleziona la lingua/gabella pocket-tts
  (`english`, `italian`, `italian_24l`, …); default da config

Response: binario audio con `Content-Type` corretto (`audio/mpeg`, `audio/opus`,
`audio/aac`, `audio/flac`, `audio/wav`, `audio/pcm`), streaming chunked quando possibile
(`wav`/`pcm` via `generate_audio_stream`; i formati compressi vengono bufferizzati e
codificati al termine della generazione).

Errori nel formato OpenAI:
`{"error": {"message": ..., "type": "invalid_request_error", "code": ...}}` con 400
(testo vuoto, formato non supportato, voce sconosciuta) e 500 (guasto generazione).

### `GET /health`

Stato modello, lingua attiva, metriche base (n. richieste, RTF medio, profondità coda).

### `GET /v1/models`

`tts-1`, `tts-1-hd`, `gpt-4o-mini-tts` con owner `pocket-tts`.

### Estensioni private — voci

#### `GET /v1/voices`

Catalogo delle voci disponibili. Response:

```json
{
  "object": "list",
  "data": [
    {
      "id": "alloy",
      "aliases": ["alloy", "alba"],
      "source": "builtin",            // builtin | custom
      "language": "en",
      "license": "https://huggingface.co/kyutai/tts-voices/blob/main/alba-mackenna/...",
      "cached": true                  // ModelState già codificato in questo processo
    },
    {
      "id": "mario",
      "aliases": ["mario"],
      "source": "custom",
      "language": "it",
      "license": null,
      "cached": false,
      "safetensors": true             // esportato su disco, ricaricamento rapido
    }
  ]
}
```

Permette ai client di scoprire i valori validi di `voice` senza hardcodarli.

#### `POST /v1/voices` (multipart/form-data)

Voice cloning on-demand:

| Campo | Tipo | Note |
| --- | --- | --- |
| `name` | string | richiesto; diventa l'alias usabile in `voice` |
| `file` | file | richiesto; wav/mp3/flac (l'extensione decide il formato) |
| `language` | string | opzionale; tag informativo mostrato in `GET /v1/voices` |

Comportamento: codifica il prompt audio in `ModelState` (operazione lenta), esporta in
`.safetensors` nella dir cache (`~/.cache/pocket_tts/voices/<name>.safetensors`), registra
l'alias persistente (file `voices.json` accanto alla cache) e risponde:

```json
{
  "id": "mario",
  "source": "custom",
  "language": "it",
  "cached": true,
  "safetensors": true
}
```

Errori: 400 (nome mancante/duplicato, file non riconosciuto), 413 (file troppo grande,
limite configurabile, default 25 MB).

#### `DELETE /v1/voices/{name}`

Rimuove la voce custom: alias dal registro + file `.safetensors`. 404 se inesistente,
405 se si tenta di cancellare una voce `builtin`.

## 4. Download, cache e warmup

Flusso di onboarding senza sorprese:

1. **All'avvio** (`load_model()`): se `~/.cache/pocket_tts` non ha i pesi, il download
   avviene una volta (su HF se path `hf://`). Comando dedicato `pocket-tts-openai warmup`
   che: carica il modello, scarica+codifica le voci della mappa e le esporta in
   `.safetensors`, genera un testo di prova.
2. **In build Docker**: il warmup viene eseguito in fase di build con i modelli/voci
   copiati in un volume persistente, così il container parte "caldo" senza download.
3. **A runtime, prima richiesta con una voce nuova**: download (se non in cache disco) +
   codifica `ModelState` (lenta) → aggiunta alla cache LRU in-process.
4. **Richieste successive**: cache LRU hit (istantaneo); se la voce era stata esportata in
   `.safetensors`, ricaricamento rapido anche dopo un riavvio.
5. **Voci custom on-demand** (`hf://`/URL/upload): scaricate, codificate e caché-ate;
   log che segnala che la prima richiesta è lenta.

Config: `POCKET_TTS_WARMUP_VOICES=giovanni,alba` (prefetch all'avvio), TTL/LRU sulla cache
voce, quantize int8 opzionale per ridurre la RAM.

## 5. Mappa voci (alias OpenAI → Kyutai)

| OpenAI | Pocket TTS | Note |
| --- | --- | --- |
| `alloy` | `alba` | en |
| `echo` | `charles` | en |
| `fable` | `eponine` | en (britannica) |
| `onyx` | `bill_boerst` | en |
| `nova` | `eve` | en |
| `shimmer` | `fantine` | en |
| `coral` (estensione) | `giovanni` | **it** — voce italiana predefinita |
| nome libero | come da catalogo / `hf://` / file | passthrough |

Mappa sovrascrivibile via YAML (`voices.yaml`) o env `POCKET_TTS_VOICE_MAP=alloy=alba,…`.
Licenze per-voce riportate in `/v1/voices` e README (fonte: `kyutai/tts-voices` su HF).

## 6. Struttura progetto

```
pocket-tts-openai/
├── pyproject.toml            # uv/pip; extra-index CPU per torch su Linux
├── README.md                 # IT/EN
├── Dockerfile                # python:3.12-slim + ffmpeg; warmup in build
├── docker-compose.yml
├── src/pocket_tts_openai/
│   ├── server.py             # app factory + uvicorn entrypoint
│   ├── routes_speech.py      # /v1/audio/speech, /v1/models, /health
│   ├── routes_voices.py      # GET/POST/DELETE /v1/voices (cloning + catalogo)
│   ├── engine.py             # wrapper TTSModel: modello singleton, lock, coda, stream
│   ├── voices.py             # alias, catalogo, cloning, cache LRU + safetensors
│   ├── encoders.py           # wav/pcm nativi; mp3/opus/aac/flac via ffmpeg
│   ├── errors.py             # errori in formato OpenAI
│   └── config.py             # env/CLI: porta, lingua default, voice map, api key, quantize
├── tests/
│   ├── test_api_contract.py  # shape risposta, content-type, errori (modello mockato)
│   ├── test_engine.py        # serializzazione, lock, cache voce
│   ├── test_encoders.py      # ffmpeg presente/assente
│   └── test_integration.py   # skippable: modello reale + client SDK openai
└── examples/
    ├── openai_sdk.py
    └── curl.sh
```

## 7. Milestone

1. **[x] M1 — Scheletro funzionante**: pyproject, config, `engine.py` con modello singleton +
   lock + coda FIFO, `/v1/audio/speech` limitato a `wav`+`pcm`, mappa voci fissa,
   `/health`, `/v1/models`. Test contratto con modello mockato.
2. **M2 — Formati compressi**: `encoders.py` con ffmpeg (rilevamento runtime), fallback 400
   documentato; test sui byte-header di ogni formato.
3. **[x] M3 — Voci**: cache LRU, prefetch/warmup all'avvio (`POCKET_TTS_WARMUP_VOICES`),
   esportazione `.safetensors`, `GET/POST/DELETE /v1/voices` (catalogo + voice cloning
   con persistenza in `voices.json`).
4. **[x] M4 — Streaming**: chunked transfer per `wav`/`pcm` via `generate_audio_stream`
   (primo chunk ~200 ms), opzione `stream` nel body; formati compressi bufferizzati.
5. **M5 — Packaging** *(parziale)*: ✅ Dockerfile CPU-only multi-stage (`deploy/Dockerfile`, uv guide, python:3.14-slim, non-root, volume `/data` → `HF_HOME` + registry voci), ✅ `.dockerignore`, ✅ `docker-compose.yml`, ✅ pipeline GHCR `.github/workflows/docker-publish.yml` (test gate + buildx + attestazioni). *Rimane:* warmup in build, esempi SDK openai, README IT/EN, `pocket-tts-openai warmup` CLI.
6. **M6 — Rifiniture**: metriche `/health` (RTF, profondità coda), quantize opzionale,
   benchmark latenza, supporto `italian_24l` testato.

## 8. Evoluzione concorrenza (post-v1, non implementata ora)

Quando la serializzazione con lock diventa un collo di bottiglia reale:

- **Pool di worker (target)**: N istanze `TTSModel` (~200 MB fp32 / ~100 MB int8) in thread
  separati (gli op PyTorch CPU rilasciano il GIL), ognuna con coda + lock + cache voce
  propria; routing per voce verso il worker che ha già il `ModelState` caricato. Stessa
  voce serializza, voci diverse vanno in parallelo. Attenzione: `torch.set_num_threads()`
  è process-global → impostarlo basso (1–2) con N worker.
- **Backpressure**: limite coda per worker + 429 con `Retry-After`; cancellazione della
  generazione alla disconnessione del client.
- **Process pool / scale-out**: worker process separati per isolamento e molti core;
  oppure N container dietro load balancer (le cache voce si popolano per container, mitigato
  dai `.safetensors`).

## 9. Rischio / note

- **Modello stateful batch=1** → richieste concorrenti serializzate per design (v1):
  documentare chiaramente nel README che la v1 non parallelizza e che lo scaling orizzontale
  è N istanze dietro LB.
- **Voce non caché-ata** → prima richiesta lenta (codifica prompt audio); mitigato da
  prefetch all'avvio e cache LRU.
- **ffmpeg assente** → solo `wav`/`pcm`; messaggio d'errore che suggerisce l'installazione.
- **Licenze voci**: riportare le licenze per-voce del catalogo Kyutai in README e
  `/v1/voices`.
- **Pause via testo non supportate** (issue #6 upstream): documentare come nota.
- **`instructions`/`speed` ignorati**: differenza di comportamento rispetto a
  `gpt-4o-mini-tts` ufficiale — da dichiarare esplicitamente nel README.
