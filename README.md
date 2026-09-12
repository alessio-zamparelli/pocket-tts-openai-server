# pocket-tts-openai

OpenAI-compatible TTS server (`POST /v1/audio/speech`) powered by
[Kyutai's pocket-tts](https://github.com/kyutai-labs/pocket-tts) — 100M-param
speech synthesis on CPU.

> Work in progress — see `./PLAN.md` for the full plan. M1 (skeleton) in progress.

## Dev

```sh
uv sync                                # deps without the engine (tests use a fake model)
uv run pytest                          # test suite
uv sync --extra engine                 # + pocket-tts (CPU-only torch on Linux, see pyproject)
uv run pocket-tts-openai               # serve on :8000
```
