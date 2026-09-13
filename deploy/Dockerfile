# syntax=docker/dockerfile:1
#
# pocket-tts-openai — OpenAI-compatible TTS server (CPU-only).
#
# Built with uv following the official integration guide:
#   https://docs.astral.sh/uv/guides/integration/docker/
#
# CPU-only by design: `torch` resolves from the PyTorch CPU wheel index via
# uv.lock (see [tool.uv.sources] in pyproject.toml) — no CUDA/NVIDIA layers.
#
# Build:    docker build -f deploy/Dockerfile -t pocket-tts-openai .
# Run:      docker run --rm -p 8080:8000 -v tts-data:/data pocket-tts-openai

###############################################################################
# Stage 1 — builder: resolve deps + project into a venv
###############################################################################
FROM python:3.14-slim AS builder

# Pin uv to the exact version used to (re)generate uv.lock.
COPY --from=ghcr.io/astral-sh/uv:0.12.13 /uv /uvx /bin/

# Use the system Python already present in the slim image (3.14).
ENV UV_PYTHON_DOWNLOADS=0 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

# Layer 1 — install ALL locked dependencies (incl. the `engine` extra, which
# pulls pocket-tts + CPU torch) without the project. This layer is cached and
# only invalidates when pyproject.toml / uv.lock change.
COPY pyproject.toml uv.lock ./
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-install-project --no-dev --no-editable --extra engine

# Layer 2 — copy the project source and install it non-editable so the venv is
# fully self-contained (no dependency on /app/src at runtime).
COPY README.md ./
COPY src ./src
RUN --mount=type=cache,target=/root/.cache/uv \
    uv sync --locked --no-dev --no-editable --extra engine

###############################################################################
# Stage 2 — runtime: minimal, non-root, persistent /data volume
###############################################################################
FROM python:3.14-slim AS runtime

# OpenMP runtime needed by torch (CPU) for multithreading, plus ffmpeg for the
# compressed /v1/audio/speech formats (mp3/opus/aac/flac). Debian ships ffmpeg
# with the libmp3lame and libopus encoders; without it those formats return a
# clear 400 (wav/pcm stay stdlib-only).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Reuse the system Python across stages; put the app venv on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    POCKET_TTS_HOST=0.0.0.0 \
    POCKET_TTS_PORT=8000 \
    POCKET_TTS_CACHE_DIR=/data/cache \
    HF_HOME=/data/hf

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Non-root runtime user; /data must stay writable for model weights + voices.
RUN useradd --create-home --uid 10001 tts \
    && mkdir -p /data/cache/voices /data/hf

# Seed the voice registry with the pre-cloned Italian voice `mtc` (see the
# repo `voices/` dir). Docker copies this into the named volume on first
# mount, so a fresh container already lists `mtc` in GET /v1/voices with no
# runtime clone.
COPY voices/registry.json /data/cache/voices/registry.json
COPY voices/mtc.safetensors /data/cache/voices/mtc.safetensors

RUN chown -R tts:tts /data

# Persistent volume: model weights (HF_HOME=/data/hf) + voice registry
# (POCKET_TTS_CACHE_DIR=/data/cache -> /data/cache/voices). Mount it to avoid
# re-downloading weights and to keep custom voices across restarts.
VOLUME ["/data"]

USER tts
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

ENTRYPOINT ["pocket-tts-openai"]
