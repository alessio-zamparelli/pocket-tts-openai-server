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
# Stage 1.5 — whispercpp: build the native whisper-server STT sidecar (CPU-only,
# no CUDA/NVIDIA). Whisper.cpp is built once from a pinned release; OpenVINO is
# a documented compile-time follow-up (-DWHISPER_OPENVINO=1) not enabled here.
###############################################################################
FROM debian:bookworm-slim AS whispercpp

ARG WHISPER_CPP_REF=v1.7.4

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates git cmake make g++ \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
# NOTE: examples/CMakeLists gates the whole examples tree on
# WHISPER_BUILD_EXAMPLES and adds examples/server unconditionally inside it;
# WHISPER_BUILD_SERVER is currently only an (unused) option, so we keep
# WHISPER_BUILD_EXAMPLES=ON (default) to build whisper-server and copy only
# that one binary out — the extra examples never reach the runtime image.
RUN git clone --depth 1 --branch ${WHISPER_CPP_REF} https://github.com/ggerganov/whisper.cpp.git \
    # BUILD_SHARED_LIBS defaults to ON on Linux, which would link
    # whisper-server against libwhisper.so.1 (and libggml*.so) and made the
    # copied-out binary abort with exit 127:
    #   error while loading shared libraries: libwhisper.so.1: cannot open
    # - Fix: build the whisper libraries statically (libwhisper.a, libggml.a)
    #   so the single whisper-server binary is self-contained and only needs
    #   the system libs present in the runtime stage (libgomp1, libstdc++).
    && cmake -S whisper.cpp -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DWHISPER_BUILD_TESTS=OFF \
        -DBUILD_SHARED_LIBS=OFF \
    && cmake --build build --config Release --target whisper-server -j"$(nproc)" \
    && mkdir -p /out/bin \
    && cp build/bin/whisper-server /out/bin/

###############################################################################
# Stage 2 — runtime: minimal, non-root, persistent /data volume
###############################################################################
FROM python:3.14-slim AS runtime

# OpenMP runtime needed by torch (CPU) for multithreading; ffmpeg decodes the
# mp3/m4a/webm audio clips the STT endpoints accept (whisper.cpp reads wav).
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Reuse the system Python across stages; put the app venv on PATH.
ENV PATH="/app/.venv/bin:$PATH" \
    UV_PYTHON_DOWNLOADS=0 \
    PYTHONUNBUFFERED=1 \
    STTS_HOST=0.0.0.0 \
    STTS_PORT=8000 \
    STTS_CACHE_DIR=/data/cache \
    HF_HOME=/data/hf

WORKDIR /app
COPY --from=builder /app/.venv /app/.venv

# Native whisper-server STT sidecar: /usr/local/bin is on PATH for the tts user.
COPY --from=whispercpp /out/bin/whisper-server /usr/local/bin/whisper-server
# STT model GGUFs live under /data/cache/stt-models (persistent volume).
ENV STTS_STT_BIN=whisper-server \
    STTS_STT_MODEL_DIR=/data/cache/stt-models

# Non-root runtime user; /data must stay writable for model weights + voices.
RUN useradd --create-home --uid 10001 tts \
    && mkdir -p /data/cache /data/hf \
    && chown -R tts:tts /data

# Persistent volume: model weights (HF_HOME=/data/hf) + voice registry
# (STTS_CACHE_DIR=/data/cache -> /data/cache/voices). Mount it to avoid
# re-downloading weights and to keep custom voices across restarts.
VOLUME ["/data"]

USER tts
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=5s --start-period=120s --retries=3 \
    CMD ["python", "-c", "import urllib.request,sys; urllib.request.urlopen('http://127.0.0.1:8000/health', timeout=4).read()"]

ENTRYPOINT ["pocket-tts-openai"]
