"""FastAPI app factory and entrypoint.

``create_app`` accepts an optional ``Config`` and ``TTSEngine``; tests inject a
fake engine, production loads pocket-tts in a background thread on startup.
"""

from __future__ import annotations

import logging
import threading
from contextlib import asynccontextmanager
from typing import cast

import uvicorn
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from pathlib import Path

from .config import Config
from .engine import TTSEngine, load_engine
from .errors import OpenAIError, invalid_api_key
from .routes_speech import health, models, speech
from .routes_voices import router as voices_router
from .voices import KYUTAI_CATALOG

logger = logging.getLogger(__name__)


def create_app(config: Config | None = None, engine: TTSEngine | None = None) -> FastAPI:
    """Build the OpenAI-compatible app. ``engine=None`` defers model loading to
    startup (production); tests pass a fake engine to stay fast and offline."""
    config = config or Config.from_env()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        if engine is not None:
            app.state.engine = engine
        else:
            app.state.engine = None
            thread = threading.Thread(target=_load_engine_background, args=(app, config), daemon=True)
            thread.start()
            if config.idle_unload_s > 0:
                # RAM reclamation watchdog: evict the idle model (PLAN-idle-unload.md).
                stop = threading.Event()
                app.state._idle_stop = stop
                wd = threading.Thread(
                    target=_idle_watchdog, args=(app, config, stop), daemon=True
                )
                wd.start()
        try:
            yield
        finally:
            stop = getattr(app.state, "_idle_stop", None)
            if stop is not None:
                stop.set()
            # Load/unload symmetry: release the resident model at shutdown so the
            # lifecycle is fully observable in logs (plan: load/unload logging).
            # Only engines this server built itself (``_auto_loaded``) are
            # released — injected engines may be reused (tests) or owned by a
            # supervisor, so they are left resident but still logged.
            engine_obj = getattr(app.state, "engine", None)
            if engine_obj is not None:
                if getattr(engine_obj, "_auto_loaded", False):
                    logger.info("shutting down: unloading model…")
                    try:
                        engine_obj.unload(reason="server shutdown")
                    except Exception:  # pragma: no cover - defensive shutdown path
                        logger.exception("error unloading model during shutdown")
                else:
                    logger.info("shutting down: engine left resident (not owned by this server)")

    app = FastAPI(title="pocket-tts-openai", version="0.1.0", lifespan=lifespan)
    app.state.config = config

    def _openai_error_handler(request: Request, exc: Exception) -> JSONResponse:
        # Starlette dispatches this handler only for the registered exception type.
        err = cast(OpenAIError, exc)
        return JSONResponse(status_code=err.status, content=err.to_payload())

    app.add_exception_handler(OpenAIError, _openai_error_handler)

    if config.api_key:

        @app.middleware("http")
        async def check_api_key(request: Request, call_next):
            if request.url.path.startswith("/v1/"):
                auth = request.headers.get("Authorization", "")
                if auth != f"Bearer {config.api_key}":
                    return JSONResponse(
                        status_code=401,
                        content=invalid_api_key(
                            "Missing or invalid API key. Pass 'Authorization: Bearer <key>'."
                        ).to_payload(),
                    )
            return await call_next(request)

    # response_model=None: speech() returns a raw binary Response, not JSON
    app.post("/v1/audio/speech", response_model=None)(speech)
    app.get("/v1/models")(models)
    app.get("/health")(health)
    # Voice catalog + cloning (private extension).
    app.include_router(voices_router, prefix="/v1/voices")
    return app


def _idle_watchdog(app: FastAPI, config: Config, stop: threading.Event) -> None:
    """Periodically evict the model after ``POCKET_TTS_IDLE_UNLOAD_S`` without
    API requests. Health probes do NOT touch the engine's idle timer, so they
    never reset the window (see PLAN-idle-unload.md).
    """
    interval = min(config.idle_poll_s, max(5, config.idle_unload_s // 2))
    while not stop.wait(interval):
        engine = getattr(app.state, "engine", None)
        if engine is None:
            continue  # still loading
        try:
            engine.maybe_unload()
        except Exception:  # pragma: no cover - defensive
            logger.exception("idle-unload watchdog failed")


def _load_engine_background(app: FastAPI, config: Config) -> None:
    """Load pocket-tts off the event loop; /speech returns 503 until it lands."""
    try:
        engine = load_engine(config)
        app.state.engine = engine
        logger.info("engine ready: sample_rate=%d voices=%d", engine.sample_rate, len(KYUTAI_CATALOG))
        engine.warmup()
    except Exception:
        logger.exception("engine failed to load; /speech will keep returning 503")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
