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

from .config import Config
from .engine import TTSEngine, load_engine
from .errors import OpenAIError, invalid_api_key
from .routes_speech import health, models, speech
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
        yield

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
    return app


def _load_engine_background(app: FastAPI, config: Config) -> None:
    """Load pocket-tts off the event loop; /speech returns 503 until it lands."""
    try:
        app.state.engine = load_engine(config)
        logger.info("engine ready: sample_rate=%d voices=%d", app.state.engine.sample_rate, len(KYUTAI_CATALOG))
    except Exception:
        logger.exception("engine failed to load; /speech will keep returning 503")


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    config = Config.from_env()
    app = create_app(config)
    uvicorn.run(app, host=config.host, port=config.port)


if __name__ == "__main__":
    main()
