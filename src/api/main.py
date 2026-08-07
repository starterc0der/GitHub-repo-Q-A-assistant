from __future__ import annotations

import logging
import time
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import pipeline, router
from src.config import settings

# uvicorn only configures its own loggers; without this the app's logger.info() calls
# fall through to Python's WARNING-only lastResort handler and vanish.
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s")

# qdrant-client logs every request through httpx at INFO — ~20 lines per query, which
# buries the pipeline's own output.
for noisy in ("httpx", "httpcore", "qdrant_client"):
    logging.getLogger(noisy).setLevel(logging.WARNING)

logger = logging.getLogger(__name__)

ALLOWED_ORIGIN = "http://localhost:5183"  # Vite dev server



@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Warm the models here rather than charging ~18s of lazy loading to the first query."""
    if settings.preload_models:
        started = time.monotonic()
        logger.info("Preloading models (set PRELOAD_MODELS=false to skip)…")
        pipeline.embedder.model  # noqa: B018 — property access is the load
        logger.info("  embedder ready (%.1fs)", time.monotonic() - started)
        pipeline.cross_reranker.model  # noqa: B018
        logger.info("Models ready in %.1fs", time.monotonic() - started)
    yield


app = FastAPI(title="GitHub Repo Q&A Assistant", lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=[ALLOWED_ORIGIN],
    allow_methods=["*"],
    allow_headers=["*"],
)
app.include_router(router)


@app.exception_handler(Exception)
async def unhandled_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    # A handler for the base Exception class runs inside Starlette's ServerErrorMiddleware,
    # which sits OUTSIDE CORSMiddleware (unlike HTTPException handlers) — so its response
    # never passes back through CORSMiddleware and reaches the browser with no CORS headers.
    # fetch() then rejects with an opaque "Failed to fetch" instead of the real error. Set
    # the header directly here rather than relying on middleware ordering.
    logger.exception("Unhandled error on %s %s", request.method, request.url.path)
    response = JSONResponse(status_code=500, content={"detail": str(exc)})
    if request.headers.get("origin") == ALLOWED_ORIGIN:
        response.headers["Access-Control-Allow-Origin"] = ALLOWED_ORIGIN
        response.headers["Vary"] = "Origin"
    return response
