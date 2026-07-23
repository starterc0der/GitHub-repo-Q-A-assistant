from __future__ import annotations

import logging

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from src.api.routes import router

logger = logging.getLogger(__name__)

ALLOWED_ORIGIN = "http://localhost:5183"  # Vite dev server

app = FastAPI(title="GitHub Repo Q&A Assistant")
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
