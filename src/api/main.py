from __future__ import annotations

from fastapi import FastAPI

from src.api.routes import router

app = FastAPI(title="GitHub Repo Q&A Assistant")
app.include_router(router)
