from __future__ import annotations

import logging

import requests

from src.config import Settings

logger = logging.getLogger(__name__)


class LLMClient:
    """Thin wrapper over Ollama's HTTP API — the only file that knows the wire format."""

    def __init__(self, settings: Settings):
        self.base_url = settings.ollama_base_url
        self.model = settings.ollama_model

    def complete(self, prompt: str, system: str | None = None) -> str:
        payload = {"model": self.model, "prompt": prompt, "stream": False}
        if system:
            payload["system"] = system
        try:
            response = requests.post(f"{self.base_url}/api/generate", json=payload, timeout=120)
            response.raise_for_status()
        except requests.RequestException as exc:
            logger.error("Ollama request failed: %s", exc)
            raise RuntimeError(
                f"Could not reach Ollama at {self.base_url} (model={self.model}): {exc}"
            ) from exc
        return response.json()["response"]
