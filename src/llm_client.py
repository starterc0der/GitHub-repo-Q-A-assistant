from __future__ import annotations

import logging
import random

import requests

from src.cancellation import check, interruptible_sleep

logger = logging.getLogger(__name__)

MAX_TOKENS = 4096


class LLMClient:
    """Thin wrapper over any OpenAI-compatible /chat/completions endpoint.

    One implementation instead of one per vendor: Gemini, OpenAI, Groq, Ollama,
    OpenRouter, Together, DeepSeek and others all serve this shape, so switching provider
    is a change of LLM_BASE_URL rather than a change of code.

    Callers depend on two things: complete() returns a string, and every failure is a
    RuntimeError they can catch to degrade gracefully.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120,
                 max_retries: int = 7):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries

    def complete(
        self, prompt: str, system: str | None = None, history: list[tuple[str, str]] | None = None
    ) -> str:
        messages = (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": role, "content": content} for role, content in (history or [])]
            + [{"role": "user", "content": prompt}]
        )
        payload = {"model": self.model, "messages": messages, "max_tokens": MAX_TOKENS}
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            check()
            try:
                response = requests.post(
                    self.url, headers=headers, json=payload, timeout=self.timeout
                )
            except requests.RequestException as exc:
                raise RuntimeError(f"Could not reach {self.url} ({self.model}): {exc}") from exc

            # A per-minute burst clears in seconds; a daily quota does not clear until the
            # provider's reset, so retrying just burns the ladder to reach the same 429.
            if response.status_code == 429 and self._is_daily_quota(response):
                raise RuntimeError(
                    f"Daily quota exhausted for {self.model}. Retrying will not help — "
                    f"wait for the provider's reset or switch LLM_MODEL."
                )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries - 1:
                    break
                delay = self._retry_delay(response, attempt)
                logger.warning(
                    "%s %s (%s), retrying in %.1fs [%d/%d]",
                    self.model, response.status_code, self.url,
                    delay, attempt + 1, self.max_retries,
                )
                interruptible_sleep(delay)
                continue

            if not response.ok:
                raise RuntimeError(
                    f"LLM call failed ({self.model}): {response.status_code} {response.text[:200]}"
                )
            return self._extract_text(response.json())

        raise RuntimeError(
            f"LLM still failing after {self.max_retries} attempts ({self.model}): "
            f"{response.status_code} {response.text[:200]}"
        )

    def _is_daily_quota(self, response: requests.Response) -> bool:
        text = response.text.lower()
        return "perday" in text or "per day" in text or "daily" in text

    def _retry_delay(self, response: requests.Response, attempt: int) -> float:
        """Prefer the server's own hint — a per-minute quota can take a full 60s to reset,
        which pure exponential backoff would undershoot."""
        header = response.headers.get("Retry-After")
        if header and header.strip().isdigit():
            return min(float(header), 65.0) + random.uniform(0, 1)
        try:
            for detail in response.json().get("error", {}).get("details", []):
                raw = detail.get("retryDelay")
                if raw:
                    return min(float(str(raw).rstrip("s")), 65.0) + random.uniform(0, 1)
        except (ValueError, AttributeError, TypeError):
            pass
        return min(2**attempt, 65) + random.uniform(0, 1)

    def _extract_text(self, body: dict) -> str:
        """A filtered or empty response is a well-formed 200 with no text — surface it as
        RuntimeError so callers hit their fallback instead of embedding ''."""
        choices = body.get("choices") or []
        if not choices:
            raise RuntimeError(f"LLM returned no choices ({self.model})")
        return (choices[0].get("message") or {}).get("content") or ""
