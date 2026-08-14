from __future__ import annotations

import json
import logging
import random
from collections.abc import Iterator

import requests

from src.cancellation import check, interruptible_sleep

logger = logging.getLogger(__name__)

MAX_TOKENS = 8192


class LLMClient:
    """Thin wrapper over any OpenAI-compatible /chat/completions endpoint.

    One implementation instead of one per vendor: Gemini, OpenAI, Groq, Ollama,
    OpenRouter, Together, DeepSeek and others all serve this shape, so switching provider
    is a change of LLM_BASE_URL rather than a change of code.

    Callers depend on two things: complete()/stream() return/yield text, and every
    failure is a RuntimeError they can catch to degrade gracefully.
    """

    def __init__(self, base_url: str, api_key: str, model: str, timeout: int = 120,
                 max_retries: int = 7):
        self.url = f"{base_url.rstrip('/')}/chat/completions"
        self.api_key = api_key
        self.model = model
        self.timeout = timeout
        self.max_retries = max_retries
        # Side-channel so callers (rewriter, decomposer, ...) don't need their return
        # signatures changed — Pipeline reads this right after each call instead.
        self.last_usage: dict[str, int] | None = None

    def complete(
        self, prompt: str, system: str | None = None, history: list[tuple[str, str]] | None = None
    ) -> str:
        payload = {
            "model": self.model, "messages": self._build_messages(prompt, system, history),
            "max_tokens": MAX_TOKENS,
        }
        response = self._post_with_retry(payload)
        body = response.json()
        self.last_usage = body.get("usage")
        return self._extract_text(body)

    def stream(
        self, prompt: str, system: str | None = None, history: list[tuple[str, str]] | None = None
    ) -> Iterator[str]:
        """Same request as complete(), but yields text deltas as they arrive instead of
        waiting for the full response. The retry ladder in _post_with_retry only covers
        the connection attempt — once tokens have started arriving, a dropped stream
        raises immediately rather than retrying, since a partial generation can't be
        cleanly resumed without risking duplicated or garbled text."""
        self.last_usage = None
        payload = {
            "model": self.model, "messages": self._build_messages(prompt, system, history),
            "max_tokens": MAX_TOKENS, "stream": True,
            "stream_options": {"include_usage": True},
        }
        response = self._post_with_retry(payload)
        yield from self._iter_deltas(response)

    def _build_messages(
        self, prompt: str, system: str | None, history: list[tuple[str, str]] | None
    ) -> list[dict]:
        return (
            ([{"role": "system", "content": system}] if system else [])
            + [{"role": role, "content": content} for role, content in (history or [])]
            + [{"role": "user", "content": prompt}]
        )

    def _post_with_retry(self, payload: dict) -> requests.Response:
        """Retries connection failures and retryable HTTP statuses (429/5xx) with
        backoff; raises RuntimeError for anything else. The caller reads the body —
        once, or (payload["stream"]) incrementally — this only gets a usable response."""
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        for attempt in range(self.max_retries):
            check()
            try:
                response = requests.post(
                    self.url, headers=headers, json=payload,
                    timeout=self.timeout, stream=payload.get("stream", False),
                )
            except requests.RequestException as exc:
                # A read timeout or connection drop is usually transient (a momentarily
                # slow provider, a network blip) — worth the same retry ladder as a 429/5xx
                # instead of failing the whole request on the first hiccup.
                if attempt == self.max_retries - 1:
                    raise RuntimeError(f"Could not reach {self.url} ({self.model}): {exc}") from exc
                delay = min(2**attempt, 65) + random.uniform(0, 1)
                logger.warning(
                    "%s request to %s failed, retrying in %.1fs [%d/%d]: %s",
                    self.model, self.url, delay, attempt + 1, self.max_retries, exc,
                )
                interruptible_sleep(delay)
                continue

            # A per-minute burst clears in seconds; a daily quota does not clear until the
            # provider's reset, so retrying just burns the ladder to reach the same 429.
            if response.status_code == 429 and self._is_daily_quota(response):
                raise RuntimeError(
                    f"Daily quota exhausted for {self.model}. Retrying will not help — "
                    f"wait for the provider's reset or switch LLM_MODEL."
                )

            if response.status_code == 429 or response.status_code >= 500:
                if attempt == self.max_retries - 1:
                    raise RuntimeError(
                        f"LLM still failing after {self.max_retries} attempts ({self.model}): "
                        f"{response.status_code} {response.text[:200]}"
                    )
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
            return response

        raise RuntimeError(f"LLM still failing after {self.max_retries} attempts ({self.model})")

    def _iter_deltas(self, response: requests.Response) -> Iterator[str]:
        """Parses an OpenAI-style SSE stream (`data: {...}` lines, ending in
        `data: [DONE]`) into text deltas."""
        got_any = False
        try:
            for line in response.iter_lines(decode_unicode=True):
                check()
                if not line or not line.startswith("data:"):
                    continue
                data = line[len("data:"):].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except ValueError:
                    continue
                if chunk.get("usage"):
                    self.last_usage = chunk["usage"]
                try:
                    delta = chunk["choices"][0]["delta"].get("content")
                except (KeyError, IndexError):
                    continue
                if delta:
                    got_any = True
                    yield delta
        except requests.RequestException as exc:
            if got_any:
                raise RuntimeError(f"Stream from {self.model} dropped mid-answer: {exc}") from exc
            raise RuntimeError(f"Could not reach {self.url} ({self.model}): {exc}") from exc

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
