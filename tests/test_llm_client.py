from __future__ import annotations

import json
from unittest.mock import Mock, patch

import requests

from src.llm_client import LLMClient


def _ok_response(text: str = "hi", usage: dict | None = None) -> Mock:
    response = Mock()
    response.status_code = 200
    response.ok = True
    body = {"choices": [{"message": {"content": text}}]}
    if usage is not None:
        body["usage"] = usage
    response.json.return_value = body
    return response


def _sse_lines(*deltas: str) -> list[str]:
    lines = []
    for d in deltas:
        lines.append(f"data: {json.dumps({'choices': [{'delta': {'content': d}}]})}")
        lines.append("")  # blank line between SSE events
    lines.append("data: [DONE]")
    return lines


def _sse_response(lines: list[str]) -> Mock:
    response = Mock()
    response.status_code = 200
    response.ok = True
    response.iter_lines.return_value = iter(lines)
    return response


@patch("src.llm_client.interruptible_sleep")
@patch("requests.post")
def test_complete_retries_after_a_read_timeout(mock_post: Mock, mock_sleep: Mock) -> None:
    """A ReadTimeout is a network-level exception, not an HTTP response — it must hit the
    same retry ladder as a 429/5xx instead of failing on the very first attempt."""
    mock_post.side_effect = [requests.exceptions.ReadTimeout("timed out"), _ok_response("recovered")]

    client = LLMClient("https://api.example.com", "key", "test-model", max_retries=3)
    result = client.complete("hello")

    assert result == "recovered"
    assert mock_post.call_count == 2
    mock_sleep.assert_called_once()


@patch("src.llm_client.interruptible_sleep")
@patch("requests.post")
def test_complete_raises_after_exhausting_retries_on_repeated_timeouts(
    mock_post: Mock, mock_sleep: Mock
) -> None:
    mock_post.side_effect = requests.exceptions.ReadTimeout("timed out")

    client = LLMClient("https://api.example.com", "key", "test-model", max_retries=3)
    try:
        client.complete("hello")
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "test-model" in str(exc)

    assert mock_post.call_count == 3
    assert mock_sleep.call_count == 2


@patch("requests.post")
def test_complete_succeeds_without_retry_on_first_try(mock_post: Mock) -> None:
    mock_post.return_value = _ok_response("hi there")

    client = LLMClient("https://api.example.com", "key", "test-model")
    result = client.complete("hello")

    assert result == "hi there"
    assert mock_post.call_count == 1


@patch("requests.post")
def test_complete_captures_token_usage(mock_post: Mock) -> None:
    mock_post.return_value = _ok_response("hi there", usage={"prompt_tokens": 12, "completion_tokens": 4})

    client = LLMClient("https://api.example.com", "key", "test-model")
    client.complete("hello")

    assert client.last_usage == {"prompt_tokens": 12, "completion_tokens": 4}


@patch("requests.post")
def test_complete_leaves_last_usage_none_when_provider_omits_it(mock_post: Mock) -> None:
    mock_post.return_value = _ok_response("hi there")

    client = LLMClient("https://api.example.com", "key", "test-model")
    client.complete("hello")

    assert client.last_usage is None


@patch("requests.post")
def test_stream_requests_and_captures_trailing_usage_chunk(mock_post: Mock) -> None:
    lines = [
        f"data: {json.dumps({'choices': [{'delta': {'content': 'Hello'}}]})}",
        "",
        # The trailing usage-only frame: no "choices" text, just token counts.
        f"data: {json.dumps({'choices': [], 'usage': {'prompt_tokens': 30, 'completion_tokens': 8}})}",
        "",
        "data: [DONE]",
    ]
    mock_post.return_value = _sse_response(lines)

    client = LLMClient("https://api.example.com", "key", "test-model")
    result = list(client.stream("hi"))

    assert result == ["Hello"]
    assert client.last_usage == {"prompt_tokens": 30, "completion_tokens": 8}
    assert mock_post.call_args.kwargs["json"]["stream_options"] == {"include_usage": True}


@patch("requests.post")
def test_stream_yields_deltas_in_order(mock_post: Mock) -> None:
    mock_post.return_value = _sse_response(_sse_lines("Hello", ", ", "world"))

    client = LLMClient("https://api.example.com", "key", "test-model")
    result = list(client.stream("hi"))

    assert result == ["Hello", ", ", "world"]
    assert mock_post.call_args.kwargs["stream"] is True


@patch("requests.post")
def test_stream_raises_if_dropped_mid_answer(mock_post: Mock) -> None:
    """A stream that starts, yields real content, then drops must raise instead of
    silently truncating — the caller (and user) needs to know the answer is incomplete."""
    def broken_lines():
        yield _sse_lines("partial")[0]
        raise requests.exceptions.ChunkedEncodingError("connection broken")

    response = Mock()
    response.status_code = 200
    response.ok = True
    response.iter_lines.return_value = broken_lines()
    mock_post.return_value = response

    client = LLMClient("https://api.example.com", "key", "test-model")
    gen = client.stream("hi")

    assert next(gen) == "partial"
    try:
        next(gen)
        assert False, "expected RuntimeError"
    except RuntimeError as exc:
        assert "dropped mid-answer" in str(exc)


@patch("src.llm_client.interruptible_sleep")
@patch("requests.post")
def test_stream_retries_connection_failure_before_streaming_starts(
    mock_post: Mock, mock_sleep: Mock
) -> None:
    """Retries are safe before any token has arrived — same ladder as complete()."""
    mock_post.side_effect = [
        requests.exceptions.ReadTimeout("timed out"),
        _sse_response(_sse_lines("recovered")),
    ]

    client = LLMClient("https://api.example.com", "key", "test-model", max_retries=3)
    result = list(client.stream("hi"))

    assert result == ["recovered"]
    assert mock_post.call_count == 2
