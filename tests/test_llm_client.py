from __future__ import annotations

from unittest.mock import Mock, patch

import requests

from src.llm_client import LLMClient


def _ok_response(text: str = "hi") -> Mock:
    response = Mock()
    response.status_code = 200
    response.ok = True
    response.json.return_value = {"choices": [{"message": {"content": text}}]}
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
