from __future__ import annotations

from unittest.mock import MagicMock, patch

import psycopg2
import redis

from src.connectors.testing import test_postgres as check_postgres, test_redis as check_redis


@patch("src.connectors.testing.redis.Redis")
def test_redis_success(mock_redis_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_redis_cls.return_value = mock_client

    ok, message = check_redis("localhost", 6379, "hunter2")

    assert ok is True
    assert "Connected to localhost:6379" in message
    mock_client.ping.assert_called_once()
    mock_client.close.assert_called_once()


@patch("src.connectors.testing.redis.Redis")
def test_redis_passes_username_through_for_acl_users(mock_redis_cls: MagicMock) -> None:
    mock_redis_cls.return_value = MagicMock()

    check_redis("localhost", 6379, "hunter2", username="watcodev")

    assert mock_redis_cls.call_args.kwargs["username"] == "watcodev"


@patch("src.connectors.testing.redis.Redis")
def test_redis_no_username_passes_none(mock_redis_cls: MagicMock) -> None:
    mock_redis_cls.return_value = MagicMock()

    check_redis("localhost", 6379, "hunter2")

    assert mock_redis_cls.call_args.kwargs["username"] is None


@patch("src.connectors.testing.redis.Redis")
def test_redis_wrong_password(mock_redis_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.ping.side_effect = redis.AuthenticationError("bad password")
    mock_redis_cls.return_value = mock_client

    ok, message = check_redis("localhost", 6379, "wrong")

    assert ok is False
    assert "Authentication failed" in message


@patch("src.connectors.testing.redis.Redis")
def test_redis_unreachable_host(mock_redis_cls: MagicMock) -> None:
    mock_client = MagicMock()
    mock_client.ping.side_effect = redis.ConnectionError("could not connect to host")
    mock_redis_cls.return_value = mock_client

    ok, message = check_redis("nonexistent.internal", 6379, "")

    assert ok is False
    assert "Could not connect" in message


@patch("src.connectors.testing.psycopg2.connect")
def test_postgres_success(mock_connect: MagicMock) -> None:
    mock_conn = MagicMock()
    mock_connect.return_value = mock_conn

    ok, message = check_postgres("localhost", 5432, "telemetry", "svc", "hunter2")

    assert ok is True
    assert "Connected to telemetry@localhost:5432" in message
    mock_conn.close.assert_called_once()


@patch("src.connectors.testing.psycopg2.connect")
def test_postgres_wrong_password(mock_connect: MagicMock) -> None:
    mock_connect.side_effect = psycopg2.OperationalError(
        "FATAL: password authentication failed for user \"svc\""
    )

    ok, message = check_postgres("localhost", 5432, "telemetry", "svc", "wrong")

    assert ok is False
    assert "Authentication failed" in message


@patch("src.connectors.testing.psycopg2.connect")
def test_postgres_unreachable_host(mock_connect: MagicMock) -> None:
    mock_connect.side_effect = psycopg2.OperationalError(
        "could not translate host name \"nonexistent.internal\" to address"
    )

    ok, message = check_postgres("nonexistent.internal", 5432, "telemetry", "svc", "x")

    assert ok is False
    assert "Could not connect" in message
