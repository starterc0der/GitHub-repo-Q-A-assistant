from __future__ import annotations

from unittest.mock import patch

import pytest
from cryptography.fernet import Fernet
from fastapi import HTTPException

from src.api.connector_routes import (
    ConnectorCredentialsRequest,
    CreateConnectorRequest,
    ReplaceCredentialsRequest,
    create_connector,
    delete_connector,
    dry_run_test,
    list_connectors,
    replace_credentials,
    retest_connector,
)
from src.config import settings
from src.db import connect, init_db, new_id, now


@pytest.fixture(autouse=True)
def _isolated_db_and_key(tmp_path, monkeypatch):
    db_path = str(tmp_path / "app.db")
    monkeypatch.setattr(settings, "db_path", db_path)
    monkeypatch.setattr(settings, "connector_encryption_key", Fernet.generate_key().decode())
    init_db(db_path)


def _make_space() -> str:
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
    return space_id


def _redis_request(**overrides) -> CreateConnectorRequest:
    fields = dict(kind="redis", host="redis.internal", port=6379, password="hunter2", name="Prod Redis")
    fields.update(overrides)
    return CreateConnectorRequest(**fields)


def test_redis_username_reaches_the_connectivity_check() -> None:
    # ACL-secured Redis (e.g. a managed/hosted instance) authenticates a specific user,
    # not just a password — dropping the username here means a correctly-configured ACL
    # user fails authentication for no real reason.
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")) as mock_test:
        dry_run_test(space_id, ConnectorCredentialsRequest(kind="redis", host="h", port=6379, password="p", username="watcodev"))

    assert mock_test.call_args.kwargs["username"] == "watcodev"


def test_dry_run_test_does_not_persist_anything() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")):
        result = dry_run_test(space_id, ConnectorCredentialsRequest(kind="redis", host="h", port=6379, password="p"))

    assert result == {"ok": True, "message": "Connected."}
    assert list_connectors(space_id)["connectors"] == []


def test_create_connector_rejects_a_failing_test() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(False, "Authentication failed.")):
        with pytest.raises(HTTPException) as exc_info:
            create_connector(space_id, _redis_request())

    assert exc_info.value.status_code == 400
    assert list_connectors(space_id)["connectors"] == []  # nothing saved on a failed test


def test_create_connector_saves_and_encrypts_the_password() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")):
        created = create_connector(space_id, _redis_request())

    assert created["status"] == "connected"
    assert created["name"] == "Prod Redis"
    assert "password" not in created and "encrypted_password" not in created

    with connect(settings.db_path) as conn:
        stored = conn.execute(
            "SELECT encrypted_password FROM connectors WHERE id=?", (created["id"],)
        ).fetchone()
    assert stored["encrypted_password"] != "hunter2"  # never plaintext at rest

    listed = list_connectors(space_id)["connectors"]
    assert len(listed) == 1
    assert listed[0]["id"] == created["id"]
    assert len(listed[0]["history"]) == 1


def test_retest_connector_uses_the_stored_decrypted_password() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")) as mock_test:
        created = create_connector(space_id, _redis_request())

    with patch("src.api.connector_routes.test_redis", return_value=(False, "now down")) as mock_retest:
        result = retest_connector(created["id"])

    # The decrypted password round-tripped correctly into the retest call.
    assert mock_retest.call_args.args[2] == "hunter2"
    assert result["status"] == "error"
    assert len(result["history"]) == 2


def test_replace_credentials_keeps_host_and_tests_the_new_password() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")):
        created = create_connector(space_id, _redis_request())

    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")) as mock_test:
        updated = replace_credentials(created["id"], ReplaceCredentialsRequest(password="new-pass"))

    assert mock_test.call_args.args[0] == "redis.internal"  # host unchanged
    assert mock_test.call_args.args[2] == "new-pass"
    assert updated["host"] == "redis.internal"


def test_replace_credentials_rejects_a_failing_test_and_keeps_old_password() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")):
        created = create_connector(space_id, _redis_request())
    with connect(settings.db_path) as conn:
        before = conn.execute(
            "SELECT encrypted_password FROM connectors WHERE id=?", (created["id"],)
        ).fetchone()["encrypted_password"]

    with patch("src.api.connector_routes.test_redis", return_value=(False, "bad password")):
        with pytest.raises(HTTPException):
            replace_credentials(created["id"], ReplaceCredentialsRequest(password="wrong"))

    with connect(settings.db_path) as conn:
        after = conn.execute(
            "SELECT encrypted_password FROM connectors WHERE id=?", (created["id"],)
        ).fetchone()["encrypted_password"]
    assert before == after


def test_delete_connector_removes_it() -> None:
    space_id = _make_space()
    with patch("src.api.connector_routes.test_redis", return_value=(True, "Connected.")):
        created = create_connector(space_id, _redis_request())

    delete_connector(created["id"])

    assert list_connectors(space_id)["connectors"] == []


def test_postgres_connector_requires_a_database_name() -> None:
    space_id = _make_space()
    request = CreateConnectorRequest(kind="postgres", host="h", port=5432, password="p", name="DB")

    with pytest.raises(HTTPException) as exc_info:
        create_connector(space_id, request)

    assert "Database name is required" in exc_info.value.detail
