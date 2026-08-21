from __future__ import annotations

import pytest
from fastapi import HTTPException

from src.auth import (
    assert_space_access,
    create_token,
    current_user,
    decode_token,
    hash_password,
    require_admin,
    verify_password,
)
from src.config import settings
from src.db import connect, init_db, new_id, now


@pytest.fixture(autouse=True)
def _isolated_db_and_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "jwt_secret", "test-secret-that-is-at-least-32-bytes-long")
    init_db(settings.db_path)


def _make_user(role: str = "user", token_version: int = 0) -> dict:
    user_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO users (id, email, name, password_hash, role, token_version, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            (user_id, f"{user_id}@example.com", "Test User", hash_password("hunter2"), role, token_version, now()),
        )
    return {"id": user_id, "role": role, "token_version": token_version}


class _FakeRequest:
    def __init__(self, token: str | None) -> None:
        self.cookies = {"access_token": token} if token else {}


def test_hash_password_round_trip() -> None:
    hashed = hash_password("hunter2")

    assert hashed != "hunter2"
    assert verify_password("hunter2", hashed)
    assert not verify_password("wrong-password", hashed)


def test_create_and_decode_token() -> None:
    token = create_token("user-1", "admin", 0)

    payload = decode_token(token)

    assert payload["sub"] == "user-1"
    assert payload["role"] == "admin"
    assert payload["tv"] == 0


def test_decode_token_rejects_garbage() -> None:
    assert decode_token("not-a-real-token") is None


def test_current_user_rejects_missing_cookie() -> None:
    with pytest.raises(HTTPException) as exc:
        current_user(_FakeRequest(None))
    assert exc.value.status_code == 401


def test_current_user_accepts_a_valid_token() -> None:
    user = _make_user(role="admin", token_version=0)
    token = create_token(user["id"], "admin", 0)

    result = current_user(_FakeRequest(token))

    assert result["id"] == user["id"]
    assert result["role"] == "admin"


def test_current_user_rejects_a_stale_token_version() -> None:
    # Simulates a role change / access revocation bumping token_version after the
    # token was issued — the old token must stop working immediately, not on expiry.
    user = _make_user(role="admin", token_version=0)
    token = create_token(user["id"], "admin", 0)
    with connect(settings.db_path) as conn:
        conn.execute("UPDATE users SET token_version=1 WHERE id=?", (user["id"],))

    with pytest.raises(HTTPException) as exc:
        current_user(_FakeRequest(token))
    assert exc.value.status_code == 401


def test_require_admin_rejects_a_regular_user() -> None:
    with pytest.raises(HTTPException) as exc:
        require_admin(user={"id": "u1", "role": "user"})
    assert exc.value.status_code == 403


def test_require_admin_accepts_an_admin() -> None:
    admin = {"id": "u1", "role": "admin"}
    assert require_admin(user=admin) is admin


def test_assert_space_access_allows_admin_to_any_space() -> None:
    assert_space_access("any-space-id", {"id": "u1", "role": "admin"})


def test_assert_space_access_rejects_unassigned_regular_user() -> None:
    user = _make_user(role="user")

    with pytest.raises(HTTPException) as exc:
        assert_space_access("some-space", user)
    assert exc.value.status_code == 403


def test_assert_space_access_allows_assigned_regular_user() -> None:
    user = _make_user(role="user")
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Demo", now(), now()),
        )
        conn.execute(
            "INSERT INTO space_members (space_id, user_id, created_at) VALUES (?, ?, ?)",
            (space_id, user["id"], now()),
        )

    assert_space_access(space_id, user)  # does not raise
