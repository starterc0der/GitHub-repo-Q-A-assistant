from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.main import app
from src.config import settings
from src.db import connect, init_db, new_id, now


@pytest.fixture(autouse=True)
def _isolated_db_and_secret(tmp_path, monkeypatch):
    monkeypatch.setattr(settings, "db_path", str(tmp_path / "app.db"))
    monkeypatch.setattr(settings, "jwt_secret", "x" * 32)
    init_db(settings.db_path)


def _seed_preexisting_space() -> str:
    space_id = new_id()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO spaces (id, name, created_at, updated_at) VALUES (?, ?, ?, ?)",
            (space_id, "Preexisting Space", now(), now()),
        )
    return space_id


def _signup(client: TestClient, email: str) -> dict:
    r = client.post("/auth/signup", json={"email": email, "name": "Name", "password": "hunter2pass"})
    assert r.status_code == 200, r.text
    return r.json()


def test_first_signup_becomes_admin_and_inherits_existing_spaces() -> None:
    space_id = _seed_preexisting_space()
    admin = TestClient(app)

    user = _signup(admin, "admin@example.com")

    assert user["role"] == "admin"
    r = admin.get("/spaces")
    assert [s["id"] for s in r.json()["spaces"]] == [space_id]


def test_second_signup_is_a_regular_user_with_no_spaces() -> None:
    _seed_preexisting_space()
    _signup(TestClient(app), "admin@example.com")
    bob = TestClient(app)

    user = _signup(bob, "bob@example.com")

    assert user["role"] == "user"
    assert bob.get("/spaces").json()["spaces"] == []


def test_duplicate_email_signup_is_rejected() -> None:
    client = TestClient(app)
    _signup(client, "dup@example.com")

    r = client.post("/auth/signup", json={"email": "dup@example.com", "name": "Again", "password": "hunter2pass"})

    assert r.status_code == 409


def test_login_with_wrong_password_is_rejected() -> None:
    client = TestClient(app)
    _signup(client, "user@example.com")

    r = TestClient(app).post("/auth/login", json={"email": "user@example.com", "password": "wrong-password"})

    assert r.status_code == 401


def test_regular_user_is_rejected_from_admin_only_routes() -> None:
    space_id = _seed_preexisting_space()
    _signup(TestClient(app), "admin@example.com")
    bob = TestClient(app)
    _signup(bob, "bob@example.com")

    assert bob.post("/spaces", json={"name": "Bob's space"}).status_code == 403
    assert bob.get(f"/spaces/{space_id}/insights").status_code == 403
    assert bob.get(f"/spaces/{space_id}/connectors").status_code == 403
    assert bob.get(f"/spaces/{space_id}/sources").status_code == 403
    assert bob.get("/users").status_code == 403


def test_regular_user_cannot_access_an_unassigned_space() -> None:
    space_id = _seed_preexisting_space()
    _signup(TestClient(app), "admin@example.com")
    bob = TestClient(app)
    _signup(bob, "bob@example.com")

    assert bob.get(f"/spaces/{space_id}").status_code == 403
    assert bob.post(f"/spaces/{space_id}/chats").status_code == 403


def test_admin_can_assign_a_space_and_the_user_gains_access() -> None:
    space_id = _seed_preexisting_space()
    admin = TestClient(app)
    _signup(admin, "admin@example.com")
    bob = TestClient(app)
    bob_user = _signup(bob, "bob@example.com")

    r = admin.post(f"/users/{bob_user['id']}/spaces/{space_id}")
    assert r.status_code == 200

    assert bob.get(f"/spaces/{space_id}").status_code == 200
    assert bob.post(f"/spaces/{space_id}/chats").status_code == 200


def test_unassigning_a_space_revokes_the_users_existing_session() -> None:
    # Not merely losing access to that one space — the session itself is invalidated
    # via token_version, so every route rejects the stale cookie immediately.
    space_id = _seed_preexisting_space()
    admin = TestClient(app)
    _signup(admin, "admin@example.com")
    bob = TestClient(app)
    bob_user = _signup(bob, "bob@example.com")
    admin.post(f"/users/{bob_user['id']}/spaces/{space_id}")
    assert bob.get(f"/spaces/{space_id}").status_code == 200

    r = admin.delete(f"/users/{bob_user['id']}/spaces/{space_id}")
    assert r.status_code == 200

    assert bob.get(f"/spaces/{space_id}").status_code == 401


def test_role_change_revokes_the_existing_session_but_a_fresh_login_reflects_it() -> None:
    space_id = _seed_preexisting_space()
    admin = TestClient(app)
    _signup(admin, "admin@example.com")
    bob = TestClient(app)
    bob_user = _signup(bob, "bob@example.com")

    r = admin.patch(f"/users/{bob_user['id']}", json={"role": "admin"})
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert bob.get("/spaces").status_code == 401  # stale cookie, old role/token_version

    bob2 = TestClient(app)
    r = bob2.post("/auth/login", json={"email": "bob@example.com", "password": "hunter2pass"})
    assert r.status_code == 200 and r.json()["role"] == "admin"
    assert bob2.get(f"/spaces/{space_id}").status_code == 200  # admin now, no explicit assignment needed
    assert bob2.get("/users").status_code == 200


def test_logged_out_request_is_rejected() -> None:
    r = TestClient(app).get("/spaces")

    assert r.status_code == 401
