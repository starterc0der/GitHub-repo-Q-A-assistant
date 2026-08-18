from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from src.api.routes import _space_row
from src.config import settings
from src.connectors.testing import test_postgres, test_redis
from src.crypto import decrypt, encrypt
from src.db import connect, new_id, now

router = APIRouter()

# Public row shape never includes encrypted_password or any plaintext credential —
# every SELECT below names its columns explicitly rather than using SELECT *, so a
# future column addition can't silently leak into an API response.
_PUBLIC_COLUMNS = (
    "id, space_id, kind, name, host, port, database, username, db_index, tls, ssl, "
    "status, last_tested_at, created_at"
)


class ConnectorCredentialsRequest(BaseModel):
    kind: str
    host: str
    port: int
    database: str | None = None
    username: str | None = None
    password: str
    db_index: int | None = None
    tls: bool = False
    ssl: bool = False


class CreateConnectorRequest(ConnectorCredentialsRequest):
    name: str = ""


class ReplaceCredentialsRequest(BaseModel):
    username: str | None = None
    password: str


def _run_test(req: ConnectorCredentialsRequest) -> tuple[bool, str]:
    if req.kind == "redis":
        return test_redis(req.host, req.port, req.password, req.db_index or 0, req.tls, username=req.username)
    if req.kind == "postgres":
        if not req.database:
            return False, "Database name is required."
        return test_postgres(req.host, req.port, req.database, req.username or "", req.password, req.ssl)
    return False, f"Unknown connector kind {req.kind!r}."


def _connector_row(connector_id: str) -> dict[str, object]:
    with connect(settings.db_path) as conn:
        row = conn.execute(
            f"SELECT {_PUBLIC_COLUMNS} FROM connectors WHERE id=?", (connector_id,)
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=404, detail=f"No connector {connector_id!r}")
    return dict(row)


def _history(connector_id: str, limit: int = 5) -> list[dict[str, object]]:
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            "SELECT ok, message, tested_at FROM connector_test_history "
            "WHERE connector_id=? ORDER BY tested_at DESC LIMIT ?",
            (connector_id, limit),
        ).fetchall()
    return [dict(r) for r in rows]


def _record_test(connector_id: str, ok: bool, message: str) -> None:
    timestamp = now()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO connector_test_history (id, connector_id, ok, message, tested_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (new_id(), connector_id, int(ok), message, timestamp),
        )
        conn.execute(
            "UPDATE connectors SET status=?, last_tested_at=? WHERE id=?",
            ("connected" if ok else "error", timestamp, connector_id),
        )


@router.get("/spaces/{space_id}/connectors")
def list_connectors(space_id: str) -> dict[str, object]:
    _space_row(space_id)
    with connect(settings.db_path) as conn:
        rows = conn.execute(
            f"SELECT {_PUBLIC_COLUMNS} FROM connectors WHERE space_id=? ORDER BY created_at DESC",
            (space_id,),
        ).fetchall()
    connectors = [dict(r) for r in rows]
    for c in connectors:
        c["history"] = _history(c["id"])
    return {"connectors": connectors}


@router.post("/spaces/{space_id}/connectors/test")
def dry_run_test(space_id: str, request: ConnectorCredentialsRequest) -> dict[str, object]:
    """Tests the given credentials without persisting anything — the interactive "Test
    connection" button in the add/replace form. The actual save below re-tests
    server-side regardless; this endpoint exists purely for inline UI feedback before
    the user commits to saving."""
    _space_row(space_id)
    ok, message = _run_test(request)
    return {"ok": ok, "message": message}


@router.post("/spaces/{space_id}/connectors")
def create_connector(space_id: str, request: CreateConnectorRequest) -> dict[str, object]:
    """Never trusts a client-reported test result — tests again here, and only persists
    (with the password encrypted) if THIS test passes. A connector can never be saved
    in an unverified state."""
    _space_row(space_id)
    ok, message = _run_test(request)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    connector_id = new_id()
    default_name = "Redis connector" if request.kind == "redis" else "Postgres connector"
    timestamp = now()
    with connect(settings.db_path) as conn:
        conn.execute(
            "INSERT INTO connectors (id, space_id, kind, name, host, port, database, "
            "username, encrypted_password, db_index, tls, ssl, status, last_tested_at, created_at) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'connected', ?, ?)",
            (
                connector_id, space_id, request.kind, request.name.strip() or default_name,
                request.host, request.port, request.database, request.username,
                encrypt(request.password, settings.connector_encryption_key),
                request.db_index, int(request.tls), int(request.ssl), timestamp, timestamp,
            ),
        )
        conn.execute(
            "INSERT INTO connector_test_history (id, connector_id, ok, message, tested_at) "
            "VALUES (?, ?, 1, ?, ?)",
            (new_id(), connector_id, message, timestamp),
        )
    return {**_connector_row(connector_id), "history": _history(connector_id)}


@router.post("/connectors/{connector_id}/test")
def retest_connector(connector_id: str) -> dict[str, object]:
    """Re-tests a saved connector's STORED credentials — the detail view's manual
    re-verify, not the add-form's dry run."""
    row = _connector_row(connector_id)
    with connect(settings.db_path) as conn:
        encrypted = conn.execute(
            "SELECT encrypted_password FROM connectors WHERE id=?", (connector_id,)
        ).fetchone()["encrypted_password"]
    password = decrypt(encrypted, settings.connector_encryption_key)
    req = ConnectorCredentialsRequest(
        kind=row["kind"], host=row["host"], port=row["port"], database=row["database"],
        username=row["username"], password=password, db_index=row["db_index"],
        tls=bool(row["tls"]), ssl=bool(row["ssl"]),
    )
    ok, message = _run_test(req)
    _record_test(connector_id, ok, message)
    return {**_connector_row(connector_id), "history": _history(connector_id)}


@router.post("/connectors/{connector_id}/replace")
def replace_credentials(connector_id: str, request: ReplaceCredentialsRequest) -> dict[str, object]:
    """Host/port/database stay fixed — only the secret (and username, for Postgres)
    changes. Tests the NEW credentials before committing, same "never save unverified"
    rule create_connector follows."""
    row = _connector_row(connector_id)
    req = ConnectorCredentialsRequest(
        kind=row["kind"], host=row["host"], port=row["port"], database=row["database"],
        username=request.username or row["username"], password=request.password,
        db_index=row["db_index"], tls=bool(row["tls"]), ssl=bool(row["ssl"]),
    )
    ok, message = _run_test(req)
    if not ok:
        raise HTTPException(status_code=400, detail=message)

    with connect(settings.db_path) as conn:
        conn.execute(
            "UPDATE connectors SET username=?, encrypted_password=? WHERE id=?",
            (req.username, encrypt(request.password, settings.connector_encryption_key), connector_id),
        )
    _record_test(connector_id, ok, message)
    return {**_connector_row(connector_id), "history": _history(connector_id)}


@router.delete("/connectors/{connector_id}")
def delete_connector(connector_id: str) -> dict[str, str]:
    _connector_row(connector_id)
    with connect(settings.db_path) as conn:
        conn.execute("DELETE FROM connectors WHERE id=?", (connector_id,))
    return {"status": "deleted"}
