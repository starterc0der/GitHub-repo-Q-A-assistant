from __future__ import annotations

import sqlite3

from fastapi import APIRouter, Depends, HTTPException, Response
from pydantic import BaseModel

from src.auth import (
    clear_auth_cookie,
    create_token,
    current_user,
    hash_password,
    set_auth_cookie,
    verify_password,
)
from src.config import settings
from src.db import connect, new_id, now

router = APIRouter(prefix="/auth")


class SignupRequest(BaseModel):
    email: str
    name: str
    password: str


class LoginRequest(BaseModel):
    email: str
    password: str


def _public_user(row: dict) -> dict:
    return {"id": row["id"], "email": row["email"], "name": row["name"], "role": row["role"]}


@router.post("/signup")
def signup(request: SignupRequest, response: Response) -> dict:
    email = request.email.strip().lower()
    if "@" not in email or not email:
        raise HTTPException(status_code=400, detail="Enter a valid email address")
    if not request.name.strip():
        raise HTTPException(status_code=400, detail="Name is required")
    if len(request.password) < 8:
        raise HTTPException(status_code=400, detail="Password must be at least 8 characters")

    user_id = new_id()
    with connect(settings.db_path) as conn:
        is_first_user = conn.execute("SELECT COUNT(*) AS n FROM users").fetchone()["n"] == 0
        role = "admin" if is_first_user else "user"
        try:
            conn.execute(
                "INSERT INTO users (id, email, name, password_hash, role, token_version, created_at) "
                "VALUES (?, ?, ?, ?, ?, 0, ?)",
                (user_id, email, request.name.strip(), hash_password(request.password), role, now()),
            )
        except sqlite3.IntegrityError:
            raise HTTPException(status_code=409, detail="An account with this email already exists") from None
        if is_first_user:
            # Nothing was accessible to anyone before auth existed — the first account
            # inherits every pre-existing space rather than orphaning them.
            conn.execute(
                "INSERT INTO space_members (space_id, user_id, created_at) "
                "SELECT id, ?, ? FROM spaces", (user_id, now()),
            )
        row = dict(conn.execute("SELECT * FROM users WHERE id=?", (user_id,)).fetchone())

    token = create_token(row["id"], row["role"], row["token_version"])
    set_auth_cookie(response, token)
    return _public_user(row)


@router.post("/login")
def login(request: LoginRequest, response: Response) -> dict:
    email = request.email.strip().lower()
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE email=?", (email,)).fetchone()
    if row is None or not verify_password(request.password, row["password_hash"]):
        raise HTTPException(status_code=401, detail="Invalid email or password")

    row = dict(row)
    token = create_token(row["id"], row["role"], row["token_version"])
    set_auth_cookie(response, token)
    return _public_user(row)


@router.post("/logout")
def logout(response: Response) -> dict:
    clear_auth_cookie(response)
    return {"status": "logged_out"}


@router.get("/me")
def me(user: dict = Depends(current_user)) -> dict:
    return _public_user(user)
