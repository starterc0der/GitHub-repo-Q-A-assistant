from __future__ import annotations

import os
import time

import jwt
from cryptography.exceptions import InvalidKey
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from fastapi import Depends, HTTPException, Request, Response

from src.config import settings
from src.db import connect

# OWASP 2023 minimum for PBKDF2-HMAC-SHA256.
PBKDF2_ITERATIONS = 600_000
TOKEN_TTL_SECONDS = 7 * 24 * 3600
COOKIE_NAME = "access_token"


def hash_password(password: str) -> str:
    salt = os.urandom(16)
    digest = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=salt, iterations=PBKDF2_ITERATIONS
    ).derive(password.encode())
    return f"pbkdf2_sha256${PBKDF2_ITERATIONS}${salt.hex()}${digest.hex()}"


def verify_password(password: str, stored: str) -> bool:
    try:
        algo, iterations, salt_hex, digest_hex = stored.split("$")
    except ValueError:
        return False
    if algo != "pbkdf2_sha256":
        return False
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(), length=32, salt=bytes.fromhex(salt_hex), iterations=int(iterations)
    )
    try:
        kdf.verify(password.encode(), bytes.fromhex(digest_hex))
        return True
    except InvalidKey:
        return False


def _require_jwt_secret() -> str:
    if not settings.jwt_secret:
        raise RuntimeError(
            "jwt_secret is not set — refusing to sign or verify login tokens. Generate "
            'one with: python3 -c "import secrets; print(secrets.token_hex(32))" and put '
            "it in .env as JWT_SECRET."
        )
    return settings.jwt_secret


def create_token(user_id: str, role: str, token_version: int) -> str:
    payload = {
        "sub": user_id, "role": role, "tv": token_version,
        "exp": int(time.time()) + TOKEN_TTL_SECONDS,
    }
    return jwt.encode(payload, _require_jwt_secret(), algorithm="HS256")


def decode_token(token: str) -> dict | None:
    try:
        return jwt.decode(token, _require_jwt_secret(), algorithms=["HS256"])
    except jwt.PyJWTError:
        return None


def set_auth_cookie(response: Response, token: str) -> None:
    response.set_cookie(
        COOKIE_NAME, token, max_age=TOKEN_TTL_SECONDS,
        httponly=True, samesite="lax", secure=False,
    )


def clear_auth_cookie(response: Response) -> None:
    response.delete_cookie(COOKIE_NAME)


def current_user(request: Request) -> dict:
    token = request.cookies.get(COOKIE_NAME)
    if not token:
        raise HTTPException(status_code=401, detail="Not logged in")
    payload = decode_token(token)
    if payload is None:
        raise HTTPException(status_code=401, detail="Invalid or expired session")
    with connect(settings.db_path) as conn:
        row = conn.execute("SELECT * FROM users WHERE id=?", (payload["sub"],)).fetchone()
    # token_version mismatch means the role changed or access was revoked since this
    # token was issued — reject it immediately rather than waiting for it to expire.
    if row is None or row["token_version"] != payload["tv"]:
        raise HTTPException(status_code=401, detail="Session no longer valid")
    return dict(row)


def require_admin(user: dict = Depends(current_user)) -> dict:
    if user["role"] != "admin":
        raise HTTPException(status_code=403, detail="Admin access required")
    return user


def assert_space_access(space_id: str, user: dict) -> None:
    if user["role"] == "admin":
        return
    with connect(settings.db_path) as conn:
        row = conn.execute(
            "SELECT 1 FROM space_members WHERE space_id=? AND user_id=?",
            (space_id, user["id"]),
        ).fetchone()
    if row is None:
        raise HTTPException(status_code=403, detail="You don't have access to this space")
