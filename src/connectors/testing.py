from __future__ import annotations

import psycopg2
import redis


def test_redis(
    host: str, port: int, password: str, db_index: int = 0, tls: bool = False, username: str | None = None
) -> tuple[bool, str]:
    """Real connectivity check — PING, nothing else. Never raises; every failure path
    (bad host, refused connection, wrong password, TLS mismatch) comes back as
    (False, a message safe to show the user), same contract test_postgres below has."""
    client = None
    try:
        client = redis.Redis(
            host=host, port=port, username=username or None, password=password or None, db=db_index, ssl=tls,
            socket_connect_timeout=5, socket_timeout=5,
            # RESP2, not the redis-py default of RESP3 — RESP3 makes the handshake send
            # HELLO ... AUTH ... instead of a plain AUTH, which a restricted ACL user
            # (allowed PING/GET/etc. but not HELLO) rejects even though their actual
            # permissions are fine. PING doesn't need RESP3's extra types.
            protocol=2,
        )
        client.ping()
        return True, f"Connected to {host}:{port}{' (TLS)' if tls else ''}."
    except redis.AuthenticationError:
        return False, "Authentication failed — check the password."
    except Exception as exc:
        # redis-py's connection failures surface as a mix of redis.RedisError,
        # socket.error, and ssl.SSLError depending on where the network call fails —
        # no single exception type covers "couldn't connect", so this is the boundary.
        return False, f"Could not connect: {exc}"
    finally:
        if client is not None:
            client.close()


def test_postgres(
    host: str, port: int, database: str, username: str, password: str, ssl: bool = False
) -> tuple[bool, str]:
    """Real connectivity check — connect, then immediately close. Never raises, same
    contract as test_redis above."""
    conn = None
    try:
        conn = psycopg2.connect(
            host=host, port=port, dbname=database, user=username, password=password,
            sslmode="require" if ssl else "prefer", connect_timeout=5,
        )
        return True, f"Connected to {database}@{host}:{port}{' (SSL)' if ssl else ''}."
    except psycopg2.OperationalError as exc:
        message = str(exc).strip()
        if "password authentication failed" in message or "authentication failed" in message:
            return False, "Authentication failed — check username and password."
        return False, f"Could not connect: {message}"
    except Exception as exc:
        return False, f"Could not connect: {exc}"
    finally:
        if conn is not None:
            conn.close()
