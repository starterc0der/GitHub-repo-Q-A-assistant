from __future__ import annotations

from cryptography.fernet import Fernet, InvalidToken


def _fernet(key: str) -> Fernet:
    if not key:
        raise RuntimeError(
            "connector_encryption_key is not set — refusing to store or read connector "
            "credentials in plaintext. Generate one with: python3 -c \"from "
            "cryptography.fernet import Fernet; print(Fernet.generate_key().decode())\" "
            "and put it in .env as CONNECTOR_ENCRYPTION_KEY."
        )
    try:
        return Fernet(key.encode())
    except (ValueError, TypeError) as exc:
        raise RuntimeError(
            "connector_encryption_key is set but isn't a valid Fernet key — regenerate "
            "with Fernet.generate_key()."
        ) from exc


def encrypt(plaintext: str, key: str) -> str:
    return _fernet(key).encrypt(plaintext.encode()).decode()


def decrypt(ciphertext: str, key: str) -> str:
    try:
        return _fernet(key).decrypt(ciphertext.encode()).decode()
    except InvalidToken as exc:
        # Wrong/rotated key, or the row predates the current key — never surface the
        # raw ciphertext or crash the caller into a stack trace that might get logged.
        raise RuntimeError("Could not decrypt this connector's credentials — the "
                            "encryption key may have changed since it was saved.") from exc
