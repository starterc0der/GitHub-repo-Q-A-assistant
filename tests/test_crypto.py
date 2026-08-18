from __future__ import annotations

import pytest
from cryptography.fernet import Fernet

from src.crypto import decrypt, encrypt

KEY = Fernet.generate_key().decode()


def test_encrypt_decrypt_round_trip() -> None:
    ciphertext = encrypt("hunter2", KEY)

    assert ciphertext != "hunter2"
    assert decrypt(ciphertext, KEY) == "hunter2"


def test_encrypt_raises_without_a_key() -> None:
    with pytest.raises(RuntimeError, match="not set"):
        encrypt("hunter2", "")


def test_encrypt_raises_on_an_invalid_key() -> None:
    with pytest.raises(RuntimeError, match="isn't a valid Fernet key"):
        encrypt("hunter2", "not-a-real-key")


def test_decrypt_raises_when_the_key_no_longer_matches() -> None:
    ciphertext = encrypt("hunter2", KEY)
    other_key = Fernet.generate_key().decode()

    with pytest.raises(RuntimeError, match="Could not decrypt"):
        decrypt(ciphertext, other_key)
