import os
from pathlib import Path

from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC

_KEY_SIZE = 32
_NONCE_SIZE = 12
_PBKDF2_ITERATIONS = 100_000


def generate_key() -> bytes:
    return os.urandom(_KEY_SIZE)


def derive_key(passphrase: str, salt: bytes) -> bytes:
    kdf = PBKDF2HMAC(
        algorithm=hashes.SHA256(),
        length=_KEY_SIZE,
        salt=salt,
        iterations=_PBKDF2_ITERATIONS,
    )
    return kdf.derive(passphrase.encode("utf-8"))


def encrypt_bytes(data: bytes, key: bytes) -> tuple[bytes, bytes]:
    nonce = os.urandom(_NONCE_SIZE)
    ciphertext = AESGCM(key).encrypt(nonce, data, None)
    return ciphertext, nonce


def decrypt_bytes(ciphertext: bytes, nonce: bytes, key: bytes) -> bytes:
    return AESGCM(key).decrypt(nonce, ciphertext, None)


def encrypt_weights(path: Path, key: bytes) -> tuple[bytes, bytes]:
    return encrypt_bytes(Path(path).read_bytes(), key)
