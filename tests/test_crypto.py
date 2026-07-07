import pytest
from cryptography.exceptions import InvalidTag

from zkllms import crypto


def test_encrypt_then_decrypt_round_trips():
    key = crypto.generate_key()
    plaintext = b"model weights and secret inputs"

    ciphertext, nonce = crypto.encrypt_bytes(plaintext, key)
    recovered = crypto.decrypt_bytes(ciphertext, nonce, key)

    assert recovered == plaintext


def test_derive_key_is_deterministic_for_same_passphrase_and_salt():
    salt = b"a-fixed-salt-value"

    first = crypto.derive_key("correct horse", salt)
    second = crypto.derive_key("correct horse", salt)

    assert first == second
    assert len(first) == 32


def test_derive_key_differs_when_salt_differs():
    key_a = crypto.derive_key("correct horse", b"salt-one")
    key_b = crypto.derive_key("correct horse", b"salt-two")

    assert key_a != key_b


def test_decrypt_with_wrong_key_is_rejected():
    ciphertext, nonce = crypto.encrypt_bytes(b"secret", crypto.generate_key())

    with pytest.raises(InvalidTag):
        crypto.decrypt_bytes(ciphertext, nonce, crypto.generate_key())


def test_each_encryption_uses_a_fresh_nonce():
    key = crypto.generate_key()

    first_ct, first_nonce = crypto.encrypt_bytes(b"same", key)
    second_ct, second_nonce = crypto.encrypt_bytes(b"same", key)

    assert first_nonce != second_nonce
    assert first_ct != second_ct


def test_encrypt_weights_reads_file_and_round_trips(tmp_path):
    weights = tmp_path / "model.onnx"
    weights.write_bytes(b"\x00\x01onnx-weight-blob\x02\x03")
    key = crypto.generate_key()

    ciphertext, nonce = crypto.encrypt_weights(weights, key)

    assert crypto.decrypt_bytes(ciphertext, nonce, key) == weights.read_bytes()
