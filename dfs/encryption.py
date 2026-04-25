import os
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

class EncryptionManager:
    def generate_key(self) -> bytes:
        return AESGCM.generate_key(bit_length=256)

    def encrypt(self, data: bytes, key: bytes):
        aesgcm = AESGCM(key)
        nonce = os.urandom(12)
        ciphertext = aesgcm.encrypt(nonce, data, None)
        return ciphertext, nonce

    def decrypt(self, ciphertext: bytes, nonce: bytes, key: bytes):
        aesgcm = AESGCM(key)
        return aesgcm.decrypt(nonce, ciphertext, None)