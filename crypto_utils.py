from __future__ import annotations

"""
Simple application-level encryption helpers using Fernet (symmetric crypto).

We encrypt:
  - chunk texts before writing them to any persistent vector store (Chroma)
  - and decrypt after retrieval, before passing to the RAG pipeline / UI.

The same key is reused across:
  - Phase 1 KB Chroma store
  - Phase 2 per-session report store

Key management:
  - looks for environment variable RAG_FERNET_KEY
  - else looks for ./secret.key file (project root)
  - else generates a new key and saves it to ./secret.key
"""

import os
from pathlib import Path
from functools import lru_cache

from cryptography.fernet import Fernet


KEY_ENV_VAR = "RAG_FERNET_KEY"
KEY_FILE_PATH = Path("./secret.key")


@lru_cache(maxsize=1)
def get_fernet() -> Fernet:
    # 1) env var
    key = os.getenv(KEY_ENV_VAR)
    if key:
        return Fernet(key.encode("utf-8"))

    # 2) key file
    if KEY_FILE_PATH.exists():
        raw = KEY_FILE_PATH.read_bytes().strip()
        return Fernet(raw)

    # 3) generate & persist (development convenience)
    new_key = Fernet.generate_key()
    KEY_FILE_PATH.write_bytes(new_key)
    return Fernet(new_key)


def encrypt_text(plain: str) -> str:
    if plain is None:
        return ""
    f = get_fernet()
    token = f.encrypt(plain.encode("utf-8"))
    return token.decode("utf-8")


def decrypt_text(ciphertext: str) -> str:
    if not ciphertext:
        return ""
    f = get_fernet()
    try:
        plain = f.decrypt(ciphertext.encode("utf-8"))
        return plain.decode("utf-8")
    except Exception:
        # if data was stored unencrypted earlier, fall back to raw text
        return ciphertext

