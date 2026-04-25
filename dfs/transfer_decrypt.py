import os
import sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM


def load_transfer_key() -> bytes:
    hex_key = os.environ.get("TRANSFER_KEY")
    if not hex_key:
        print("❌ TRANSFER_KEY environment variable not set.")
        sys.exit(1)
    try:
        key = bytes.fromhex(hex_key)
    except ValueError:
        print("❌ TRANSFER_KEY is not valid hex.")
        sys.exit(1)
    if len(key) != 32:
        print(f"❌ TRANSFER_KEY must be 32 bytes (64 hex chars). Got {len(key)} bytes.")
        sys.exit(1)
    return key


def main():
    transfer_key = load_transfer_key()

    # Find the one report file
    reports = [
        f for f in os.listdir(".")
        if f.startswith("report_") and f.endswith(".transfer.bin")
    ]

    if not reports:
        print("❌ No report file found in current directory.")
        sys.exit(1)

    bundle_path = reports[0]

    with open(bundle_path, "rb") as f:
        bundle = f.read()

    nonce      = bundle[:12]
    ciphertext = bundle[12:]

    try:
        plaintext = AESGCM(transfer_key).decrypt(nonce, ciphertext, None)
    except Exception:
        print("❌ Decryption failed. Wrong key or corrupted file.")
        sys.exit(1)

    with open("report.txt", "wb") as f:
        f.write(plaintext)

    print(f"✅ Decrypted → report.txt  ({len(plaintext)} bytes)")


if __name__ == "__main__":
    main()