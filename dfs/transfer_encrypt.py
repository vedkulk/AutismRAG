import os
import sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from compression_manager import CompressionManager
from encryption import EncryptionManager
from key_manager import KeyManager
from erasure_manager import merge_fragments
from storage_manager import StorageManager
from metadata_manager import MetadataManager


# ─────────────────────────────────────────
# Load transfer key from environment
# ─────────────────────────────────────────
def load_transfer_key() -> bytes:
    hex_key = os.environ.get("TRANSFER_KEY")
    if not hex_key:
        print("❌ TRANSFER_KEY environment variable not set.")
        print("   Set it with: export TRANSFER_KEY=<64 hex chars>")
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


# ─────────────────────────────────────────
# Reconstruct plaintext from storage
# ─────────────────────────────────────────
def reconstruct_from_storage(username: str, file_id: str) -> bytes:
    cm = CompressionManager()
    em = EncryptionManager()
    km = KeyManager()
    sm = StorageManager(num_nodes=5)
    mm = MetadataManager()

    metadata = mm.load_metadata(username, file_id)

    # Sort fragments by index
    sorted_paths = sorted(
        metadata["fragments"],
        key=lambda x: int(x.split("_frag_")[1].split(".")[0])
    )

    fragments = sm.retrieve_fragments(sorted_paths)
    merged = merge_fragments(fragments)

    shares = sm.retrieve_key_shares(metadata["key_shares"][:3])
    key = km.reconstruct_key(shares)

    decrypted = em.decrypt(
        merged,
        bytes.fromhex(metadata["nonce"]),
        key
    )

    plaintext = cm.decompress(decrypted)
    return plaintext


# ─────────────────────────────────────────
# Encrypt plaintext with transfer key
# Bundle format: [12-byte nonce][ciphertext]
# ─────────────────────────────────────────
def transfer_encrypt(plaintext: bytes, transfer_key: bytes) -> bytes:
    nonce = os.urandom(12)
    ciphertext = AESGCM(transfer_key).encrypt(nonce, plaintext, None)
    return nonce + ciphertext


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    username = input("Enter username: ").strip()
    file_id  = input("Enter file_id to transfer: ").strip()

    print("\n🔄 Reconstructing file from storage...")
    plaintext = reconstruct_from_storage(username, file_id)
    print(f"✅ Reconstructed {len(plaintext)} bytes")

    transfer_key = load_transfer_key()

    print("🔐 Encrypting for transfer...")
    bundle = transfer_encrypt(plaintext, transfer_key)

    output_path = f"{file_id}.transfer.bin"
    with open(output_path, "wb") as f:
        f.write(bundle)

    print(f"✅ Transfer bundle saved: {output_path}")
    print(f"   Size: {len(bundle)} bytes")
    print(f"\n📦 Send '{output_path}' to Device B via your pipeline.")


if __name__ == "__main__":
    main()
