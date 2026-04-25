import os
import re
import sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

from compression_manager import CompressionManager
from encryption import EncryptionManager
from key_manager import KeyManager
from erasure_manager import merge_fragments
from storage_manager import StorageManager
from metadata_manager import MetadataManager


# ─────────────────────────────────────────
# Transfer key loader
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
# Transfer encrypt
# Bundle format: [12-byte nonce][ciphertext]
# ─────────────────────────────────────────
def transfer_encrypt(plaintext: bytes, transfer_key: bytes) -> bytes:
    nonce = os.urandom(12)
    ciphertext = AESGCM(transfer_key).encrypt(nonce, plaintext, None)
    return nonce + ciphertext


# ─────────────────────────────────────────
# List user files
# ─────────────────────────────────────────
def list_user_files(username, base_path="metadata"):
    user_path = os.path.join(base_path, username)

    if not os.path.exists(user_path):
        print("No files found.")
        return []

    pattern = re.compile(r".*__\d{4}-\d{2}-\d{2}__.*")

    files = []
    for file in os.listdir(user_path):
        name = os.path.splitext(file)[0]
        if pattern.match(name):
            files.append(file)

    files.sort(reverse=True)
    return files


# ─────────────────────────────────────────
# Core reconstruct + transfer encrypt
# ─────────────────────────────────────────
def reconstruct_file(username, file_id, transfer_key: bytes):
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

    # 🔐 Re-encrypt for transfer — never touch disk as plaintext
    bundle = transfer_encrypt(plaintext, transfer_key)

    output_file = f"report_{username}_{file_id}.transfer.bin"
    with open(output_file, "wb") as f:
        f.write(bundle)

    print(f"✅ Transfer bundle saved: {output_file}  ({len(bundle)} bytes)")
    print(f"   📦 Send to Device B and decrypt with transfer_decrypt.py")


# ─────────────────────────────────────────
# Main
# ─────────────────────────────────────────
def main():
    transfer_key = load_transfer_key()

    username = input("Enter username: ").strip()

    files = list_user_files(username)

    if not files:
        print("No valid files.")
        return

    print("\nOptions:")
    print("1. Retrieve single file")
    print("2. List files")
    print("3. Retrieve ALL files")

    choice = input("Select option: ").strip()

    # -------------------------
    # OPTION 1 → SINGLE FILE
    # -------------------------
    if choice == "1":
        print("\nAvailable files:")
        for i, file in enumerate(files):
            print(f"{i + 1}. {os.path.splitext(file)[0]}")

        idx = int(input("Select file number: ")) - 1
        file_id = os.path.splitext(files[idx])[0]

        reconstruct_file(username, file_id, transfer_key)

    # -------------------------
    # OPTION 2 → LIST ONLY
    # -------------------------
    elif choice == "2":
        print("\nAvailable files:")
        for file in files:
            print(os.path.splitext(file)[0])

    # -------------------------
    # OPTION 3 → ALL FILES
    # -------------------------
    elif choice == "3":
        print("\nReconstructing ALL files...\n")

        for file in files:
            file_id = os.path.splitext(file)[0]

            try:
                reconstruct_file(username, file_id, transfer_key)
            except Exception as e:
                print(f"❌ Failed: {file_id} -> {e}")

    else:
        print("Invalid option")


if __name__ == "__main__":
    main()