from compression_manager import CompressionManager
from encryption import EncryptionManager
from key_manager import KeyManager
from erasure_manager import split_into_fragments, merge_fragments
from storage_manager import StorageManager
from metadata_manager import MetadataManager
from datetime import datetime
import os
import re


# 🔹 Generate versioned file_id
def generate_file_id(username, base_path, name_only, custom_id, timestamp):
    custom_id = custom_id.lower().replace(" ", "")
    pattern = re.compile(rf"{re.escape(custom_id)}(_v(\d+))?$")

    max_version = 0

    # Scan ALL nodes
    if os.path.exists(base_path):
        for node in os.listdir(base_path):
            node_path = os.path.join(base_path, node)

            if not os.path.isdir(node_path):
                continue

            user_path = os.path.join(node_path, username)

            if not os.path.exists(user_path):
                continue

            for folder in os.listdir(user_path):
                if "__" not in folder:
                    continue

                last_part = folder.split("__")[-1]

                match = pattern.fullmatch(last_part)
                if match:
                    if match.group(2):  # version exists
                        version = int(match.group(2))
                        max_version = max(max_version, version)
                    else:
                        max_version = max(max_version, 1)

    if max_version == 0:
        final_custom_id = custom_id
    else:
        final_custom_id = f"{custom_id}_v{max_version + 1}"

    return f"{name_only}__{timestamp}__{final_custom_id}"


# 🔹 List available files
def list_available_files():
    script_dir = os.path.dirname(os.path.abspath(__file__))

    files = [f for f in os.listdir(script_dir) if f.endswith(".txt")]

    if not files:
        print("No .txt files found!")
        return None

    print("\nAvailable files:")
    for i, file in enumerate(files):
        size = os.path.getsize(os.path.join(script_dir, file))
        print(f"{i + 1}. {file} ({size} bytes)")

    return files, script_dir

def main():
    username = input("Enter username: ").strip()

    # 🔹 Select file
    result = list_available_files()
    if not result:
        return

    files, base_dir = result

    try:
        choice = int(input("Select file number: ")) - 1
    except ValueError:
        print("Invalid input!")
        return

    if choice < 0 or choice >= len(files):
        print("Invalid choice!")
        return

    selected_file = os.path.join(base_dir, files[choice])

    # 🔹 File info
    filename = os.path.basename(selected_file)
    name_only = os.path.splitext(filename)[0]

    # 🔹 Custom label
    custom_id = input("Enter custom label (e.g., lungvisit1): ").strip()

    # 🔹 Timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d__%H-%M-%S")

    # 🔹 Generate file_id
    file_id = generate_file_id(
        username,
        "storage",
        name_only,
        custom_id,
        timestamp
    )

    num_fragments = 5
    

    # 🔹 Initialize managers
    cm = CompressionManager()
    em = EncryptionManager()
    km = KeyManager()
    sm = StorageManager(num_nodes=5)
    mm = MetadataManager()

    # 🔹 Read file
    with open(selected_file, "rb") as f:
        data = f.read()

    # 🔹 Compress
    compressed = cm.compress(data)

    # 🔹 Generate key
    key = em.generate_key()

    # 🔹 Split key
    shares = km.split_key(key, num_shares=5, threshold=3)

    # 🔹 Encrypt
    ciphertext, nonce = em.encrypt(compressed, key)

    # 🔹 Split into fragments
    num_fragments = len(sm.nodes) * 2   # e.g. 6 fragments for 3 nodes
    # fragments = split_into_fragments(encoded_data, num_fragments=num_fragments)
    fragments = split_into_fragments(ciphertext, num_fragments)
    # 🔹 Store fragments
    fragment_locations = sm.store_fragments(username, file_id, fragments)

    # 🔹 Store key shares
    key_locations = sm.store_key_shares(username, file_id, shares)

    # 🔹 Save metadata
    metadata = {
        "username": username,
        "file_id": file_id,
        "original_filename": filename,
        "custom_id": custom_id,
        "timestamp": timestamp,
        "fragments": fragment_locations,
        "key_shares": key_locations,
        "nonce": nonce.hex()
    }

    mm.save_metadata(username, file_id, metadata)

    print("\n--- Stored Successfully ---\n")

    # =========================
    # 🔄 Retrieval Phase
    # =========================

    loaded_metadata = mm.load_metadata(username, file_id)

    retrieved_fragments = sm.retrieve_fragments(loaded_metadata["fragments"])

    merged_data = merge_fragments(retrieved_fragments)

    recovered_ciphertext = merged_data

    retrieved_shares = sm.retrieve_key_shares(
        loaded_metadata["key_shares"][:3]
    )

    recovered_key = km.reconstruct_key(retrieved_shares)

    decrypted = em.decrypt(
        recovered_ciphertext,
        bytes.fromhex(loaded_metadata["nonce"]),
        recovered_key
    )

    final_data = cm.decompress(decrypted)

    print("Restored correctly:", final_data == data)

    # 🔹 Debug: show stored folders per node
    print("\nStored folders:")
    if os.path.exists("storage"):
        for node in os.listdir("storage"):
            path = os.path.join("storage", node, username)
            if os.path.exists(path):
                print(f"{node} -> {os.listdir(path)}")

    print("Generated file_id:", file_id)


if __name__ == "__main__":
    main()