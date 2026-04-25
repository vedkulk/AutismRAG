import json
import os

class MetadataManager:
    def __init__(self, base_path="metadata"):
        self.base_path = base_path
        os.makedirs(base_path, exist_ok=True)

    def _get_metadata_path(self, username, file_id):
        user_dir = os.path.join(self.base_path, username)
        os.makedirs(user_dir, exist_ok=True)
        return os.path.join(user_dir, f"{file_id}.json")

    def save_metadata(self, username, file_id, metadata):
        path = self._get_metadata_path(username, file_id)
        with open(path, "w") as f:
            json.dump(metadata, f, indent=4)

    def load_metadata(self, username, file_id):
        path = self._get_metadata_path(username, file_id)
        with open(path, "r") as f:
            return json.load(f)