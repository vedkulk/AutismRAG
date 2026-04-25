import os


class StorageManager:
    def __init__(self, base_path="storage", num_nodes=5):
        self.base_path = base_path
        self.num_nodes = num_nodes
        self.nodes = [os.path.join(base_path, f"node_{i}") for i in range(num_nodes)]

        self._setup_nodes()

    def _setup_nodes(self):
        for node in self.nodes:
            os.makedirs(node, exist_ok=True)

    def _get_user_paths(self, node, username, file_id):
        base = os.path.join(node, username, file_id)
        fragments_path = os.path.join(base, "fragments")
        keys_path = os.path.join(base, "keys")

        os.makedirs(fragments_path, exist_ok=True)
        os.makedirs(keys_path, exist_ok=True)

        return fragments_path, keys_path

    def store_fragments(self, username, file_id, fragments):
        locations = []

        for i, fragment in enumerate(fragments):
            node_path = self.nodes[i % self.num_nodes]
            fragments_path, _ = self._get_user_paths(node_path, username, file_id)

            file_path = os.path.join(fragments_path, f"{username}_frag_{i}.bin")

            with open(file_path, "wb") as f:
                f.write(fragment)

            locations.append(file_path)

        return locations

    def retrieve_fragments(self, locations):
        fragments = []

        for path in locations:
            with open(path, "rb") as f:
                fragments.append(f.read())

        return fragments

    def store_key_shares(self, username, file_id, shares):
        locations = []

        for i, share in enumerate(shares):
            node_path = self.nodes[i % self.num_nodes]
            _, keys_path = self._get_user_paths(node_path, username, file_id)

            file_path = os.path.join(keys_path, f"{username}_key_{i}.txt")

            with open(file_path, "w") as f:
                f.write(share)

            locations.append(file_path)

        return locations

    def retrieve_key_shares(self, locations):
        shares = []

        for path in locations:
            with open(path, "r") as f:
                shares.append(f.read())

        return shares