from shamir_mnemonic import generate_mnemonics, combine_mnemonics

class KeyManager:
    def split_key(self, key: bytes, num_shares: int, threshold: int):
        group_threshold = 1
        groups = [(threshold, num_shares)]

        mnemonics = generate_mnemonics(
            group_threshold,
            groups,
            key
        )

        return mnemonics[0]

    def reconstruct_key(self, shares):
        return combine_mnemonics(shares)