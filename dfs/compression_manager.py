import zstandard as zstd

class CompressionManager:
    def __init__(self, level=3):
        self.compressor = zstd.ZstdCompressor(level=level)
        self.decompressor = zstd.ZstdDecompressor()

    def compress(self, data: bytes) -> bytes:
        return self.compressor.compress(data)

    def decompress(self, compressed_data: bytes) -> bytes:
        return self.decompressor.decompress(compressed_data)