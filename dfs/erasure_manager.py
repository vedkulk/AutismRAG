def split_into_fragments(data: bytes, num_fragments: int):
    size = len(data)
    chunk_size = size // num_fragments

    fragments = []

    for i in range(num_fragments):
        start = i * chunk_size
        if i == num_fragments - 1:
            fragments.append(data[start:])
        else:
            fragments.append(data[start:start + chunk_size])

    return fragments


def merge_fragments(fragments):
    return b"".join(fragments)