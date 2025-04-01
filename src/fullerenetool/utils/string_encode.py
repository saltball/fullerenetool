import base64


def compress_integer_list(int_list):
    if not int_list or len(int_list) == 0:
        return "none"

    # simple connect with '_'
    simple_str = "_".join(map(str, int_list))
    if len(simple_str) <= 100:
        return simple_str

    # compress
    compressed_bytes = bytearray()
    for num in int_list:
        # encode
        while num >= 128:
            compressed_bytes.append((num & 0x7F) | 0x80)
            num >>= 7
        compressed_bytes.append(num)

    # encode with base64
    compressed_str = base64.b64encode(compressed_bytes).decode("utf-8")

    return compressed_str


def decompress_string_to_integer_list(compressed_str):
    if compressed_str == "none":
        return []

    try:
        # for string containing '_', return the list
        if "_" in compressed_str:
            return list(map(int, compressed_str.split("_")))

        # decode base64
        decoded_bytes = base64.b64decode(compressed_str)
        # decode
        int_list = []
        index = 0
        while index < len(decoded_bytes):
            value = 0
            shift = 0
            while True:
                byte = decoded_bytes[index]
                value |= (byte & 0x7F) << shift
                shift += 7
                index += 1
                if not byte & 0x80:
                    break
            int_list.append(value)

        return int_list
    except Exception as e:
        print(f"Error during decompression: {e}")
        return []


if __name__ == "__main__":
    # Example usage
    int_list = [213123, 1231231, 2131231] * 128
    encoded_bytes = compress_integer_list(int_list)
    print(encoded_bytes)
    decoded_list = decompress_string_to_integer_list(encoded_bytes)
    print(decoded_list)
