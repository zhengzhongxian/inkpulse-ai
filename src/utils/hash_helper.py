import hashlib

def calculate_sha256(data: bytes) -> str:
    """Calculate the SHA-256 hash of binary data."""
    sha256_hash = hashlib.sha256()
    sha256_hash.update(data)
    return sha256_hash.hexdigest()
