"""SHA-256 helpers — used to detect snapshot drift between collections."""
from __future__ import annotations

import hashlib
from pathlib import Path


def sha256_bytes(data: bytes) -> str:
    """Return `sha256:<hex>` for the given bytes."""
    return "sha256:" + hashlib.sha256(data).hexdigest()


def sha256_file(path: Path, chunk_size: int = 65536) -> str:
    """Stream-hash a file. Returns `sha256:<hex>`."""
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(chunk_size):
            h.update(chunk)
    return "sha256:" + h.hexdigest()
