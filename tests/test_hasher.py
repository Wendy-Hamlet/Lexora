from __future__ import annotations

from lexora.collect.hasher import sha256_bytes, sha256_file


def test_sha256_bytes_format():
    h = sha256_bytes(b"hello")
    assert h.startswith("sha256:")
    assert len(h) == len("sha256:") + 64


def test_sha256_file_matches_bytes(tmp_path):
    p = tmp_path / "x.bin"
    p.write_bytes(b"hello")
    assert sha256_file(p) == sha256_bytes(b"hello")
