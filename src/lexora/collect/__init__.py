"""Stage 1 — collection.

Loads jurisdiction profiles, crawls portals (http / sitemap / playwright / api),
hashes raw bytes, and writes immutable RawDocument records.

Owner: <to be assigned>.
"""
from lexora.collect.hasher import sha256_bytes, sha256_file
from lexora.collect.profile_loader import load_profile

__all__ = ["load_profile", "sha256_bytes", "sha256_file"]
