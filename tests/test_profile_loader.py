"""Tests for jurisdiction profile loading."""
from __future__ import annotations

from pathlib import Path

import pytest

from lexora.collect.profile_loader import load_all_profiles, load_profile
from lexora.models.source import LegalSystem, SourceType

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs" / "jurisdictions"


@pytest.mark.parametrize("iso", ["sg", "au", "my"])
def test_shipped_profiles_load(iso):
    profile = load_profile(CONFIG_DIR / f"{iso}.yaml")
    assert profile.iso_code.lower() == iso
    assert profile.legal_system in LegalSystem
    assert any(p.source_type is SourceType.primary for p in profile.portals), (
        f"{iso} must have at least one primary portal"
    )


def test_load_all_skips_template():
    profiles = load_all_profiles(CONFIG_DIR)
    assert set(profiles.keys()) == {"SG", "AU", "MY"}
