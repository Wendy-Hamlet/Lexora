"""Load and validate a jurisdiction YAML profile into a SourceProfile."""
from __future__ import annotations

from pathlib import Path

import yaml

from lexora.models.source import SourceProfile


def load_profile(path: Path) -> SourceProfile:
    """Load configs/jurisdictions/<iso>.yaml and validate with Pydantic."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return SourceProfile.model_validate(data)


def load_all_profiles(directory: Path) -> dict[str, SourceProfile]:
    """Load every <iso>.yaml in the given directory. Skips _template.yaml."""
    profiles: dict[str, SourceProfile] = {}
    for yaml_path in sorted(Path(directory).glob("*.yaml")):
        if yaml_path.stem.startswith("_"):
            continue
        profile = load_profile(yaml_path)
        profiles[profile.iso_code] = profile
    return profiles
