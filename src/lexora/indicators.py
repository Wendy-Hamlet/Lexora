"""Load the RDTII indicator definitions from configs/rdtii_indicators.yaml."""
from __future__ import annotations

from pathlib import Path

import yaml

from lexora.models.indicator import RDTIIIndicator


def load_indicators(path: Path) -> list[RDTIIIndicator]:
    """Flatten the pillar-grouped YAML into a list of RDTIIIndicator."""
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    indicators: list[RDTIIIndicator] = []
    for pillar in data.get("pillars", []):
        pillar_num = int(pillar["pillar"])
        for ind in pillar.get("indicators", []):
            indicators.append(
                RDTIIIndicator(
                    id=str(ind["id"]),
                    pillar=pillar_num,
                    name=ind["name"],
                    description=ind["description"].strip(),
                    keywords=list(ind.get("keywords", []) or []),
                )
            )
    return indicators


__all__ = ["load_indicators"]
