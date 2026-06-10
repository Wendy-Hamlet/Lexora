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
                    rdtii_id=str(ind["rdtii_id"]),
                    submission_id=str(ind["submission_id"]),
                    pillar=pillar_num,
                    name=ind["name"],
                    description=ind["description"].strip(),
                    scoring_criteria=str(ind.get("scoring_criteria", "") or "").strip(),
                    possible_scores=[float(s) for s in ind.get("possible_scores", []) or []],
                    keywords=list(ind.get("keywords", []) or []),
                    discovery_queries=list(ind.get("discovery_queries", []) or []),
                )
            )
    return indicators


__all__ = ["load_indicators"]
