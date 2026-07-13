"""Load the RDTII indicator definitions from configs/rdtii_indicators.yaml."""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from lexora.models.indicator import RDTIIIndicator

# The hackathon's mandatory scope. The YAML carries all 12 RDTII pillars (61 regulatory
# indicators) so the engine is not hard-wired to data policy, but a run must not silently
# widen to all of them: discovery would fan out over tariffs, procurement, patents and
# telecoms, and only 6/7 have lawyer-validated gold. Other pillars are opt-in, per run.
DEFAULT_PILLARS: tuple[int, ...] = (6, 7)


def load_indicators(
    path: Path, *, pillars: Sequence[int] | None = None
) -> list[RDTIIIndicator]:
    """Flatten the pillar-grouped YAML into a list of RDTIIIndicator.

    ``pillars`` selects which RDTII pillars to load; it defaults to the mandatory Round-1
    scope (6 and 7). Pass an explicit sequence to widen it (e.g. ``pillars=[8]`` to pilot
    internet-intermediary liability), or ``pillars=range(1, 13)`` for everything.
    """
    wanted = set(DEFAULT_PILLARS if pillars is None else pillars)
    with Path(path).open("r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    indicators: list[RDTIIIndicator] = []
    for pillar in data.get("pillars", []):
        pillar_num = int(pillar["pillar"])
        if pillar_num not in wanted:
            continue
        for ind in pillar.get("indicators", []):
            indicators.append(
                RDTIIIndicator(
                    rdtii_id=str(ind["rdtii_id"]),
                    submission_id=str(ind["submission_id"]),
                    pillar=pillar_num,
                    name=ind["name"],
                    description=ind["description"].strip(),
                    long_definition=str(ind.get("long_definition", "") or "").strip(),
                    scoring_criteria=str(ind.get("scoring_criteria", "") or "").strip(),
                    possible_scores=[float(s) for s in ind.get("possible_scores", []) or []],
                    keywords=list(ind.get("keywords", []) or []),
                    discovery_queries=list(ind.get("discovery_queries", []) or []),
                )
            )
    return indicators


__all__ = ["DEFAULT_PILLARS", "load_indicators"]
