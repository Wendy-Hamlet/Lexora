"""STRI adapter (WS-S, S-3) — OECD Digital STRI.

The OECD STRI is the Pillar 6 secondary index named by the RDTII guide (p.49).
(The World Bank-WTO STRI was dropped on 2026-06-22: the guide places it under
Pillars 2/3/5 — FDI/procurement/telecom — which are out of our P6/P7 scope.)

This index is not cleanly machine-fetchable (OECD's viz API is an opaque
interleaved columnar blob) and changes only ~annually, so it is served from a
curated, human-verified snapshot (``configs/secondary_stri_snapshot.yaml``)
rather than a live scrape — the robust choice agreed for S-3.

Being indices (not law-name pointers), an STRI value informs P6 at the coarse
"restrictions exist here" level: value > 0 -> presence='yes' (a P6 corroboration
for the coverage cross-check + provenance), value == 0 -> presence='no'. It never
seeds discovery (no law name). An unfilled (null) snapshot entry emits nothing —
the snapshot is never fabricated.
"""
from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import yaml

from lexora.collect.secondary.base import register_source
from lexora.models.indicator import RDTIIIndicator
from lexora.models.secondary import Presence, SecondarySignal

_SNAPSHOT_PATH = Path(__file__).resolve().parents[4] / "configs" / "secondary_stri_snapshot.yaml"


def load_stri_snapshot(path: Path | None = None) -> dict:
    """Load the snapshot into ``{source_key: {"name","scale","indicators",
    "economies": {ISO: {value, as_of, source_url}}}}``."""
    path = Path(path) if path is not None else _SNAPSHOT_PATH
    with path.open("r", encoding="utf-8") as fh:
        data = yaml.safe_load(fh) or {}
    out: dict = {}
    for s in data.get("snapshots", []) or []:
        out[str(s["source_key"])] = {
            "name": str(s.get("source_name", "")),
            "scale": str(s.get("scale", "")),
            "indicators": tuple(str(i) for i in s.get("indicators", []) or []),
            "economies": {str(k).upper(): (v or {}) for k, v in (s.get("economies", {}) or {}).items()},
        }
    return out


def _emit(
    source_key: str,
    economy: str,
    indicators: Sequence[RDTIIIndicator],
    *,
    snapshot: dict | None = None,
) -> list[SecondarySignal]:
    snap = snapshot if snapshot is not None else load_stri_snapshot()
    src = snap.get(source_key)
    if not src:
        return []
    entry = src["economies"].get(economy.upper())
    if not entry or entry.get("value") is None:  # unfilled -> never fabricate, skip
        return []
    value = float(entry["value"])
    presence = Presence.yes if value > 0 else Presence.no
    wanted = {i.submission_id for i in indicators}
    targets = [i for i in src["indicators"] if not wanted or i in wanted]
    snippet = (
        f"{src['name']}: {value} ({src['scale']}"
        + (f", as of {entry['as_of']}" if entry.get("as_of") else "")
        + ")"
    )
    return [
        SecondarySignal(
            economy=economy.upper(),
            indicator_id=ind_id,
            source_name=src["name"],
            presence=presence,
            source_url=str(entry.get("source_url", "")),
            raw_snippet=snippet,
        )
        for ind_id in targets
    ]


@register_source("oecd_dstri")
def oecd_dstri(economy: str, indicators: Sequence[RDTIIIndicator], *, snapshot: dict | None = None,
               **_) -> list[SecondarySignal]:
    """OECD Digital STRI (AU only) -> P6 restriction-presence signals."""
    return _emit("oecd_dstri", economy, indicators, snapshot=snapshot)


__all__ = ["oecd_dstri", "load_stri_snapshot"]
