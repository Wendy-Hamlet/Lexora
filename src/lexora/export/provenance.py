"""Whether a run's answers came from the sources or from a recording — and saying so.

The live pitch runs `--offline`, replaying a recorded corpus so the engine produces real
output in seconds without depending on three portals staying reachable. That is honest
engineering and a dishonest artifact if it is not labelled: a reviewer looking at the CSV,
or at the screen, cannot tell a replayed run from a cold live one, and the difference is
exactly what they are there to judge.

WHERE THE LINE IS. A run is a DEMONSTRATION when the network was replayed
(``LEXORA_HTTP_CACHE=replay``): nothing was fetched, so nothing can be claimed as current.
A warm VERDICT cache is a different thing and is not a demonstration -- the judge cache is
keyed on the rendered prompt and the clause text, so a hit is the same model answering the
same question, and a submission run may legitimately use it. Both facts are reported; only
the first renames the artifacts.
"""
from __future__ import annotations

from lexora.collect import http_cache

BANNER = "DEMONSTRATION RUN — replayed from a recorded corpus, NOT a live fetch"
PREFIX = "DEMO_"


def is_demonstration() -> bool:
    """True when this run served its documents from a recording instead of the network."""
    return http_cache.mode() == http_cache.REPLAY


def provenance(*, judged: int = 0, from_cache: int = 0) -> dict:
    """The facts a reader needs to weigh the output, in one place.

    ``judged`` / ``from_cache`` are the verifier's counters: clauses actually put to the
    model, versus clauses answered from the verdict store.
    """
    total = judged + from_cache
    return {
        "demonstration": is_demonstration(),
        "network": {"off": "live", "record": "live (recorded)",
                    "replay": "replayed from recording"}.get(http_cache.mode(), http_cache.mode()),
        "clauses_judged_live": judged,
        "clauses_from_verdict_cache": from_cache,
        "verdict_cache_share": round(from_cache / total, 3) if total else 0.0,
    }


def label(name: str) -> str:
    """Prefix an artifact filename on a demonstration run, so a replayed CSV can never be
    filed as a submission by mistake. Idempotent."""
    if not is_demonstration() or name.startswith(PREFIX):
        return name
    return PREFIX + name


def console_notice(*, judged: int = 0, from_cache: int = 0) -> str:
    """The line a run prints about its own provenance. Always says something: a live run
    stating that it was live is what makes the demonstration banner meaningful."""
    p = provenance(judged=judged, from_cache=from_cache)
    lead = f"!! {BANNER}" if p["demonstration"] else f"Provenance: network {p['network']}"
    if p["clauses_judged_live"] or p["clauses_from_verdict_cache"]:
        lead += (f"; {p['clauses_judged_live']} clause(s) judged live, "
                 f"{p['clauses_from_verdict_cache']} from the verdict cache")
    return lead


__all__ = ["BANNER", "PREFIX", "console_notice", "is_demonstration", "label", "provenance"]
