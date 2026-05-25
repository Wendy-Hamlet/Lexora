"""FastAPI application factory.

Run with: `lexora serve --port 8001`
"""
from __future__ import annotations

from fastapi import FastAPI

from lexora import __version__

api = FastAPI(
    title="Lexora Audit API",
    version=__version__,
    description="Verifiable mapping of digital-trade regulations to RDTII.",
)


@api.get("/healthz")
def healthz() -> dict[str, str]:
    return {"status": "ok", "version": __version__}


@api.get("/jurisdictions")
def list_jurisdictions() -> list[dict]:  # pragma: no cover
    """Return loaded jurisdiction profiles.

    TODO: read from configs/jurisdictions/ via profile_loader.load_all_profiles.
    """
    raise NotImplementedError("Implement /jurisdictions.")
