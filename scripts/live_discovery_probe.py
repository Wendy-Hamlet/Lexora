"""Live discovery probe — runs the discovery layer against real portals and
writes results to JSON (so non-ASCII titles never hit the Windows console)."""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from lexora.collect.discovery import discover  # noqa: E402
from lexora.collect.profile_loader import load_profile  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
OUT = REPO / "outputs" / "live_discovery.json"


def main() -> None:
    report: dict = {}
    for iso in ("my", "sg", "au"):
        profile = load_profile(REPO / "configs" / "jurisdictions" / f"{iso}.yaml")
        portal = profile.portals[0]  # the primary source portal
        force_browser = iso in ("sg", "au")  # SG=403, AU=SPA
        t0 = time.time()
        try:
            results = discover(portal, limit=5, force_browser=force_browser, timeout=60.0)
            report[iso] = {
                "portal": portal.name,
                "url": str(portal.url),
                "query": portal.search_query,
                "force_browser": force_browser,
                "elapsed_s": round(time.time() - t0, 1),
                "candidates": [
                    {"score": r.score, "via": r.via, "pdf": r.is_pdf_link,
                     "title": r.title, "url": r.url}
                    for r in results
                ],
            }
        except Exception as exc:  # record, don't crash the whole probe
            report[iso] = {"portal": portal.name, "error": f"{type(exc).__name__}: {exc}"}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    # ASCII-safe console summary
    for iso, data in report.items():
        n = len(data.get("candidates", [])) if "candidates" in data else "ERR"
        print(f"{iso}: {n} candidates  ({data.get('error', 'ok')})")


if __name__ == "__main__":
    main()
