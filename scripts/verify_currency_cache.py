"""Offline end-to-end check of the amendment-currency layer on REAL cached PDFs.

Drives `_apply_currency_flags` over a principal Act + its real amending Act (both
already in `data/raw/`), with the LLM-FIRST extractor and the per-amending-Act thread
pool, then prints the currency fields each citation ends up carrying. This exercises
the exact production Tier-2 path (LLM-first extraction, regex fallback, multi-key
bridge, parallel extraction) without touching the network — the cache stands in for
a live fetch.

Usage:
    LEXORA_AMENDMENT_LLM=1 python scripts/verify_currency_cache.py
    python scripts/verify_currency_cache.py            # regex-only (no LLM backend)
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pypdf import PdfReader  # noqa: E402

from lexora.cite.amendments import (  # noqa: E402
    affected_sections,
    classify_version,
    detect_amends_target,
    has_global_rename,
)
from lexora.cite.amendments_llm import llm_enabled, make_amendment_extractor  # noqa: E402
from lexora.models.citation import Citation, DiscoveryTag, ReviewStatus  # noqa: E402
from lexora.models.source import (  # noqa: E402
    LegalSystem,
    RawDocument,
    SourceProfile,
    SourceType,
)
from lexora.pipeline import (  # noqa: E402
    DemoArtifacts,
    _apply_currency_flags,
    _extract_amendment_instructions,
)

RAW = Path("data/raw/my")
PRINCIPAL_SHA = "cc1ef72d1"   # PDPA 2010, Act 709 — ORIGINAL "as made" (mislabels "2010")
AMENDMENT_SHA = "835960086"   # PDPA (Amendment) Act 2024, Act A1727 — the real gazette


def _find(prefix: str) -> Path:
    hits = sorted(RAW.glob(f"{prefix}*.pdf"))
    if not hits:
        raise SystemExit(f"no cached PDF under {RAW} starting {prefix}")
    return hits[0]


def _text(p: Path) -> str:
    return "\n".join((pg.extract_text() or "") for pg in PdfReader(str(p)).pages)


def _artifact(sha: str, title: str, text: str, number: str, last_amended: str) -> DemoArtifacts:
    doc = RawDocument(
        document_id=sha, source_url=f"https://lom.agc.gov.my/{sha}", http_status=200,
        retrieval_timestamp=datetime.now(timezone.utc), sha256=sha,
        content_type="application/pdf", bytes_path=f"{sha}.pdf", portal_name="AGC",
        jurisdiction="Malaysia", source_type=SourceType.primary, title=title,
        law_number=number, last_amended=last_amended,
    )
    return DemoArtifacts(document=doc, clauses=[], citations=[], document_text=text)


def _cit(sha: str, section: str, quote: str) -> Citation:
    return Citation(
        economy="Malaysia", title="Personal Data Protection Act 2010", law_number="Act 709",
        last_amended="2010", indicator_id="P7-I1", article_path=f"S. {section}",
        discovery_tag=DiscoveryTag.known, page_or_dom_anchor="p.1", quote=quote,
        source_url=f"https://lom.agc.gov.my/{sha}", confidence=0.9, clause_id=f"c{section}",
        retrieval_timestamp=datetime.now(timezone.utc), document_hash=sha,
        jurisdiction="Malaysia", legal_form="statute", char_start=0, char_end=5,
        review_status=ReviewStatus.verified,
    )


def main() -> None:
    p_path, a_path = _find(PRINCIPAL_SHA), _find(AMENDMENT_SHA)
    print(f"principal : {p_path.name}")
    print(f"amendment : {a_path.name}")
    p_text, a_text = _text(p_path), _text(a_path)

    # Confirm the FULL-text classification (the 2-page scan misread these).
    print(f"\nprincipal version : {classify_version(p_text).value}")
    print(f"amendment version : {classify_version(a_text).value}"
          f"   amends: {detect_amends_target(a_text)}")

    extractor = make_amendment_extractor(use_llm=llm_enabled())
    backend = "LLM-first" if extractor._client is not None else "regex-only (no LLM backend)"
    print(f"extractor : {backend}")

    t0 = time.perf_counter()
    instrs = _extract_amendment_instructions(a_text, extractor)
    dt = time.perf_counter() - t0
    print(f"\ninstructions extracted: {len(instrs)} in {dt:.1f}s "
          f"(global_rename={has_global_rename(instrs)})")
    secs = affected_sections(instrs)
    print(f"affected sections: {sorted(secs)[:25]}")

    # Build citations: a few sections the amendment touches + one it does not, each
    # with a real PDPA snippet that uses the old term "data user" (to also exercise
    # the global-rename annotation).
    touched = sorted(secs)[:3]
    quote = "A data user shall, in respect of personal data, comply with ..."
    cits = [_cit(PRINCIPAL_SHA, s, quote) for s in touched]
    cits.append(_cit(PRINCIPAL_SHA, "999", quote))  # untouched control

    principal = _artifact(PRINCIPAL_SHA, "Personal Data Protection Act 2010",
                          p_text, "Act 709", "2010")
    amendment = _artifact(AMENDMENT_SHA, "Personal Data Protection (Amendment) Act 2024",
                          a_text, "Act A1727", "2024")
    profile = SourceProfile(jurisdiction="Malaysia", iso_code="MY", primary_language="en",
                            legal_system=LegalSystem.common)
    _apply_currency_flags([principal, amendment], cits, profile,
                          extractor=extractor, workers=4)

    print("\n=== citation currency fields ===")
    for c in cits:
        amend_txt = (c.amendment_text[:80] + "…") if c.amendment_text else "—"
        print(f"\n{c.article_path}")
        print(f"  currency_status        : {c.currency_status}")
        print(f"  review_status          : {c.review_status.value}")
        print(f"  source_version         : {c.source_version}")
        print(f"  amended_by             : {c.amended_by or '—'}")
        print(f"  incorporated_to        : {c.amendments_incorporated_to or '—'}")
        print(f"  amendment_text         : {amend_txt}")
        print(f"  notes                  : {c.notes or '—'}")
    if extractor._client is not None:
        print(f"\nLLM usage: {extractor.extracted} amending Act(s) extracted, "
              f"{extractor.rejected} instruction(s) rejected by source-check, "
              f"{extractor.error_count} backend error(s)")


if __name__ == "__main__":
    os.environ.setdefault("LEXORA_LLM_MAX_RETRIES", "1")
    main()
