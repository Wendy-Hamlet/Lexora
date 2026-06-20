"""P6/P7 boundary rules — official scope discriminators retrieval cannot infer.

Retrieval maps a clause to an indicator by vocabulary overlap; it cannot tell a
minimum-retention DUTY from a retention LIMITATION, a conditional-flow regime from
an outright ban, or a measure on government data from one in scope. These predicates
encode the boundaries the RDTII guide adjudicates (internal-guide Annex FAQ pp.9/13)
as a TIGHTENING filter: they only DROP a wrongly-surfaced (indicator, clause) pair,
never add one — consistent with the verbatim / verifier philosophy.

Default is ADMIT: an indicator with no rule, or a clause that trips no exclusion,
passes through unchanged. Rules are deliberately conservative (exclude only on a
clear contrary signal) to avoid dropping correct citations.

Currently implemented: 7.3 (minimum retention vs retention limitation) and 6.1/6.4
(outright ban / localisation vs conditional flow). The P6/7.3 government-data
carve-out needs scope understanding a keyword rule cannot do safely, so it is left
to the LLM verifier (verification lane), not this deterministic layer.
"""
from __future__ import annotations

import re


def _has(text: str, markers: tuple[str, ...]) -> bool:
    return any(m in text for m in markers)


# --- 7.3 Minimum period of data retention -----------------------------------
# Official: scores a MINIMUM retention duty ("keep/retain for at least ..."). A
# retention LIMITATION ("cease retaining once the purpose ends", "not be kept
# longer than necessary") is the OPPOSITE and must NOT map here (guide p.13;
# e.g. SG PDPA s.25 is a limitation, deliberately not 7.3).
_RETENTION_MIN = (
    "at least", "minimum", "not less than", "no less than", "for a period of",
    "for a minimum", "must be kept for", "must be retained for", "must keep",
    "must retain", "shall keep", "shall retain", "retain the", "retention period of",
)
_RETENTION_LIMIT = (
    "cease to retain", "cease retaining", "no longer than", "not longer than",
    "not be kept longer", "shall not be kept", "must not be kept", "as soon as",
    "no longer necessary", "longer than is necessary", "longer than necessary",
)


def _admits_7_3(text: str) -> bool:
    """Exclude only a clear retention LIMITATION that carries no minimum-duration
    signal; admit minimum-retention duties and neutral clauses."""
    return not (_has(text, _RETENTION_LIMIT) and not _has(text, _RETENTION_MIN))


# --- 6.1 Ban / local processing  vs  6.4 Conditional flow -------------------
# Official (guide p.13): an OUTRIGHT prohibition / local-processing mandate is 6.1;
# a transfer PERMITTED SUBJECT TO CONDITIONS (adequacy, safeguards, comparable
# protection, consent, an approved/whitelisted destination) is 6.4 — even when the
# clause is phrased as "must not transfer ... except <condition>" (e.g. SG PDPA
# s.26, MY PDPA s.129 both → 6.4). So the presence of a substantive transfer
# CONDITION routes a clause to 6.4 regardless of accompanying ban language.
_TRANSFER_CONDITION = (
    "comparable to the protection", "comparable level of protection",
    "appropriate safeguards", "adequate level of protection", "adequacy",
    "appropriate protection", "with the consent", "consent of the individual",
    "the individual consents", "given consent", "prescribed", "prescribed conditions",
    "subject to the conditions", "binding corporate rules", "standard contractual",
    "approved by", "specified by the minister", "place as specified", "as specified by",
)
# Outright ban / data-localisation markers (no transfer permitted, or must stay local).
_TRANSFER_BAN = (
    "must not transfer", "shall not transfer", "may not transfer", "prohibited from transferring",
    "ban on transfer", "must not be transferred", "may not be transferred",
    "must be held", "must be stored", "stored within", "held within", "processed within",
    "processed locally", "local processing", "must not be held or processed outside",
    "must not be processed outside", "only be held", "only be processed",
    # active-voice data-localisation (e.g. My Health Records Act s.77)
    "must not hold", "must not process", "not hold or process",
)


def _admits_6_1(text: str) -> bool:
    """A conditional cross-border transfer belongs to 6.4 — exclude it from the 6.1
    ban indicator. Admit outright bans / localisation (no transfer condition)."""
    return not _has(text, _TRANSFER_CONDITION)


def _admits_6_4(text: str) -> bool:
    """A pure ban / localisation with no transfer condition belongs to 6.1 — exclude
    it from 6.4. Admit anything carrying a transfer condition."""
    return not (_has(text, _TRANSFER_BAN) and not _has(text, _TRANSFER_CONDITION))


_RULES = {
    "6.1": _admits_6_1,
    "6.4": _admits_6_4,
    "7.3": _admits_7_3,
}


def admits_clause(rdtii_id: str, clause_text: str) -> bool:
    """False when a boundary rule excludes ``clause_text`` from indicator
    ``rdtii_id``; True otherwise (no rule, or the clause passes)."""
    rule = _RULES.get(rdtii_id)
    if rule is None:
        return True
    return rule(re.sub(r"\s+", " ", clause_text).lower())


__all__ = ["admits_clause"]
