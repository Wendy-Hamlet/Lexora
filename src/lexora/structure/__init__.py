"""Stage 3 — structure.

Rebuilds article / section / paragraph paths from extracted pages.
Ambiguous headings stay as raw blocks with a review flag — no silent merging.

Note: the MVP intentionally uses a small canonical citation schema rather than
full Akoma Ntoso XML. Akoma Ntoso adapters can be added later as an export
target without changing this module's contract.
"""
