"""What a run costs, from the invoice.

Kept out of the bench script so the live run and the benchmark price the same tokens the
same way. Every rate below is a BILLED LINE divided by its BILLED TOKEN COUNT on the
2026-07-28 invoice, not a number copied off a price page — the two disagreed once already
(DeepSeek-V4-Flash was guessed at 0.5/0.1 and is really 1.0/0.2).

Three rates, not two. Prompt tokens the provider served from its prefix cache bill at a
fraction of fresh input, and the judge re-sends an identical ~3k-token indicator catalogue
on every one of ~22k calls — whether that block is cached is the difference between ¥650
and ¥250 for a full run. A two-rate model does not just overstate the bill, it hides the
single biggest lever on it.
"""
from __future__ import annotations

# CNY per 1M tokens: (fresh input, cached input, output).
RATES: dict[str, tuple[float, float, float]] = {
    "GLM-5.2": (8.0, 2.0, 28.0),           # confirmed to the cent on all three lines
    "GLM-4.5-Flash": (0.0, 0.0, 0.0),      # genuinely free, all three lines billed 0
    "DeepSeek-V4-Flash": (1.0, 0.2, 2.0),  # was guessed at 0.5/0.1; the invoice says 1.0/0.2
    # Qwen has NO cached-input line on the invoice at all, which corroborates the measured
    # 0% prefix-cache hit rate: Paratera does not cache this family. Priced as input-only.
    "Qwen3.5-35B-A3B": (1.6, 1.6, 12.8),
}


def cost_cny(model: str, prompt: int, cached: int, completion: int) -> float | None:
    """CNY for one model's usage, or ``None`` when that model is not on the invoice.

    ``cached`` is the part of ``prompt`` the provider served from its prefix cache, so it
    is subtracted rather than added. ``None`` (not 0.0) for an unpriced model, so a caller
    reports "tokens, cost unknown" instead of a confident and wrong zero.
    """
    rate = RATES.get(model)
    if rate is None:
        return None
    fresh_in, cached_in, out = rate
    cached = max(0, min(cached, prompt))
    return (
        (prompt - cached) / 1e6 * fresh_in
        + cached / 1e6 * cached_in
        + completion / 1e6 * out
    )


__all__ = ["RATES", "cost_cny"]
