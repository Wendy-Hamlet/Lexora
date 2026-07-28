"""HTML exporter — the reviewer's read-only console, as one self-contained file.

The brief says judges include non-technical reviewers, and describes the policy judge's
loop literally: open the output, filter Discovery Tag = NEW, read the Mapping Rationale,
click the Source URL, check the Verbatim Snippet against it. In a spreadsheet that loop is
four programs and a lot of horizontal scrolling; a 900-character quote in a CSV cell is not
something anyone reads.

So every run also writes this: one HTML file, no server, no build step, no external asset,
that puts the same rows in front of that loop — filter by economy / indicator / discovery
tag, search the full text, and read each provision as a card with its quote laid out in
full and its source one click away.

It is a *view*, not a second source of truth. Every value is read from the same Citation
objects the CSV is written from, so the two cannot disagree; if a number here differs from
the CSV, the bug is in this file and the CSV wins. Nothing is computed here that is not
already in the row — in particular no score, no ranking, and no text that the verbatim
contract did not produce.

Self-contained is a requirement, not a convenience: it has to open from a USB stick on a
laptop with no network, which is the situation a live demo is one bad conference wifi away
from.
"""
from __future__ import annotations

import html
from collections.abc import Iterable
from datetime import datetime
from pathlib import Path

from lexora.export import provenance
from lexora.export.law_name import normalize_law_name
from lexora.models.citation import Citation

_CSS = """
:root{--bg:#fafafa;--card:#fff;--line:#e2e2e2;--ink:#1a1a1a;--dim:#777;--accent:#0a58ca}
*{box-sizing:border-box}
body{font-family:-apple-system,"Segoe UI",Roboto,"Helvetica Neue",Arial,sans-serif;
 margin:0;padding:0 20px 60px;color:var(--ink);background:var(--bg);line-height:1.55}
.demo-banner{background:#b42318;color:#fff;font-weight:700;letter-spacing:.02em;
 text-align:center;padding:11px 16px;margin:0 -20px 4px;font-size:15px;
 border-bottom:3px solid #7a1610}
header{max-width:1080px;margin:0 auto;padding:22px 0 6px}
h1{margin:.1em 0;font-size:26px}
.sub{color:var(--dim);margin:0 0 14px;font-size:14px}
.stats{display:flex;flex-wrap:wrap;gap:10px;margin:14px 0}
.stat{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:8px 14px;min-width:92px}
.stat b{display:block;font-size:22px;color:var(--accent)}
.stat span{font-size:12px;color:var(--dim)}
.bar{position:sticky;top:0;z-index:9;background:var(--bg);border-bottom:1px solid var(--line);
 padding:10px 0;margin-bottom:10px}
.bar .inner{max-width:1080px;margin:0 auto;display:flex;flex-wrap:wrap;gap:6px;align-items:center}
.chip{font:inherit;font-size:12px;padding:3px 10px;border:1px solid #ccc;background:#fff;
 border-radius:12px;cursor:pointer}
.chip.on{background:var(--accent);color:#fff;border-color:var(--accent)}
.grp{display:flex;flex-wrap:wrap;gap:5px;align-items:center}
.grp>em{font-style:normal;font-size:11px;color:var(--dim);margin-right:2px}
#q{font:inherit;font-size:13px;padding:5px 9px;border:1px solid #ccc;border-radius:6px;
 flex:1;min-width:180px}
#count{font-size:12px;color:var(--dim);margin-left:auto;white-space:nowrap}
main{max-width:1080px;margin:0 auto}
.card{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px 14px;
 margin:9px 0;box-shadow:0 1px 2px rgba(0,0,0,.04)}
.chead{display:flex;flex-wrap:wrap;gap:8px;align-items:baseline}
.sec{font-weight:700;color:var(--accent);font-family:ui-monospace,Consolas,monospace}
.law{flex:1;min-width:220px}
.badges{margin-left:auto;white-space:nowrap}
.tag{font-size:11px;padding:1px 7px;border-radius:10px;margin-left:4px}
.t-NEW{background:#e6f4ea;color:#1a7f37;font-weight:700}
.t-KNOWN{background:#eef;color:#337}
.t-ind{background:#f3e8ff;color:#6b21a8;font-weight:700}
.t-conf{background:#f2f2f2;color:#555;font-family:ui-monospace,Consolas,monospace}
.t-stale{background:#fff4e5;color:#9a3412;font-weight:700}
.meta{color:var(--dim);font-size:12px;margin:3px 0}
.ratio{margin:6px 0;font-size:14px}
pre.verb{white-space:pre-wrap;word-break:break-word;background:#f6f6f6;border:1px solid #eee;
 border-radius:6px;padding:9px;font-size:12.5px;max-height:320px;overflow:auto;margin:6px 0}
.foot{font-size:13px}
.foot a{color:var(--accent);text-decoration:none;word-break:break-all}
.foot a:hover{text-decoration:underline}
.notes{font-size:12.5px;color:#a0522d;margin-top:4px}
.hide{display:none}
.empty{color:var(--dim);padding:40px 0;text-align:center}
@media print{.bar{position:static}.card{break-inside:avoid}}
"""

_JS = """
const cards=[...document.querySelectorAll('.card')];
const state={economy:new Set(),indicator:new Set(),tag:new Set(),q:''};
function apply(){
 let n=0;
 for(const c of cards){
  const d=c.dataset;
  const ok=(!state.economy.size||state.economy.has(d.economy))
        &&(!state.indicator.size||state.indicator.has(d.indicator))
        &&(!state.tag.size||state.tag.has(d.tag))
        &&(!state.q||d.hay.includes(state.q));
  c.classList.toggle('hide',!ok); if(ok)n++;
 }
 document.getElementById('count').textContent=n+' / '+cards.length+' provisions shown';
 document.getElementById('empty').classList.toggle('hide',n>0);
}
for(const b of document.querySelectorAll('.chip[data-facet]')){
 b.onclick=()=>{const s=state[b.dataset.facet];
  b.classList.toggle('on')?s.add(b.dataset.value):s.delete(b.dataset.value);apply();};
}
document.getElementById('q').oninput=e=>{state.q=e.target.value.toLowerCase();apply();};
document.getElementById('reset').onclick=()=>{
 for(const k of ['economy','indicator','tag'])state[k].clear();
 for(const b of document.querySelectorAll('.chip[data-facet]'))b.classList.remove('on');
 document.getElementById('q').value='';state.q='';apply();};
apply();
"""


def _text(citation: Citation, attr: str) -> str:
    v = getattr(citation, attr, None)
    if v is None:
        return ""
    return str(getattr(v, "value", v))


def _chip(facet: str, value: str, label: str | None = None) -> str:
    return (f'<button class="chip" data-facet="{facet}" data-value="{html.escape(value)}">'
            f'{html.escape(label or value)}</button>')


def _card(citation: Citation) -> str:
    e = html.escape
    economy = _text(citation, "economy")
    indicator = _text(citation, "indicator_id")
    tag = _text(citation, "discovery_tag")
    law = normalize_law_name(_text(citation, "title"))
    article = _text(citation, "article_path") or "(unlocated)"
    quote = _text(citation, "quote")
    rationale = _text(citation, "mapping_rationale")
    url = _text(citation, "source_url")
    confidence = _text(citation, "confidence")
    currency = _text(citation, "currency_status")
    notes = _text(citation, "notes")

    # Everything the free-text box searches, lower-cased once here rather than per keystroke.
    hay = " ".join([economy, indicator, law, article, quote, rationale,
                    _text(citation, "law_number"), notes]).lower()

    bits = [f'<div class="card" data-economy="{e(economy)}" data-indicator="{e(indicator)}"'
            f' data-tag="{e(tag)}" data-hay="{e(hay)}">']
    bits.append('<div class="chead">')
    bits.append(f'<span class="sec">{e(article)}</span>')
    bits.append(f'<span class="law">{e(law)}</span>')
    bits.append('<span class="badges">')
    bits.append(f'<span class="tag t-ind">{e(indicator)}</span>')
    if tag:
        bits.append(f'<span class="tag t-{e(tag)}">{e(tag)}</span>')
    if confidence:
        bits.append(f'<span class="tag t-conf">{e(confidence)}</span>')
    if currency and currency.upper() not in ("CURRENT", "NONE", ""):
        bits.append(f'<span class="tag t-stale">{e(currency)}</span>')
    bits.append("</span></div>")

    meta = [x for x in (economy, _text(citation, "law_number"),
                        f'last amended {_text(citation, "last_amended")}'
                        if _text(citation, "last_amended") else "",
                        _text(citation, "page_or_dom_anchor")) if x]
    if meta:
        bits.append(f'<div class="meta">{e(" · ".join(meta))}</div>')
    if rationale:
        bits.append(f'<div class="ratio">{e(rationale)}</div>')
    if quote:
        bits.append(f'<pre class="verb">{e(quote)}</pre>')
    if url:
        bits.append(f'<div class="foot">source: <a href="{e(url)}" '
                    f'target="_blank" rel="noopener">{e(url)}</a></div>')
    # The offsets are what make the quote checkable rather than merely quoted.
    offsets = (_text(citation, "char_start"), _text(citation, "char_end"))
    doc_hash = _text(citation, "document_hash")
    if offsets[0] and doc_hash:
        bits.append(f'<div class="meta">chars {e(offsets[0])}–{e(offsets[1])} of '
                    f'{e(doc_hash[:23])}…</div>')
    if notes:
        bits.append(f'<div class="notes">{e(notes)}</div>')
    bits.append("</div>")
    return "".join(bits)


def to_html(citations: Iterable[Citation], out_path: Path, *,
            title: str = "Lexora — RDTII provision map") -> int:
    """Write the self-contained reviewer console. Returns the number of cards."""
    rows = list(citations)
    e = html.escape

    economies = sorted({_text(c, "economy") for c in rows} - {""})
    indicators = sorted({_text(c, "indicator_id") for c in rows} - {""})
    tags = sorted({_text(c, "discovery_tag") for c in rows} - {""})
    laws = len({normalize_law_name(_text(c, "title")) for c in rows} - {""})
    n_new = sum(1 for c in rows if _text(c, "discovery_tag") == "NEW")

    stats = [("provisions", len(rows)), ("NEW", n_new), ("KNOWN", len(rows) - n_new),
             ("indicators", len(indicators)), ("laws", laws), ("economies", len(economies))]

    parts = [
        "<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">",
        '<meta name="viewport" content="width=device-width, initial-scale=1">',
        f"<title>{e(title)}</title><style>{_CSS}</style></head><body>",
    ]
    # The page is what a reviewer looks at during the live pitch, so it is where the
    # replayed/live distinction has to be impossible to miss -- before any number is read,
    # not in a footnote under it.
    if provenance.is_demonstration():
        parts.append(f'<div class="demo-banner">{e(provenance.BANNER)}</div>')
    parts += [
        "<header>", f"<h1>{e(title)}</h1>",
        f'<p class="sub">Generated {datetime.now().strftime("%Y-%m-%d %H:%M")} · '
        "every row carries its verbatim text, character offsets and official source URL. "
        "This page is a view of the submission CSV, not a second source of truth.</p>",
        '<div class="stats">',
    ]
    for label, value in stats:
        parts.append(f'<div class="stat"><b>{value}</b><span>{e(label)}</span></div>')
    parts.append("</div></header>")

    parts.append('<div class="bar"><div class="inner">')
    if len(economies) > 1:
        parts.append('<div class="grp"><em>economy</em>'
                     + "".join(_chip("economy", x) for x in economies) + "</div>")
    parts.append('<div class="grp"><em>tag</em>'
                 + "".join(_chip("tag", x) for x in tags) + "</div>")
    parts.append('<div class="grp"><em>indicator</em>'
                 + "".join(_chip("indicator", x) for x in indicators) + "</div>")
    parts.append('<input id="q" type="search" placeholder="search law, section, quote…">')
    parts.append('<button class="chip" id="reset">reset</button>')
    parts.append('<span id="count"></span>')
    parts.append("</div></div>")

    parts.append("<main>")
    parts.extend(_card(c) for c in rows)
    parts.append('<div class="empty hide" id="empty">No provision matches these filters.</div>')
    parts.append("</main>")
    parts.append(f"<script>{_JS}</script></body></html>")

    Path(out_path).write_text("".join(parts), encoding="utf-8")
    return len(rows)


__all__ = ["to_html"]
