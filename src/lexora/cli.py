"""Top-level Lexora CLI.

Usage:
    lexora demo     --jurisdiction sg --pdf path/to/pdpa.pdf --source-url URL
    lexora collect  --jurisdiction SG
    lexora extract  --document-id <id>
    lexora classify --indicator 6.1 --jurisdiction SG
    lexora export   --format jsonld --out citations.jsonl
    lexora serve    # FastAPI audit UI
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.table import Table

app = typer.Typer(
    name="lexora",
    add_completion=False,
    no_args_is_help=True,
    help="Lexora — verifiable mapping of digital-trade regulations to the RDTII framework.",
)
console = Console()


@app.command()
def collect(
    jurisdiction: str = typer.Option(..., "--jurisdiction", "-j", help="ISO code, e.g. SG"),
    config_dir: Path = typer.Option(Path("configs/jurisdictions"), "--config-dir"),
    dry_run: bool = typer.Option(False, "--dry-run"),
) -> None:
    """Crawl the portals defined in the jurisdiction profile and store raw documents."""
    from lexora.collect.crawler import crawl
    from lexora.collect.profile_loader import load_profile

    profile = load_profile(config_dir / f"{jurisdiction.lower()}.yaml")
    console.print(f"[bold]Loaded profile:[/bold] {profile.jurisdiction} ({profile.iso_code})")
    console.print(f"  portals: {len(profile.portals)}")
    if dry_run:
        for p in profile.portals:
            console.print(f"  • {p.source_type.value:9} {p.fetch_method.value:5} {p.url}")
        return

    dest = Path("data") / "raw" / profile.iso_code.lower()
    table = Table(title=f"Crawl — {profile.jurisdiction}")
    table.add_column("status", justify="right")
    table.add_column("type")
    table.add_column("content-type")
    table.add_column("bytes", justify="right")
    table.add_column("portal")
    for result in crawl(profile, dest_dir=dest):
        d = result.document
        colour = "green" if 200 <= d.http_status < 300 else "red"
        table.add_row(
            f"[{colour}]{d.http_status}[/{colour}]",
            "PDF" if result.is_pdf() else ("HTML" if result.is_html() else "other"),
            d.content_type,
            str(len(result.body)),
            d.portal_name,
        )
    console.print(table)
    console.print(f"[green]Raw documents stored under[/green] {dest}")


@app.command()
def discover(
    jurisdiction: str = typer.Option(..., "--jurisdiction", "-j", help="ISO code, e.g. SG"),
    query: Optional[str] = typer.Option(None, "--query", "-q", help="Override the portal search query"),
    config_dir: Path = typer.Option(Path("configs/jurisdictions"), "--config-dir"),
    limit: int = typer.Option(5, "--limit", help="Top candidates per portal"),
    browser: bool = typer.Option(False, "--browser", help="Render every portal with a headless browser"),
) -> None:
    """Autonomously discover candidate instrument URLs on each portal.

    This is the discovery half of the mandatory crawl: instead of being handed a
    URL, Lexora searches each portal and ranks the links most likely to be the
    primary legal instrument. JS / anti-bot portals (SG SSO 403, AU SPA) are
    rendered with a headless browser automatically.
    """
    from lexora.collect.discovery import discover as run_discovery
    from lexora.collect.profile_loader import load_profile
    from lexora.collect.strategies import connector_for

    profile = load_profile(config_dir / f"{jurisdiction.lower()}.yaml")
    console.print(f"[bold]Discovery — {profile.jurisdiction} ({profile.iso_code})[/bold]")

    for portal in profile.portals:
        # Regulator portals (PDPC, …) expose a guidance corpus, not a search box:
        # use the registered connector to harvest it rather than the generic search.
        connector = connector_for(portal)
        if connector is not None:
            results = connector(
                portal, [], limit=limit, known_instruments=profile.known_instruments,
                known_instrument_ids=profile.known_instrument_ids,
            )
        else:
            results = run_discovery(
                portal, query=query, limit=limit, force_browser=browser,
                known_instruments=profile.known_instruments,
                known_instrument_ids=profile.known_instrument_ids,
            )
        table = Table(title=f"{portal.name}  ·  {portal.source_type.value}", show_lines=False)
        table.add_column("score", justify="right")
        table.add_column("via")
        table.add_column("tag")
        table.add_column("×", justify="right")
        table.add_column("type")
        table.add_column("title")
        table.add_column("url")
        for r in results:
            title = (r.title[:48] + "…") if len(r.title) > 48 else r.title
            tag = r.discovery_tag or ""
            tag_disp = f"[green]{tag}[/green]" if tag == "KNOWN" else (f"[yellow]{tag}[/yellow]" if tag else "")
            table.add_row(
                f"{r.score:.2f}", r.via, tag_disp, str(r.n_variants),
                "PDF" if r.is_pdf_link else "page", title, r.url,
            )
        if not results:
            table.add_row("—", "—", "—", "—", "—", "[dim]no candidates[/dim]", "")
        console.print(table)


@app.command()
def map(  # noqa: A001 - CLI verb
    jurisdiction: str = typer.Option(..., "--jurisdiction", "-j", help="ISO code, e.g. my"),
    portal_index: int = typer.Option(0, "--portal-index", help="Which profile portal to use"),
    config_dir: Path = typer.Option(Path("configs/jurisdictions"), "--config-dir"),
    indicators_path: Path = typer.Option(Path("configs/rdtii_indicators.yaml"), "--indicators"),
    out: Path = typer.Option(Path("outputs") / "map.jsonld", "--out", "-o"),
    top_k: int = typer.Option(3, "--top-k", help="Max sections emitted per indicator per "
                              "instrument (multi-section; gated by --rel-floor)"),
    min_score: float = typer.Option(0.35, "--min-score", help="BM25 clause-relevance floor"),
    rel_floor: float = typer.Option(0.6, "--rel-floor", help="Multi-section precision gate: a "
                                    "secondary section is kept only if its relevance is >= this "
                                    "fraction of the top section's (0 disables)"),
    budget: int = typer.Option(20, "--budget", help="Max instruments to map per jurisdiction"),
    verify: bool = typer.Option(False, "--verify/--no-verify",
                                help="Tighten mappings with the LLM verifier (needs an "
                                     "OpenAI-compatible endpoint; see LEXORA_LLM_* env)"),
    rationale_llm: bool = typer.Option(False, "--rationale-llm/--no-rationale-llm",
                                       help="Author the Mapping Rationale column with the LLM "
                                            "(template fallback + verbatim-copy guard); same "
                                            "endpoint as --verify. Default: deterministic template."),
    metadata_llm: bool = typer.Option(False, "--metadata-llm/--no-metadata-llm",
                                      help="Extract Law Number / Last Amended from the document "
                                           "text with the LLM (source-verified) when the portal "
                                           "channel and curated anchor do not supply them."),
    amendment_llm: bool = typer.Option(False, "--amendment-llm/--no-amendment-llm",
                                       help="Extract amendment instructions (Tier-2 provision "
                                            "adjudication) with the LLM, source-verified, falling "
                                            "back to the regex parser. More complete on real "
                                            "drafting than the regex floor; same endpoint as "
                                            "--verify. Also enabled by LEXORA_AMENDMENT_LLM=1."),
) -> None:
    """Fully autonomous MULTI-instrument map: per indicator, discover the family of
    instruments (flagship law + sectoral statutes), fetch each one's full text, and
    emit verbatim-validated citations as submission CSV + JSON-LD.

    No URL is handed in — Lexora searches the portal with each indicator's concept
    phrases, assembles a working set of instruments, and maps them all."""
    from lexora.cite.amendments_llm import llm_enabled, make_amendment_extractor
    from lexora.cite.metadata import make_metadata_extractor
    from lexora.cite.rationale import make_rationale_generator
    from lexora.classify.verifier import make_verifier
    from lexora.collect.profile_loader import load_profile
    from lexora.export.csv_exporter import to_csv
    from lexora.export.jsonld_exporter import to_jsonld
    from lexora.indicators import load_indicators
    from lexora.pipeline import run_pipeline_map

    profile = load_profile(config_dir / f"{jurisdiction.lower()}.yaml")
    indicators = load_indicators(indicators_path)
    portal = profile.portals[portal_index]
    verifier = make_verifier(use_llm=verify)
    if verify and verifier is None:
        console.print("[yellow]--verify requested but the LLM backend is unavailable "
                      "(install the [llm] extra); continuing with BM25 + verbatim only.[/yellow]")
    rationale_gen = make_rationale_generator(use_llm=rationale_llm)
    if rationale_llm and rationale_gen._client is None:
        console.print("[yellow]--rationale-llm requested but the LLM backend is unavailable; "
                      "using the deterministic template rationale.[/yellow]")
    meta_extractor = make_metadata_extractor(use_llm=metadata_llm)
    if metadata_llm and meta_extractor._client is None:
        console.print("[yellow]--metadata-llm requested but the LLM backend is unavailable; "
                      "using portal metadata + curated anchor only.[/yellow]")
    # LLM-first amendment extraction (Tier-2). Honour the flag OR the env switch so an
    # e2e run can opt in without re-plumbing; falls back to the regex parser if down.
    want_amend_llm = amendment_llm or llm_enabled()
    amendment_extractor = make_amendment_extractor(use_llm=want_amend_llm)
    if want_amend_llm and amendment_extractor._client is None:
        console.print("[yellow]--amendment-llm requested but the LLM backend is unavailable; "
                      "using the regex amendment parser only.[/yellow]")
    console.print(f"[bold]Autonomous multi-map — {profile.jurisdiction} ({profile.iso_code})[/bold]")
    console.print(f"  portal: {portal.name} · {len(indicators)} indicators · budget {budget}"
                  f"{' · LLM verifier ON' if verifier is not None else ''}"
                  f"{' · LLM rationale ON' if rationale_gen._client is not None else ''}"
                  f"{' · LLM metadata ON' if meta_extractor._client is not None else ''}"
                  f"{' · LLM amendment ON' if amendment_extractor._client is not None else ''}")

    result = run_pipeline_map(
        portal=portal, profile=profile, indicators=indicators,
        top_k=top_k, min_score=min_score, rel_floor=rel_floor, budget=budget,
        verifier=verifier, rationale_gen=rationale_gen, meta_extractor=meta_extractor,
        amendment_extractor=amendment_extractor,
    )
    if not result.discovered:
        console.print("[red]No instruments discovered.[/red]")
        raise typer.Exit(1)
    if verifier is not None and getattr(verifier, "error_count", 0):
        console.print(
            "[yellow]LLM verifier backend errors encountered: "
            f"{verifier.error_count} judgement(s) failed"
            f" (last error: {verifier.last_error_type or 'unknown'}). "
            "The run continued without fabricating citations.[/yellow]"
        )

    # Working set: which instruments did discovery assemble?
    disc = Table(title=f"Working set — {len(result.discovered)} instrument(s)")
    disc.add_column("score", justify="right")
    disc.add_column("via")
    disc.add_column("tag")
    disc.add_column("title")
    for r in result.discovered:
        tag = r.discovery_tag or ""
        tag_disp = f"[green]{tag}[/green]" if tag == "KNOWN" else (f"[yellow]{tag}[/yellow]" if tag else "")
        title = (r.title[:54] + "…") if len(r.title) > 54 else r.title
        disc.add_row(f"{r.score:.2f}", r.via, tag_disp, title)
    console.print(disc)

    ok_docs = sum(1 for d in result.documents if 200 <= d.document.http_status < 300)
    console.print(
        f"  fetched {ok_docs}/{len(result.documents)} full texts · "
        f"{len(result.citations)} citation(s) across instruments"
    )

    out.parent.mkdir(parents=True, exist_ok=True)
    n = to_jsonld(result.citations, out)
    csv_out = out.with_suffix(".csv")
    to_csv(result.citations, csv_out)

    table = Table(title=f"Lexora map — {profile.jurisdiction}", show_lines=False)
    table.add_column("indicator")
    table.add_column("instrument")
    table.add_column("clause")
    table.add_column("tag")
    table.add_column("conf", justify="right")
    table.add_column("quote (first 60 chars)")
    for c in sorted(result.citations, key=lambda c: c.indicator_id):
        quote = c.quote.replace("\n", " ")
        inst = (c.title[:28] + "…") if len(c.title) > 28 else c.title
        table.add_row(
            c.indicator_id, inst, c.article_path, c.discovery_tag.value,
            f"{c.confidence:.2f}", (quote[:57] + "...") if len(quote) > 60 else quote,
        )
    console.print(table)
    console.print(f"[green]Wrote {n} citation(s)[/green] to {csv_out} (submission CSV) and {out} (JSON-LD)")


@app.command()
def extract(document_id: str = typer.Option(..., "--document-id", "-d")) -> None:
    """Run extraction (HTML / PDF text / OCR) on a raw document."""
    raise NotImplementedError("extract() — implement in lexora.extract")


@app.command()
def classify(
    indicator: str = typer.Option(..., "--indicator", "-i", help="RDTII indicator id e.g. 6.1"),
    jurisdiction: str = typer.Option(..., "--jurisdiction", "-j"),
) -> None:
    """Retrieve candidate clauses and run the LLM verifier."""
    raise NotImplementedError("classify() — implement in lexora.classify")


@app.command()
def export(
    fmt: str = typer.Option("jsonld", "--format", "-f", help="jsonld | csv"),
    out: Path = typer.Option(..., "--out", "-o"),
) -> None:
    """Export verified citations."""
    raise NotImplementedError("export() — implement in lexora.export")


@app.command()
def demo(
    jurisdiction: str = typer.Option(..., "--jurisdiction", "-j", help="ISO code, e.g. sg"),
    pdf: Optional[Path] = typer.Option(None, "--pdf", "-p", help="Local PDF (manual-upload fallback)"),
    url: Optional[str] = typer.Option(None, "--url", help="LIVE-fetch this URL (PDF or HTML)"),
    source_url: Optional[str] = typer.Option(None, "--source-url", "-u", help="Canonical URL (required with --pdf)"),
    browser: bool = typer.Option(False, "--browser", help="With --url: escalate to a headless browser on 403/429 (e.g. SG SSO)"),
    portal_name: str = typer.Option("manual-upload", "--portal"),
    title: Optional[str] = typer.Option(None, "--title"),
    legal_form: str = typer.Option("statute", "--legal-form"),
    config_dir: Path = typer.Option(Path("configs/jurisdictions"), "--config-dir"),
    indicators_path: Path = typer.Option(Path("configs/rdtii_indicators.yaml"), "--indicators"),
    out: Path = typer.Option(Path("outputs") / "demo.jsonld", "--out", "-o"),
    top_k: int = typer.Option(1, "--top-k"),
    min_score: float = typer.Option(0.5, "--min-score"),
) -> None:
    """End-to-end: live-fetch a URL (--url) or ingest a local PDF (--pdf), and
    emit verbatim-validated citations as submission CSV + JSON-LD."""
    from lexora.collect.profile_loader import load_profile
    from lexora.export.csv_exporter import to_csv
    from lexora.export.jsonld_exporter import to_jsonld
    from lexora.indicators import load_indicators
    from lexora.pipeline import run_demo_pipeline, run_pipeline_from_url

    if bool(pdf) == bool(url):
        raise typer.BadParameter("provide exactly one of --pdf or --url")
    if pdf and not source_url:
        raise typer.BadParameter("--source-url is required with --pdf")

    profile = load_profile(config_dir / f"{jurisdiction.lower()}.yaml")
    indicators = load_indicators(indicators_path)
    console.print(f"[bold]Jurisdiction:[/bold] {profile.jurisdiction} ({profile.iso_code})")
    console.print(f"[bold]Indicators loaded:[/bold] {len(indicators)}")

    if url:
        console.print(f"[bold]Live fetch:[/bold] {url}")
        artifacts = run_pipeline_from_url(
            url=url,
            profile=profile,
            indicators=indicators,
            portal_name=portal_name if portal_name != "manual-upload" else "live-fetch",
            title=title,
            legal_form=legal_form,
            top_k=top_k,
            min_score=min_score,
            browser_fallback=browser,
        )
        d = artifacts.document
        colour = "green" if 200 <= d.http_status < 300 else "red"
        console.print(
            f"  HTTP [{colour}]{d.http_status}[/{colour}] · {d.content_type} · "
            f"{len(artifacts.pages)} page(s) / {len(artifacts.blocks)} block(s)"
        )
    else:
        artifacts = run_demo_pipeline(
            pdf_path=pdf,
            profile=profile,
            indicators=indicators,
            source_url=source_url,
            portal_name=portal_name,
            title=title,
            legal_form=legal_form,
            top_k=top_k,
            min_score=min_score,
        )

    out.parent.mkdir(parents=True, exist_ok=True)
    n = to_jsonld(artifacts.citations, out)
    csv_out = out.with_suffix(".csv")
    to_csv(artifacts.citations, csv_out)

    table = Table(title=f"Lexora demo — {profile.jurisdiction}", show_lines=False)
    table.add_column("indicator")
    table.add_column("clause")
    table.add_column("status")
    table.add_column("conf", justify="right")
    table.add_column("quote (first 80 chars)")
    for c in artifacts.citations:
        quote = c.quote.replace("\n", " ")
        if len(quote) > 80:
            quote = quote[:77] + "..."
        table.add_row(
            c.indicator_id,
            c.article_path,
            c.review_status.value,
            f"{c.confidence:.2f}",
            quote,
        )
    console.print(table)
    units = len(artifacts.pages) + len(artifacts.blocks)
    console.print(
        f"[green]Wrote {n} citation(s)[/green] to {csv_out} (submission CSV) and {out} (JSON-LD)  "
        f"(clauses={len(artifacts.clauses)}, units={units}, "
        f"doc={artifacts.document.document_id})"
    )


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8001, "--port"),
) -> None:
    """Launch the FastAPI audit UI."""
    import uvicorn

    uvicorn.run("lexora.api.app:api", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
