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
    """Crawl sources defined in the jurisdiction profile and store raw documents."""
    from lexora.collect.profile_loader import load_profile

    profile = load_profile(config_dir / f"{jurisdiction.lower()}.yaml")
    console.print(f"[bold]Loaded profile:[/bold] {profile.jurisdiction} ({profile.iso_code})")
    console.print(f"  portals: {len(profile.portals)}")
    if dry_run:
        return
    raise NotImplementedError("crawl() — implement in lexora.collect.crawler")


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
    pdf: Path = typer.Option(..., "--pdf", "-p", exists=True, dir_okay=False, readable=True),
    source_url: str = typer.Option(..., "--source-url", "-u", help="Canonical URL of the source instrument"),
    portal_name: str = typer.Option("manual-upload", "--portal"),
    title: str | None = typer.Option(None, "--title"),
    legal_form: str = typer.Option("statute", "--legal-form"),
    config_dir: Path = typer.Option(Path("configs/jurisdictions"), "--config-dir"),
    indicators_path: Path = typer.Option(Path("configs/rdtii_indicators.yaml"), "--indicators"),
    out: Path = typer.Option(Path("outputs") / "demo.jsonld", "--out", "-o"),
    top_k: int = typer.Option(1, "--top-k"),
    min_score: float = typer.Option(0.5, "--min-score"),
) -> None:
    """Slice 0 end-to-end: ingest one PDF and emit verbatim-validated citations."""
    from lexora.collect.profile_loader import load_profile
    from lexora.export.jsonld_exporter import to_jsonld
    from lexora.indicators import load_indicators
    from lexora.pipeline import run_demo_pipeline

    profile = load_profile(config_dir / f"{jurisdiction.lower()}.yaml")
    indicators = load_indicators(indicators_path)
    console.print(f"[bold]Jurisdiction:[/bold] {profile.jurisdiction} ({profile.iso_code})")
    console.print(f"[bold]Indicators loaded:[/bold] {len(indicators)}")

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
    console.print(
        f"[green]Wrote {n} citation(s) to {out}[/green]  "
        f"(clauses={len(artifacts.clauses)}, pages={len(artifacts.pages)}, "
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
