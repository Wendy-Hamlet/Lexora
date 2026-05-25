"""Top-level Lexora CLI.

Usage:
    lexora collect --jurisdiction SG
    lexora extract --document-id <id>
    lexora classify --indicator 6.1 --jurisdiction SG
    lexora export --format jsonld --out citations.jsonl
    lexora serve   # FastAPI audit UI
"""
from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

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
def serve(
    host: str = typer.Option("127.0.0.1", "--host"),
    port: int = typer.Option(8001, "--port"),
) -> None:
    """Launch the FastAPI audit UI."""
    import uvicorn

    uvicorn.run("lexora.api.app:api", host=host, port=port, reload=False)


if __name__ == "__main__":
    app()
