"""CLI entry point for cuoptopt-agent.

Usage:
    cuoptopt-agent "Improve presolver for dense LP systems on L40" --model nvidia
    cuoptopt-agent "Optimize VRP routing kernel memory access" --model claude
    cuoptopt-agent "Speed up branch-and-bound on H100" --model gpt --max-iter 3
"""

from __future__ import annotations

import logging
from pathlib import Path

import typer
from rich.console import Console
from rich.logging import RichHandler

app = typer.Typer(
    name="cuoptopt-agent",
    help=(
        "Agentic optimization loop for NVIDIA cuOpt. "
        "Searches literature, implements code improvements, validates against benchmarks, "
        "and opens a GitHub PR upon human approval."
    ),
    add_completion=False,
)

console = Console()

# Resolve paths relative to this file so the CLI works from any cwd
_HERE = Path(__file__).parent          # cuoptopt-agent/cuoptopt_agent/
_AGENT_DIR = _HERE.parent              # cuoptopt-agent/
_REPO_ROOT = _AGENT_DIR.parent         # repo root


def _setup_logging(verbose: bool) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(message)s",
        handlers=[RichHandler(rich_tracebacks=True, show_path=False)],
    )


@app.command()
def main(
    query: str = typer.Argument(
        ...,
        help='Natural-language optimization goal, e.g. "Improve presolver for dense LP on L40".',
    ),
    model: str = typer.Option(
        "claude",
        "--model", "-m",
        help="LLM backend to use: claude | gpt | nvidia",
        show_default=True,
    ),
    max_iter: int = typer.Option(
        5,
        "--max-iter",
        help="Maximum number of implement-test-evaluate cycles before giving up.",
        show_default=True,
    ),
    config_dir: Path = typer.Option(
        _AGENT_DIR / "config",
        "--config-dir",
        help="Directory containing models.yaml and thresholds.yaml.",
        show_default=True,
    ),
    repo_root: Path = typer.Option(
        _REPO_ROOT,
        "--repo-root",
        help="Root of the cuOpt repository.",
        show_default=True,
    ),
    skip_research: bool = typer.Option(
        False,
        "--skip-research",
        help="Skip Google Scholar / arxiv search (useful for offline runs).",
    ),
    verbose: bool = typer.Option(
        False,
        "--verbose", "-v",
        help="Enable debug logging.",
    ),
) -> None:
    """Run the cuoptopt-agent optimization loop."""
    _setup_logging(verbose)

    # Validate model key
    valid_models = ("claude", "gpt", "nvidia")
    if model not in valid_models:
        console.print(
            f"[red]Unknown model '{model}'. Choose from: {', '.join(valid_models)}[/red]"
        )
        raise typer.Exit(1)

    if not config_dir.is_dir():
        console.print(f"[red]Config directory not found: {config_dir}[/red]")
        raise typer.Exit(1)

    if not repo_root.is_dir():
        console.print(f"[red]Repo root not found: {repo_root}[/red]")
        raise typer.Exit(1)

    from .orchestrator import run

    run(
        query=query,
        model_key=model,
        config_dir=config_dir,
        repo_root=repo_root,
        max_iter=max_iter,
        skip_research=skip_research,
    )


if __name__ == "__main__":
    app()
