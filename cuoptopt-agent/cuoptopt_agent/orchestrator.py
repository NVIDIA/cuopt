"""Top-level workflow controller.

Implements the full optimization loop:

  1. Load relevant skills
  2. Search literature (Scholar + arxiv)
  3. Ask LLM to implement changes
  4. Run benchmarks (baseline then candidate)
  5. Evaluate regression:
     - Speed regression  → auto-reject, loop
     - Quality regression → ask human, loop or continue
     - No regression     → ask human for final approval
  6. On approval: create branch, commit, push, open PR
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path

import yaml
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.prompt import Confirm, Prompt

from .git_utils import (
    commit_changes,
    create_branch,
    create_pull_request,
    get_current_branch,
    make_branch_name,
    push_branch,
)
from .implementation import (
    ImplementationResult,
    PriorFailure,
    generate_and_apply_changes,
    revert_all_changes,
)
from .models import LLMClient
from .research import Paper, format_papers_for_prompt, search_literature
from .skill_loader import LoadedSkill, find_skills
from .testing import BenchmarkRun, compare_runs, run_benchmark

console = Console()
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Orchestration state
# ---------------------------------------------------------------------------

@dataclass
class AgentState:
    query: str
    model_key: str
    config_dir: Path
    repo_root: Path
    max_iter: int
    # filled in as we go
    skills: list[LoadedSkill] = field(default_factory=list)
    papers: list[Paper] = field(default_factory=list)
    baseline: BenchmarkRun | None = None
    prior_failures: list[PriorFailure] = field(default_factory=list)
    iteration: int = 0


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _load_yaml(path: Path) -> dict:  # type: ignore[type-arg]
    with open(path) as f:
        return yaml.safe_load(f) or {}


def _section(title: str) -> None:
    console.rule(f"[bold cyan]{title}[/bold cyan]")


def _print_md(text: str) -> None:
    console.print(Markdown(text))


# ---------------------------------------------------------------------------
# Main orchestrator
# ---------------------------------------------------------------------------

def run(
    query: str,
    model_key: str,
    config_dir: Path,
    repo_root: Path,
    max_iter: int = 5,
    skip_research: bool = False,
) -> None:
    """Entry point called from main.py."""
    thresholds = _load_yaml(config_dir / "thresholds.yaml")
    models_cfg_path = str(config_dir / "models.yaml")

    client = LLMClient(model_key=model_key, config_path=models_cfg_path)
    state = AgentState(
        query=query,
        model_key=model_key,
        config_dir=config_dir,
        repo_root=repo_root,
        max_iter=max_iter,
    )

    # ── Step 1: Load skills ────────────────────────────────────────────────
    _section("Loading Skills")
    state.skills = find_skills(query, repo_root, top_n=6)
    if state.skills:
        console.print(f"[green]Loaded {len(state.skills)} skill(s):[/green]")
        for s in state.skills:
            console.print(f"  • {s.name}  (score={s.score:.3f})")
    else:
        console.print("[yellow]No matching skills found.[/yellow]")

    # ── Step 2: Literature research ────────────────────────────────────────
    if skip_research:
        console.print("[dim]Skipping literature research (--skip-research).[/dim]")
    else:
        _section("Searching Literature")
        console.print("Searching Google Scholar and arxiv…")
        state.papers = search_literature(query, max_scholar=5, max_arxiv=5)
        console.print(f"[green]Found {len(state.papers)} paper(s).[/green]")
        if state.papers:
            _print_md(format_papers_for_prompt(state.papers[:3]))

    # ── Step 3: Baseline benchmark ─────────────────────────────────────────
    _section("Baseline Benchmark")
    console.print("Running baseline benchmarks (this may take a while)…")
    state.baseline = run_benchmark("baseline", repo_root, thresholds)
    ok_count = sum(1 for r in state.baseline.results if r.status == "ok")
    console.print(f"[green]Baseline: {ok_count}/{len(state.baseline.results)} instances completed.[/green]")

    # ── Main loop ──────────────────────────────────────────────────────────
    original_branch = get_current_branch(repo_root)

    for iteration in range(1, max_iter + 1):
        state.iteration = iteration
        _section(f"Implementation Loop — Iteration {iteration}/{max_iter}")

        # Step 3a: Generate & apply changes
        impl_result = generate_and_apply_changes(
            query=query,
            skills=state.skills,
            papers=state.papers,
            prior_failures=state.prior_failures,
            client=client,
            repo_root=repo_root,
        )

        if impl_result.reasoning:
            console.print(Panel(impl_result.reasoning, title="LLM Reasoning", border_style="blue"))

        if not impl_result.applied:
            errors = "\n".join(impl_result.apply_errors) or "LLM returned no diff blocks."
            console.print(f"[red]Changes could not be applied:[/red]\n{errors}")
            state.prior_failures.append(PriorFailure(
                iteration=iteration,
                changes=impl_result.changes,
                reason="apply_error",
                details=errors,
            ))
            if iteration == max_iter:
                _abort("Maximum iterations reached without a successful patch application.")
            continue

        console.print(f"[green]Applied {len(impl_result.changes)} diff(s).[/green]")

        # Step 3b: Candidate benchmark
        _section("Candidate Benchmark")
        console.print("Running candidate benchmarks…")
        candidate = run_benchmark("candidate", repo_root, thresholds)

        # Step 3c: Regression check
        report = compare_runs(
            state.baseline,
            candidate,
            thresholds.get("speed_tolerance_pct", 5.0),
            thresholds.get("quality_tolerance_pct", 1.0),
        )
        console.print(Panel(report.summary(), title="Regression Report", border_style="yellow"))
        _print_md(report.per_instance_table())

        if report.speed_regression:
            console.print("[red]Speed regression detected — auto-rejecting and reverting.[/red]")
            revert_all_changes(repo_root, impl_result.changes)
            state.prior_failures.append(PriorFailure(
                iteration=iteration,
                changes=impl_result.changes,
                reason="speed_regression",
                details=report.summary(),
            ))
            if iteration == max_iter:
                _abort("Maximum iterations reached. No improvement found.")
            continue

        if report.quality_regression:
            console.print("[yellow]Quality regression detected. Human review required.[/yellow]")
            _print_md(report.per_instance_table())
            accept = Confirm.ask(
                f"Quality degraded by {report.quality_delta_pct:.2f}%. "
                "Accept this tradeoff and proceed?"
            )
            if not accept:
                console.print("[red]Quality regression denied — reverting.[/red]")
                revert_all_changes(repo_root, impl_result.changes)
                state.prior_failures.append(PriorFailure(
                    iteration=iteration,
                    changes=impl_result.changes,
                    reason="quality_regression_denied",
                    details=report.summary(),
                ))
                if iteration == max_iter:
                    _abort("Maximum iterations reached. No improvement accepted.")
                continue

        # ── Step 4: Final human approval ───────────────────────────────────
        _section("Final Approval")
        console.print(Panel(report.summary(), title="Final Results", border_style="green"))
        console.print(f"\nModified files:")
        for c in impl_result.changes:
            console.print(f"  • {c.file_path}")

        approved = Confirm.ask("\nAccept these changes and create a GitHub PR?")
        if not approved:
            console.print("[red]Changes rejected by user — reverting.[/red]")
            revert_all_changes(repo_root, impl_result.changes)
            state.prior_failures.append(PriorFailure(
                iteration=iteration,
                changes=impl_result.changes,
                reason="user_rejected",
                details="User declined at final approval step.",
            ))
            if iteration == max_iter:
                _abort("Maximum iterations reached. No improvement accepted.")
            continue

        # ── Step 5: Create branch, commit, push, open PR ───────────────────
        _section("Creating Pull Request")
        branch_name = make_branch_name(query)
        create_branch(branch_name, repo_root)
        commit_changes(impl_result.changes, query, repo_root)
        push_branch(branch_name, repo_root)

        papers_text = format_papers_for_prompt(state.papers)
        pr_url = create_pull_request(
            branch_name=branch_name,
            base_branch=original_branch,
            query=query,
            report=report,
            papers_text=papers_text,
            repo_root=repo_root,
        )

        console.print(f"\n[bold green]Pull request created:[/bold green] {pr_url}")
        return  # success

    _abort(f"Reached maximum of {max_iter} iterations without a successful outcome.")


def _abort(message: str) -> None:
    console.print(f"\n[bold red]Agent stopped:[/bold red] {message}")
    raise SystemExit(1)
