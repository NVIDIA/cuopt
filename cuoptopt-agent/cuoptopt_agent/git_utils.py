"""Git and GitHub utilities.

Responsibilities:
- Infer a branch name from the query and current date
- Create the branch, commit all changes
- Open a pull request via PyGithub with a rich description
"""

from __future__ import annotations

import logging
import os
import re
import subprocess
from datetime import date
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .implementation import Change
    from .testing import RegressionReport

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Branch name inference
# ---------------------------------------------------------------------------

_TYPE_KEYWORDS: dict[str, list[str]] = {
    "presolve":    ["presolver", "presolve", "bound-tightening", "probing", "redundancy"],
    "mip-solver":  ["branch-and-bound", "cutting-plane", "bab", "milp", "mip", "integer"],
    "lp-solver":   ["simplex", "pdlp", "lp", "linear-program", "dual", "primal"],
    "routing":     ["vrp", "tsp", "pdp", "routing", "vehicle"],
    "cuda-kernel": ["cuda", "kernel", "gpu", "warp", "shared-memory", "tensor-core", "sm"],
    "qp-solver":   ["qp", "quadratic", "portfolio"],
    "memory":      ["memory", "coalescing", "bandwidth", "cache", "hbm"],
}


def infer_branch_type(query: str) -> str:
    lower = query.lower()
    for label, keywords in _TYPE_KEYWORDS.items():
        if any(kw in lower for kw in keywords):
            return label
    return "optimization"


def make_branch_name(query: str) -> str:
    today = date.today().isoformat()          # YYYY-MM-DD
    branch_type = infer_branch_type(query)
    # Sanitize: alphanumeric + hyphens only
    safe_type = re.sub(r"[^a-z0-9-]", "-", branch_type).strip("-")
    return f"{today}-{safe_type}"


# ---------------------------------------------------------------------------
# Git helpers
# ---------------------------------------------------------------------------

def _git(args: list[str], cwd: Path, check: bool = True) -> subprocess.CompletedProcess:  # type: ignore[type-arg]
    return subprocess.run(
        ["git"] + args,
        cwd=cwd,
        capture_output=True,
        text=True,
        check=check,
    )


def get_current_branch(repo_root: Path) -> str:
    r = _git(["rev-parse", "--abbrev-ref", "HEAD"], cwd=repo_root)
    return r.stdout.strip()


def create_branch(branch_name: str, repo_root: Path) -> None:
    _git(["checkout", "-b", branch_name], cwd=repo_root)
    logger.info("Created branch: %s", branch_name)


def commit_changes(changes: list["Change"], query: str, repo_root: Path) -> str:
    """Stage and commit all modified files. Returns the commit SHA."""
    files = [c.file_path for c in changes]
    _git(["add", "--"] + files, cwd=repo_root)

    subject = f"perf: {query[:72]}"
    body = (
        "Automated change produced by cuoptopt-agent.\n\n"
        "Modified files:\n" + "\n".join(f"  - {f}" for f in files)
    )
    _git(["commit", "-m", f"{subject}\n\n{body}"], cwd=repo_root)
    sha = _git(["rev-parse", "HEAD"], cwd=repo_root).stdout.strip()
    logger.info("Committed as %s", sha)
    return sha


def push_branch(branch_name: str, repo_root: Path) -> None:
    _git(["push", "-u", "origin", branch_name], cwd=repo_root)
    logger.info("Pushed %s to origin", branch_name)


# ---------------------------------------------------------------------------
# GitHub PR creation
# ---------------------------------------------------------------------------

def _get_repo_slug(repo_root: Path) -> tuple[str, str]:
    """Return (owner, repo) from the origin remote URL."""
    r = _git(["remote", "get-url", "origin"], cwd=repo_root)
    url = r.stdout.strip()
    # ssh: git@github.com:owner/repo.git  or  https://github.com/owner/repo.git
    m = re.search(r"[:/]([^/]+)/([^/]+?)(?:\.git)?$", url)
    if not m:
        raise ValueError(f"Cannot parse GitHub slug from remote URL: {url}")
    return m.group(1), m.group(2)


def create_pull_request(
    branch_name: str,
    base_branch: str,
    query: str,
    report: "RegressionReport",
    papers_text: str,
    repo_root: Path,
) -> str:
    """Create a GitHub PR and return the PR URL."""
    token = os.environ.get("GITHUB_TOKEN", "")
    if not token:
        raise EnvironmentError(
            "GITHUB_TOKEN environment variable is not set. "
            "Export a personal access token with repo scope."
        )

    from github import Github  # type: ignore[import]

    owner, repo_name = _get_repo_slug(repo_root)
    gh = Github(token)
    repo = gh.get_repo(f"{owner}/{repo_name}")

    title = f"perf: {query[:72]}"
    body = _build_pr_body(query, report, papers_text, branch_name)

    pr = repo.create_pull(
        title=title,
        body=body,
        head=branch_name,
        base=base_branch,
    )
    logger.info("Pull request created: %s", pr.html_url)
    return pr.html_url


def _build_pr_body(
    query: str,
    report: "RegressionReport",
    papers_text: str,
    branch_name: str,
) -> str:
    return f"""\
## Summary

**Optimization query:** {query}

This PR was produced by `cuoptopt-agent`. It implements the code changes \
described below, which were validated against the cuOpt benchmark suite.

## Benchmark Results

{report.per_instance_table()}

**Aggregate:**
```
{report.summary()}
```

## Literature

{papers_text}

---
_Branch: `{branch_name}` | Generated by cuoptopt-agent_
"""
