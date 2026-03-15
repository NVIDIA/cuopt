"""LLM-driven code modification agent.

Sends the query, loaded skills, research papers, and any prior failure context
to the LLM. Expects the response to contain one or more ``<diff>`` blocks
(unified diff format) that the agent applies to the repository using the
``patch`` utility.

Prompt contract
---------------
The LLM is instructed to respond with:

    <file>path/to/file.cu</file>
    <diff>
    --- a/path/to/file.cu
    +++ b/path/to/file.cu
    @@ ... @@
    ...
    </diff>

Multiple <diff> blocks are supported. Any explanatory text outside these tags
is captured as ``reasoning`` for logging and reassessment context.
"""

from __future__ import annotations

import logging
import re
import subprocess
import tempfile
from dataclasses import dataclass, field
from pathlib import Path

from .models import LLMClient
from .research import Paper, format_papers_for_prompt
from .skill_loader import LoadedSkill, format_skills_for_prompt

logger = logging.getLogger(__name__)

_DIFF_RE = re.compile(r"<diff>(.*?)</diff>", re.DOTALL | re.IGNORECASE)
_FILE_RE = re.compile(r"<file>(.*?)</file>", re.DOTALL | re.IGNORECASE)


# ---------------------------------------------------------------------------
# Data models
# ---------------------------------------------------------------------------

@dataclass
class Change:
    file_path: str
    diff_text: str


@dataclass
class ImplementationResult:
    changes: list[Change] = field(default_factory=list)
    reasoning: str = ""
    raw_response: str = ""
    applied: bool = False
    apply_errors: list[str] = field(default_factory=list)


@dataclass
class PriorFailure:
    iteration: int
    changes: list[Change]
    reason: str      # "speed_regression" | "quality_regression_denied" | "apply_error"
    details: str


# ---------------------------------------------------------------------------
# Prompt construction
# ---------------------------------------------------------------------------

_SYSTEM_PROMPT = """\
You are a GPU software engineer working on NVIDIA cuOpt, a high-performance \
combinatorial optimization solver. Your task is to improve the cuOpt codebase \
based on the provided optimization query, leveraging domain-specific skills \
and recent literature.

RESPONSE FORMAT (strictly required):
- For each file you want to modify, output a block in this exact format:

<file>relative/path/from/repo/root.ext</file>
<diff>
--- a/relative/path/from/repo/root.ext
+++ b/relative/path/from/repo/root.ext
@@ -LINE,COUNT +LINE,COUNT @@
 context line
-removed line
+added line
 context line
</diff>

Rules:
- Output ONLY unified diff format inside <diff> tags.
- Diffs must apply cleanly with `patch -p1`.
- Paths are relative to the repository root (not cuoptopt-agent/).
- You may output multiple <file>/<diff> pairs.
- Place your reasoning and explanation OUTSIDE the tags (before or after).
- Do NOT include any code outside of diff blocks — only diffs modify files.
- Be conservative: prefer minimal, focused changes over large rewrites.
"""


def _build_user_prompt(
    query: str,
    skills: list[LoadedSkill],
    papers: list[Paper],
    prior_failures: list[PriorFailure],
    repo_structure_hint: str,
) -> str:
    parts: list[str] = []

    parts.append(f"## Optimization Query\n\n{query}")

    if skills:
        parts.append("## Relevant Skills\n\n" + format_skills_for_prompt(skills))

    if papers:
        parts.append("## Recent Literature\n\n" + format_papers_for_prompt(papers))

    if repo_structure_hint:
        parts.append(f"## Repository Structure (relevant paths)\n\n```\n{repo_structure_hint}\n```")

    if prior_failures:
        parts.append("## Prior Attempts (rejected — do NOT repeat these approaches)\n")
        for f in prior_failures:
            files_changed = ", ".join(c.file_path for c in f.changes) or "none"
            parts.append(
                f"### Iteration {f.iteration}\n"
                f"- Files changed: {files_changed}\n"
                f"- Rejection reason: {f.reason}\n"
                f"- Details: {f.details}\n"
            )

    parts.append(
        "## Task\n\n"
        "Analyze the query, skills, and literature. Produce minimal, targeted diffs "
        "that improve cuOpt performance or solution quality for the stated goal. "
        "Explain your approach before the diff blocks."
    )

    return "\n\n---\n\n".join(parts)


# ---------------------------------------------------------------------------
# Diff parsing & application
# ---------------------------------------------------------------------------

def _parse_response(response: str) -> tuple[list[Change], str]:
    """Extract Change objects and reasoning text from LLM response."""
    changes: list[Change] = []
    diff_spans: list[tuple[int, int]] = []

    for m in _DIFF_RE.finditer(response):
        diff_text = m.group(1).strip()
        # Try to find the preceding <file> tag
        preceding = response[: m.start()]
        file_match = list(_FILE_RE.finditer(preceding))
        file_path = file_match[-1].group(1).strip() if file_match else _guess_path_from_diff(diff_text)
        changes.append(Change(file_path=file_path, diff_text=diff_text))
        diff_spans.append((m.start(), m.end()))

    # Reasoning: everything not inside tags
    reasoning = _DIFF_RE.sub("", _FILE_RE.sub("", response)).strip()
    return changes, reasoning


def _guess_path_from_diff(diff_text: str) -> str:
    for line in diff_text.splitlines():
        if line.startswith("+++ b/"):
            return line[6:].strip()
        if line.startswith("--- a/"):
            return line[6:].strip()
    return "unknown"


def _apply_diff(repo_root: Path, diff_text: str) -> str | None:
    """Apply a unified diff via `patch -p1`. Returns error string or None on success."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_text)
        patch_path = f.name

    try:
        result = subprocess.run(
            ["patch", "-p1", "--input", patch_path, "--forward"],
            cwd=repo_root,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            return result.stdout + result.stderr
        return None
    finally:
        Path(patch_path).unlink(missing_ok=True)


def _revert_diff(repo_root: Path, diff_text: str) -> None:
    """Revert a previously applied diff."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".patch", delete=False) as f:
        f.write(diff_text)
        patch_path = f.name

    try:
        subprocess.run(
            ["patch", "-p1", "--input", patch_path, "--reverse", "--forward"],
            cwd=repo_root,
            capture_output=True,
        )
    finally:
        Path(patch_path).unlink(missing_ok=True)


# ---------------------------------------------------------------------------
# Repository structure hint
# ---------------------------------------------------------------------------

def _get_repo_structure_hint(repo_root: Path) -> str:
    """Return a concise listing of the most relevant source directories."""
    important_paths = [
        "cpp/src",
        "cpp/include",
        "python/cuopt",
        "python/libcuopt",
    ]
    lines: list[str] = []
    for rel in important_paths:
        p = repo_root / rel
        if p.is_dir():
            sub = sorted(p.iterdir())[:12]
            for item in sub:
                lines.append(f"{rel}/{item.name}{'/' if item.is_dir() else ''}")
    return "\n".join(lines[:60])


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_and_apply_changes(
    query: str,
    skills: list[LoadedSkill],
    papers: list[Paper],
    prior_failures: list[PriorFailure],
    client: LLMClient,
    repo_root: Path,
) -> ImplementationResult:
    """Ask the LLM for diffs, parse them, and apply them to the repository."""
    repo_hint = _get_repo_structure_hint(repo_root)
    user_prompt = _build_user_prompt(query, skills, papers, prior_failures, repo_hint)

    logger.info("Requesting implementation from LLM (%s)…", client.cfg["model"])
    raw_response = client.complete(system=_SYSTEM_PROMPT, user=user_prompt)

    changes, reasoning = _parse_response(raw_response)

    result = ImplementationResult(
        changes=changes,
        reasoning=reasoning,
        raw_response=raw_response,
    )

    if not changes:
        logger.warning("LLM returned no diff blocks.")
        return result

    # Apply each diff
    applied_so_far: list[Change] = []
    for change in changes:
        err = _apply_diff(repo_root, change.diff_text)
        if err:
            result.apply_errors.append(f"{change.file_path}: {err}")
            logger.error("Failed to apply diff to %s:\n%s", change.file_path, err)
            # Revert successfully applied diffs so the tree stays clean
            for prev in reversed(applied_so_far):
                _revert_diff(repo_root, prev.diff_text)
            return result
        applied_so_far.append(change)
        logger.info("Applied diff to %s", change.file_path)

    result.applied = True
    return result


def revert_all_changes(repo_root: Path, changes: list[Change]) -> None:
    """Revert all changes from a previous ``ImplementationResult``."""
    for change in reversed(changes):
        _revert_diff(repo_root, change.diff_text)
        logger.info("Reverted %s", change.file_path)
