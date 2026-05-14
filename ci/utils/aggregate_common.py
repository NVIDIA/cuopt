#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Shared helpers for the nightly and PR aggregators.

Both aggregators consume per-matrix summary JSONs produced by
``nightly_report.py`` and merge them into a single view.  The merge logic,
S3 listing, and HTML escaping are identical in both cases and live here.

Renderers (HTML dashboard for nightly; Markdown comment for PRs) stay in the
respective aggregator scripts since their output formats diverge.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

# Ensure ci/utils is importable when invoked from a sibling script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_helpers import s3_download, s3_list  # noqa: E402


def download_summaries(
    s3_prefix: str,
    local_dir: str | os.PathLike[str],
    s3_fallback_prefix: str = "",
) -> list[dict[str, Any]]:
    """Download all JSON summaries from an S3 prefix into a local directory.

    If ``s3_fallback_prefix`` is set and no summaries are found at
    ``s3_prefix``, retries with the fallback (used when the run-scoped
    path is empty because uploads landed under the branch-scoped path).

    Args:
        s3_prefix: Primary S3 URI prefix to list (e.g.,
            ``s3://bucket/ci_test_reports/pr/run-12345/``).
        local_dir: Local directory to download into.  Created if absent.
        s3_fallback_prefix: Optional secondary prefix to try when
            ``s3_prefix`` yields no ``*.json`` summaries.

    Returns:
        List of loaded summary dicts.  Files that fail to parse are
        skipped with a warning to stderr; the list contains only
        successfully loaded entries.

    Raises:
        This function does not raise.  Underlying S3 / IO / JSON parse
        errors are caught and logged.
    """
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    uris = s3_list(s3_prefix)
    json_uris = [
        u
        for u in uris
        if u.endswith(".json") and not u.endswith("/consolidated.json")
    ]

    if (
        not json_uris
        and s3_fallback_prefix
        and s3_fallback_prefix != s3_prefix
    ):
        print(
            f"No summaries at {s3_prefix}, trying fallback: {s3_fallback_prefix}"
        )
        uris = s3_list(s3_fallback_prefix)
        json_uris = [
            u
            for u in uris
            if u.endswith(".json") and not u.endswith("/consolidated.json")
        ]
        if json_uris:
            s3_prefix = s3_fallback_prefix

    print(f"Found {len(json_uris)} summary file(s) at {s3_prefix}")

    summaries = []
    for uri in json_uris:
        filename = uri.rsplit("/", 1)[-1]
        local_path = str(local_dir / filename)
        if s3_download(uri, local_path):
            try:
                with open(local_path) as f:
                    summaries.append(json.load(f))
            except (json.JSONDecodeError, OSError) as exc:
                print(
                    f"WARNING: Failed to parse {local_path}: {exc}",
                    file=sys.stderr,
                )
    return summaries


def load_local_summaries(
    local_dir: str | os.PathLike[str],
) -> list[dict[str, Any]]:
    """Load JSON summaries from a local directory (for testing without S3).

    Args:
        local_dir: Directory containing ``*.json`` per-matrix summaries.

    Returns:
        List of loaded summary dicts.  Files that fail to parse are
        skipped with a warning to stderr.

    Raises:
        This function does not raise.  IO / JSON parse errors are caught
        and logged.
    """
    local_dir = Path(local_dir)
    summaries = []
    for json_file in sorted(local_dir.glob("*.json")):
        try:
            with open(json_file) as f:
                summaries.append(json.load(f))
        except (json.JSONDecodeError, OSError) as exc:
            print(
                f"WARNING: Failed to parse {json_file}: {exc}", file=sys.stderr
            )
    return summaries


def aggregate_summaries(
    summaries: list[dict[str, Any]],
) -> dict[str, Any]:
    """Merge per-matrix summaries into a single consolidated view.

    Args:
        summaries: List of per-matrix summary dicts as produced by
            ``nightly_report.py`` (either nightly or PR mode).  Each
            dict is expected to provide at least ``test_type``,
            ``matrix_label``, ``counts``, and the per-bucket failure
            lists; missing fields default to safe values.

    Returns:
        Consolidated dict with keys:

          - ``matrix_grid``: list of per-matrix status dicts (sorted
            by ``test_type`` then ``matrix_label``).
          - ``totals``: aggregate test counts across all matrices.
          - ``all_new_failures``, ``all_recurring_failures``,
            ``all_flaky_tests``, ``all_resolved_tests``: merged failure
            lists with per-entry ``test_type`` / ``matrix_label``
            context added.
          - ``has_new_flaky``: True iff any summary flagged a new flaky.

    Raises:
        This function does not raise.  Malformed entries are tolerated.
    """
    grid = []
    totals = {
        "total": 0,
        "passed": 0,
        "failed": 0,
        "flaky": 0,
        "skipped": 0,
        "resolved": 0,
    }
    all_new_failures = []
    all_recurring_failures = []
    all_flaky_tests = []
    all_resolved_tests = []
    any_new_flaky = False

    for s in summaries:
        test_type = s.get("test_type", "unknown")
        matrix_label = s.get("matrix_label", "unknown")
        counts = s.get("counts", {})

        failed = counts.get("failed", 0)
        flaky = counts.get("flaky", 0)
        has_new = s.get("has_new_failures", False)
        if s.get("has_new_flaky", False):
            any_new_flaky = True

        if failed > 0:
            status = "failed-new" if has_new else "failed-recurring"
        elif flaky > 0:
            status = "flaky"
        elif counts.get("total", 0) == 0:
            status = "no-results"
        else:
            status = "passed"

        grid.append(
            {
                "test_type": test_type,
                "matrix_label": matrix_label,
                "status": status,
                "counts": counts,
                "sha": s.get("sha", ""),
            }
        )

        for key in totals:
            totals[key] += counts.get(key, 0)

        ctx = {"test_type": test_type, "matrix_label": matrix_label}
        for entry in s.get("new_failures", []):
            all_new_failures.append({**entry, **ctx})
        for entry in s.get("recurring_failures", []):
            all_recurring_failures.append({**entry, **ctx})
        for entry in s.get("flaky_tests", []):
            all_flaky_tests.append({**entry, **ctx})
        for entry in s.get("resolved_tests", []):
            all_resolved_tests.append({**entry, **ctx})

    grid.sort(key=lambda g: (g["test_type"], g["matrix_label"]))

    return {
        "matrix_grid": grid,
        "totals": totals,
        "all_new_failures": all_new_failures,
        "all_recurring_failures": all_recurring_failures,
        "all_flaky_tests": all_flaky_tests,
        "all_resolved_tests": all_resolved_tests,
        "has_new_flaky": any_new_flaky,
    }


def html_escape(text: Any) -> str:
    """Escape HTML special characters in ``text``.

    Args:
        text: Any value; converted to ``str`` before escaping.

    Returns:
        ``str(text)`` with ``&``, ``<``, ``>`` and ``"`` replaced by
        their HTML entity equivalents.  Safe for inclusion in HTML
        attribute values and element bodies.

    Raises:
        This function does not raise.
    """
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )
