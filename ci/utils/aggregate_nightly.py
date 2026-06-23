#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Aggregate per-matrix nightly test summaries into a single consolidated report.

Runs as a post-test job after all matrix CI jobs finish.  It:
  1. Lists all JSON summaries uploaded to S3 for today's date
  2. Downloads and merges them
  3. Builds a matrix grid (test_type x matrix_label → status)
  4. Generates a consolidated JSON, HTML report, and Slack payload
  5. Uploads the consolidated report to S3

Usage:
  python ci/utils/aggregate_nightly.py \\
      --s3-summaries-prefix s3://bucket/ci_test_reports/nightly/summaries/2026-04-13/ \\
      --s3-reports-prefix s3://bucket/ci_test_reports/nightly/reports/2026-04-13/ \\
      --output-dir /tmp/aggregate-output \\
      --date 2026-04-13 \\
      --branch main
"""

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

# Ensure ci/utils is importable when invoked as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_helpers import s3_download, s3_upload, s3_list  # noqa: E402


# ---------------------------------------------------------------------------
# Download and merge summaries
# ---------------------------------------------------------------------------


def download_summaries(s3_prefix, local_dir, s3_fallback_prefix=""):
    """Download all JSON summaries from S3 prefix into local_dir.
    If s3_fallback_prefix is set and no summaries found at s3_prefix,
    retries with the fallback (used when RAPIDS_BRANCH in rapidsai
    containers doesn't match the branch input).
    Returns list of loaded summary dicts."""
    local_dir = Path(local_dir)
    local_dir.mkdir(parents=True, exist_ok=True)

    uris = s3_list(s3_prefix)
    json_uris = [
        u
        for u in uris
        if u.endswith(".json") and not u.endswith("/consolidated.json")
    ]

    # Fallback: search the parent date prefix if branch-specific path is empty
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


def load_local_summaries(local_dir):
    """Load summaries from a local directory (for testing without S3)."""
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


# ---------------------------------------------------------------------------
# Aggregation
# ---------------------------------------------------------------------------


def aggregate_summaries(summaries):
    """Merge per-matrix summaries into a consolidated view.

    Returns a dict with:
      - matrix_grid: list of {test_type, matrix_label, status, counts, ...}
      - totals: aggregate counts
      - all_new_failures, all_recurring_failures, all_flaky_tests,
        all_resolved_tests: merged lists with matrix context added
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

        # Determine job status
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

        # Accumulate totals
        for key in totals:
            totals[key] += counts.get(key, 0)

        # Merge failure lists with matrix context
        ctx = {"test_type": test_type, "matrix_label": matrix_label}
        for entry in s.get("new_failures", []):
            all_new_failures.append({**entry, **ctx})
        for entry in s.get("recurring_failures", []):
            all_recurring_failures.append({**entry, **ctx})
        for entry in s.get("flaky_tests", []):
            all_flaky_tests.append({**entry, **ctx})
        for entry in s.get("resolved_tests", []):
            all_resolved_tests.append({**entry, **ctx})

    # Sort grid for consistent display
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


# ---------------------------------------------------------------------------
# Consolidated JSON
# ---------------------------------------------------------------------------


def parse_workflow_jobs(workflow_jobs_path):
    """Parse GitHub Actions workflow job statuses from JSON file.
    Returns all jobs (except nightly-summary itself) with name,
    conclusion, URL, and whether they are tracked by per-matrix
    S3 summaries."""
    if not workflow_jobs_path or not Path(workflow_jobs_path).exists():
        return []

    # Job name prefixes that are covered by per-matrix S3 reports.
    # These jobs also have detailed test results; other jobs only have
    # a pass/fail status at the workflow level.
    TRACKED_PREFIXES = (
        "conda-cpp-tests",
        "conda-python-tests",
        "wheel-tests-cuopt-server",
        "wheel-tests-cuopt",
    )

    try:
        with open(workflow_jobs_path) as f:
            data = json.load(f)
        jobs_list = data.get("jobs", [])
        result = []
        for job in jobs_list:
            name = job.get("name", "")
            # Skip the nightly-summary job itself
            if "nightly-summary" in name.lower():
                continue
            # Skip helper jobs (compute-matrix, etc.)
            if "compute-matrix" in name.lower():
                continue
            tracked = any(name.startswith(p) for p in TRACKED_PREFIXES)
            result.append(
                {
                    "name": name,
                    "conclusion": job.get("conclusion", "unknown"),
                    "status": job.get("status", "unknown"),
                    "url": job.get("html_url", ""),
                    "has_test_details": tracked,
                }
            )
        return result
    except (json.JSONDecodeError, OSError) as exc:
        print(
            f"WARNING: Failed to parse workflow jobs: {exc}",
            file=sys.stderr,
        )
        return []


def generate_consolidated_json(
    agg, date_str, branch, github_run_url="", workflow_jobs=None
):
    """Generate the consolidated JSON for Slack and dashboard."""
    total_jobs = len(agg["matrix_grid"])
    failed_jobs = sum(
        1 for g in agg["matrix_grid"] if g["status"].startswith("failed")
    )
    flaky_jobs = sum(1 for g in agg["matrix_grid"] if g["status"] == "flaky")
    passed_jobs = sum(1 for g in agg["matrix_grid"] if g["status"] == "passed")

    # Workflow-level CI job statuses
    wf_jobs = workflow_jobs or []
    failed_ci_jobs = [j for j in wf_jobs if j["conclusion"] == "failure"]
    # Jobs without per-matrix S3 tracking (notebooks, JuMP, etc.)
    untracked_failed = [
        j for j in failed_ci_jobs if not j.get("has_test_details", False)
    ]

    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "date": date_str,
        "branch": branch,
        "github_run_url": github_run_url,
        "job_summary": {
            "total": total_jobs,
            "passed": passed_jobs,
            "failed": failed_jobs,
            "flaky": flaky_jobs,
        },
        "test_totals": agg["totals"],
        "has_new_failures": len(agg["all_new_failures"]) > 0,
        "has_new_flaky": agg.get("has_new_flaky", False),
        "matrix_grid": agg["matrix_grid"],
        "new_failures": agg["all_new_failures"],
        "recurring_failures": agg["all_recurring_failures"],
        "flaky_tests": agg["all_flaky_tests"],
        "resolved_tests": agg["all_resolved_tests"],
        "workflow_jobs": wf_jobs,
        "failed_ci_jobs": failed_ci_jobs,
        "untracked_failed_ci_jobs": untracked_failed,
    }


# ---------------------------------------------------------------------------
# Consolidated HTML
# ---------------------------------------------------------------------------


def _html_escape(text):
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def _status_badge(status):
    """Return an HTML badge for a matrix cell status."""
    colors = {
        "passed": ("#388e3c", "PASS"),
        "failed-new": ("#d32f2f", "NEW FAIL"),
        "failed-recurring": ("#e65100", "RECURRING"),
        "flaky": ("#f9a825", "FLAKY"),
        "no-results": ("#757575", "NO DATA"),
    }
    bg, label = colors.get(status, ("#757575", status.upper()))
    text_color = "#212121" if status == "flaky" else "#fff"
    return (
        f'<span style="display:inline-block;padding:3px 8px;border-radius:4px;'
        f'background:{bg};color:{text_color};font-size:0.75rem;font-weight:600">'
        f"{label}</span>"
    )


def generate_consolidated_html(
    agg,
    date_str,
    branch,
    github_run_url="",
    s3_reports_prefix="",
):
    """Generate a consolidated HTML dashboard for all matrix combos."""
    total_jobs = len(agg["matrix_grid"])
    failed_jobs = sum(
        1 for g in agg["matrix_grid"] if g["status"].startswith("failed")
    )

    if failed_jobs > 0:
        bar_color = "#d32f2f"
        bar_text = f"{failed_jobs} of {total_jobs} matrix jobs have failures"
    elif any(g["status"] == "flaky" for g in agg["matrix_grid"]):
        bar_color = "#f9a825"
        bar_text = "All jobs passed (flaky tests detected)"
    else:
        bar_color = "#388e3c"
        bar_text = f"All {total_jobs} matrix jobs passed"

    totals = agg["totals"]

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>cuOpt Nightly — {_html_escape(branch)} — {_html_escape(date_str)}</title>
<style>
  :root {{ --fail: #d32f2f; --pass: #388e3c; --flaky: #f9a825; --skip: #757575;
           --bg: #fafafa; --card: #fff; --border: #e0e0e0; --text: #212121; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text);
          padding: 24px; max-width: 1400px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .meta {{ color: #616161; font-size: 0.85rem; margin-bottom: 16px; }}
  .meta a {{ color: #1565c0; }}
  .status-bar {{ padding: 12px 16px; border-radius: 8px; color: #fff;
                 font-weight: 600; margin-bottom: 20px; font-size: 1.1rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(130px, 1fr));
                   gap: 12px; margin-bottom: 24px; }}
  .summary-card {{ background: var(--card); border: 1px solid var(--border);
                   border-radius: 8px; padding: 14px; text-align: center; }}
  .summary-card .num {{ font-size: 1.8rem; font-weight: 700; }}
  .summary-card .lbl {{ font-size: 0.75rem; color: #757575; text-transform: uppercase; }}
  .num.pass {{ color: var(--pass); }}  .num.fail {{ color: var(--fail); }}
  .num.flaky {{ color: var(--flaky); }}  .num.skip {{ color: var(--skip); }}
  section {{ margin-bottom: 24px; }}
  h2 {{ font-size: 1.15rem; margin-bottom: 10px; padding-bottom: 4px;
        border-bottom: 2px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #f5f5f5; text-align: left; padding: 8px 10px; font-weight: 600;
        position: sticky; top: 0; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:hover td {{ background: #f5f5f5; }}
  details {{ margin-top: 4px; }}
  details summary {{ cursor: pointer; color: #1565c0; font-size: 0.8rem; }}
  pre.error {{ background: #263238; color: #e0e0e0; padding: 12px; border-radius: 6px;
               font-size: 0.78rem; overflow-x: auto; white-space: pre-wrap;
               word-break: break-word; max-height: 300px; margin-top: 6px; }}
  .matrix-link {{ color: #1565c0; text-decoration: none; }}
  .matrix-link:hover {{ text-decoration: underline; }}
</style>
</head>
<body>
<h1>cuOpt Nightly Tests — {_html_escape(branch)}</h1>
<div class="meta">
  Date: <strong>{_html_escape(date_str)}</strong>""")

    if github_run_url:
        parts.append(
            f' &nbsp;|&nbsp; <a href="{_html_escape(github_run_url)}">'
            f"GitHub Actions Run</a>"
        )

    parts.append(f"""</div>
<div class="status-bar" style="background:{bar_color}">{bar_text}</div>
<div class="summary-grid">
  <div class="summary-card"><div class="num">{totals["total"]}</div><div class="lbl">Total Tests</div></div>
  <div class="summary-card"><div class="num pass">{totals["passed"]}</div><div class="lbl">Passed</div></div>
  <div class="summary-card"><div class="num fail">{totals["failed"]}</div><div class="lbl">Failed</div></div>
  <div class="summary-card"><div class="num flaky">{totals["flaky"]}</div><div class="lbl">Flaky</div></div>
  <div class="summary-card"><div class="num skip">{totals["skipped"]}</div><div class="lbl">Skipped</div></div>
  <div class="summary-card"><div class="num pass">{totals["resolved"]}</div><div class="lbl">Stabilized</div></div>
</div>""")

    # Helper: build a GitHub source link for test names when suite looks like a file path
    def _test_name_html(entry):
        """Return HTML for the test name, linked to source if suite looks like a file path."""
        name_escaped = _html_escape(entry["name"])
        suite = entry.get("suite", "")
        # Find the sha from the matching grid entry
        sha = "unknown"
        for g in agg["matrix_grid"]:
            if (
                g["test_type"] == entry.get("test_type")
                and g["matrix_label"] == entry.get("matrix_label")
                and g.get("sha")
            ):
                sha = g["sha"]
                break
        if (
            sha != "unknown"
            and suite
            and ("/" in suite or suite.endswith(".py"))
        ):
            url = f"https://github.com/NVIDIA/cuopt/blob/{_html_escape(sha)}/{_html_escape(suite)}"
            return f'<a href="{url}" style="color:#1565c0;text-decoration:none"><code>{name_escaped}</code></a>'
        return f"<code>{name_escaped}</code>"

    def _error_summary(message, max_len=200):
        """Extract the most useful part of an error message for display.
        Prefers the last line (usually the assertion) over the first
        (usually the test method signature)."""
        if not message:
            return ""
        lines = [
            ln.strip() for ln in message.strip().splitlines() if ln.strip()
        ]
        # Use the last non-empty line (typically the assertion/error)
        if lines:
            summary = lines[-1]
            # If the last line is very short, include the previous line too
            if len(summary) < 40 and len(lines) > 1:
                summary = lines[-2] + " — " + summary
        else:
            summary = message
        if len(summary) > max_len:
            summary = summary[:max_len] + "..."
        return summary

    # --- New failures ---
    if agg["all_new_failures"]:
        parts.append("<section><h2>New Failures</h2><table>")
        parts.append(
            "<tr><th>Test Type</th><th>Matrix</th><th>Suite</th>"
            "<th>Test</th><th>Error</th></tr>"
        )
        for e in agg["all_new_failures"]:
            msg = _html_escape(e.get("message", ""))
            short = _html_escape(_error_summary(e.get("message", "")))
            parts.append(
                f"<tr><td>{_html_escape(e['test_type'])}</td>"
                f"<td><code>{_html_escape(e['matrix_label'])}</code></td>"
                f"<td>{_html_escape(e['suite'])}</td>"
                f"<td>{_test_name_html(e)}</td>"
                f"<td><details><summary>{short}</summary>"
                f'<pre class="error">{msg}</pre></details></td></tr>'
            )
        parts.append("</table></section>")

    # --- Flaky ---
    if agg["all_flaky_tests"]:
        parts.append("<section><h2>Flaky Tests</h2><table>")
        parts.append(
            "<tr><th>Test Type</th><th>Matrix</th><th>Suite</th>"
            "<th>Test</th><th>Retries</th><th>Error</th></tr>"
        )
        for e in agg["all_flaky_tests"]:
            msg = _html_escape(e.get("message", ""))
            short = _html_escape(_error_summary(e.get("message", "")))
            parts.append(
                f"<tr><td>{_html_escape(e['test_type'])}</td>"
                f"<td><code>{_html_escape(e['matrix_label'])}</code></td>"
                f"<td>{_html_escape(e['suite'])}</td>"
                f"<td><code>{_html_escape(e['name'])}</code></td>"
                f"<td>{e.get('retry_count', '?')}</td>"
                f"<td><details><summary>{short}</summary>"
                f'<pre class="error">{msg}</pre></details></td></tr>'
            )
        parts.append("</table></section>")

    # --- Recurring failures ---
    if agg["all_recurring_failures"]:
        parts.append("<section><h2>Recurring Failures</h2><table>")
        parts.append(
            "<tr><th>Test Type</th><th>Matrix</th><th>Suite</th>"
            "<th>Test</th><th>Since</th><th>Error</th></tr>"
        )
        for e in agg["all_recurring_failures"]:
            msg = _html_escape(e.get("message", ""))
            short = _html_escape(_error_summary(e.get("message", "")))
            parts.append(
                f"<tr><td>{_html_escape(e['test_type'])}</td>"
                f"<td><code>{_html_escape(e['matrix_label'])}</code></td>"
                f"<td>{_html_escape(e['suite'])}</td>"
                f"<td>{_test_name_html(e)}</td>"
                f"<td>{_html_escape(e.get('first_seen', '?'))}</td>"
                f"<td><details><summary>{short}</summary>"
                f'<pre class="error">{msg}</pre></details></td></tr>'
            )
        parts.append("</table></section>")

    # --- Resolved ---
    if agg["all_resolved_tests"]:
        parts.append("<section><h2>Stabilized Tests</h2><table>")
        parts.append(
            "<tr><th>Test Type</th><th>Matrix</th><th>Suite</th>"
            "<th>Test</th><th>Failing Since</th><th>Count</th></tr>"
        )
        for e in agg["all_resolved_tests"]:
            parts.append(
                f"<tr><td>{_html_escape(e['test_type'])}</td>"
                f"<td><code>{_html_escape(e['matrix_label'])}</code></td>"
                f"<td>{_html_escape(e['suite'])}</td>"
                f"<td><code>{_html_escape(e['name'])}</code></td>"
                f"<td>{_html_escape(e.get('first_seen', '?'))}</td>"
                f"<td>{e.get('failure_count', '?')}</td></tr>"
            )
        parts.append("</table></section>")

    if (
        not agg["all_new_failures"]
        and not agg["all_recurring_failures"]
        and not agg["all_flaky_tests"]
        and not agg["all_resolved_tests"]
    ):
        parts.append(
            '<p style="color:#9e9e9e;font-style:italic;padding:16px">'
            "All tests passed across all matrices!</p>"
        )

    # --- Matrix grid (at the end) ---
    parts.append("<section><h2>Matrix Overview</h2><table>")
    parts.append(
        "<tr><th>Test Type</th><th>Matrix</th><th>Status</th>"
        "<th>Passed</th><th>Failed</th><th>Flaky</th><th>Total</th><th>Report</th></tr>"
    )
    for g in agg["matrix_grid"]:
        counts = g["counts"]
        report_link = ""
        if s3_reports_prefix:
            report_filename = f"{g['test_type']}-{g['matrix_label']}.html"
            prefix = s3_reports_prefix.rstrip("/") + "/"
            report_link = (
                f'<a class="matrix-link" href="{_html_escape(prefix)}'
                f'{_html_escape(report_filename)}">View</a>'
            )
        parts.append(
            f"<tr><td><strong>{_html_escape(g['test_type'])}</strong></td>"
            f"<td><code>{_html_escape(g['matrix_label'])}</code></td>"
            f"<td>{_status_badge(g['status'])}</td>"
            f"<td>{counts.get('passed', 0)}</td>"
            f"<td>{counts.get('failed', 0)}</td>"
            f"<td>{counts.get('flaky', 0)}</td>"
            f"<td>{counts.get('total', 0)}</td>"
            f"<td>{report_link}</td></tr>"
        )
    parts.append("</table></section>")

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Index management
# ---------------------------------------------------------------------------

MAX_INDEX_DAYS = 90  # Keep at most 90 days in the index


def update_index(s3_index_uri, date_str, consolidated, output_dir):
    """Download index.json, add today's entry, prune old entries, re-upload."""
    local_index = str(output_dir / "index.json")

    # Download existing index (or start fresh)
    index = {"_schema_version": 1, "dates": {}}
    if s3_download(s3_index_uri, local_index):
        try:
            with open(local_index) as f:
                loaded = json.load(f)
                if "dates" in loaded:
                    index = loaded
        except (json.JSONDecodeError, OSError):
            pass

    # Add today's entry keyed by date/branch for multi-branch support
    branch = consolidated.get("branch", "main")
    entry_key = f"{date_str}/{branch}"
    index["dates"][entry_key] = {
        "date": date_str,
        "branch": branch,
        "job_summary": consolidated.get("job_summary", {}),
        "test_totals": consolidated.get("test_totals", {}),
        "has_new_failures": consolidated.get("has_new_failures", False),
        "github_run_url": consolidated.get("github_run_url", ""),
    }

    # Prune to last N entries
    dates_sorted = sorted(index["dates"].keys(), reverse=True)
    if len(dates_sorted) > MAX_INDEX_DAYS:
        for old_key in dates_sorted[MAX_INDEX_DAYS:]:
            del index["dates"][old_key]

    # Write and upload
    with open(local_index, "w") as f:
        json.dump(index, f, indent=2, sort_keys=True)
        f.write("\n")
    print(f"Updated index.json with {len(index['dates'])} date(s)")

    s3_upload(local_index, s3_index_uri)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Aggregate per-matrix nightly test summaries"
    )
    parser.add_argument(
        "--s3-summaries-prefix",
        default="",
        help="S3 prefix for per-matrix JSON summaries (e.g., s3://bucket/.../summaries/2026-04-13/)",
    )
    parser.add_argument(
        "--s3-summaries-fallback",
        default="",
        help="Fallback S3 prefix if no summaries found at primary prefix",
    )
    parser.add_argument(
        "--s3-reports-prefix",
        default="",
        help="S3 prefix where per-matrix HTML reports live (for linking)",
    )
    parser.add_argument(
        "--s3-output-uri",
        default="",
        help="S3 URI to upload the consolidated JSON",
    )
    parser.add_argument(
        "--s3-html-output-uri",
        default="",
        help="S3 URI to upload the consolidated HTML report",
    )
    parser.add_argument(
        "--s3-index-uri",
        default="",
        help="S3 URI for the index.json that tracks all available dates (read + write)",
    )
    parser.add_argument(
        "--s3-dashboard-uri",
        default="",
        help="S3 URI to upload the dashboard HTML (e.g., s3://bucket/.../dashboard/index.html)",
    )
    parser.add_argument(
        "--dashboard-dir",
        default="",
        help="Local directory containing dashboard files to upload",
    )
    parser.add_argument(
        "--local-summaries-dir",
        default="",
        help="Local directory with JSON summaries (alternative to S3, for testing)",
    )
    parser.add_argument(
        "--output-dir",
        default="aggregate-output",
        help="Local directory to write output files",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date for this run (YYYY-MM-DD)",
    )
    parser.add_argument("--branch", default="main", help="Branch name")
    parser.add_argument(
        "--github-run-url",
        default="",
        help="URL to the GitHub Actions run",
    )
    parser.add_argument(
        "--workflow-jobs",
        default="",
        help="Path to JSON file with GitHub Actions workflow job statuses",
    )

    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ---- Step 1: Collect summaries ----
    if args.local_summaries_dir:
        summaries = load_local_summaries(args.local_summaries_dir)
    elif args.s3_summaries_prefix:
        download_dir = output_dir / "downloaded_summaries"
        summaries = download_summaries(
            args.s3_summaries_prefix, download_dir, args.s3_summaries_fallback
        )
    else:
        print(
            "ERROR: Provide --s3-summaries-prefix or --local-summaries-dir",
            file=sys.stderr,
        )
        return 1

    if not summaries:
        print(
            "WARNING: No summaries found. Generating empty report.",
            file=sys.stderr,
        )

    print(f"Loaded {len(summaries)} matrix summary file(s)")

    # ---- Step 2: Aggregate ----
    agg = aggregate_summaries(summaries)
    print(
        f"Matrix grid: {len(agg['matrix_grid'])} jobs — "
        f"{sum(1 for g in agg['matrix_grid'] if g['status'] == 'passed')} passed, "
        f"{sum(1 for g in agg['matrix_grid'] if g['status'].startswith('failed'))} failed, "
        f"{sum(1 for g in agg['matrix_grid'] if g['status'] == 'flaky')} flaky"
    )

    # ---- Step 2b: Parse workflow job statuses ----
    workflow_jobs = parse_workflow_jobs(args.workflow_jobs)
    if workflow_jobs:
        failed_wf = [j for j in workflow_jobs if j["conclusion"] == "failure"]
        print(
            f"Workflow jobs: {len(workflow_jobs)} total, "
            f"{len(failed_wf)} failed"
        )

    # ---- Step 3: Generate outputs ----
    consolidated = generate_consolidated_json(
        agg,
        args.date,
        args.branch,
        args.github_run_url,
        workflow_jobs,
    )

    json_path = output_dir / "consolidated_summary.json"
    json_path.write_text(json.dumps(consolidated, indent=2) + "\n")
    print(f"Consolidated JSON written to {json_path}")

    html_report = generate_consolidated_html(
        agg,
        args.date,
        args.branch,
        args.github_run_url,
        args.s3_reports_prefix,
    )
    html_path = output_dir / "consolidated_report.html"
    html_path.write_text(html_report)
    print(f"Consolidated HTML written to {html_path}")

    # ---- Step 4: Upload to S3 ----
    if args.s3_output_uri:
        s3_upload(str(json_path), args.s3_output_uri)
    if args.s3_html_output_uri:
        s3_upload(str(html_path), args.s3_html_output_uri)

    # ---- Step 5: Update index.json ----
    if args.s3_index_uri:
        update_index(
            args.s3_index_uri,
            args.date,
            consolidated,
            output_dir,
        )

    # ---- Step 6: Upload dashboard (self-contained with embedded data) ----
    if args.s3_dashboard_uri and args.dashboard_dir:
        dashboard_file = Path(args.dashboard_dir) / "index.html"
        if dashboard_file.exists():
            # Read the index.json we just uploaded/created
            index_path = output_dir / "index.json"
            index_data = {}
            if index_path.exists():
                with open(index_path) as f:
                    index_data = json.load(f)

            # Inject data into dashboard HTML so it works without S3 fetches
            dashboard_html = dashboard_file.read_text()
            # Escape </ sequences to prevent premature </script> closing
            # when test names or error messages contain HTML-like content
            safe_index = json.dumps(index_data).replace("</", r"<\/")
            safe_consolidated = json.dumps(consolidated).replace("</", r"<\/")
            inject_script = (
                "<script>\n"
                "// Embedded data — injected by aggregate_nightly.py\n"
                f"window.__EMBEDDED_INDEX__ = {safe_index};\n"
                f"window.__EMBEDDED_CONSOLIDATED__ = {safe_consolidated};\n"
                "</script>\n"
            )
            # Insert before </head>
            dashboard_html = dashboard_html.replace(
                "</head>", inject_script + "</head>"
            )

            embedded_path = output_dir / "dashboard.html"
            embedded_path.write_text(dashboard_html)
            s3_upload(str(embedded_path), args.s3_dashboard_uri)
            print("Dashboard uploaded with embedded data")
        else:
            print(
                f"WARNING: Dashboard not found at {dashboard_file}",
                file=sys.stderr,
            )

    return 0


if __name__ == "__main__":
    sys.exit(main())
