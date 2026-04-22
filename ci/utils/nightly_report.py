#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Nightly test report generator for cuOpt CI.

Parses JUnit XML test results, classifies failures as flaky vs genuine,
maintains a failure history database on S3, and outputs:
  - HTML report (detailed, uploaded to S3 and linked from Slack)
  - Markdown summary (for $GITHUB_STEP_SUMMARY or terminal)
  - JSON summary (for downstream consumers like Slack notifier and dashboard)

Each CI matrix job (CUDA version x Python version x architecture) runs this
script independently.  The --test-type and --matrix-label flags identify the
job so that history and summaries are stored per-matrix-combo.

History lifecycle:
  1. Download history from S3 (falls back to empty if not found)
  2. Classify this run's results
  3. Update history: mark new failures, bump recurring counts, resolve stabilized tests
  4. Upload updated history back to S3
  5. Generate reports (HTML, Markdown, JSON, GitHub Step Summary)
  6. Upload per-run JSON snapshot to S3 summaries dir (for aggregation)

Usage:
  python ci/utils/nightly_report.py \\
      --results-dir test-results/ \\
      --output-dir report-output/ \\
      --sha abc123 \\
      --test-type python \\
      --matrix-label cuda12.9-py3.12-x86_64 \\
      --s3-history-uri s3://bucket/ci_test_reports/nightly/history/python-main-cuda12.9-py3.12-x86_64.json \\
      --s3-summary-uri s3://bucket/ci_test_reports/nightly/summaries/2026-04-13/python-cuda12.9-py3.12-x86_64.json
"""

import argparse
import json
import os
import sys
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from xml.etree import ElementTree

# Ensure ci/utils is importable when invoked as a script
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from s3_helpers import s3_download, s3_upload  # noqa: E402

EMPTY_HISTORY = {"_schema_version": 2, "tests": {}}

# A test that resolves then fails again within this window is considered
# "bouncing" (intermittently flaky) rather than a new failure.
BOUNCE_WINDOW_DAYS = int(os.environ.get("CUOPT_BOUNCE_WINDOW_DAYS", 14))

# Number of failure/resolve cycles that classify a test as cross-run flaky.
BOUNCE_THRESHOLD = int(os.environ.get("CUOPT_BOUNCE_THRESHOLD", 2))


# ---------------------------------------------------------------------------
# JUnit XML parsing
# ---------------------------------------------------------------------------


def parse_junit_xml(xml_path):
    """Parse a JUnit XML file and return a list of test result dicts."""
    results = []
    try:
        tree = ElementTree.parse(xml_path)
    except ElementTree.ParseError as e:
        print(f"WARNING: Failed to parse {xml_path}: {e}", file=sys.stderr)
        return results

    root = tree.getroot()

    if root.tag == "testsuites":
        suites = root.findall("testsuite")
    elif root.tag == "testsuite":
        suites = [root]
    else:
        return results

    for suite in suites:
        suite_name = suite.get("name", os.path.basename(xml_path))
        for testcase in suite.findall("testcase"):
            name = testcase.get("name", "unknown")
            classname = testcase.get("classname", "")
            time_taken = testcase.get("time", "0")

            failure = testcase.find("failure")
            error = testcase.find("error")
            skipped = testcase.find("skipped")

            if skipped is not None:
                status = "skipped"
                message = skipped.get("message", "")
            elif failure is not None:
                status = "failed"
                message = failure.get("message", "")
                if failure.text:
                    message = failure.text.strip()
            elif error is not None:
                status = "error"
                message = error.get("message", "")
                if error.text:
                    message = error.text.strip()
            else:
                status = "passed"
                message = ""

            results.append(
                {
                    "suite": suite_name,
                    "classname": classname,
                    "name": name,
                    "status": status,
                    "time": time_taken,
                    "message": message,
                    "source_file": str(xml_path),
                }
            )

    return results


def collect_all_results(results_dir):
    """Collect test results from all JUnit XML files in a directory."""
    results_dir = Path(results_dir)
    all_results = []
    for xml_file in sorted(results_dir.rglob("*.xml")):
        all_results.extend(parse_junit_xml(xml_file))
    return all_results


# ---------------------------------------------------------------------------
# Classification
# ---------------------------------------------------------------------------


def classify_failures(results):
    """
    Classify test results into passed, failed, flaky, skipped, and error.

    pytest-rerunfailures records reruns as additional <testcase> entries.
    A test that failed then passed on rerun is flaky.
    """
    test_groups = defaultdict(list)
    for r in results:
        # Group by classname+name (not suite) so rerun entries from
        # supplementary XML files match the main XML entries
        key = f"{r['classname']}::{r['name']}"
        test_groups[key].append(r)

    classified = {
        "passed": [],
        "failed": [],
        "flaky": [],
        "skipped": [],
        "error": [],
    }

    for key, entries in test_groups.items():
        statuses = [e["status"] for e in entries]

        if all(s == "skipped" for s in statuses):
            classified["skipped"].append(entries[0])
        elif any(s == "passed" for s in statuses):
            if any(s in ("failed", "error") for s in statuses):
                entry = entries[-1].copy()
                entry["status"] = "flaky"
                entry["retry_count"] = sum(
                    1 for s in statuses if s in ("failed", "error")
                )
                # Capture the error message from the failed attempt
                # (entries[-1] is the passing entry with no message)
                failed = [
                    e for e in entries if e["status"] in ("failed", "error")
                ]
                if failed:
                    entry["message"] = failed[-1].get("message", "")
                classified["flaky"].append(entry)
            else:
                classified["passed"].append(entries[-1])
        elif any(s == "error" for s in statuses):
            classified["error"].append(entries[-1])
        else:
            classified["failed"].append(entries[-1])

    return classified


# ---------------------------------------------------------------------------
# History management
# ---------------------------------------------------------------------------


def load_history(history_path):
    """Load failure history from a local JSON file."""
    try:
        with open(history_path) as f:
            data = json.load(f)
            if "tests" in data:
                return data
    except (FileNotFoundError, json.JSONDecodeError):
        pass
    return dict(EMPTY_HISTORY)


def _days_between(date_a, date_b):
    """Return absolute number of days between two YYYY-MM-DD strings."""
    try:
        a = datetime.strptime(date_a, "%Y-%m-%d")
        b = datetime.strptime(date_b, "%Y-%m-%d")
        return abs((a - b).days)
    except (ValueError, TypeError):
        return 999


def _is_recent_resolve(rec, date_str):
    """Check if a test was resolved recently (within bounce window)."""
    resolved_date = rec.get("resolved_date", "")
    if not resolved_date:
        return False
    return _days_between(resolved_date, date_str) <= BOUNCE_WINDOW_DAYS


def update_history(history, classified, sha, date_str):
    """
    Update failure history with this run's results.

    Returns (history, new_failures, recurring_failures, resolved_tests,
    new_flaky_tests).

    Classification logic:
      - "new failure": never seen before (no history entry at all)
      - "recurring": was already active (failing on previous runs)
      - "bouncing": was resolved recently but failed again — reactivated
        as recurring (not new), and marked cross-run flaky after 2+ bounces
      - "resolved": was active, now passes — notified once, then silent
        on subsequent passes
    """
    tests = history.setdefault("tests", {})
    new_failures = []
    recurring_failures = []
    resolved_tests = []
    new_flaky_tests = []

    # --- Genuine failures ---
    for entry in classified["failed"] + classified["error"]:
        test_key = f"{entry['suite']}::{entry['classname']}::{entry['name']}"

        if test_key in tests:
            rec = tests[test_key]

            if rec["status"] == "active":
                # Still failing — bump count
                rec["last_seen_date"] = date_str
                rec["last_seen_sha"] = sha
                rec["failure_count"] += 1
                recurring_failures.append(
                    {**entry, "first_seen": rec["first_seen_date"]}
                )
            elif rec["status"] == "resolved" and _is_recent_resolve(
                rec, date_str
            ):
                # Bouncing: resolved recently but failed again.
                # Reactivate as recurring, not new. Track the bounce.
                rec["status"] = "active"
                rec["last_seen_date"] = date_str
                rec["last_seen_sha"] = sha
                rec["failure_count"] += 1
                rec["bounce_count"] = rec.get("bounce_count", 0) + 1
                if rec["bounce_count"] >= BOUNCE_THRESHOLD:
                    rec["is_flaky"] = True
                recurring_failures.append(
                    {
                        **entry,
                        "first_seen": rec["first_seen_date"],
                        "is_bouncing": True,
                    }
                )
            else:
                # Resolved long ago — treat as new cycle but keep history
                rec["status"] = "active"
                rec["last_seen_date"] = date_str
                rec["last_seen_sha"] = sha
                rec["failure_count"] += 1
                rec["bounce_count"] = rec.get("bounce_count", 0) + 1
                new_failures.append(entry)
        else:
            # Truly new — never seen before
            tests[test_key] = {
                "suite": entry["suite"],
                "classname": entry["classname"],
                "name": entry["name"],
                "first_seen_date": date_str,
                "first_seen_sha": sha,
                "last_seen_date": date_str,
                "last_seen_sha": sha,
                "failure_count": 1,
                "is_flaky": False,
                "bounce_count": 0,
                "status": "active",
            }
            new_failures.append(entry)

    # --- Flaky tests (passed on retry within this run) ---
    for entry in classified["flaky"]:
        test_key = f"{entry['suite']}::{entry['classname']}::{entry['name']}"
        if test_key in tests:
            rec = tests[test_key]
            rec["last_seen_date"] = date_str
            rec["last_seen_sha"] = sha
            rec["failure_count"] += 1
            rec["is_flaky"] = True
            # If it was resolved, reactivate — it's still unstable
            if rec["status"] == "resolved":
                rec["status"] = "active"
                rec["bounce_count"] = rec.get("bounce_count", 0) + 1
        else:
            tests[test_key] = {
                "suite": entry["suite"],
                "classname": entry["classname"],
                "name": entry["name"],
                "first_seen_date": date_str,
                "first_seen_sha": sha,
                "last_seen_date": date_str,
                "last_seen_sha": sha,
                "failure_count": 1,
                "is_flaky": True,
                "bounce_count": 0,
                "status": "active",
            }
            new_flaky_tests.append(entry)

    # --- Resolve stabilized tests ---
    passed_keys = set()
    for entry in classified["passed"]:
        test_key = f"{entry['suite']}::{entry['classname']}::{entry['name']}"
        passed_keys.add(test_key)

    for test_key in passed_keys:
        if test_key in tests and tests[test_key]["status"] == "active":
            rec = tests[test_key]
            rec["status"] = "resolved"
            rec["resolved_date"] = date_str
            rec["resolved_sha"] = sha
            resolved_tests.append(
                {
                    "suite": rec["suite"],
                    "classname": rec["classname"],
                    "name": rec["name"],
                    "first_seen": rec["first_seen_date"],
                    "failure_count": rec["failure_count"],
                    "bounce_count": rec.get("bounce_count", 0),
                    "was_flaky": rec.get("is_flaky", False),
                }
            )
        # If already "resolved" and passes again — no notification.
        # The resolved notification was sent once when it first stabilized.

    return (
        history,
        new_failures,
        recurring_failures,
        resolved_tests,
        new_flaky_tests,
    )


def save_history(history, history_path):
    """Write history to a local JSON file."""
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2, sort_keys=True)
        f.write("\n")


# ---------------------------------------------------------------------------
# Report generation
# ---------------------------------------------------------------------------


def generate_markdown_report(
    classified,
    new_failures,
    recurring_failures,
    resolved_tests,
    history,
    test_type="",
    matrix_label="",
    sha="",
    date_str="",
):
    """Generate a Markdown summary report."""
    lines = []
    title = "# Nightly Test Report"
    if test_type:
        title += f" — {test_type}"
    if matrix_label:
        title += f" [{matrix_label}]"
    lines.append(title)
    lines.append("")
    if date_str or sha:
        meta_parts = []
        if date_str:
            meta_parts.append(f"**Date:** {date_str}")
        if sha:
            meta_parts.append(f"**Commit:** `{sha[:12]}`")
        if matrix_label:
            meta_parts.append(f"**Matrix:** {matrix_label}")
        lines.append(" | ".join(meta_parts))
        lines.append("")

    total_passed = len(classified["passed"])
    total_failed = len(classified["failed"]) + len(classified["error"])
    total_flaky = len(classified["flaky"])
    total_skipped = len(classified["skipped"])
    total = total_passed + total_failed + total_flaky + total_skipped

    lines.append("## Summary")
    lines.append("")
    lines.append("| Metric | Count |")
    lines.append("|--------|-------|")
    lines.append(f"| Total tests | {total} |")
    lines.append(f"| Passed | {total_passed} |")
    lines.append(f"| **Genuine failures** | **{total_failed}** |")
    lines.append(f"| Flaky (passed on retry) | {total_flaky} |")
    lines.append(f"| Skipped | {total_skipped} |")
    if resolved_tests:
        lines.append(
            f"| **Stabilized (were failing, now pass)** | **{len(resolved_tests)}** |"
        )
    lines.append("")

    # -- New genuine failures (highest priority) --
    if new_failures:
        lines.append("## NEW Failures (not previously seen)")
        lines.append("")
        lines.append("| Suite | Test | Error |")
        lines.append("|-------|------|-------|")
        for entry in new_failures:
            short_msg = (
                entry.get("message", "")[:80]
                .replace("\n", " ")
                .replace("|", "\\|")
            )
            lines.append(
                f"| {entry['suite']} | `{entry['name']}` | {short_msg} |"
            )
        lines.append("")

    # -- Recurring failures --
    if recurring_failures:
        lines.append("## Recurring Failures")
        lines.append("")
        lines.append("| Suite | Test | First seen | Failure count | Error |")
        lines.append("|-------|------|------------|---------------|-------|")
        for entry in recurring_failures:
            short_msg = (
                entry.get("message", "")[:60]
                .replace("\n", " ")
                .replace("|", "\\|")
            )
            first_seen = entry.get("first_seen", "unknown")
            test_key = (
                f"{entry['suite']}::{entry['classname']}::{entry['name']}"
            )
            count = (
                history.get("tests", {})
                .get(test_key, {})
                .get("failure_count", "?")
            )
            lines.append(
                f"| {entry['suite']} | `{entry['name']}` | {first_seen} | {count} | {short_msg} |"
            )
        lines.append("")

    # -- Stabilized tests --
    if resolved_tests:
        lines.append("## Stabilized Tests (were failing, now passing)")
        lines.append("")
        lines.append(
            "| Suite | Test | Was failing since | Total failure count | Was flaky? |"
        )
        lines.append(
            "|-------|------|-------------------|---------------------|------------|"
        )
        for entry in resolved_tests:
            flaky_badge = "Yes" if entry.get("was_flaky") else "No"
            lines.append(
                f"| {entry['suite']} | `{entry['name']}` | {entry['first_seen']} "
                f"| {entry['failure_count']} | {flaky_badge} |"
            )
        lines.append("")

    # -- Flaky tests --
    if classified["flaky"]:
        lines.append("## Flaky Tests (passed on retry)")
        lines.append("")
        lines.append("| Suite | Test | Retries needed | Error |")
        lines.append("|-------|------|----------------|-------|")
        for entry in classified["flaky"]:
            retry_count = entry.get("retry_count", "?")
            short_msg = (
                entry.get("message", "")[:80]
                .replace("\n", " ")
                .replace("|", "\\|")
            )
            lines.append(
                f"| {entry['suite']} | `{entry['name']}` | {retry_count} | {short_msg} |"
            )
        lines.append("")

    # -- Detailed errors --
    all_failures = classified["failed"] + classified["error"]
    if all_failures:
        lines.append("## All Failure Details")
        lines.append("")
        for entry in all_failures:
            lines.append(f"### `{entry['classname']}::{entry['name']}`")
            lines.append(f"- **Suite**: {entry['suite']}")
            lines.append(f"- **Source**: {entry['source_file']}")
            msg = entry.get("message", "").strip()
            if msg:
                lines.append("- **Error**:")
                lines.append("```")
                for line in msg.split("\n")[:20]:
                    lines.append(line)
                lines.append("```")
            lines.append("")

    if not all_failures and not classified["flaky"] and not resolved_tests:
        lines.append("All tests passed! No failures or flaky tests detected.")
        lines.append("")

    return "\n".join(lines)


def generate_json_summary(
    classified,
    new_failures,
    recurring_failures,
    resolved_tests,
    new_flaky_tests=None,
    test_type="",
    matrix_label="",
    sha="",
    date_str="",
):
    """Generate a JSON summary for downstream tools (Slack notifier, dashboard)."""
    if new_flaky_tests is None:
        new_flaky_tests = []
    new_flaky_keys = {
        f"{e['classname']}::{e['name']}" for e in new_flaky_tests
    }
    return {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "test_type": test_type,
        "matrix_label": matrix_label,
        "sha": sha,
        "date": date_str,
        "counts": {
            "total": sum(len(v) for v in classified.values()),
            "passed": len(classified["passed"]),
            "failed": len(classified["failed"]) + len(classified["error"]),
            "flaky": len(classified["flaky"]),
            "skipped": len(classified["skipped"]),
            "resolved": len(resolved_tests),
        },
        "has_new_failures": len(new_failures) > 0,
        "has_new_flaky": len(new_flaky_tests) > 0,
        "new_failures": [
            {
                "suite": e["suite"],
                "name": e["name"],
                "classname": e["classname"],
                "message": e.get("message", ""),
            }
            for e in new_failures
        ],
        "recurring_failures": [
            {
                "suite": e["suite"],
                "name": e["name"],
                "classname": e["classname"],
                "first_seen": e.get("first_seen", "unknown"),
                "message": e.get("message", ""),
            }
            for e in recurring_failures
        ],
        "flaky_tests": [
            {
                "suite": e["suite"],
                "name": e["name"],
                "classname": e["classname"],
                "retry_count": e.get("retry_count", 0),
                "message": e.get("message", ""),
                "is_new": f"{e['classname']}::{e['name']}" in new_flaky_keys,
            }
            for e in classified["flaky"]
        ],
        "resolved_tests": [
            {
                "suite": e["suite"],
                "name": e["name"],
                "classname": e["classname"],
                "first_seen": e.get("first_seen", "unknown"),
                "failure_count": e.get("failure_count", 0),
                "was_flaky": e.get("was_flaky", False),
            }
            for e in resolved_tests
        ],
    }


# ---------------------------------------------------------------------------
# HTML report
# ---------------------------------------------------------------------------


def _html_escape(text):
    """Escape HTML special characters."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def generate_html_report(
    classified,
    new_failures,
    recurring_failures,
    resolved_tests,
    history,
    test_type="",
    matrix_label="",
    sha="",
    date_str="",
):
    """Generate a self-contained HTML report with detailed failure info."""
    total_passed = len(classified["passed"])
    total_failed = len(classified["failed"]) + len(classified["error"])
    total_flaky = len(classified["flaky"])
    total_skipped = len(classified["skipped"])
    total = total_passed + total_failed + total_flaky + total_skipped

    title = "Nightly Test Report"
    if test_type:
        title += f" &mdash; {_html_escape(test_type)}"
    if matrix_label:
        title += f" [{_html_escape(matrix_label)}]"

    # Determine overall status color
    if total_failed > 0:
        status_color = "#d32f2f"
        status_text = f"{total_failed} failure(s)"
    elif total_flaky > 0:
        status_color = "#f9a825"
        status_text = "All passed (flaky detected)"
    else:
        status_color = "#388e3c"
        status_text = "All passed"

    parts = []
    parts.append(f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>
  :root {{ --fail: #d32f2f; --pass: #388e3c; --flaky: #f9a825; --skip: #757575;
           --bg: #fafafa; --card: #fff; --border: #e0e0e0; --text: #212121; }}
  * {{ margin: 0; padding: 0; box-sizing: border-box; }}
  body {{ font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto,
          Helvetica, Arial, sans-serif; background: var(--bg); color: var(--text);
          padding: 24px; max-width: 1200px; margin: 0 auto; }}
  h1 {{ font-size: 1.5rem; margin-bottom: 4px; }}
  .meta {{ color: #616161; font-size: 0.85rem; margin-bottom: 16px; }}
  .meta code {{ background: #eeeeee; padding: 2px 6px; border-radius: 3px; font-size: 0.8rem; }}
  .status-bar {{ padding: 12px 16px; border-radius: 8px; color: #fff;
                 font-weight: 600; margin-bottom: 20px; font-size: 1.1rem; }}
  .summary-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(140px, 1fr));
                   gap: 12px; margin-bottom: 24px; }}
  .summary-card {{ background: var(--card); border: 1px solid var(--border);
                   border-radius: 8px; padding: 16px; text-align: center; }}
  .summary-card .num {{ font-size: 2rem; font-weight: 700; }}
  .summary-card .lbl {{ font-size: 0.8rem; color: #757575; text-transform: uppercase; }}
  .num.pass {{ color: var(--pass); }}  .num.fail {{ color: var(--fail); }}
  .num.flaky {{ color: var(--flaky); }}  .num.skip {{ color: var(--skip); }}
  section {{ margin-bottom: 24px; }}
  h2 {{ font-size: 1.15rem; margin-bottom: 10px; padding-bottom: 4px;
        border-bottom: 2px solid var(--border); }}
  table {{ width: 100%; border-collapse: collapse; font-size: 0.85rem; }}
  th {{ background: #f5f5f5; text-align: left; padding: 8px 10px; font-weight: 600; }}
  td {{ padding: 8px 10px; border-bottom: 1px solid var(--border); vertical-align: top; }}
  tr:hover td {{ background: #f5f5f5; }}
  .badge {{ display: inline-block; padding: 2px 8px; border-radius: 4px;
            font-size: 0.75rem; font-weight: 600; color: #fff; }}
  .badge-new {{ background: var(--fail); }}
  .badge-recurring {{ background: #e65100; }}
  .badge-flaky {{ background: var(--flaky); color: #212121; }}
  .badge-resolved {{ background: var(--pass); }}
  details {{ margin-top: 4px; }}
  details summary {{ cursor: pointer; color: #1565c0; font-size: 0.8rem; }}
  pre.error {{ background: #263238; color: #e0e0e0; padding: 12px; border-radius: 6px;
               font-size: 0.78rem; overflow-x: auto; white-space: pre-wrap;
               word-break: break-word; max-height: 300px; margin-top: 6px; }}
  .empty {{ color: #9e9e9e; font-style: italic; padding: 16px; }}
</style>
</head>
<body>
<h1>{title}</h1>
<div class="meta">""")

    meta_parts = []
    if date_str:
        meta_parts.append(f"Date: <strong>{_html_escape(date_str)}</strong>")
    if sha:
        meta_parts.append(f"Commit: <code>{_html_escape(sha[:12])}</code>")
    if matrix_label:
        meta_parts.append(
            f"Matrix: <strong>{_html_escape(matrix_label)}</strong>"
        )
    parts.append(" &nbsp;|&nbsp; ".join(meta_parts))

    parts.append(f"""</div>
<div class="status-bar" style="background:{status_color}">{status_text}</div>
<div class="summary-grid">
  <div class="summary-card"><div class="num">{total}</div><div class="lbl">Total</div></div>
  <div class="summary-card"><div class="num pass">{total_passed}</div><div class="lbl">Passed</div></div>
  <div class="summary-card"><div class="num fail">{total_failed}</div><div class="lbl">Failed</div></div>
  <div class="summary-card"><div class="num flaky">{total_flaky}</div><div class="lbl">Flaky</div></div>
  <div class="summary-card"><div class="num skip">{total_skipped}</div><div class="lbl">Skipped</div></div>
  <div class="summary-card"><div class="num pass">{len(resolved_tests)}</div><div class="lbl">Stabilized</div></div>
</div>""")

    # --- New failures ---
    if new_failures:
        parts.append("<section><h2>New Failures</h2><table>")
        parts.append("<tr><th>Suite</th><th>Test</th><th>Error</th></tr>")
        for e in new_failures:
            msg = _html_escape(e.get("message", ""))
            short = _html_escape(e.get("message", "")[:100])
            parts.append(
                f"<tr><td>{_html_escape(e['suite'])}</td>"
                f"<td><code>{_html_escape(e['name'])}</code> "
                f'<span class="badge badge-new">NEW</span></td>'
                f"<td><details><summary>{short}</summary>"
                f'<pre class="error">{msg}</pre></details></td></tr>'
            )
        parts.append("</table></section>")

    # --- Recurring failures ---
    if recurring_failures:
        parts.append("<section><h2>Recurring Failures</h2><table>")
        parts.append(
            "<tr><th>Suite</th><th>Test</th><th>First Seen</th>"
            "<th>Count</th><th>Error</th></tr>"
        )
        for e in recurring_failures:
            msg = _html_escape(e.get("message", ""))
            short = _html_escape(e.get("message", "")[:100])
            first_seen = _html_escape(e.get("first_seen", "unknown"))
            test_key = f"{e['suite']}::{e['classname']}::{e['name']}"
            count = (
                history.get("tests", {})
                .get(test_key, {})
                .get("failure_count", "?")
            )
            parts.append(
                f"<tr><td>{_html_escape(e['suite'])}</td>"
                f"<td><code>{_html_escape(e['name'])}</code> "
                f'<span class="badge badge-recurring">RECURRING</span></td>'
                f"<td>{first_seen}</td><td>{count}</td>"
                f"<td><details><summary>{short}</summary>"
                f'<pre class="error">{msg}</pre></details></td></tr>'
            )
        parts.append("</table></section>")

    # --- Stabilized ---
    if resolved_tests:
        parts.append("<section><h2>Stabilized Tests</h2><table>")
        parts.append(
            "<tr><th>Suite</th><th>Test</th><th>Failing Since</th>"
            "<th>Failure Count</th><th>Was Flaky?</th></tr>"
        )
        for e in resolved_tests:
            flaky_tag = "Yes" if e.get("was_flaky") else "No"
            parts.append(
                f"<tr><td>{_html_escape(e['suite'])}</td>"
                f"<td><code>{_html_escape(e['name'])}</code> "
                f'<span class="badge badge-resolved">FIXED</span></td>'
                f"<td>{_html_escape(e.get('first_seen', '?'))}</td>"
                f"<td>{e.get('failure_count', '?')}</td>"
                f"<td>{flaky_tag}</td></tr>"
            )
        parts.append("</table></section>")

    # --- Flaky ---
    if classified["flaky"]:
        parts.append("<section><h2>Flaky Tests (passed on retry)</h2><table>")
        parts.append(
            "<tr><th>Suite</th><th>Test</th><th>Retries</th>"
            "<th>Error</th></tr>"
        )
        for e in classified["flaky"]:
            msg = _html_escape(e.get("message", ""))
            raw_msg = e.get("message", "").strip()
            # Use last non-empty line as the short summary (typically the assertion)
            lines = [ln for ln in raw_msg.splitlines() if ln.strip()]
            short = _html_escape(lines[-1][:150] if lines else "")
            parts.append(
                f"<tr><td>{_html_escape(e['suite'])}</td>"
                f"<td><code>{_html_escape(e['name'])}</code> "
                f'<span class="badge badge-flaky">FLAKY</span></td>'
                f"<td>{e.get('retry_count', '?')}</td>"
                f"<td><details><summary>{short}</summary>"
                f'<pre class="error">{msg}</pre></details></td></tr>'
            )
        parts.append("</table></section>")

    # --- All failure details ---
    all_failures = classified["failed"] + classified["error"]
    if all_failures:
        parts.append("<section><h2>All Failure Details</h2>")
        for e in all_failures:
            msg = _html_escape(e.get("message", "").strip())
            parts.append(
                f'<h3 style="font-size:0.95rem;margin-top:16px">'
                f"<code>{_html_escape(e['classname'])}::{_html_escape(e['name'])}</code></h3>"
                f'<p style="font-size:0.82rem;color:#616161">'
                f"Suite: {_html_escape(e['suite'])} &nbsp;|&nbsp; "
                f"Source: {_html_escape(e['source_file'])}</p>"
            )
            if msg:
                parts.append(f'<pre class="error">{msg}</pre>')
        parts.append("</section>")

    if not all_failures and not classified["flaky"] and not resolved_tests:
        parts.append(
            '<p class="empty">All tests passed! No failures or flaky tests detected.</p>'
        )

    parts.append("</body></html>")
    return "\n".join(parts)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main():
    parser = argparse.ArgumentParser(
        description="Generate nightly test failure report from JUnit XML results"
    )
    parser.add_argument(
        "--results-dir",
        required=True,
        help="Directory containing JUnit XML test result files",
    )
    parser.add_argument(
        "--output-dir",
        default="report-output",
        help="Directory to write report files to",
    )
    parser.add_argument(
        "--sha",
        default=os.environ.get("GITHUB_SHA", "unknown"),
        help="Git commit SHA for this run",
    )
    parser.add_argument(
        "--date",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        help="Date for this run (YYYY-MM-DD)",
    )
    parser.add_argument(
        "--test-type",
        default="unknown",
        help=(
            "Test type identifier (e.g., cpp, python, wheel-python, "
            "wheel-server, notebooks)"
        ),
    )
    parser.add_argument(
        "--matrix-label",
        default="",
        help=(
            "Matrix combination label (e.g., cuda12.9-py3.12-x86_64). "
            "Included in reports and JSON summary to identify the CI job."
        ),
    )
    parser.add_argument(
        "--s3-history-uri",
        default="",
        help=(
            "S3 URI for persistent failure history JSON. "
            "Downloaded before analysis, uploaded after update. "
            "Example: s3://bucket/ci_test_reports/nightly/history/"
            "python-main-cuda12.9-py3.12-x86_64.json"
        ),
    )
    parser.add_argument(
        "--s3-history-seed-uri",
        default="",
        help=(
            "S3 URI to seed history from when this branch has no history yet "
            "(e.g., first nightly on a new release branch). Typically points "
            "to main's history so known failures are inherited, not re-reported "
            "as new. Only used if --s3-history-uri download fails."
        ),
    )
    parser.add_argument(
        "--s3-summary-uri",
        default="",
        help=(
            "S3 URI to upload this run's JSON snapshot for aggregation. "
            "Scoped by run ID to prevent cross-run pollution. "
            "Example: s3://bucket/.../summaries/2026-04-13/run-12345/"
            "python-cuda12.9-py3.12-x86_64.json"
        ),
    )
    parser.add_argument(
        "--s3-summary-branch-uri",
        default="",
        help=(
            "S3 URI to also upload the JSON snapshot under the branch path "
            "for manual browsing. Optional — same content as --s3-summary-uri."
        ),
    )
    parser.add_argument(
        "--s3-html-uri",
        default="",
        help=(
            "S3 URI to upload the HTML report. "
            "Example: s3://bucket/ci_test_reports/nightly/reports/"
            "2026-04-13/python-cuda12.9-py3.12-x86_64.html"
        ),
    )
    parser.add_argument(
        "--github-step-summary",
        default=os.environ.get("GITHUB_STEP_SUMMARY", ""),
        help="Path to write GitHub Actions step summary",
    )

    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    local_history_path = str(output_dir / "test_failure_history.json")

    # ---- Step 1: Download history from S3 ----
    if args.s3_history_uri:
        if not s3_download(args.s3_history_uri, local_history_path):
            # No history for this branch yet — seed from parent (e.g., main)
            # so known failures are inherited and not re-reported as new.
            if args.s3_history_seed_uri and s3_download(
                args.s3_history_seed_uri, local_history_path
            ):
                print(
                    f"Seeded history from {args.s3_history_seed_uri} "
                    f"(first run on this branch)"
                )

    # ---- Step 2: Collect and classify results ----
    print(f"Collecting test results from {args.results_dir} ...")
    results = collect_all_results(args.results_dir)
    if not results:
        print("WARNING: No test results found.", file=sys.stderr)

    print(f"Found {len(results)} test case entries across all XML files")
    classified = classify_failures(results)

    print(
        f"Classification: {len(classified['passed'])} passed, "
        f"{len(classified['failed'])} failed, "
        f"{len(classified['error'])} errors, "
        f"{len(classified['flaky'])} flaky, "
        f"{len(classified['skipped'])} skipped"
    )

    # ---- Step 3: Update history ----
    history = load_history(local_history_path)
    (
        history,
        new_failures,
        recurring_failures,
        resolved_tests,
        new_flaky_tests,
    ) = update_history(history, classified, args.sha, args.date)

    if new_flaky_tests:
        print(
            f"NEW FLAKY: {len(new_flaky_tests)} test(s) flaky for the first time"
        )
    if resolved_tests:
        print(
            f"Stabilized: {len(resolved_tests)} previously-failing test(s) now pass"
        )

    save_history(history, local_history_path)
    print(f"Updated local history at {local_history_path}")

    # ---- Step 4: Upload history back to S3 ----
    if args.s3_history_uri:
        s3_upload(local_history_path, args.s3_history_uri)

    # ---- Step 5: Generate reports ----
    report_kwargs = dict(
        test_type=args.test_type,
        matrix_label=args.matrix_label,
        sha=args.sha,
        date_str=args.date,
    )

    md_report = generate_markdown_report(
        classified,
        new_failures,
        recurring_failures,
        resolved_tests,
        history,
        **report_kwargs,
    )
    md_path = output_dir / "nightly_report.md"
    md_path.write_text(md_report)
    print(f"Markdown report written to {md_path}")

    html_report = generate_html_report(
        classified,
        new_failures,
        recurring_failures,
        resolved_tests,
        history,
        **report_kwargs,
    )
    html_path = output_dir / "nightly_report.html"
    html_path.write_text(html_report)
    print(f"HTML report written to {html_path}")

    json_summary = generate_json_summary(
        classified,
        new_failures,
        recurring_failures,
        resolved_tests,
        new_flaky_tests,
        **report_kwargs,
    )
    json_path = output_dir / "nightly_summary.json"
    json_path.write_text(json.dumps(json_summary, indent=2) + "\n")
    print(f"JSON summary written to {json_path}")

    if args.github_step_summary:
        with open(args.github_step_summary, "a") as f:
            f.write(md_report)
        print(f"Wrote GitHub Step Summary to {args.github_step_summary}")

    # ---- Step 6: Upload per-run snapshot and HTML to S3 ----
    s3_ok = True
    if args.s3_summary_uri:
        if not s3_upload(str(json_path), args.s3_summary_uri):
            print(
                "ERROR: Failed to upload JSON summary to S3. "
                "The nightly aggregate will NOT include this job's results.",
                file=sys.stderr,
            )
            s3_ok = False

    # Also upload to branch-scoped path for manual browsing
    if (
        args.s3_summary_branch_uri
        and args.s3_summary_branch_uri != args.s3_summary_uri
    ):
        if not s3_upload(str(json_path), args.s3_summary_branch_uri):
            # Non-critical — the run-scoped copy is what the aggregate needs
            print(
                "WARNING: Failed to upload branch-scoped JSON summary.",
                file=sys.stderr,
            )

    if args.s3_html_uri:
        if not s3_upload(str(html_path), args.s3_html_uri):
            print(
                "WARNING: Failed to upload HTML report to S3.",
                file=sys.stderr,
            )
            s3_ok = False

    if s3_ok and (args.s3_summary_uri or args.s3_html_uri):
        print("S3 uploads completed successfully.")

    # ---- Exit code ----
    genuine_failures = len(classified["failed"]) + len(classified["error"])
    if genuine_failures > 0:
        print(
            f"\nFAILED: {genuine_failures} genuine test failure(s) detected."
        )
        return 1
    if classified["flaky"]:
        print(
            f"\nWARNING: All tests passed but {len(classified['flaky'])} flaky test(s) detected."
        )
    else:
        print("\nAll tests passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
