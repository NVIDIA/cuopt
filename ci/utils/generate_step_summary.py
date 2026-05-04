# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Generate a GitHub Step Summary (Markdown) from a consolidated nightly summary JSON.

Prints Markdown to stdout suitable for appending to $GITHUB_STEP_SUMMARY.

Usage:
    python3 generate_step_summary.py <consolidated_summary.json>
"""

import json
import sys


def main():
    with open(sys.argv[1]) as f:
        d = json.load(f)

    totals = d.get("test_totals", {})
    grid = d.get("matrix_grid", [])
    new_f = d.get("new_failures", [])
    recur = d.get("recurring_failures", [])
    flaky = d.get("flaky_tests", [])
    resolved = d.get("resolved_tests", [])

    print(
        "# Nightly Test Summary \u2014 %s \u2014 %s"
        % (d.get("branch", ""), d.get("date", ""))
    )
    print()
    print("| Metric | Count |")
    print("|--------|-------|")
    print("| Total | %d |" % totals.get("total", 0))
    print("| Passed | %d |" % totals.get("passed", 0))
    print("| **Failed** | **%d** |" % totals.get("failed", 0))
    print("| Flaky | %d |" % totals.get("flaky", 0))
    print("| Skipped | %d |" % totals.get("skipped", 0))
    print("| Stabilized | %d |" % totals.get("resolved", 0))
    print()
    if new_f:
        print("## New Failures")
        print("| Test Type | Matrix | Test | Error |")
        print("|-----------|--------|------|-------|")
        for e in new_f[:20]:
            msg = (
                (e.get("message", "")[:80])
                .replace("\n", " ")
                .replace("|", "\\|")
            )
            print(
                "| %s | %s | `%s` | %s |"
                % (
                    e.get("test_type", ""),
                    e.get("matrix_label", ""),
                    e["name"],
                    msg,
                )
            )
        print()
    if flaky:
        print("## Flaky Tests")
        print("| Test Type | Matrix | Test | Retries |")
        print("|-----------|--------|------|---------|")
        for e in flaky[:20]:
            print(
                "| %s | %s | `%s` | %s |"
                % (
                    e.get("test_type", ""),
                    e.get("matrix_label", ""),
                    e["name"],
                    e.get("retry_count", "?"),
                )
            )
        print()
    if recur:
        print("## Recurring Failures")
        print("| Test Type | Matrix | Test | Since |")
        print("|-----------|--------|------|-------|")
        for e in recur[:20]:
            print(
                "| %s | %s | `%s` | %s |"
                % (
                    e.get("test_type", ""),
                    e.get("matrix_label", ""),
                    e["name"],
                    e.get("first_seen", "?"),
                )
            )
        print()
    if resolved:
        print("## Stabilized Tests")
        for e in resolved[:10]:
            print(
                "- `%s` (%s) \u2014 was failing %sx"
                % (
                    e["name"],
                    e.get("matrix_label", ""),
                    e.get("failure_count", "?"),
                )
            )
        print()
    print("## Matrix Overview")
    print("| Test Type | Matrix | Status | Passed | Failed | Flaky |")
    print("|-----------|--------|--------|--------|--------|-------|")
    for g in grid:
        c = g.get("counts", {})
        print(
            "| %s | %s | %s | %d | %d | %d |"
            % (
                g["test_type"],
                g["matrix_label"],
                g["status"],
                c.get("passed", 0),
                c.get("failed", 0),
                c.get("flaky", 0),
            )
        )


if __name__ == "__main__":
    main()
