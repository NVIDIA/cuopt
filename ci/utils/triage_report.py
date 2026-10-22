#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the triage agent's findings and write a draft report.

Fails if the agent output is malformed or leaves a cluster unaccounted for.

Usage: triage_report.py <triage_input.json> <triage.json>
"""

import json
import os
import re
import sys

from triage_input import redact

CATEGORIES = {"product-bug", "infra", "dataset", "flaky", "unknown"}
ACTIONS = {"create", "comment", "ignore"}
LIST_FIELDS = ("merged_signatures", "suspect_files")
TEXT_FIELDS = ("signature", "title", "summary", "hypothesis")


def validate(finding, known, seen):
    errors = []
    if not isinstance(finding, dict):
        return ["finding is not an object"]
    for field in TEXT_FIELDS:
        value = finding.get(field)
        if not isinstance(value, str) or not value.strip():
            errors.append(f"{field} missing or empty")
    for field in LIST_FIELDS:
        if not isinstance(finding.get(field), list):
            errors.append(f"{field} missing or not a list")
    if finding.get("category") not in CATEGORIES:
        errors.append(f"bad category {finding.get('category')!r}")
    if finding.get("action") not in ACTIONS:
        errors.append(f"bad action {finding.get('action')!r}")
    if errors:
        return errors
    for sig in [finding["signature"], *finding["merged_signatures"]]:
        if sig not in known:
            errors.append(f"unknown signature {sig!r}")
        elif sig in seen:
            errors.append(f"signature {sig!r} used twice")
        seen.add(sig)
    return errors


def render(finding, clusters):
    def clean(value, limit):
        return redact(value)[:limit]

    lines = [
        f"- **{finding['category']}** / {finding['action']}: "
        f"{clean(finding['title'], 120)}",
        "",
        "<details><summary>Details</summary>",
        "",
        clean(finding["summary"], 1500),
        "",
        f"**Likely cause:** {clean(finding['hypothesis'], 1500)}",
        "",
    ]
    for c in clusters:
        fence = "`" * (
            max(map(len, re.findall(r"`+", c["error"])), default=2) + 1
        )
        lines += [
            f"**Failed in:** {', '.join(c['matrices']) or '-'}",
            f"{fence}\n{c['error']}\n{fence}",
        ]
    files = [clean(str(p), 200) for p in finding["suspect_files"][:10]]
    lines += [f"**Files:** {', '.join(files) or '-'}", "", "</details>"]
    return "\n".join(lines)


def main():
    with open(sys.argv[1]) as f:
        data = json.load(f)
    clusters = {c["signature"]: c for c in data["clusters"]}
    with open(sys.argv[2]) as f:
        findings = json.load(f)["findings"]

    seen = set()
    errors = []
    for i, finding in enumerate(findings):
        errors += [
            f"finding {i}: {e}" for e in validate(finding, clusters, seen)
        ]
    errors += [
        f"cluster {s} has no finding" for s in sorted(set(clusters) - seen)
    ]
    if errors:
        sys.exit(redact("Invalid triage output:\n" + "\n".join(errors))[:2000])

    header = ["## Nightly triage (draft, nothing filed)", ""]
    if data["omitted"]:
        header += [
            f"**{data['omitted']} smaller failure clusters were not analysed.**",
            "",
        ]
    report = "\n".join(
        header
        + [
            render(
                f,
                [
                    clusters[s]
                    for s in [f["signature"], *f["merged_signatures"]]
                ],
            )
            for f in findings
        ]
    )
    with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
        f.write(report + "\n")


if __name__ == "__main__":
    main()
