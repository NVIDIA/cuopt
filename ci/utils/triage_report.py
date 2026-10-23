#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Validate the triage agent's findings and write a draft report.

Fails if the agent output is invalid (see triage_validate.py).

Usage: triage_report.py <triage_input.json> <triage.json>
"""

import html
import json
import os
import re
import sys
from typing import Any

from triage_common import redact
from triage_validate import validate

MARKDOWN_SPECIAL = re.compile(r"([\\`*_\[\]#~|])")


def clean(value: str, limit: int, one_line: bool = False) -> str:
    text = redact(value)[:limit]
    if one_line:
        text = " ".join(text.split())
    return MARKDOWN_SPECIAL.sub(r"\\\1", html.escape(text, quote=False))


def render(finding: dict[str, Any], clusters: list[dict[str, Any]]) -> str:
    lines = [
        f"- **{finding['category']}** / {finding['action']}: "
        f"{clean(finding['title'], 120, True)}",
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
            f"**Failed in:** {clean(', '.join(c['matrices']), 600, True) or '-'}",
            f"{fence}\n{c['error']}\n{fence}",
        ]
    files = [clean(str(p), 200, True) for p in finding["suspect_files"][:10]]
    lines += [f"**Files:** {', '.join(files) or '-'}", "", "</details>"]
    return "\n".join(lines)


def main() -> None:
    with open(sys.argv[1]) as f:
        data = json.load(f)
    clusters = {c["signature"]: c for c in data["clusters"]}
    with open(sys.argv[2]) as f:
        doc = json.load(f)
    errors = validate(set(clusters), doc)
    if errors:
        sys.exit(redact("Invalid triage output:\n" + "\n".join(errors))[:2000])
    findings = doc["findings"]

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
