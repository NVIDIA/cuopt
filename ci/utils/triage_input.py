#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Group nightly failures into redacted clusters for the triage agent.

Usage: triage_input.py <consolidated_summary.json> <triage_input.json>
"""

import hashlib
import json
import os
import re
import sys

MAX_CLUSTERS = 40
MAX_TESTS = 20
MAX_ERROR_CHARS = 600

REDACTIONS = [
    (re.compile(r"s3://\S+"), "<s3-uri>"),
    (re.compile(r"https?://\S+", re.I), "<url>"),
    (re.compile(r"\b(?:bearer|basic)\s+[\w.~+/=-]+", re.I), "<credential>"),
    (
        re.compile(r"\b(?:xox[abp]-|ghp_|gho_|ghs_|sk-|AKIA)[\w\-]{8,}"),
        "<token>",
    ),
    (re.compile(r"\bsk-\.{2,3}\w*"), "<token>"),
    (re.compile(r"\bKey=\S+"), "Key=<redacted>"),
    (re.compile(r"/home/[^/\s]+"), "/home/<user>"),
    (re.compile(r"\b\d{12}\b"), "<id>"),
]


def redact(text):
    for secret in os.environ.get("TRIAGE_REDACT", "").split():
        text = text.replace(secret, "<redacted>")
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text


FRAME = re.compile(r"^\s*#\d+ in ")
OUTCOME = re.compile(
    r"Which is|Actual:|Reason=|xception|SIG[A-Z]+|bad_alloc|Segmentation"
    r"|ValidationError|CUOPT_ERROR_TYPE|[A-Za-z]Error\b"
)


def normalize(line, keep_numbers):
    line = re.sub(r"0x[0-9a-fA-F]+", "<hex>", line)
    line = re.sub(r"/\S+", "<path>", line)
    return line if keep_numbers else re.sub(r"\d+(\.\d+)?", "<n>", line)


def signature(message, flaky):
    """Hash the lines that state the outcome, not the test-specific call text.

    Flaky failures never share a cluster with real ones.
    """
    lines = [
        line[:200]
        for line in message.splitlines()
        if line.strip() and not FRAME.match(line)
    ]
    key = [line for line in lines if OUTCOME.search(line)][:3] or lines[:3]
    text = "\n".join(normalize(line, "Which is" in line) for line in key)
    text = f"{'flaky' if flaky else 'failing'}\n{text}"
    return hashlib.sha1(text.encode()).hexdigest()[:12]


def build_clusters(summary):
    clusters = {}

    def add(message, label, matrix, test):
        c = clusters.setdefault(
            signature(message, label == "flaky"),
            {
                "error": redact(message)[:MAX_ERROR_CHARS],
                "labels": set(),
                "matrices": set(),
                "tests": [],
                "failures": 0,
            },
        )
        c["failures"] += 1
        c["labels"].add(label)
        if matrix:
            c["matrices"].add(matrix)
        if test and test not in c["tests"]:
            c["tests"].append(test)

    for label, key in (
        ("new", "new_failures"),
        ("recurring", "recurring_failures"),
        ("flaky", "flaky_tests"),
    ):
        for e in summary[key]:
            test = redact(f"{e['classname']}.{e['name']}".strip("."))
            add(
                e["message"],
                label,
                redact(e["matrix_label"])[:100],
                test[:200],
            )
    for job in summary["untracked_failed_ci_jobs"]:
        add(f"CI job failed: {job['name']}", "ci-job", "", "")

    biggest = sorted(
        clusters.items(), key=lambda kv: kv[1]["failures"], reverse=True
    )
    kept = [
        {
            "signature": sig,
            "error": c["error"],
            "labels": sorted(c["labels"]),
            "matrices": sorted(c["matrices"]),
            "failures": c["failures"],
            "tests": c["tests"][:MAX_TESTS],
        }
        for sig, c in biggest[:MAX_CLUSTERS]
    ]
    return kept, len(biggest) - len(kept)


if __name__ == "__main__":
    with open(sys.argv[1]) as f:
        clusters, omitted = build_clusters(json.load(f))
    with open(sys.argv[2], "w") as f:
        json.dump({"omitted": omitted, "clusters": clusters}, f, indent=2)
    print(f"{len(clusters)} failure clusters ({omitted} omitted)")
