# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Redaction shared by the nightly triage scripts."""

import os
import re

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


def redact(text: str) -> str:
    for secret in os.environ.get("TRIAGE_REDACT", "").split():
        text = text.replace(secret, "<redacted>")
    for pattern, replacement in REDACTIONS:
        text = pattern.sub(replacement, text)
    return text
