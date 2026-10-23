#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check the triage agent's findings against the schema and the clusters.

Usage: triage_validate.py <triage_input.json> <triage.json>
"""

import json
import os
import sys
from typing import Any

from jsonschema import Draft202012Validator
from triage_common import redact

SCHEMA_PATH = os.path.join(os.path.dirname(__file__), "triage_schema.json")


def validate(known: set[str], doc: Any) -> list[str]:
    """Return the problems in doc; every cluster must be in one finding."""
    with open(SCHEMA_PATH) as f:
        schema = json.load(f)
    errors = [
        f"{'/'.join(map(str, e.absolute_path)) or 'root'}: {e.message}"
        for e in Draft202012Validator(schema).iter_errors(doc)
    ]
    if errors:
        return errors
    seen: set[str] = set()
    for i, finding in enumerate(doc["findings"]):
        for sig in [finding["signature"], *finding["merged_signatures"]]:
            if sig not in known:
                errors.append(f"finding {i}: unknown signature {sig!r}")
            elif sig in seen:
                errors.append(f"finding {i}: signature {sig!r} used twice")
            seen.add(sig)
    return errors + [
        f"cluster {s} has no finding" for s in sorted(known - seen)
    ]


def main() -> None:
    with open(sys.argv[1]) as f:
        known = {c["signature"] for c in json.load(f)["clusters"]}
    with open(sys.argv[2]) as f:
        errors = validate(known, json.load(f))
    if errors:
        sys.exit(redact("Invalid triage output:\n" + "\n".join(errors))[:2000])
    print("triage.json is valid")


if __name__ == "__main__":
    main()
