#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
JUnit XML helpers for CI test runner scripts.

Extracts test names from JUnit XML files for crash isolation and retry logic.
Called from shell scripts via: python3 ci/utils/junit_helpers.py <command> <args>

Commands:
  failed  <xml_file> [--sep SEP]   Print failed/errored test names
  passed  <xml_file> [--sep SEP]   Print passed test names (excludes skipped)
  gtest-list                        Parse gtest --gtest_list_tests from stdin
"""

import sys
from xml.etree import ElementTree


def extract_tests(xml_path, status="failed", sep=".", include_skipped=False):
    """Extract test names from a JUnit XML file.

    Args:
        xml_path: Path to JUnit XML file.
        status: "failed" to extract failures/errors, "passed" for passes.
        sep: Separator between classname and name ("." for gtest, "::" for pytest).
        include_skipped: If False, skipped tests are excluded from "passed" results.
    """
    try:
        tree = ElementTree.parse(xml_path)
    except (ElementTree.ParseError, FileNotFoundError, OSError):
        return

    for tc in tree.iter("testcase"):
        cls = tc.get("classname", "")
        name = tc.get("name", "")
        if not cls or not name:
            continue

        has_failure = tc.find("failure") is not None
        has_error = tc.find("error") is not None
        has_skipped = tc.find("skipped") is not None

        if status == "failed" and (has_failure or has_error):
            print(f"{cls}{sep}{name}")
        elif status == "passed":
            if not has_failure and not has_error:
                if include_skipped or not has_skipped:
                    print(f"{cls}{sep}{name}")


def parse_gtest_list():
    """Parse gtest --gtest_list_tests output from stdin into Suite.TestName."""
    suite = ""
    for line in sys.stdin:
        line = line.rstrip()
        if not line or line.startswith("#"):
            continue
        if not line.startswith(" "):
            suite = line.rstrip(".")
        else:
            print(f"{suite}.{line.strip().split()[0]}")


def main():
    if len(sys.argv) < 2:
        print(f"Usage: {sys.argv[0]} <command> [args]", file=sys.stderr)
        sys.exit(1)

    cmd = sys.argv[1]

    if cmd in ("failed", "passed"):
        if len(sys.argv) < 3:
            print(
                f"Usage: {sys.argv[0]} {cmd} <xml_file> [--sep SEP]",
                file=sys.stderr,
            )
            sys.exit(1)
        xml_path = sys.argv[2]
        sep = "."
        for i, arg in enumerate(sys.argv[3:], 3):
            if arg == "--sep" and i + 1 < len(sys.argv):
                sep = sys.argv[i + 1]
        extract_tests(xml_path, status=cmd, sep=sep)

    elif cmd == "gtest-list":
        parse_gtest_list()

    else:
        print(f"Unknown command: {cmd}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
