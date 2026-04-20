# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Pytest plugin: write rerun failures to a supplementary JUnit XML.

pytest-rerunfailures v14+ only records the final outcome in JUnit XML.
This plugin collects rerun (failed) attempts and writes them to a
separate XML file so nightly_report.py can classify flaky tests
(tests that failed then passed on retry).

Usage: pytest -p cuopt_rerun_xml ...
Requires RAPIDS_TESTS_DIR env var for output location.
"""

import os
from collections import defaultdict
from xml.etree.ElementTree import Element, ElementTree, SubElement

import pytest

# Collect rerun failure reports keyed by nodeid
_rerun_failures = defaultdict(list)
_final_outcomes = {}


@pytest.hookimpl(hookwrapper=True)
def pytest_runtest_protocol(item, nextitem):
    """Track each test's final outcome."""
    yield


@pytest.hookimpl(trylast=True)
def pytest_runtest_logreport(report):
    """Collect reports — track reruns and final outcomes."""
    if report.when != "call":
        return
    node_id = report.nodeid
    if report.outcome == "rerun":
        # This is a failed attempt that will be retried
        msg = ""
        if report.longrepr:
            msg = str(report.longrepr)[:500]
        _rerun_failures[node_id].append(msg)
    else:
        _final_outcomes[node_id] = report.outcome


def pytest_sessionfinish(session, exitstatus):
    """Write supplementary XML for flaky tests (failed then passed)."""
    if not _rerun_failures:
        return

    output_dir = os.environ.get("RAPIDS_TESTS_DIR", "")
    if not output_dir:
        return

    testsuites = Element("testsuites")
    suite = SubElement(testsuites, "testsuite", name="pytest-reruns")
    count = 0

    for node_id, failure_messages in _rerun_failures.items():
        final = _final_outcomes.get(node_id, "")
        if final != "passed":
            # Test didn't eventually pass — not flaky, just failed
            continue

        # Flaky: failed on rerun attempts, passed on final
        parts = node_id.rsplit("::", 1)
        if len(parts) == 2:
            classname = parts[0].replace("/", ".").replace(".py", "")
            name = parts[1]
        else:
            classname = ""
            name = node_id

        for msg in failure_messages:
            tc = SubElement(
                suite,
                "testcase",
                classname=classname,
                name=name,
                time="0",
            )
            fail = SubElement(tc, "failure", message=msg[:200])
            fail.text = msg
            count += 1

    if count > 0:
        suite.set("tests", str(count))
        suite.set("failures", str(count))
        out_path = os.path.join(output_dir, "junit-pytest-reruns.xml")
        ElementTree(testsuites).write(
            out_path, xml_declaration=True, encoding="unicode"
        )
        print(f"\nWrote {count} rerun failure entries to {out_path}")
