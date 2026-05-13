# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TEMPORARY — DO NOT MERGE.
#
# This file verifies the PR test-classification comment end to end:
#
#   1. `test_pr_comment_smoke_always_fails` — a plain assertion failure.
#      Confirms the classifier emits a normal `pr_classification=new`
#      entry under "NEW failures".
#
#   2. `test_zz_pr_comment_smoke_segfault` — sends SIGSEGV to the pytest
#      process to confirm PR #1191's crash-marker path: pytest dies
#      mid-run, `write_pytest_crash_marker` writes <xml>-crash.xml, and
#      the classifier surfaces a PROCESS_CRASH entry under NEW failures
#      with a message containing "SIGSEGV".  Named with a `zz_` prefix
#      so it runs after the assert test in pytest's definition order
#      and the assert failure still makes it into the partial JUnit XML.
#
# Remove this file before merging PR #1194.

import os
import signal
import sys

import pytest


def test_pr_comment_smoke_always_fails():
    """Intentionally fails to exercise the pr-test-summary workflow."""
    assert False, (
        "intentional failure to verify the PR comment classifier — "
        "this test should not exist on main; remove with PR #1194"
    )


@pytest.mark.skipif(
    sys.version_info[:2] != (3, 11),
    reason=(
        "Smoke crash is scoped to the py3.11 matrix so the CAUTION block "
        "in the PR comment can be verified to appear on exactly one "
        "matrix entry while other Python versions stay green-on-crash."
    ),
)
def test_zz_pr_comment_smoke_segfault():
    """Intentionally crashes the pytest process to exercise the
    crash-marker path added in PR #1191.  Should produce a
    PROCESS_CRASH entry in the PR comment with a SIGSEGV message.
    """
    os.kill(os.getpid(), signal.SIGSEGV)
