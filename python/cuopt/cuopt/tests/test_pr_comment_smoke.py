# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# TEMPORARY — DO NOT MERGE.
#
# This file exists solely to verify the PR test-classification comment
# end to end: it forces one always-failing test so the PR-mode classifier
# emits a `pr_classification=new` entry that lands under "NEW failures"
# in the sticky PR comment.  Remove this file before merging PR #1194.


def test_pr_comment_smoke_always_fails():
    """Intentionally fails to exercise the pr-test-summary workflow."""
    assert False, (
        "intentional failure to verify the PR comment classifier — "
        "this test should not exist on main; remove with PR #1194"
    )
