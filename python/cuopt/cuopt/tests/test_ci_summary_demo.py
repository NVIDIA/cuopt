# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Intentional failures to exercise the PR test summary comment feature.
# Remove after verifying the summary comment shows correct output.


def test_ci_summary_demo_failure():
    assert False, (
        "Intentional failure: remove after verifying PR summary comment."
    )


def test_ci_summary_demo_failure_2():
    raise RuntimeError(
        "Intentional error: remove after verifying PR summary comment."
    )
