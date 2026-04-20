# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""
Intentionally flaky test to validate the CI flaky detection mechanism.
Fails on first run, passes on retry (pytest-rerunfailures).
Uses a temp file as a run counter.
Remove this test once the flaky detection pipeline is validated.
"""

import os
import tempfile


MARKER = os.path.join(tempfile.gettempdir(), "cuopt_flaky_validation_python")


def test_flaky_fails_first_passes_on_retry():
    if os.path.exists(MARKER):
        # Second run — pass and clean up
        os.remove(MARKER)
        assert True, "Passed on retry (flaky validation working)"
    else:
        # First run — create marker and fail
        with open(MARKER, "w") as f:
            f.write("first_attempt")
        assert False, "Intentional first-run failure for flaky detection validation"
