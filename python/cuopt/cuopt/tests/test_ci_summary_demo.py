# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import sys

import pytest


@pytest.mark.skipif(sys.version_info < (3, 14), reason="Python 3.14 only")
def test_ci_summary_demo():
    pytest.fail(
        "Intentional failure: remove after verifying PR summary comment."
    )
