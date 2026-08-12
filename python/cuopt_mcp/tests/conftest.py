# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest


@pytest.fixture
def anyio_backend():
    """Run @pytest.mark.anyio tests on asyncio only.

    anyio's plugin would otherwise parametrise across trio as well, which is
    not a dependency of this package.
    """
    return "asyncio"
