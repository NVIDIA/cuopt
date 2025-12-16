# SPDX-FileCopyrightText: Copyright (c) 2023-2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import pytest

from cuopt import routing
from cuopt.utilities import InputValidationError


def test_solve_infeasible_with_challenging_breaks():
    with pytest.raises(InputValidationError) as err:
        routing.DataModel(0, 0, 0)
    assert str(err.value) == "The data model needs at least one location"
