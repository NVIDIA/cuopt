# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from cuopt_sh_client import SolverMethod


def test_primal_solver_method_matches_native_value():
    assert SolverMethod.PrimalSimplex == 4
