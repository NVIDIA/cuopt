# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from libcpp.memory cimport unique_ptr

from cuopt.linear_programming.solver.solver cimport solver_ret_t
cdef object build_solution_from_unique_ptr(
    unique_ptr[solver_ret_t] sol_ret_ptr,
    object variable_names)
