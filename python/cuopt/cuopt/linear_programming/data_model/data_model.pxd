# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stddef cimport size_t
from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cuopt/mathematical_optimization/io/mps_data_model.hpp" namespace "cuopt::mathematical_optimization::io" nogil: # noqa

    cdef cppclass mps_data_model_t[i_t, f_t]:
        cppclass quadratic_constraint_t:
            int constraint_row_index
            string constraint_row_name
            char constraint_row_type
            vector[double] linear_values
            vector[int] linear_indices
            double rhs_value
            vector[int] rows
            vector[int] cols
            vector[double] vals


cdef extern from "cuopt/mathematical_optimization/io/data_model_view.hpp" namespace "cuopt::mathematical_optimization::io" nogil: # noqa

    cdef cppclass data_model_view_t[i_t, f_t]:
        void set_maximize(bool maximize) except +
        void set_csr_constraint_matrix(
            const f_t* A_values, i_t size_values,
            const i_t* A_indices, i_t size_indices,
            const i_t* A_offsets, i_t size_offsets) except +
        void set_constraint_bounds(const f_t* b, i_t size) except +
        void set_objective_coefficients(const f_t* c, i_t size) except +
        void set_objective_scaling_factor(
            f_t objective_scaling_factor) except +
        void set_objective_offset(
            f_t objective_offset) except +
        void set_quadratic_objective_matrix(
            const f_t* Q_values, i_t size_values,
            const i_t* Q_indices, i_t size_indices,
            const i_t* Q_offsets, i_t size_offsets) except +
        void set_variable_lower_bounds(
            const f_t* variable_lower_bounds,
            i_t size) except +
        void set_variable_upper_bounds(
            const f_t* variable_upper_bounds,
            i_t size) except +
        void set_constraint_lower_bounds(
            const f_t* constraint_lower_bounds,
            i_t size) except +
        void set_constraint_upper_bounds(
            const f_t* constraint_upper_bounds,
            i_t size) except +
        void set_initial_primal_solution(
            const f_t* initial_primal_solution,
            i_t size) except +
        void set_initial_dual_solution(
            const f_t* initial_dual_solution,
            i_t size) except +
        void set_row_types(const char* row_types, i_t size) except +
        void set_variable_types(const char* var_types, i_t size) except +
        void set_variable_names(const vector[string] variables_names) except +
        void set_row_names(const vector[string] row_names) except +
        void set_problem_name(const string problem_name) except +
        void set_objective_name(const string objective_name) except +
        void set_quadratic_constraints(
            vector[mps_data_model_t[i_t, f_t].quadratic_constraint_t] constraints) except +


cdef extern from "cuopt/mathematical_optimization/io/writer.hpp" namespace "cuopt::mathematical_optimization::io" nogil: # noqa

    cdef void write_mps(
        const data_model_view_t[int, double] data_model,
        const string user_problem_file) except +
