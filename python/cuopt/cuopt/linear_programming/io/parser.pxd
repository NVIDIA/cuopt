# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "cuopt/mathematical_optimization/io/mps_data_model.hpp" namespace "cuopt::mathematical_optimization::io": # noqa

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

        bool maximize_
        vector[f_t] A_
        vector[i_t] A_indices_
        vector[i_t] A_offsets_
        vector[f_t] b_
        vector[f_t] c_
        f_t objective_scaling_factor_
        f_t objective_offset_
        vector[f_t] Q_objective_values_
        vector[i_t] Q_objective_indices_
        vector[i_t] Q_objective_offsets_
        vector[f_t] variable_lower_bounds_
        vector[f_t] variable_upper_bounds_
        vector[f_t] constraint_lower_bounds_
        vector[f_t] constraint_upper_bounds_
        vector[char] var_types_
        vector[string] var_names_
        vector[string] row_names_
        vector[char] row_types_
        string objective_name_
        string problem_name_
        const vector[quadratic_constraint_t]& get_quadratic_constraints() const

cdef extern from "cuopt/mathematical_optimization/io/utilities/cython_parser.hpp" namespace "cuopt::cython": # noqa

    cdef unique_ptr[mps_data_model_t[int, double]] call_read(
        const string& file_path,
        bool fixed_mps_format
    ) except +

    cdef unique_ptr[mps_data_model_t[int, double]] call_parse_mps(
        const string& mps_file_path,
        bool fixed_mps_format
    ) except +
