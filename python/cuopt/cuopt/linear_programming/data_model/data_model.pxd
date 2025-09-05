# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp cimport bool
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "mps_parser/data_model_view.hpp" namespace "cuopt::mps_parser" nogil: # noqa

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


cdef extern from "mps_parser/writer.hpp" namespace "cuopt::mps_parser" nogil: # noqa

    cdef void write_mps(
        const data_model_view_t[int, double] data_model,
        const string user_problem_file) except +
