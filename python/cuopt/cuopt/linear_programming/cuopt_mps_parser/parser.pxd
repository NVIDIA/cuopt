# SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION
# & AFFILIATES. All rights reserved.
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
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector


cdef extern from "mps_parser/mps_data_model.hpp" namespace "cuopt::mps_parser": # noqa

    cdef cppclass mps_data_model_t[i_t, f_t]:
        bool maximize_
        vector[f_t] A_
        vector[i_t] A_indices_
        vector[i_t] A_offsets_
        vector[f_t] b_
        vector[f_t] c_
        f_t objective_scaling_factor_
        f_t objective_offset_
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

cdef extern from "mps_parser/utilities/cython_mps_parser.hpp" namespace "cuopt::cython": # noqa

    cdef unique_ptr[mps_data_model_t[int, double]] call_parse_mps(
        const string& mps_file_path,
        bool fixed_mps_format
    ) except +
