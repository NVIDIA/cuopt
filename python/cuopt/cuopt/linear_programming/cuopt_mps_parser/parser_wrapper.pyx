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

from cuopt_mps_parser.utilities import catch_mps_parser_exception

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from .parser cimport call_parse_mps

import warnings

import numpy as np
from data_model import DataModel


def type_cast(np_obj, np_type, name):
    msg = "Casting " + name + " from " + str(np_obj.dtype) + " to " + str(np.dtype(np_type))  # noqa
    warnings.warn(msg)
    np_obj = np_obj.astype(np.dtype(np_type))
    return np_obj


@catch_mps_parser_exception
def ParseMps(mps_file_path, fixed_mps_formats):
    data_model = DataModel()

    dm_ret_ptr = move(
        call_parse_mps(
            mps_file_path.encode('utf-8'),
            fixed_mps_formats
        )
    )
    dm_ret = move(dm_ret_ptr.get()[0])

    A_values_data = dm_ret.A_.data()
    A_values_size = dm_ret.A_.size()
    cdef double[:] A_values_ = <double[:A_values_size]>A_values_data
    A_values = np.asarray(A_values_).copy()

    A_indices_data = dm_ret.A_indices_.data()
    A_indices_size = dm_ret.A_indices_.size()
    cdef int[:] A_indices_ = <int[:A_indices_size]>A_indices_data
    A_indices = np.asarray(A_indices_).copy()

    A_offsets_data = dm_ret.A_offsets_.data()
    A_offsets_size = dm_ret.A_offsets_.size()
    cdef int[:] A_offsets_ = <int[:A_offsets_size]>A_offsets_data
    A_offsets = np.asarray(A_offsets_).copy()

    b_data = dm_ret.b_.data()
    b_size = dm_ret.b_.size()
    cdef double[:] b_ = <double[:b_size]>b_data
    b = np.asarray(b_).copy()

    c_data = dm_ret.c_.data()
    c_size = dm_ret.c_.size()
    cdef double[:] c_ = <double[:c_size]>c_data
    c = np.asarray(c_).copy()

    variable_lower_bounds_data = dm_ret.variable_lower_bounds_.data()
    variable_lower_bounds_size = dm_ret.variable_lower_bounds_.size()
    cdef double[:] variable_lower_bounds_ = <double[:variable_lower_bounds_size]>variable_lower_bounds_data # noqa
    variable_lower_bounds = np.asarray(variable_lower_bounds_).copy()

    variable_upper_bounds_data = dm_ret.variable_upper_bounds_.data()
    variable_upper_bounds_size = dm_ret.variable_upper_bounds_.size()
    cdef double[:] variable_upper_bounds_ = <double[:variable_upper_bounds_size]>variable_upper_bounds_data # noqa
    variable_upper_bounds = np.asarray(variable_upper_bounds_).copy()

    constraint_lower_bounds_data = dm_ret.constraint_lower_bounds_.data()
    constraint_lower_bounds_size = dm_ret.constraint_lower_bounds_.size()
    cdef double[:] constraint_lower_bounds_ = <double[:constraint_lower_bounds_size]>constraint_lower_bounds_data # noqa
    constraint_lower_bounds = np.asarray(constraint_lower_bounds_).copy()

    constraint_upper_bounds_data = dm_ret.constraint_upper_bounds_.data()
    constraint_upper_bounds_size = dm_ret.constraint_upper_bounds_.size()
    cdef double[:] constraint_upper_bounds_ = <double[:constraint_upper_bounds_size]>constraint_upper_bounds_data # noqa
    constraint_upper_bounds = np.asarray(constraint_upper_bounds_).copy()

    var_types_data = dm_ret.var_types_.data()
    var_types_size = dm_ret.var_types_.size()
    cdef char[:] var_types_ = <char[:var_types_size]>var_types_data # noqa
    var_types = np.asarray(var_types_, dtype='str').copy()
    row_types_data = dm_ret.row_types_.data()
    row_types_size = dm_ret.row_types_.size()
    cdef char[:] row_types_
    if row_types_size > 0:
        row_types_ = <char[:row_types_size]>row_types_data # noqa
        row_types = np.asarray(row_types_, dtype='str').copy()
    else:
        row_types = None
    var_names_ = np.asarray([i.decode() for i in dm_ret.var_names_])
    row_names_ = np.asarray([i.decode() for i in dm_ret.row_names_])

    data_model.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
    data_model.set_constraint_bounds(b)
    data_model.set_objective_coefficients(c)
    data_model.set_variable_lower_bounds(variable_lower_bounds)
    data_model.set_variable_upper_bounds(variable_upper_bounds)
    data_model.set_constraint_lower_bounds(constraint_lower_bounds)
    data_model.set_constraint_upper_bounds(constraint_upper_bounds)
    data_model.set_maximize(dm_ret.maximize_)
    data_model.set_objective_scaling_factor(dm_ret.objective_scaling_factor_)
    data_model.set_objective_offset(dm_ret.objective_offset_)
    data_model.set_variable_types(var_types)
    if row_types is not None:
        data_model.set_row_types(row_types)
    data_model.set_variable_names(var_names_)
    data_model.set_row_names(row_names_)

    return data_model
