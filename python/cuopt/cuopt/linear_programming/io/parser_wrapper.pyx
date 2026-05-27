# SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from cuopt.linear_programming.io.utilities import (
    catch_io_exception,
)

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.utility cimport move

from .parser cimport call_read, call_parse_mps, mps_data_model_t
import warnings

import numpy as np
from cuopt.linear_programming.data_model import DataModel


def type_cast(np_obj, np_type, name):
    msg = "Casting " + name + " from " + str(np_obj.dtype) + " to " + str(np.dtype(np_type))  # noqa
    warnings.warn(msg)
    np_obj = np_obj.astype(np.dtype(np_type))
    return np_obj


# Copies the C++ data model behind `dm` into the Python-side `data_model`.
# Copies every field on mps_data_model_t into the Python DataModel.
cdef _marshal_data_model(mps_data_model_t[int, double]* dm, data_model):
    A_values_data = dm.A_.data()
    A_values_size = dm.A_.size()
    cdef double[:] A_values_ = <double[:A_values_size]>A_values_data
    A_values = np.asarray(A_values_).copy()

    A_indices_data = dm.A_indices_.data()
    A_indices_size = dm.A_indices_.size()
    cdef int[:] A_indices_ = <int[:A_indices_size]>A_indices_data
    A_indices = np.asarray(A_indices_).copy()

    A_offsets_data = dm.A_offsets_.data()
    A_offsets_size = dm.A_offsets_.size()
    cdef int[:] A_offsets_ = <int[:A_offsets_size]>A_offsets_data
    A_offsets = np.asarray(A_offsets_).copy()

    b_data = dm.b_.data()
    b_size = dm.b_.size()
    cdef double[:] b_ = <double[:b_size]>b_data
    b = np.asarray(b_).copy()

    c_data = dm.c_.data()
    c_size = dm.c_.size()
    cdef double[:] c_ = <double[:c_size]>c_data
    c = np.asarray(c_).copy()

    Q_values_size = dm.Q_objective_values_.size()
    if Q_values_size > 0:
        Q_values_data = dm.Q_objective_values_.data()
        Q_values = np.asarray(<double[:Q_values_size]>Q_values_data).copy()
    else:
        Q_values = np.array([], dtype=np.float64)

    Q_indices_size = dm.Q_objective_indices_.size()
    if Q_indices_size > 0:
        Q_indices_data = dm.Q_objective_indices_.data()
        Q_indices = np.asarray(<int[:Q_indices_size]>Q_indices_data).copy()
    else:
        Q_indices = np.array([], dtype=np.int32)

    Q_offsets_size = dm.Q_objective_offsets_.size()
    if Q_offsets_size > 0:
        Q_offsets_data = dm.Q_objective_offsets_.data()
        Q_offsets = np.asarray(<int[:Q_offsets_size]>Q_offsets_data).copy()
    else:
        Q_offsets = np.array([], dtype=np.int32)

    variable_lower_bounds_data = dm.variable_lower_bounds_.data()
    variable_lower_bounds_size = dm.variable_lower_bounds_.size()
    cdef double[:] variable_lower_bounds_ = <double[:variable_lower_bounds_size]>variable_lower_bounds_data # noqa
    variable_lower_bounds = np.asarray(variable_lower_bounds_).copy()

    variable_upper_bounds_data = dm.variable_upper_bounds_.data()
    variable_upper_bounds_size = dm.variable_upper_bounds_.size()
    cdef double[:] variable_upper_bounds_ = <double[:variable_upper_bounds_size]>variable_upper_bounds_data # noqa
    variable_upper_bounds = np.asarray(variable_upper_bounds_).copy()

    constraint_lower_bounds_data = dm.constraint_lower_bounds_.data()
    constraint_lower_bounds_size = dm.constraint_lower_bounds_.size()
    cdef double[:] constraint_lower_bounds_ = <double[:constraint_lower_bounds_size]>constraint_lower_bounds_data # noqa
    constraint_lower_bounds = np.asarray(constraint_lower_bounds_).copy()

    constraint_upper_bounds_data = dm.constraint_upper_bounds_.data()
    constraint_upper_bounds_size = dm.constraint_upper_bounds_.size()
    cdef double[:] constraint_upper_bounds_ = <double[:constraint_upper_bounds_size]>constraint_upper_bounds_data # noqa
    constraint_upper_bounds = np.asarray(constraint_upper_bounds_).copy()

    var_types_data = dm.var_types_.data()
    var_types_size = dm.var_types_.size()
    cdef char[:] var_types_ = <char[:var_types_size]>var_types_data # noqa
    var_types = np.asarray(var_types_, dtype='str').copy()
    row_types_data = dm.row_types_.data()
    row_types_size = dm.row_types_.size()
    cdef char[:] row_types_
    if row_types_size > 0:
        row_types_ = <char[:row_types_size]>row_types_data # noqa
        row_types = np.asarray(row_types_, dtype='str').copy()
    else:
        row_types = None
    var_names_ = np.asarray([i.decode() for i in dm.var_names_])
    row_names_ = np.asarray([i.decode() for i in dm.row_names_])

    data_model.set_csr_constraint_matrix(A_values, A_indices, A_offsets)
    data_model.set_constraint_bounds(b)
    data_model.set_objective_coefficients(c)
    data_model.set_variable_lower_bounds(variable_lower_bounds)
    data_model.set_variable_upper_bounds(variable_upper_bounds)
    data_model.set_constraint_lower_bounds(constraint_lower_bounds)
    data_model.set_constraint_upper_bounds(constraint_upper_bounds)
    data_model.set_maximize(dm.maximize_)
    data_model.set_objective_scaling_factor(dm.objective_scaling_factor_)
    data_model.set_objective_offset(dm.objective_offset_)
    data_model.set_quadratic_objective_matrix(Q_values, Q_indices, Q_offsets)
    data_model.set_variable_types(var_types)
    if row_types is not None:
        data_model.set_row_types(row_types)
    data_model.set_variable_names(var_names_)
    data_model.set_row_names(row_names_)
    data_model.set_objective_name(dm.objective_name_.decode())
    data_model.set_problem_name(dm.problem_name_.decode())

    return data_model


@catch_io_exception
def Read(file_path, fixed_mps_format=False):
    data_model = DataModel()
    dm_ret_ptr = move(
        call_read(
            file_path.encode('utf-8'),
            fixed_mps_format,
        )
    )
    return _marshal_data_model(dm_ret_ptr.get(), data_model)


@catch_io_exception
def ParseMps(mps_file_path, fixed_mps_format=False):
    data_model = DataModel()
    dm_ret_ptr = move(
        call_parse_mps(
            mps_file_path.encode('utf-8'),
            fixed_mps_format,
        )
    )
    return _marshal_data_model(dm_ret_ptr.get(), data_model)
