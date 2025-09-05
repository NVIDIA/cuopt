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

from .data_model cimport data_model_view_t, write_mps

import warnings

import numpy as np

import cudf

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move


def type_cast(np_obj, np_type, name):
    obj_type = np_obj.dtype

    if ((np.issubdtype(np_type, np.floating) and
         (not np.issubdtype(obj_type, np.floating)))
       or (np.issubdtype(np_type, np.integer) and
           (not np.issubdtype(obj_type, np.integer)))):
        msg = "Casting " + name + " from " + str(obj_type) + " to " + str(np.dtype(np_type))  # noqa
        warnings.warn(msg)
    np_obj = np_obj.astype(np.dtype(np_type))
    return np_obj


def get_data_ptr(array):
    if isinstance(array, cudf.Series):
        return array.__cuda_array_interface__['data'][0]
    elif isinstance(array, np.ndarray):
        return array.__array_interface__['data'][0]
    else:
        raise Exception(
            "get_data_ptr must be called with cudf.Series or np.ndarray"
        )


cdef class DataModel:

    def __init__(self):
        self.c_data_model_view.reset(new data_model_view_t[int, double]())

        self.maximize = False
        self.A_values = np.array([])
        self.A_indices = np.array([])
        self.A_offsets = np.array([])
        self.b = np.array([])
        self.c = np.array([])
        self.objective_scaling_factor = 1.0
        self.objective_offset = 0.0
        self.variable_lower_bounds = np.array([])
        self.variable_upper_bounds = np.array([])
        self.constraint_lower_bounds = np.array([])
        self.constraint_upper_bounds = np.array([])
        self.initial_primal_solution = np.array([])
        self.initial_dual_solution = np.array([])
        self.host_row_types = np.array([], dtype='U1')
        self.ascii_row_types = np.array([])
        self.objective_name = ""
        self.problem_name = ""
        self.variable_types = np.array([])
        self.variable_names = np.array([])
        self.row_names = np.array([])

    def set_maximize(self, maximize):
        self.maximize = maximize

    def set_csr_constraint_matrix(self, A_values, A_indices, A_offsets):
        self.A_values = type_cast(A_values, np.float64, "A_values")
        self.A_indices = type_cast(A_indices, np.int32, "A_indices")
        self.A_offsets = type_cast(A_offsets, np.int32, "A_offsets")

    def set_constraint_bounds(self, b):
        self.b = type_cast(b, np.float64, "b")

    def set_objective_coefficients(self, c):
        self.c = type_cast(c, np.float64, "c")

    def set_objective_scaling_factor(self, objective_scaling_factor):
        self.objective_scaling_factor = objective_scaling_factor

    def set_objective_offset(self, objective_offset):
        self.objective_offset = objective_offset

    def set_variable_lower_bounds(self, variable_lower_bounds):
        self.variable_lower_bounds = type_cast(
            variable_lower_bounds, np.float64, "variable_lower_bounds"
        )

    def set_variable_upper_bounds(self, variable_upper_bounds):
        self.variable_upper_bounds = type_cast(
            variable_upper_bounds, np.float64, "variable_upper_bounds"
        )

    def set_constraint_lower_bounds(self, constraint_lower_bounds):
        self.constraint_lower_bounds = type_cast(
            constraint_lower_bounds, np.float64, "constraint_lower_bounds"
        )

    def set_constraint_upper_bounds(self, constraint_upper_bounds):
        self.constraint_upper_bounds = type_cast(
            constraint_upper_bounds, np.float64, "constraint_upper_bounds"
        )

    def set_row_types(self, row_types):
        self.host_row_types = row_types
        ascii_values = [ord(char) for char in row_types]
        self.ascii_row_types = np.array(ascii_values, dtype=np.int8)

    def set_initial_primal_solution(self, initial_primal_solution):
        self.initial_primal_solution = type_cast(
            initial_primal_solution, np.float64, "initial_primal_solution"
        )

    def set_initial_dual_solution(self, initial_dual_solution):
        self.initial_dual_solution = type_cast(
            initial_dual_solution, np.float64, "initial_dual_solution"
        )

    def set_variable_types(self, variable_types):
        self.variable_types = variable_types

    def set_variable_names(self, variables_names):
        self.variable_names = variables_names

    def set_row_names(self, row_names):
        self.row_names = row_names

    def set_objective_name(self, objective_name):
        self.objective_name = objective_name

    def set_problem_name(self, problem_name):
        self.problem_name = problem_name

    def get_sense(self):
        return self.maximize

    def get_constraint_matrix_values(self):
        return self.A_values

    def get_constraint_matrix_indices(self):
        return self.A_indices

    def get_constraint_matrix_offsets(self):
        return self.A_offsets

    def get_constraint_bounds(self):
        return self.b

    def get_objective_coefficients(self):
        return self.c

    def get_objective_scaling_factor(self):
        return self.objective_scaling_factor

    def get_objective_offset(self):
        return self.objective_offset

    def get_variable_lower_bounds(self):
        return self.variable_lower_bounds

    def get_variable_upper_bounds(self):
        return self.variable_upper_bounds

    def get_constraint_lower_bounds(self):
        return self.constraint_lower_bounds

    def get_constraint_upper_bounds(self):
        return self.constraint_upper_bounds

    def get_row_types(self):
        return self.host_row_types

    def get_ascii_row_types(self):
        return self.ascii_row_types

    def get_initial_primal_solution(self):
        return self.initial_primal_solution

    def get_initial_dual_solution(self):
        return self.initial_dual_solution

    def get_variable_names(self):
        return self.variable_names

    def get_variable_types(self):
        return self.variable_types

    def get_row_names(self):
        return self.row_names

    def get_objective_name(self):
        return self.objective_name

    def get_problem_name(self):
        return self.problem_name

    def set_data_model_view(self):
        cdef data_model_view_t[int, double]* c_data_model_view = (
            self.c_data_model_view.get()
        )

        # Set self.fields on the C++ side if set on the Python side
        cdef uintptr_t c_A_values = (
            get_data_ptr(self.get_constraint_matrix_values())
        )
        cdef uintptr_t c_A_indices = (
            get_data_ptr(self.get_constraint_matrix_indices())
        )
        cdef uintptr_t c_A_offsets = (
            get_data_ptr(self.get_constraint_matrix_offsets())
        )
        if self.get_constraint_matrix_values().shape[0] != 0 and self.get_constraint_matrix_indices().shape[0] != 0 and self.get_constraint_matrix_offsets().shape[0] != 0: # noqa
            c_data_model_view.set_csr_constraint_matrix(
                <const double *> c_A_values,
                self.get_constraint_matrix_values().shape[0],
                <const int *> c_A_indices,
                self.get_constraint_matrix_indices().shape[0],
                <const int *> c_A_offsets,
                self.get_constraint_matrix_offsets().shape[0]
            )

        cdef uintptr_t c_b = (
            get_data_ptr(self.get_constraint_bounds())
        )
        if self.get_constraint_bounds().shape[0] != 0:
            c_data_model_view.set_constraint_bounds(
                <const double *> c_b,
                self.get_constraint_bounds().shape[0]
            )

        cdef uintptr_t c_c = (
            get_data_ptr(self.get_objective_coefficients())
        )
        if self.get_objective_coefficients().shape[0] != 0:
            c_data_model_view.set_objective_coefficients(
                <const double *> c_c,
                self.get_objective_coefficients().shape[0]
            )

        c_data_model_view.set_objective_scaling_factor(
            <double> self.get_objective_scaling_factor()
        )
        c_data_model_view.set_objective_offset(
            <double> self.get_objective_offset()
        )
        c_data_model_view.set_maximize(<bool> self.maximize)

        cdef uintptr_t c_variable_lower_bounds = (
            get_data_ptr(self.get_variable_lower_bounds())
        )
        if self.get_variable_lower_bounds().shape[0] != 0:
            c_data_model_view.set_variable_lower_bounds(
                <const double *> c_variable_lower_bounds,
                self.get_variable_lower_bounds().shape[0]
            )

        cdef uintptr_t c_variable_upper_bounds = (
            get_data_ptr(self.get_variable_upper_bounds())
        )
        if self.get_variable_upper_bounds().shape[0] != 0:
            c_data_model_view.set_variable_upper_bounds(
                <const double *> c_variable_upper_bounds,
                self.get_variable_upper_bounds().shape[0]
            )
        cdef uintptr_t c_constraint_lower_bounds = (
            get_data_ptr(self.get_constraint_lower_bounds())
        )
        if self.get_constraint_lower_bounds().shape[0] != 0:
            c_data_model_view.set_constraint_lower_bounds(
                <const double *> c_constraint_lower_bounds,
                self.get_constraint_lower_bounds().shape[0]
            )
        cdef uintptr_t c_constraint_upper_bounds = (
            get_data_ptr(self.get_constraint_upper_bounds())
        )
        if self.get_constraint_upper_bounds().shape[0] != 0:
            c_data_model_view.set_constraint_upper_bounds(
                <const double *> c_constraint_upper_bounds,
                self.get_constraint_upper_bounds().shape[0]
            )
        cdef uintptr_t c_row_types = (
            get_data_ptr(self.get_ascii_row_types())
        )
        if self.get_ascii_row_types().shape[0] != 0:
            c_data_model_view.set_row_types(
                <const char *> c_row_types,
                self.get_ascii_row_types().shape[0]
            )

        cdef uintptr_t c_var_types = (
            get_data_ptr(self.get_variable_types())
        )
        if self.get_variable_types().shape[0] != 0:
            c_data_model_view.set_variable_types(
                <const char *> c_var_types,
                self.get_variable_types().shape[0]
            )

        cdef vector[string] c_var_names
        for s in self.get_variable_names():
            c_var_names.push_back(s.encode())

        if len(self.get_variable_names()) != 0:
            c_data_model_view.set_variable_names(
                c_var_names
            )

        cdef vector[string] c_row_names
        for s in self.get_row_names():
            c_row_names.push_back(s.encode())

        if len(self.get_row_names()) != 0:
            c_data_model_view.set_row_names(
                c_row_names
            )

        if self.get_problem_name():
            c_data_model_view.set_problem_name(
                self.get_problem_name().encode()
            )

        if self.get_objective_name():
            c_data_model_view.set_objective_name(
                self.get_objective_name().encode()
            )

        # Set initial solution on the C++ side if set on the Python side
        cdef uintptr_t c_initial_primal_solution = (
            get_data_ptr(self.get_initial_primal_solution())
        )
        if self.get_initial_primal_solution().shape[0] != 0:
            c_data_model_view.set_initial_primal_solution(
                <const double *> c_initial_primal_solution,
                self.get_initial_primal_solution().shape[0]
            )
        cdef uintptr_t c_initial_dual_solution = (
            get_data_ptr(self.get_initial_dual_solution())
        )
        if self.get_initial_dual_solution().shape[0] != 0:
            c_data_model_view.set_initial_dual_solution(
                <const double *> c_initial_dual_solution,
                self.get_initial_dual_solution().shape[0]
            )

    def writeMPS(self, user_problem_file):
        self.variable_types = type_cast(
            self.variable_types, "S1", "variable_types"
        )
        self.set_data_model_view()
        write_mps(self.c_data_model_view.get()[0],
                  user_problem_file.encode('utf-8'))
