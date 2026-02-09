# SPDX-FileCopyrightText: Copyright (c) 2023-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0


# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libcpp.memory cimport unique_ptr
from libcpp.string cimport string
from libcpp.vector cimport vector

from cuopt.linear_programming.solver.solver cimport solver_settings_t

def get_solver_setting(name):
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings

    unique_solver_settings.reset(new solver_settings_t[int, double]())

    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    return c_solver_settings.get_parameter_as_string(
        name.encode('utf-8')
    ).decode('utf-8')


cpdef get_solver_parameter_names():
    cdef unique_ptr[solver_settings_t[int, double]] unique_solver_settings
    unique_solver_settings.reset(new solver_settings_t[int, double]())
    cdef solver_settings_t[int, double]* c_solver_settings = (
        unique_solver_settings.get()
    )
    cdef vector[string] parameter_names = c_solver_settings.get_parameter_names()

    cdef list py_parameter_names = []
    cdef size_t i
    for i in range(parameter_names.size()):
        # std::string -> Python str
        py_parameter_names.append(parameter_names[i].decode("utf-8"))
    return py_parameter_names

solver_params = get_solver_parameter_names()
for param in solver_params: globals()["CUOPT_"+param.upper()] = param
