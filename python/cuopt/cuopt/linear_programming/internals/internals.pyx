# SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3


from libc.stdint cimport uintptr_t
from cpython.ref cimport PyObject

import ctypes
import numpy as np


cdef extern from "cuopt/linear_programming/utilities/callbacks_implems.hpp" namespace "cuopt::internals":  # noqa
    cdef cppclass Callback:
        pass

    cdef cppclass default_get_solution_callback_t(Callback):
        void setup() except +
        void get_solution(void* data,
                          void* objective_value,
                          void* solution_bound,
                          void* user_data) except +
        PyObject* pyCallbackClass

    cdef cppclass default_set_solution_callback_t(Callback):
        void setup() except +
        void set_solution(void* data,
                          void* objective_value,
                          void* solution_bound,
                          void* user_data) except +
        PyObject* pyCallbackClass


cdef class PyCallback:

    cdef object _user_data

    def __init__(self):
        self._user_data = None

    property user_data:
        def __get__(self):
            return self._user_data
        def __set__(self, value):
            self._user_data = value

    cpdef uintptr_t get_user_data_ptr(self):
        cdef PyObject* ptr
        if self._user_data is None:
            return 0
        ptr = <PyObject*>self._user_data
        return <uintptr_t>ptr

    def get_numpy_array(self, data, shape, typestr):
        c_type = ctypes.c_float if typestr == "float32" else ctypes.c_double
        addr = int(data)
        buf = (c_type * shape).from_address(addr)
        return np.ctypeslib.as_array(buf)

cdef class GetSolutionCallback(PyCallback):

    cdef default_get_solution_callback_t native_callback

    def __init__(self):
        super().__init__()
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)


cdef class SetSolutionCallback(PyCallback):

    cdef default_set_solution_callback_t native_callback

    def __init__(self):
        super().__init__()
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
