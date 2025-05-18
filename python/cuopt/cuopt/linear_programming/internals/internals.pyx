# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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


from libc.stdint cimport uintptr_t

import numpy as np
from numba.cuda.api import from_cuda_array_interface


cdef extern from "Python.h":
    cdef cppclass PyObject


cdef extern from "cuopt/linear_programming/utilities/callbacks_implems.hpp" namespace "cuopt::internals":  # noqa
    cdef cppclass Callback:
        pass

    cdef cppclass default_get_solution_callback_t(Callback):
        void setup() except +
        void get_solution(void* data, void* objective_value) except +
        PyObject* pyCallbackClass

    cdef cppclass default_set_solution_callback_t(Callback):
        void setup() except +
        void set_solution(void* data, void* objective_value) except +
        PyObject* pyCallbackClass


cdef class PyCallback:

    def get_numba_matrix(self, data, shape, typestr):

        sizeofType = 4 if typestr == "float32" else 8
        desc = {
            'shape': (shape,),
            'strides': None,
            'typestr': typestr,
            'data': (data, True),
            'version': 3,
        }

        data = from_cuda_array_interface(desc, None, False)
        return data

    def get_numpy_array(self, data, shape, typestr):
        sizeofType = 4 if typestr == "float32" else 8
        desc = {
            'shape': (shape,),
            'strides': None,
            'typestr': typestr,
            'data': (data, False),
            'version': 3
        }
        data = desc['data'][0]
        shape = desc['shape']

        numpy_array = np.array([data], dtype=desc['typestr']).reshape(shape)
        return numpy_array

cdef class GetSolutionCallback(PyCallback):

    cdef default_get_solution_callback_t native_callback

    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)


cdef class SetSolutionCallback(PyCallback):

    cdef default_set_solution_callback_t native_callback

    def __init__(self):
        self.native_callback.pyCallbackClass = <PyObject *><void*>self

    def get_native_callback(self):
        return <uintptr_t>&(self.native_callback)
