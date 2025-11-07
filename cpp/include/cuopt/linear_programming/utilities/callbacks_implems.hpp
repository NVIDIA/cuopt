/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <Python.h>
#include <cuopt/linear_programming/utilities/internals.hpp>

#include <iostream>

namespace cuopt {
namespace internals {

class default_get_solution_callback_t : public get_solution_callback_t {
 public:
  PyObject* get_numba_matrix(void* data, std::size_t size)
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;

    if (isFloat) {
      return PyObject_CallMethod(pycl, "get_numba_matrix", "(lls)", data, size, "float32");
    } else {
      return PyObject_CallMethod(pycl, "get_numba_matrix", "(lls)", data, size, "float64");
    }
  }

  PyObject* get_numpy_array(void* data, std::size_t size)
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;
    if (isFloat) {
      return PyObject_CallMethod(pycl, "get_numpy_array", "(lls)", data, size, "float32");
    } else {
      return PyObject_CallMethod(pycl, "get_numpy_array", "(lls)", data, size, "float64");
    }
  }

  void get_solution(void* data, void* objective_value) override
  {
    PyObject* numba_matrix = get_numba_matrix(data, n_variables);
    PyObject* numpy_array  = get_numba_matrix(objective_value, 1);
    PyObject* res =
      PyObject_CallMethod(this->pyCallbackClass, "get_solution", "(OO)", numba_matrix, numpy_array);
    Py_DECREF(numba_matrix);
    Py_DECREF(numpy_array);
    Py_DECREF(res);
  }

  PyObject* pyCallbackClass;
};

class default_set_solution_callback_t : public set_solution_callback_t {
 public:
  PyObject* get_numba_matrix(void* data, std::size_t size)
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;

    if (isFloat) {
      return PyObject_CallMethod(pycl, "get_numba_matrix", "(lls)", data, size, "float32");
    } else {
      return PyObject_CallMethod(pycl, "get_numba_matrix", "(lls)", data, size, "float64");
    }
  }

  PyObject* get_numpy_array(void* data, std::size_t size)
  {
    PyObject* pycl = (PyObject*)this->pyCallbackClass;
    if (isFloat) {
      return PyObject_CallMethod(pycl, "get_numpy_array", "(lls)", data, size, "float32");
    } else {
      return PyObject_CallMethod(pycl, "get_numpy_array", "(lls)", data, size, "float64");
    }
  }

  void set_solution(void* data, void* objective_value) override
  {
    PyObject* numba_matrix = get_numba_matrix(data, n_variables);
    PyObject* numpy_array  = get_numba_matrix(objective_value, 1);
    PyObject* res =
      PyObject_CallMethod(this->pyCallbackClass, "set_solution", "(OO)", numba_matrix, numpy_array);
    Py_DECREF(numba_matrix);
    Py_DECREF(numpy_array);
    Py_DECREF(res);
  }

  PyObject* pyCallbackClass;
};

}  // namespace internals
}  // namespace cuopt
