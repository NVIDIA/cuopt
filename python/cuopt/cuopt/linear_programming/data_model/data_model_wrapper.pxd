# SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

from .data_model cimport *

import warnings

import numpy as np

from libcpp.memory cimport unique_ptr


cdef class DataModel:
    cdef unique_ptr[data_model_view_t[int, double]] c_data_model_view
