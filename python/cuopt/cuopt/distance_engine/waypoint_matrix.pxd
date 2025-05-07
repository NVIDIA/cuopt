# SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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

from libcpp.memory cimport unique_ptr
from libcpp.utility cimport pair

from pylibraft.common.handle cimport *
from rmm.librmm.device_buffer cimport device_buffer


cdef extern from "cuopt/routing/distance_engine/waypoint_matrix.hpp" namespace "cuopt::distance_engine": # noqa

    cdef cppclass waypoint_matrix_t[i_t, f_t]:
        waypoint_matrix_t()
        waypoint_matrix_t(
            const handle_t& handle,
            const i_t* offsets,
            i_t n_vertices,
            const i_t* indices,
            const f_t* weights
        ) except +
        void compute_cost_matrix(
            f_t* d_cost_matrix,
            const i_t* target_locations,
            i_t n_target_locations
        ) except +
        pair[unique_ptr[device_buffer], unique_ptr[device_buffer]] compute_waypoint_sequence( # noqa
            const i_t* target_locations,
            i_t n_target_locations,
            const i_t* locations,
            i_t n_locations
        ) except +
        void compute_shortest_path_costs(
            f_t* d_time_matrix,
            const i_t* target_locations,
            i_t n_target_locations,
            const f_t* weights
        ) except +
