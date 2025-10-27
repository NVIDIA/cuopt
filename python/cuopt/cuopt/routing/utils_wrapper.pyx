# SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
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

from pylibraft.common.handle cimport *
from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from cuopt.routing.structure.routing_utilities cimport *

from enum import IntEnum

import cupy as cp
import numpy as np
from numba import cuda

import cudf
from cudf.core.buffer import as_buffer

from libcpp.utility cimport move

from cuopt.utilities import col_from_buf


class DatasetDistribution(IntEnum):
    CLUSTERED = dataset_distribution_t.CLUSTERED
    RANDOM = dataset_distribution_t.RANDOM
    RANDOM_CLUSTERED = dataset_distribution_t.RANDOM_CLUSTERED


def generate_dataset(locations=100, asymmetric=True, min_demand=cudf.Series(),
                     max_demand=cudf.Series(), min_capacities=cudf.Series(),
                     max_capacities=cudf.Series(), min_service_time=0,
                     max_service_time=0, tw_tightness=0.0,
                     drop_return_trips=0.0, shifts=1,
                     n_vehicle_types=1, n_matrix_types=1,
                     distribution=DatasetDistribution.CLUSTERED,
                     center_box=None, seed=0):

    cdef unique_ptr[handle_t] handle_ptr
    handle_ptr.reset(new handle_t())
    handle_ = handle_ptr.get()

    min_demand = min_demand.astype(np.int16)
    max_demand = max_demand.astype(np.int16)
    min_capacities = min_capacities.astype(np.uint16)
    max_capacities = max_capacities.astype(np.uint16)
    dim = min_demand.shape[0]

    cdef uintptr_t c_min_demand = <uintptr_t>NULL
    if min_demand is not None:
        c_min_demand = min_demand.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_max_demand = <uintptr_t>NULL
    if max_demand is not None:
        c_max_demand = max_demand.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_min_capacities = <uintptr_t>NULL
    if min_capacities is not None:
        c_min_capacities = min_capacities.__cuda_array_interface__['data'][0]
    cdef uintptr_t c_max_capacities = <uintptr_t>NULL
    if max_capacities is not None:
        c_max_capacities = max_capacities.__cuda_array_interface__['data'][0]
    cdef int c_distrib_type = distribution.value

    center_box_min = 0
    center_box_max = locations / 2
    if center_box is not None:
        center_box_min = center_box[0]
        center_box_max = center_box[1]

    cdef dataset_params_t[int, float] params
    populate_dataset_params[int, float](params, locations, asymmetric,
                                        dim, <int32_t*>c_min_demand,
                                        <int32_t*>c_max_demand,
                                        <int32_t*>c_min_capacities,
                                        <int32_t*>c_max_capacities,
                                        min_service_time,
                                        max_service_time,
                                        tw_tightness,
                                        drop_return_trips,
                                        shifts,
                                        n_vehicle_types,
                                        n_matrix_types,
                                        <dataset_distribution_t>c_distrib_type,
                                        <float>center_box_min,
                                        <float>center_box_max,
                                        seed)

    g_ret_ptr = move(call_generate_dataset(handle_[0], params))
    g_ret = move(g_ret_ptr.get()[0])

    coordinates = cudf.DataFrame()
    orders = cudf.DataFrame()
    vehicles = cudf.DataFrame()
    constraints = dict()

    x_pos = DeviceBuffer.c_from_unique_ptr(move(g_ret.d_x_pos_))
    y_pos = DeviceBuffer.c_from_unique_ptr(move(g_ret.d_y_pos_))
    x_pos = as_buffer(x_pos)
    y_pos = as_buffer(y_pos)
    coordinates['x'] = col_from_buf(x_pos, np.float32)
    coordinates['y'] = col_from_buf(y_pos, np.float32)

    matrices_buf = as_buffer(
        DeviceBuffer.c_from_unique_ptr(move(g_ret.d_matrices_))
    )
    desc = matrices_buf.__cuda_array_interface__
    desc["shape"] = (n_vehicle_types, n_matrix_types, locations, locations)
    desc["typestr"] = "f4"
    matrices = cuda.from_cuda_array_interface(desc)
    matrices = cp.asarray(matrices, dtype=np.float32)
    matrices_ret = cp.array(matrices)

    # Create vehicles_df
    vehicle_earliest = DeviceBuffer.c_from_unique_ptr(
        move(g_ret.d_vehicle_earliest_time_)
    )
    vehicle_latest = DeviceBuffer.c_from_unique_ptr(
        move(g_ret.d_vehicle_latest_time_)
    )
    vehicle_drop_return_trips = DeviceBuffer.c_from_unique_ptr(
        move(g_ret.d_drop_return_trips_)
    )
    vehicle_skip_first_trips = DeviceBuffer.c_from_unique_ptr(
        move(g_ret.d_skip_first_trips_)
    )

    vehicle_earliest = as_buffer(vehicle_earliest)
    vehicle_latest = as_buffer(vehicle_latest)
    vehicle_drop_return_trips = as_buffer(vehicle_drop_return_trips)
    vehicle_skip_first_trips = as_buffer(vehicle_skip_first_trips)
    vehicles["earliest_time"] = col_from_buf(vehicle_earliest, np.int32)
    vehicles["latest_time"] = col_from_buf(vehicle_latest, np.int32)
    vehicles["drop_return_trips"] = col_from_buf(
        vehicle_drop_return_trips, np.bool_
    )
    vehicles["skip_first_trips"] = col_from_buf(
        vehicle_skip_first_trips, np.bool_
    )

    fleet_size = vehicles["earliest_time"].shape[0]
    capacities_buf = as_buffer(
        DeviceBuffer.c_from_unique_ptr(move(g_ret.d_caps_))
    )
    desc = capacities_buf.__cuda_array_interface__
    desc["shape"] = (dim, fleet_size)
    desc["typestr"] = "u2"
    capacities = cuda.from_cuda_array_interface(desc)
    capacities = cp.array(cp.asarray(capacities, dtype=np.uint16))
    for i in range(dim):
        vehicles["capacity_" + str(i)] = capacities[i]

    # Fleet order constraints
    service_times_buf = as_buffer(
        DeviceBuffer.c_from_unique_ptr(move(g_ret.d_service_time_))
    )
    desc = service_times_buf.__cuda_array_interface__
    desc["shape"] = (fleet_size, locations)
    desc["typestr"] = "i4"
    order_service_times = cuda.from_cuda_array_interface(desc)
    order_service_times = cp.asarray(order_service_times, dtype=np.int32)
    order_service_times = cp.array(order_service_times)

    constraints["order_service_times"] = order_service_times

    # Create orders df
    earliest_time = DeviceBuffer.c_from_unique_ptr(
        move(g_ret.d_earliest_time_)
    )
    earliest_time = as_buffer(earliest_time)
    latest_time = DeviceBuffer.c_from_unique_ptr(
        move(g_ret.d_latest_time_)
    )
    latest_time = as_buffer(latest_time)

    orders["earliest_time"] = col_from_buf(earliest_time, np.int32)
    orders["latest_time"] = col_from_buf(latest_time, np.int32)

    demands_buf = as_buffer(
        DeviceBuffer.c_from_unique_ptr(move(g_ret.d_demands_))
    )
    desc = demands_buf.__cuda_array_interface__
    desc["shape"] = (dim, locations)
    desc["typestr"] = "i2"
    demands = cuda.from_cuda_array_interface(desc)
    demands = cp.array(cp.asarray(demands, dtype=np.int16))
    for i in range(dim):
        orders["demand_" + str(i)] = demands[i]

    return coordinates, matrices_ret, orders, vehicles, constraints
