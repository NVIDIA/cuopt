# SPDX-FileCopyrightText: Copyright (c) 2022-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. # noqa
# SPDX-License-Identifier: Apache-2.0

# cython: profile=False
# distutils: language = c++
# cython: embedsignature = True
# cython: language_level = 3

from libc.stdint cimport uintptr_t
from libcpp.memory cimport unique_ptr
from libcpp.utility cimport move, pair

from pylibraft.common.handle cimport *
from rmm.pylibrmm.device_buffer cimport DeviceBuffer

from cuopt.distance_engine.waypoint_matrix cimport waypoint_matrix_t

import cupy as cp
import numpy as np
from numba import cuda

import cudf

from cuopt.utilities import series_from_buf

import pyarrow as pa


cdef class WaypointMatrix:

    cdef unique_ptr[waypoint_matrix_t[int, float]] c_waypoint_matrix
    cdef unique_ptr[handle_t] handle_ptr

    def __init__(self, offsets, indices, weights):
        self.handle_ptr.reset(new handle_t())
        handle_ = self.handle_ptr.get()
        self.offsets = offsets.astype(np.dtype(np.int32))
        self.indices = indices.astype(np.dtype(np.int32))
        self.weights = weights.astype(np.dtype(np.float32))

        cdef uintptr_t c_offsets = self.offsets.__array_interface__['data'][0]
        cdef uintptr_t c_indices = self.indices.__array_interface__['data'][0]
        cdef uintptr_t c_weights = self.weights.__array_interface__['data'][0]

        # -1 for the number of vertices since offsets list has an extra cell
        self.c_waypoint_matrix.reset(new waypoint_matrix_t[int, float](
            handle_[0],
            <const int *> c_offsets,
            len(self.offsets) - 1,
            <const int *>c_indices,
            <const float *>c_weights)
        )

    def compute_cost_matrix(self, target_locations):
        target_locations = target_locations.astype(np.int32)
        cdef uintptr_t c_target_locations = (
            target_locations.__array_interface__['data'][0]
        )

        cupy_array = cp.empty(len(target_locations) * len(target_locations),
                              dtype=np.float32,
                              order='C')
        cdef uintptr_t c_cost_matrix = cupy_array.data.ptr

        self.c_waypoint_matrix.get()[0].compute_cost_matrix(
            <float *> c_cost_matrix,
            <const int *> c_target_locations,
            len(target_locations)
        )

        return cudf.DataFrame(cupy_array.reshape(len(target_locations),
                                                 len(target_locations)))

    def compute_waypoint_sequence(self, target_locations, route_df):
        target_locations = target_locations.astype(np.dtype(np.int32))
        cdef uintptr_t c_target_locations = (
            target_locations.__array_interface__['data'][0]
        )

        route_df = route_df.reset_index(drop=True)
        route_locations = route_df['location'].astype(np.dtype(np.int32))
        route_locations = cp.array(route_locations.to_cupy(), order='C')
        cdef uintptr_t c_route_locations = route_locations.data.ptr

        path_info =(
            move(self.c_waypoint_matrix.get()[0].compute_waypoint_sequence(
                <const int *> c_target_locations,
                len(target_locations),
                <const int *> c_route_locations,
                len(route_locations)))
        )

        full_sequence_offset = (
            DeviceBuffer.c_from_unique_ptr(move(path_info.first))
        )
        full_path = DeviceBuffer.c_from_unique_ptr(move(path_info.second))

        route_df['sequence_offset'] = series_from_buf(
            full_sequence_offset, pa.int32()
        )
        locations = route_df["location"].replace(
            to_replace=list(range(len(target_locations))),
            value=target_locations.tolist()
        )
        route_df['location'] = locations
        waypoint_seq = series_from_buf(full_path, pa.int32())

        def create_way_point_types(routes, waypoint_seq):

            sequence_types = cudf.Series(["w"]*len(waypoint_seq))
            routes_dict = routes.groupby("truck_id").agg(list).to_pandas().to_dict() # noqa
            sequence_offsets = {k: np.array(v) for k, v in routes_dict["sequence_offset"].items()} # noqa
            task = {k: np.array(v) for k, v in routes_dict["type"].items()}
            truck_ids = routes.truck_id.unique().to_pandas()
            for tr_id in truck_ids:
                if len(sequence_types) > sequence_offsets[tr_id][-1]:
                    sequence_types.iloc[sequence_offsets[tr_id][-1]] = "-"
                sequence_types.iloc[
                    sequence_offsets[tr_id][1:]-1
                ] = task[tr_id][1:]

            return sequence_types

        return cudf.DataFrame(
            {
                "waypoint_sequence": waypoint_seq,
                "waypoint_type": create_way_point_types(
                    route_df, waypoint_seq
                )
            }
        )

    def compute_shortest_path_costs(self, target_locations, weights):

        target_locations = target_locations.astype(np.dtype(np.int32))
        weights = weights.astype(np.dtype(np.float32))

        cdef uintptr_t c_target_locations = (
            target_locations.__array_interface__['data'][0]
        )
        cdef uintptr_t c_weights = weights.__array_interface__['data'][0]

        cupy_array = cp.empty(
            len(target_locations) * len(target_locations),
            dtype=np.dtype(np.float32),
            order='C'
        )
        cdef uintptr_t c_time_matrix = cupy_array.data.ptr

        self.c_waypoint_matrix.get()[0].compute_shortest_path_costs(
            <float *> c_time_matrix,
            <const int *> c_target_locations,
            len(target_locations),
            <const float *>c_weights
        )

        return cudf.DataFrame(
            cupy_array.reshape(len(target_locations), len(target_locations))
        )
