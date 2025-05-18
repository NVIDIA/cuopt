/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights
 * reserved. SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once

#include <cuopt/error.hpp>
#include <routing/dimensions.cuh>
#include <routing/structures.hpp>

#include <rmm/exec_policy.hpp>

#include <raft/util/cuda_utils.cuh>

namespace cuopt {
namespace routing {
namespace detail {

template <typename f_t>
constexpr f_t euclidean_distance(const f_t px1, const f_t py1, const f_t px2, const f_t py2)
{
  f_t diff_x = (px1 - px2);
  f_t diff_y = (py1 - py2);
  return sqrtf(diff_x * diff_x + diff_y * diff_y);
}

template <typename i_t>
constexpr void get_row_and_col_index(i_t& row, i_t& col, i_t flat_index, i_t nodes)
{
  row = nodes - 2 - floor(sqrt(-8 * flat_index + 4 * nodes * (nodes - 1) - 7.f) / 2.0f - 0.5f);
  col = flat_index + row + 1 - nodes * (nodes - 1) / 2 + (nodes - row) * ((nodes - row) - 1) / 2;
}

template <typename i_t, typename f_t>
constexpr double lookup_dist(f_t const* table, i_t i, i_t j, size_t width)
{
  return table[i * width + j];
}

template <typename i_t, typename f_t = float>
static constexpr NodeInfo<i_t> load(i_t pos, NodeInfo<i_t> const* path_node)
{
  return path_node[pos];
}

// All values pre-loaded overload
template <typename i_t, typename f_t, bool is_device = true>
static constexpr double get_distance(const NodeInfo<i_t>& l1,
                                     const NodeInfo<i_t>& l2,
                                     const VehicleInfo<f_t, is_device>& vehicle_info)
{
  if (vehicle_info.skip_first_trip && l1.node_type() == node_type_t::DEPOT) { return 0.f; }
  if (vehicle_info.drop_return_trip && l2.node_type() == node_type_t::DEPOT) { return 0.f; }
  auto matrix = vehicle_info.matrices.get_cost_matrix(vehicle_info.type);
  return lookup_dist(matrix, l1.location(), l2.location(), vehicle_info.matrices.extent[3]);
}

// All values pre-loaded overload
template <typename i_t, typename f_t, bool is_device = true>
static constexpr double get_transit_time(const NodeInfo<i_t>& l1,
                                         const NodeInfo<i_t>& l2,
                                         const VehicleInfo<f_t, is_device>& vehicle_info,
                                         const bool use_service_time = false)
{
  if (vehicle_info.skip_first_trip && l1.node_type() == node_type_t::DEPOT) { return 0.f; }

  double transit_time = 0.;
  if (use_service_time && l1 != l2) {
    // FIXME:: We are assuming that there is at most one break. So break duration is obtained form
    // zero dimension
    double service_time =
      (l1.node_type() == node_type_t::DEPOT)
        ? 0.
        : (l1.node_type() == node_type_t::BREAK ? vehicle_info.break_durations[l1.node()]
                                                : vehicle_info.order_service_times[l1.node()]);
    transit_time += service_time;
  }

  if (vehicle_info.drop_return_trip && l2.node_type() == node_type_t::DEPOT) {
    return transit_time;
  }

  auto matrix = vehicle_info.matrices.get_time_matrix(vehicle_info.type);
  transit_time +=
    lookup_dist(matrix, l1.location(), l2.location(), vehicle_info.matrices.extent[3]);

  return transit_time;
}

template <typename i_t, typename f_t>
static constexpr double get_arc_dimension(dim_t dim,
                                          const NodeInfo<i_t>& l1,
                                          const NodeInfo<i_t>& l2,
                                          const VehicleInfo<f_t>& vehicle_info)
{
  if (dim == dim_t::DIST) { return get_distance(l1, l2, vehicle_info); }
  return get_transit_time(l1, l2, vehicle_info);
}

template <typename i_t, typename f_t, dim_t dim, bool is_device>
static constexpr double get_arc_of_dimension(const NodeInfo<i_t>& l1,
                                             const NodeInfo<i_t>& l2,
                                             const VehicleInfo<f_t, is_device>& vehicle_info)
{
  if constexpr (dim == dim_t::DIST) {
    return get_distance(l1, l2, vehicle_info);
  } else if constexpr (dim == dim_t::TIME) {
    return get_transit_time(l1, l2, vehicle_info, true);
  } else if constexpr (dim == dim_t::SERVICE_TIME) {
    return l1.is_depot() ? 0. : vehicle_info.order_service_times[l1.node()];
  } else if constexpr (dim == dim_t::MISMATCH) {
    return !l1.is_service_node() ? 0. : (double)(1 - vehicle_info.order_match[l1.node()]);
  } else if constexpr (dim == dim_t::BREAK) {
    return l1.is_break();
  } else {
    return double{};
  }
}

template <typename i_t, typename f_t, size_t dim, bool is_device = true>
static constexpr double get_arc_of_dimension(const NodeInfo<i_t>& l1,
                                             const NodeInfo<i_t>& l2,
                                             const VehicleInfo<f_t, is_device>& vehicle_info)
{
  return get_arc_of_dimension<i_t, f_t, (dim_t)dim>(l1, l2, vehicle_info);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
