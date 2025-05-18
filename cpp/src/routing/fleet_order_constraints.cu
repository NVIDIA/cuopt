/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "fleet_order_constraints.hpp"

#include <thrust/fill.h>
#include <cuopt/error.hpp>
#include <raft/core/span.hpp>
#include <utilities/copy_helpers.hpp>
#include <vector>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
rmm::device_uvector<bool> generate_vehicle_order_match_matrix(
  data_model_view_t<i_t, f_t> const& data_model, bool& is_homogenous)
{
  auto handle_ptr_     = data_model.get_handle_ptr();
  auto stream_view     = handle_ptr_->get_stream();
  const i_t fleet_size = data_model.get_fleet_size();
  const i_t n_orders   = data_model.get_num_orders();

  const bool depot_included       = data_model.get_order_locations() == nullptr;
  const auto& vehicle_order_match = data_model.get_vehicle_order_match();
  const auto& order_vehicle_match = data_model.get_order_vehicle_match();
  const i_t order_begin           = depot_included ? 1 : 0;

  if (!vehicle_order_match.empty() || !order_vehicle_match.empty()) {
    std::vector<bool> vehicle_order_match_h(n_orders * fleet_size, true);
    std::set<std::pair<i_t, i_t>> not_allowed_pairs;
    // loop over specified vehicles and set the entries corresponding to specified
    // order list to true and remaining orders to false
    for (const auto& [vehicle_id, order_ids] : vehicle_order_match) {
      const auto order_ids_vec_h = cuopt::host_copy(order_ids, stream_view);
      const auto order_ids_h =
        std::unordered_set<i_t>(order_ids_vec_h.begin(), order_ids_vec_h.end());

      for (i_t order_id = order_begin; order_id < n_orders; ++order_id) {
        if (!order_ids_h.count(order_id)) {
          vehicle_order_match_h[vehicle_id * n_orders + order_id] = false;
          not_allowed_pairs.insert({order_id, vehicle_id});
        }
      }
    }

    for (const auto& [order_id, vehicle_ids] : order_vehicle_match) {
      const auto vehicle_ids_vec_h = cuopt::host_copy(vehicle_ids, stream_view);
      const auto vehicle_ids_h =
        std::unordered_set<i_t>(vehicle_ids_vec_h.begin(), vehicle_ids_vec_h.end());
      for (i_t vehicle_id = 0; vehicle_id < fleet_size; ++vehicle_id) {
        if (!vehicle_ids_h.count(vehicle_id)) {
          vehicle_order_match_h[order_id + vehicle_id * n_orders] = false;
        } else {
          cuopt_expects(
            not_allowed_pairs.count({order_id, vehicle_id}) == 0u,
            error_type_t::ValidationError,
            "Mismatch between vehicle_order_match and order_vehicle_match constraints!");
        }
      }
    }

    if (is_homogenous && !vehicle_order_match_h.empty()) {
      for (i_t vehicle_id = 1; vehicle_id < fleet_size; ++vehicle_id) {
        if (is_homogenous) {
          for (i_t order_id = order_begin; order_id < n_orders; ++order_id) {
            if (vehicle_order_match_h[order_id + (vehicle_id - 1) * n_orders] !=
                vehicle_order_match_h[order_id + vehicle_id * n_orders]) {
              is_homogenous = false;
              break;
            }
          }
        }
      }
    }

    return cuopt::device_copy(vehicle_order_match_h, stream_view);
  }

  return rmm::device_uvector<bool>(0, stream_view);
}

template <typename i_t>
__global__ void modify_service_times(raft::device_span<i_t> service_times,
                                     raft::device_span<bool const> order_vehicle_match)
{
  cuopt_assert(service_times.size() == order_vehicle_match.size(),
               "service times and order vehicle match matrix should have same sizes");
  size_t idx = threadIdx.x + blockDim.x * blockIdx.x;
  for (; idx < service_times.size(); idx += blockDim.x * gridDim.x) {
    if (!order_vehicle_match[idx]) { service_times[idx] = std::numeric_limits<i_t>::max(); }
  }
}

template <typename i_t, typename f_t>
void populate_vehicle_order_match(data_model_view_t<i_t, f_t> const& data_model,
                                  detail::fleet_order_constraints_t<i_t>& fleet_order_constraints_,
                                  bool& is_homogenous)
{
  fleet_order_constraints_.order_match =
    generate_vehicle_order_match_matrix<i_t, f_t>(data_model, is_homogenous);
}

template void populate_vehicle_order_match(
  data_model_view_t<int, float> const& data_model,
  detail::fleet_order_constraints_t<int>& fleet_order_constraints_,
  bool& is_homogenous);
}  // namespace detail
}  // namespace routing
}  // namespace cuopt
