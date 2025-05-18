/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <cuopt/routing/assignment.hpp>
#include <cuopt/routing/cython/generator.hpp>
#include <cuopt/routing/data_model_view.hpp>
#include <cuopt/routing/routing_structures.hpp>
#include <cuopt/routing/solver_settings.hpp>

#include <raft/core/handle.hpp>

#include <memory>

namespace cuopt {
namespace cython {

template <typename i_t, typename f_t>
void populate_dataset_params(routing::generator::dataset_params_t<i_t, f_t>& params,
                             i_t n_locations,
                             bool asymmetric,
                             i_t dim,
                             routing::demand_i_t const* min_demand,
                             routing::demand_i_t const* max_demand,
                             routing::cap_i_t const* min_capacities,
                             routing::cap_i_t const* max_capacities,
                             i_t min_service_time,
                             i_t max_service_time,
                             f_t tw_tightness,
                             f_t drop_return_trips,
                             i_t n_shifts,
                             i_t n_vehicle_types,
                             i_t n_matrix_types,
                             routing::generator::dataset_distribution_t distrib,
                             f_t center_box_min,
                             f_t center_box_max,
                             i_t seed);

// aggregate for vehicle_routing() return type
// to be exposed to cython:
struct vehicle_routing_ret_t {
  int vehicle_count_;
  double total_objective_value_;
  std::map<routing::objective_t, double> objective_values_;
  std::unique_ptr<rmm::device_buffer> d_route_;
  std::unique_ptr<rmm::device_buffer> d_route_locations_;
  std::unique_ptr<rmm::device_buffer> d_arrival_stamp_;
  std::unique_ptr<rmm::device_buffer> d_truck_id_;
  std::unique_ptr<rmm::device_buffer> d_node_types_;
  std::unique_ptr<rmm::device_buffer> d_unserviced_nodes_;
  std::unique_ptr<rmm::device_buffer> d_accepted_;
  routing::solution_status_t status_;
  std::string solution_string_;
  cuopt::error_type_t error_status_;
  std::string error_message_;
};

// aggregate for dataset_t() return type
// to be exposed to cython:
struct dataset_ret_t {
  std::unique_ptr<rmm::device_buffer> d_x_pos_;
  std::unique_ptr<rmm::device_buffer> d_y_pos_;
  std::unique_ptr<rmm::device_buffer> d_matrices_;
  std::unique_ptr<rmm::device_buffer> d_earliest_time_;
  std::unique_ptr<rmm::device_buffer> d_latest_time_;
  std::unique_ptr<rmm::device_buffer> d_service_time_;
  std::unique_ptr<rmm::device_buffer> d_vehicle_earliest_time_;
  std::unique_ptr<rmm::device_buffer> d_vehicle_latest_time_;
  std::unique_ptr<rmm::device_buffer> d_drop_return_trips_;
  std::unique_ptr<rmm::device_buffer> d_skip_first_trips_;
  std::unique_ptr<rmm::device_buffer> d_vehicle_types_;
  std::unique_ptr<rmm::device_buffer> d_demands_;
  std::unique_ptr<rmm::device_buffer> d_caps_;
};

// Wrapper for solve to expose the API to cython.
std::unique_ptr<vehicle_routing_ret_t> call_solve(routing::data_model_view_t<int, float>*,
                                                  routing::solver_settings_t<int, float>*);

// Wrapper for dataset to expose the API to cython.
std::unique_ptr<dataset_ret_t> call_generate_dataset(
  raft::handle_t const& handle, routing::generator::dataset_params_t<int, float> const& params);

}  // namespace cython
}  // namespace cuopt
