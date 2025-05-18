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

#include <cuopt/routing/cython/cython.hpp>
#include <cuopt/routing/solve.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_buffer.hpp>
#include <routing/generator/generator.hpp>

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
                             i_t seed)
{
  params.n_locations       = n_locations;
  params.asymmetric        = asymmetric;
  params.dim               = dim;
  params.min_demand        = min_demand;
  params.max_demand        = max_demand;
  params.min_capacities    = min_capacities;
  params.max_capacities    = max_capacities;
  params.min_service_time  = min_service_time;
  params.max_service_time  = max_service_time;
  params.tw_tightness      = tw_tightness;
  params.drop_return_trips = drop_return_trips;
  params.n_shifts          = n_shifts;
  params.n_vehicle_types   = n_vehicle_types;
  params.n_matrix_types    = n_matrix_types;
  params.distrib           = distrib;
  params.center_box_min    = center_box_min;
  params.center_box_max    = center_box_max;
  params.seed              = seed;
}

/**
 * @brief Wrapper for vehicle_routing to expose the API to cython
 *
 * @param data_model Composable data model object
 * @param settings  Composable solver settings object
 * @return std::unique_ptr<vehicle_routing_ret_t>
 */
std::unique_ptr<vehicle_routing_ret_t> call_solve(
  routing::data_model_view_t<int, float>* data_model,
  routing::solver_settings_t<int, float>* settings)

{
  auto routing_solution = cuopt::routing::solve(*data_model, *settings);
  vehicle_routing_ret_t vr_ret{
    routing_solution.get_vehicle_count(),
    routing_solution.get_total_objective(),
    routing_solution.get_objectives(),
    std::make_unique<rmm::device_buffer>(routing_solution.get_route().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_order_locations().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_arrival_stamp().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_truck_id().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_node_types().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_unserviced_nodes().release()),
    std::make_unique<rmm::device_buffer>(routing_solution.get_accepted().release()),
    routing_solution.get_status(),
    routing_solution.get_status_string(),
    routing_solution.get_error_status().get_error_type(),
    routing_solution.get_error_status().what()};
  return std::make_unique<vehicle_routing_ret_t>(std::move(vr_ret));
}

/**
 * @brief Wrapper for dataset_t to expose the API to cython.
 * @param solver Composable solver object
 */
std::unique_ptr<dataset_ret_t> call_generate_dataset(
  raft::handle_t const& handle, routing::generator::dataset_params_t<int, float> const& params)
{
  auto data           = routing::generator::generate_dataset<int, float>(handle, params);
  auto [x_pos, y_pos] = data.get_coordinates();
  auto& fleet_info    = data.get_fleet_info();
  auto& order_info    = data.get_order_info();

  dataset_ret_t gen_ret{
    std::make_unique<rmm::device_buffer>(x_pos.release()),
    std::make_unique<rmm::device_buffer>(y_pos.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.matrices_.buffer.release()),
    std::make_unique<rmm::device_buffer>(order_info.v_earliest_time_.release()),
    std::make_unique<rmm::device_buffer>(order_info.v_latest_time_.release()),
    std::make_unique<rmm::device_buffer>(
      fleet_info.fleet_order_constraints_.order_service_times.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_earliest_time_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_latest_time_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_drop_return_trip_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_skip_first_trip_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_types_.release()),
    std::make_unique<rmm::device_buffer>(order_info.v_demand_.release()),
    std::make_unique<rmm::device_buffer>(fleet_info.v_capacities_.release())};
  return std::make_unique<dataset_ret_t>(std::move(gen_ret));
}

template void populate_dataset_params<int, float>(
  routing::generator::dataset_params_t<int, float>& params,
  int n_locations,
  bool asymmetric,
  int dim,
  routing::demand_i_t const* min_demand,
  routing::demand_i_t const* max_demand,
  routing::cap_i_t const* min_capacities,
  routing::cap_i_t const* max_capacities,
  int min_service_time,
  int max_service_time,
  float tw_tightness,
  float drop_return_trips,
  int n_shifts,
  int n_vehicle_types,
  int n_matrix_types,
  routing::generator::dataset_distribution_t distrib,
  float center_box_min,
  float center_box_max,
  int seed);

}  // namespace cython
}  // namespace cuopt
