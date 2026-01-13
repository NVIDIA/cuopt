/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <cuopt/routing/cython/cython.hpp>
#include <cuopt/routing/solve.hpp>
#include <raft/common/nvtx.hpp>
#include <raft/core/handle.hpp>
#include <rmm/device_buffer.hpp>
#include <routing/generator/generator.hpp>

#include <omp.h>
#include <chrono>

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
 * @brief Wrapper for batch vehicle_routing to expose the API to cython
 *
 * @param data_models Vector of data model pointers
 * @param settings  Composable solver settings object
 * @return std::vector<std::unique_ptr<vehicle_routing_ret_t>>
 */
std::vector<std::unique_ptr<vehicle_routing_ret_t>> call_batch_solve(
  std::vector<routing::data_model_view_t<int, float>*> data_models,
  routing::solver_settings_t<int, float>* settings)
{
  const std::size_t size = data_models.size();
  std::vector<std::unique_ptr<vehicle_routing_ret_t>> list(size);

  // Use OpenMP for parallel execution
  const int max_thread = std::min(static_cast<int>(size), omp_get_max_threads());
  rmm::cuda_stream_pool stream_pool(size, rmm::cuda_stream::flags::non_blocking);

  int device_id = raft::resource::get_device_id(*(data_models[0]->get_handle_ptr()));

#pragma omp parallel for num_threads(max_thread)
  for (std::size_t i = 0; i < size; ++i) {
    // Required in multi-GPU environments to set the device for each thread
    RAFT_CUDA_TRY(cudaSetDevice(device_id));

    auto old_stream = data_models[i]->get_handle_ptr()->get_stream();
    // Make sure previous operations are finished
    data_models[i]->get_handle_ptr()->sync_stream();

    // Set new non blocking stream for current data model
    raft::resource::set_cuda_stream(*(data_models[i]->get_handle_ptr()), stream_pool.get_stream(i));
    auto routing_solution = cuopt::routing::solve(*data_models[i], *settings);

    // Make sure current solve is finished
    stream_pool.get_stream(i).synchronize();

    // Create buffers and reassociate them with the original stream so they
    // outlive the local stream which will be destroyed at end of loop iteration
    auto make_buffer = [old_stream = old_stream](rmm::device_buffer&& buf) {
      buf.set_stream(old_stream);
      return std::make_unique<rmm::device_buffer>(std::move(buf));
    };

    vehicle_routing_ret_t vr_ret{routing_solution.get_vehicle_count(),
                                 routing_solution.get_total_objective(),
                                 routing_solution.get_objectives(),
                                 make_buffer(routing_solution.get_route().release()),
                                 make_buffer(routing_solution.get_order_locations().release()),
                                 make_buffer(routing_solution.get_arrival_stamp().release()),
                                 make_buffer(routing_solution.get_truck_id().release()),
                                 make_buffer(routing_solution.get_node_types().release()),
                                 make_buffer(routing_solution.get_unserviced_nodes().release()),
                                 make_buffer(routing_solution.get_accepted().release()),
                                 routing_solution.get_status(),
                                 routing_solution.get_status_string(),
                                 routing_solution.get_error_status().get_error_type(),
                                 routing_solution.get_error_status().what()};
    list[i] = std::make_unique<vehicle_routing_ret_t>(std::move(vr_ret));

    // Restore the old stream
    raft::resource::set_cuda_stream(*(data_models[i]->get_handle_ptr()), old_stream);
    old_stream.synchronize();
  }

  return list;
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
