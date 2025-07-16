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

#include "generator.hpp"
#include "generator_utils.cuh"

#include <thrust/iterator/constant_iterator.h>
#include <thrust/reduce.h>
#include <thrust/shuffle.h>
#include <thrust/sort.h>
#include <algorithm>
#include <cuopt/routing/solve.hpp>
#include <cuopt/routing/solver_settings.hpp>

#include <raft/random/make_blobs.cuh>
#include <raft/random/rng.cuh>

#include <limits>

namespace cuopt {
namespace routing {
namespace generator {

template <typename i_t, typename f_t>
dataset_t<i_t, f_t>::dataset_t(rmm::device_uvector<f_t>& x_pos,
                               rmm::device_uvector<f_t>& y_pos,
                               detail::order_info_t<i_t, f_t>& order_info,
                               detail::fleet_info_t<i_t, f_t>& fleet_info,
                               std::vector<break_dimension_t<i_t>>& break_container)
  : v_x_pos_(std::move(x_pos)),
    v_y_pos_(std::move(y_pos)),
    order_info_(std::move(order_info)),
    fleet_info_(std::move(fleet_info)),
    break_container_(std::move(break_container))
{
}

template <typename i_t, typename f_t>
std::tuple<rmm::device_uvector<f_t>&, rmm::device_uvector<f_t>&>
dataset_t<i_t, f_t>::get_coordinates()
{
  return std::make_tuple(std::ref(v_x_pos_), std::ref(v_y_pos_));
}

template <typename i_t, typename f_t>
detail::fleet_info_t<i_t, f_t>& dataset_t<i_t, f_t>::get_fleet_info()
{
  return fleet_info_;
}

template <typename i_t, typename f_t>
detail::order_info_t<i_t, f_t>& dataset_t<i_t, f_t>::get_order_info()
{
  return order_info_;
}

template <typename i_t, typename f_t>
std::vector<break_dimension_t<i_t>>& dataset_t<i_t, f_t>::get_vehicle_breaks()
{
  return break_container_;
}

template <typename i_t, typename f_t>
dataset_t<i_t, f_t> generate_dataset(raft::handle_t const& handle,
                                     dataset_params_t<i_t, f_t> const& params)
{
  cuopt_expects(params.n_locations >= 2,
                error_type_t::ValidationError,
                "cuOpt needs at least one depot and one location");
  cuopt_expects(
    params.dim >= 0, error_type_t::ValidationError, "Number of dimensions should be positive");
  cuopt_expects(params.drop_return_trips >= 0 && params.drop_return_trips <= 1,
                error_type_t::ValidationError,
                "drop_return_trips should be between 0 and 1");
  cuopt_expects(params.tw_tightness >= 0 && params.tw_tightness <= 1,
                error_type_t::ValidationError,
                "tw_tightness should be between 0 and 1");
  cuopt_expects(
    params.n_shifts >= 1, error_type_t::ValidationError, "There should be at least one shift");
  cuopt_expects(params.break_dim >= 0,
                error_type_t::ValidationError,
                "Number of break dimensions should be positive");
  cuopt_expects(params.n_vehicle_types > 0,
                error_type_t::ValidationError,
                "Number of vehicle types should be positive");
  cuopt_expects(params.n_matrix_types > 0 && params.n_matrix_types <= 2,
                error_type_t::ValidationError,
                "Matrix types supported are cost or transit time only.");

  auto coordinates = generate_coordinates<i_t, f_t>(handle, params);
  auto order_info  = generate_order_info<i_t, f_t>(handle, params);
  auto fleet_info  = generate_fleet_info<i_t, f_t>(handle, params, order_info);
  auto break_container =
    generate_vehicle_breaks<i_t, f_t>(handle, params, order_info, fleet_info.get_num_vehicles());

  return dataset_t<i_t, f_t>(
    std::get<0>(coordinates), std::get<1>(coordinates), order_info, fleet_info, break_container);
}
template <typename i_t, typename f_t>
detail::fleet_order_constraints_t<i_t> generate_fleet_order_constraints(
  raft::handle_t const& handle, dataset_params_t<i_t, f_t> const& params)
{
  auto n_orders   = params.n_locations;
  auto n_vehicles = params.n_locations - 1;

  detail::fleet_order_constraints_t<i_t> fleet_order_constraints(&handle, n_orders, n_vehicles);
  fleet_order_constraints.fill(0);

  raft::random::RngState r(params.seed, raft::random::GenPhilox);

  for (i_t truck_id = 0; truck_id < params.n_locations - 1; ++truck_id) {
    raft::random::uniformInt(
      r,
      fleet_order_constraints.order_service_times.data() + truck_id * n_orders + 1,
      n_orders - 1,
      params.min_service_time,
      params.max_service_time + 1,
      handle.get_stream());
  }
  return fleet_order_constraints;
}

template <typename i_t, typename f_t>
detail::fleet_info_t<i_t, f_t> generate_fleet_info(raft::handle_t const& handle,
                                                   dataset_params_t<i_t, f_t> const& params,
                                                   detail::order_info_t<i_t, f_t> const& order_info)
{
  detail::fleet_info_t<i_t, f_t> fleet_info(&handle, params.n_locations - 1);

  auto coordinates               = generate_coordinates<i_t, f_t>(handle, params);
  fleet_info.matrices_           = generate_matrices<i_t, f_t>(handle, params, coordinates);
  auto fleet_size                = fleet_info.get_num_vehicles();
  fleet_info.v_types_            = generate_vehicle_types<i_t, f_t>(handle, params, fleet_size);
  fleet_info.v_skip_first_trip_  = generate_skip_first_trips<i_t, f_t>(handle, params, fleet_size);
  fleet_info.v_drop_return_trip_ = generate_drop_return_trips<i_t, f_t>(handle, params, fleet_size);
  fleet_info.fleet_order_constraints_ = generate_fleet_order_constraints(handle, params);
  auto [vehicle_earliest, vehicle_latest] =
    generate_vehicle_time_windows<i_t, f_t>(handle, params, order_info, fleet_size);
  fleet_info.v_earliest_time_ = std::move(vehicle_earliest);
  fleet_info.v_latest_time_   = std::move(vehicle_latest);
  fleet_info.v_capacities_    = generate_vehicle_capacities<i_t, f_t>(handle, params, fleet_size);
  return fleet_info;
}

template <typename i_t, typename f_t>
detail::order_info_t<i_t, f_t> generate_order_info(raft::handle_t const& handle,
                                                   dataset_params_t<i_t, f_t> const& params)
{
  detail::order_info_t<i_t, f_t> order_info(&handle, params.n_locations - 1);
  order_info.v_demand_             = generate_demands<i_t, f_t>(handle, params);
  auto [earliest, latest, service] = generate_time_windows<i_t, f_t>(handle, params);
  order_info.v_earliest_time_      = std::move(earliest);
  order_info.v_latest_time_        = std::move(latest);

  return order_info;
}

template <typename i_t, typename f_t>
coordinates_t<f_t> generate_coordinates(raft::handle_t const& handle,
                                        dataset_params_t<i_t, f_t> const& params)
{
  rmm::device_uvector<f_t> v_x_pos(params.n_locations, handle.get_stream());
  rmm::device_uvector<f_t> v_y_pos(params.n_locations, handle.get_stream());
  i_t n_cols = 2;
  i_t n_clusters{};
  f_t cluster_std{};

  if (params.distrib == dataset_distribution_t::RANDOM) {
    n_clusters  = params.n_locations;
    cluster_std = 0.f;
  } else if (params.distrib == dataset_distribution_t::CLUSTERED) {
    n_clusters  = std::ceil(params.n_locations / 10.f);
    cluster_std = n_clusters / 30.f;
  } else if (params.distrib == dataset_distribution_t::RANDOM_CLUSTERED) {
    n_clusters  = std::ceil(params.n_locations / 5.f);
    cluster_std = params.n_locations / 90.f;
  }

  rmm::device_uvector<f_t> out_pos(2 * params.n_locations, handle.get_stream());
  rmm::device_uvector<i_t> center_labels(params.n_locations, handle.get_stream());

  raft::random::make_blobs(out_pos.data(),
                           center_labels.data(),
                           params.n_locations,
                           n_cols,
                           n_clusters,
                           handle.get_stream(),
                           false,
                           (f_t*)nullptr,
                           (f_t*)nullptr,
                           cluster_std,
                           false,
                           params.center_box_min,
                           params.center_box_max,
                           params.seed);

  raft::copy(v_x_pos.data() + 1, out_pos.data() + 1, params.n_locations - 1, handle.get_stream());
  raft::copy(v_y_pos.data() + 1,
             out_pos.data() + params.n_locations + 1,
             params.n_locations - 1,
             handle.get_stream());
  auto depot_x = params.center_box_max / 2.f;
  auto depot_y = depot_x;

  // Update depot to be in the center.
  raft::copy(v_x_pos.data(), &depot_x, 1, handle.get_stream());
  raft::copy(v_y_pos.data(), &depot_y, 1, handle.get_stream());
  return std::make_tuple(std::move(v_x_pos), std::move(v_y_pos));
}

template <typename i_t, typename f_t>
d_mdarray_t<f_t> generate_matrices(raft::handle_t const& handle,
                                   dataset_params_t<i_t, f_t> const& params,
                                   coordinates_t<f_t> const& coordinates)
{
  constexpr f_t asymmetry_scalar = 0.01;
  dim3 n_threads(32, 32);
  dim3 n_blocks(
    std::min((params.n_locations + n_threads.x - 1) / n_threads.x, (unsigned)CUDA_MAX_BLOCKS_2D),
    std::min((params.n_locations + n_threads.y - 1) / n_threads.y, (unsigned)CUDA_MAX_BLOCKS_2D));

  rmm::device_uvector<f_t> cost_matrix(params.n_locations * params.n_locations,
                                       handle.get_stream());
  rmm::device_uvector<f_t> v_rands(params.n_locations * params.n_locations, handle.get_stream());

  detail::build_cost_matrix<i_t, f_t>
    <<<n_blocks, n_threads, 0, handle.get_stream()>>>(cost_matrix.data(),
                                                      std::get<0>(coordinates).data(),
                                                      std::get<1>(coordinates).data(),
                                                      params.n_locations,
                                                      params.asymmetric,
                                                      asymmetry_scalar);
  RAFT_CHECK_CUDA(handle.get_stream());

  auto seed     = params.seed;
  auto matrices = detail::create_device_mdarray<f_t>(
    params.n_locations, params.n_vehicle_types, params.n_matrix_types, handle.get_stream());

  for (auto vehicle_type = 0; vehicle_type < params.n_vehicle_types; ++vehicle_type) {
    for (auto matrix_type = 0; matrix_type < params.n_matrix_types; ++matrix_type) {
      raft::random::RngState r(seed++, raft::random::GenPhilox);
      raft::random::uniform(r,
                            v_rands.data(),
                            v_rands.size(),
                            static_cast<f_t>(1.1),
                            static_cast<f_t>(1.5),
                            handle.get_stream());

      auto matrix_span = matrices.get_cost_matrix(vehicle_type, matrix_type);

      thrust::transform(handle.get_thrust_policy(),
                        cost_matrix.begin(),
                        cost_matrix.end(),
                        v_rands.begin(),
                        matrix_span,
                        thrust::multiplies<f_t>());
    }
  }
  return matrices;
}

template <typename i_t, typename f_t>
rmm::device_uvector<i_t> generate_random_fleet(raft::handle_t const& handle,
                                               size_t n_splits,
                                               size_t fleet_size)
{
  rmm::device_uvector<i_t> v_array(fleet_size, handle.get_stream());
  thrust::fill(handle.get_thrust_policy(), v_array.begin(), v_array.end(), 0);

  auto level_size = v_array.size() / n_splits;
  for (size_t i = 1; i < n_splits; ++i) {
    auto start = v_array.begin() + i * level_size;
    auto end   = start + level_size;
    thrust::fill(handle.get_thrust_policy(), start, end, i);
  }
  thrust::default_random_engine g;
  thrust::shuffle(handle.get_thrust_policy(), v_array.begin(), v_array.end(), g);
  return v_array;
}

template <typename i_t, typename f_t>
rmm::device_uvector<uint8_t> generate_vehicle_types(raft::handle_t const& handle,
                                                    dataset_params_t<i_t, f_t> const& params,
                                                    size_t fleet_size)
{
  return generate_random_fleet<uint8_t, f_t>(handle, params.n_vehicle_types, fleet_size);
}

template <typename i_t, typename f_t>
rmm::device_uvector<cap_i_t> generate_vehicle_capacities(raft::handle_t const& handle,
                                                         dataset_params_t<i_t, f_t> const& params,
                                                         size_t fleet_size)
{
  std::vector<cap_i_t> h_min_capacities(params.dim);
  std::vector<cap_i_t> h_max_capacities(params.dim);
  raft::copy(h_min_capacities.data(), params.min_capacities, params.dim, handle.get_stream());
  raft::copy(h_max_capacities.data(), params.max_capacities, params.dim, handle.get_stream());

  raft::random::RngState r(params.seed, raft::random::GenPhilox);

  rmm::device_uvector<cap_i_t> capacities(fleet_size * params.dim, handle.get_stream());
  for (i_t i = 0; i < params.dim; ++i) {
    raft::random::uniformInt(r,
                             capacities.data() + i * fleet_size,
                             fleet_size,
                             static_cast<cap_i_t>(h_min_capacities[i]),
                             static_cast<cap_i_t>(h_max_capacities[i] + 1),
                             handle.get_stream());
  }
  return capacities;
}

template <typename i_t, typename f_t>
rmm::device_uvector<demand_i_t> generate_demands(raft::handle_t const& handle,
                                                 dataset_params_t<i_t, f_t> const& params)
{
  std::vector<demand_i_t> h_min_demand(params.dim);
  std::vector<demand_i_t> h_max_demand(params.dim);

  raft::copy(h_min_demand.data(), params.min_demand, params.dim, handle.get_stream());
  raft::copy(h_max_demand.data(), params.max_demand, params.dim, handle.get_stream());

  raft::random::RngState r(params.seed, raft::random::GenPhilox);

  rmm::device_uvector<demand_i_t> demands(params.n_locations * params.dim, handle.get_stream());
  for (i_t i = 0; i < params.dim; ++i) {
    demands.set_element_to_zero_async(i * params.n_locations, handle.get_stream());
    raft::random::uniformInt(r,
                             demands.data() + i * params.n_locations + 1,
                             params.n_locations - 1,
                             static_cast<demand_i_t>(h_min_demand[i]),
                             static_cast<demand_i_t>(h_max_demand[i] + 1),
                             handle.get_stream());
  }
  return demands;
}

template <typename i_t, typename f_t>
rmm::device_uvector<bool> generate_drop_return_trips(raft::handle_t const& handle,
                                                     dataset_params_t<i_t, f_t> const& params,
                                                     size_t fleet_size)
{
  rmm::device_uvector<bool> drop_return_trip(fleet_size, handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), drop_return_trip.begin(), drop_return_trip.end(), 0);
  i_t end = params.drop_return_trips * drop_return_trip.size();
  thrust::fill(
    handle.get_thrust_policy(), drop_return_trip.begin(), drop_return_trip.begin() + end, 1);
  thrust::default_random_engine g;
  thrust::shuffle(handle.get_thrust_policy(), drop_return_trip.begin(), drop_return_trip.end(), g);
  return drop_return_trip;
}

template <typename i_t, typename f_t>
rmm::device_uvector<bool> generate_skip_first_trips(raft::handle_t const& handle,
                                                    dataset_params_t<i_t, f_t> const& params,
                                                    size_t fleet_size)
{
  rmm::device_uvector<bool> skip_first_trip(fleet_size, handle.get_stream());

  thrust::fill(handle.get_thrust_policy(), skip_first_trip.begin(), skip_first_trip.end(), 0);
  i_t end = params.drop_return_trips * skip_first_trip.size();
  thrust::fill(
    handle.get_thrust_policy(), skip_first_trip.begin(), skip_first_trip.begin() + end, 1);
  thrust::default_random_engine g;
  thrust::shuffle(handle.get_thrust_policy(), skip_first_trip.begin(), skip_first_trip.end(), g);
  return skip_first_trip;
}

template <typename i_t, typename f_t>
std::vector<break_dimension_t<i_t>> generate_vehicle_breaks(
  raft::handle_t const& handle,
  dataset_params_t<i_t, f_t> const& params,
  detail::order_info_t<i_t, f_t> const& order_info,
  size_t fleet_size)
{
  i_t depot_earliest, depot_latest;
  raft::copy(&depot_earliest, order_info.v_earliest_time_.data(), 1, handle.get_stream());
  raft::copy(&depot_latest, order_info.v_latest_time_.data(), 1, handle.get_stream());

  std::vector<break_dimension_t<i_t>> break_container;
  i_t break_split = std::ceil((depot_latest - depot_earliest) / (f_t)(params.break_dim + 1));
  for (i_t i = 0; i < params.break_dim; ++i) {
    i_t earliest_time =
      std::max(static_cast<i_t>(0.8 * break_split + break_split * i), depot_earliest + 1);
    i_t latest_time =
      std::min(static_cast<i_t>(1.2 * break_split + break_split * i), depot_latest - 1);
    i_t service_time = (latest_time - earliest_time) / 3;
    rmm::device_uvector<i_t> break_earliest(fleet_size, handle.get_stream());
    rmm::device_uvector<i_t> break_latest(fleet_size, handle.get_stream());
    rmm::device_uvector<i_t> break_duration(fleet_size, handle.get_stream());
    thrust::fill(
      handle.get_thrust_policy(), break_earliest.begin(), break_earliest.end(), earliest_time);
    thrust::fill(handle.get_thrust_policy(), break_latest.begin(), break_latest.end(), latest_time);
    thrust::fill(
      handle.get_thrust_policy(), break_duration.begin(), break_duration.end(), service_time);
    break_container.emplace_back(std::make_tuple(
      std::move(break_earliest), std::move(break_latest), std::move(break_duration)));
  }
  return break_container;
}

template <typename i_t, typename f_t>
vehicle_time_window_t<i_t> generate_vehicle_time_windows(
  raft::handle_t const& handle,
  dataset_params_t<i_t, f_t> const& params,
  detail::order_info_t<i_t, f_t> const& order_info,
  size_t fleet_size)
{
  i_t depot_earliest, depot_latest;
  raft::copy(&depot_earliest, order_info.v_earliest_time_.data(), 1, handle.get_stream());
  raft::copy(&depot_latest, order_info.v_latest_time_.data(), 1, handle.get_stream());
  rmm::device_uvector<i_t> vehicle_earliest(fleet_size, handle.get_stream());
  rmm::device_uvector<i_t> vehicle_latest(fleet_size, handle.get_stream());

  i_t shift_earliest = depot_earliest;
  i_t shift_latest   = std::ceil((depot_latest - depot_earliest) / (f_t)params.n_shifts);
  i_t shift_size     = fleet_size / params.n_shifts;

  for (i_t i = 0; i < params.n_shifts; ++i) {
    auto offset = i * shift_size;
    if (i == params.n_shifts - 1) shift_size = fleet_size - (i * shift_size);
    thrust::fill(handle.get_thrust_policy(),
                 vehicle_earliest.begin() + offset,
                 vehicle_earliest.begin() + offset + shift_size,
                 shift_earliest);
    thrust::fill(handle.get_thrust_policy(),
                 vehicle_latest.begin() + offset,
                 vehicle_latest.begin() + offset + shift_size,
                 shift_latest);
    shift_earliest = shift_latest;
    shift_latest += shift_latest;
  }

  // Vehicle tw is inclusive of service time so add it to provide more feasible solutions
  thrust::transform(handle.get_thrust_policy(),
                    vehicle_latest.begin(),
                    vehicle_latest.end(),
                    thrust::make_constant_iterator(params.max_service_time),
                    vehicle_latest.begin(),
                    [] __device__(i_t vehicle_time, i_t service_time) {
                      auto vehicle_latest = vehicle_time + service_time;
                      if (vehicle_latest > std::numeric_limits<i_t>::max())
                        return std::numeric_limits<i_t>::max();
                      return static_cast<i_t>(vehicle_latest);
                    });
  return std::make_tuple(std::move(vehicle_earliest), std::move(vehicle_latest));
}

template <typename i_t, typename f_t>
rmm ::device_uvector<i_t> create_service_time(raft::handle_t const& handle,
                                              dataset_params_t<i_t, f_t> const& params)
{
  rmm::device_uvector<i_t> v_service_time(params.n_locations, handle.get_stream());
  // Create service time based on the average travel time between locations without self
  // locations.
  auto depot_service = 0;
  v_service_time.set_element_async(0, depot_service, handle.get_stream());

  raft::random::RngState r(params.seed, raft::random::GenPhilox);
  raft::random::uniformInt(r,
                           v_service_time.begin() + 1,
                           v_service_time.size() - 1,
                           params.min_service_time,
                           params.max_service_time + 1,
                           handle.get_stream());
  return v_service_time;
}

template <typename i_t, typename f_t>
time_window_t<i_t> generate_time_windows(raft::handle_t const& handle,
                                         dataset_params_t<i_t, f_t> const& params)
{
  rmm::device_uvector<i_t> v_earliest_time(params.n_locations, handle.get_stream());
  rmm::device_uvector<i_t> v_latest_time(params.n_locations, handle.get_stream());

  auto n_vehicles  = params.n_locations - 1;
  auto coordinates = generate_coordinates<i_t, f_t>(handle, params);
  auto matrices    = generate_matrices<i_t, f_t>(handle, params, coordinates);

  data_model_view_t<i_t, f_t> data_model(&handle, params.n_locations, n_vehicles);
  detail::fill_data_model_matrices<i_t, f_t>(data_model, matrices);

  auto time_matrix    = matrices.get_time_matrix(0);
  auto v_service_time = create_service_time<i_t, f_t>(handle, params);
  detail::fill_time_windows<i_t, f_t>
    <<<params.n_locations, 64, 0, handle.get_stream()>>>(time_matrix,
                                                         v_earliest_time.data(),
                                                         v_latest_time.data(),
                                                         params.tw_tightness,
                                                         params.n_locations);
  handle.sync_stream();
  RAFT_CHECK_CUDA(handle.get_stream());

  return std::make_tuple(
    std::move(v_earliest_time), std::move(v_latest_time), std::move(v_service_time));
}

template class dataset_t<int, float>;
template dataset_t<int, float> generate_dataset<int, float>(raft::handle_t const&,
                                                            dataset_params_t<int, float> const&);
template rmm::device_uvector<uint8_t> generate_vehicle_types<int, float>(
  raft::handle_t const&, dataset_params_t<int, float> const&, size_t fleet_size);
}  // namespace generator
}  // namespace routing
}  // namespace cuopt
