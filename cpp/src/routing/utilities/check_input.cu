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

#include <cuopt/error.hpp>
#include <routing/utilities/check_input.hpp>

#include <raft/core/device_mdarray.hpp>
#include <raft/linalg/transpose.cuh>
#include <raft/util/cudart_utils.hpp>

#include <thrust/equal.h>
#include <thrust/extrema.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/logical.h>
#include <cuda/std/functional>

#include <unordered_set>

namespace cuopt {
namespace routing {
namespace detail {

/**
 * @brief Transforms the device vector with absolute value
 * @param v Vector to be transformed
 * @param stream_view Stream view
 */
template <typename T>
void transform_absolute(rmm::device_uvector<T>& v, rmm::cuda_stream_view stream_view)
{
  thrust::transform(
    rmm::exec_policy(stream_view), v.begin(), v.end(), v.begin(), [] __device__(const auto& x) {
      return x < 0 ? -x : x;
    });
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));
}

/**
 * @brief Checks if latest time of a delivery is earlier than earliest time of the pickup
 * @param pickup_indices Device array pointing to pickup indices
 * @param delivery_indices Device array pointing to delivery indices
 * @param earliest_time Earliest time of the nodes
 * @param latest_time Latest time of the nodes
 * @param n_requests Number of pickup delivery pairs
 * @param stream_view Stream view
 */
template <typename i_t>
bool check_pickup_tw(const i_t* pickup_indices,
                     const i_t* delivery_indices,
                     const i_t* earliest_time,
                     const i_t* latest_time,
                     size_t n_requests,
                     rmm::cuda_stream_view stream_view)
{
  typedef typename rmm::device_uvector<const i_t>::iterator IterConstInt;
  thrust::permutation_iterator<IterConstInt, IterConstInt> pickup_iter(earliest_time,
                                                                       pickup_indices);
  thrust::permutation_iterator<IterConstInt, IterConstInt> delivery_iter(latest_time,
                                                                         delivery_indices);
  auto zip_iterator = thrust::make_zip_iterator(thrust::make_tuple(pickup_iter, delivery_iter));
  bool violates_sanity =
    thrust::any_of(rmm::exec_policy(stream_view),
                   zip_iterator,
                   zip_iterator + n_requests,
                   [] __device__(const auto& x) { return thrust::get<0>(x) > thrust::get<1>(x); });
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));
  return !violates_sanity;
}

/**
 * @brief Checks if the demands of pickup delivery pairs are exact negation
 * @param pickup_indices Device array pointing to pickup indices
 * @param delivery_indices Device array pointing to delivery indices
 * @param demands Demands of the nodes
 * @param n_requests Number of pickup delivery pairs
 * @param stream_view Stream view
 */
template <typename i_t>
bool check_pickup_demands(const i_t* pickup_indices,
                          const i_t* delivery_indices,
                          const i_t* demands,
                          size_t n_requests,
                          rmm::cuda_stream_view stream_view)
{
  typedef typename rmm::device_uvector<const i_t>::iterator IterConstInt;
  thrust::permutation_iterator<IterConstInt, IterConstInt> pickup_iter(demands, pickup_indices);
  thrust::permutation_iterator<IterConstInt, IterConstInt> delivery_iter(demands, delivery_indices);
  auto zip_iterator    = thrust::make_zip_iterator(thrust::make_tuple(pickup_iter, delivery_iter));
  bool violates_sanity = thrust::any_of(
    rmm::exec_policy(stream_view),
    zip_iterator,
    zip_iterator + n_requests,
    [] __device__(const auto& x) { return thrust::get<0>(x) != -thrust::get<1>(x); });
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));
  return !violates_sanity;
}

template <typename i_t, typename v_t>
bool check_pdp_values(const i_t* pickup_indices,
                      const i_t* delivery_indices,
                      const v_t* values,
                      size_t n_requests,
                      rmm::cuda_stream_view stream_view)
{
  auto pickup_iter   = thrust::make_permutation_iterator(values, pickup_indices);
  auto delivery_iter = thrust::make_permutation_iterator(values, delivery_indices);
  auto zip_iterator  = thrust::make_zip_iterator(thrust::make_tuple(pickup_iter, delivery_iter));
  bool violates_sanity =
    thrust::any_of(rmm::exec_policy(stream_view),
                   zip_iterator,
                   zip_iterator + n_requests,
                   [] __device__(const auto& x) { return thrust::get<0>(x) != thrust::get<1>(x); });
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));
  return !violates_sanity;
}

/**
 * @brief Checks a matrix for being symmetric
 * @param matrix Matrix residing on device
 * @param width Width of the matrix
 * @param handle_ptr raft handle object
 */
template <typename i_t, typename f_t>
bool is_symmetric_matrix(f_t const* matrix, i_t width, raft::handle_t const* handle_ptr)
{
  i_t mat_size = width * width;

  auto transposed_matrix = raft::make_device_matrix<f_t>(*handle_ptr, width, width);
  raft::linalg::transpose<f_t>(*handle_ptr,
                               const_cast<f_t*>(matrix),
                               transposed_matrix.data_handle(),
                               width,
                               width,
                               handle_ptr->get_stream());
  RAFT_CUDA_TRY(cudaStreamSynchronize(handle_ptr->get_stream()));

  return thrust::equal(handle_ptr->get_thrust_policy(),
                       matrix,
                       matrix + mat_size,
                       static_cast<f_t*>(transposed_matrix.data_handle()));
}

template bool is_symmetric_matrix<int, float>(float const*, int, raft::handle_t const*);

/**
 * @brief Checks if biggest earliest time is smaller than depot latest time
 * @param v_latest_time Latest time vector
 * @param depot_earliest Earliest time of the depot
 * @param stream_view Stream view
 */
template <typename i_t>
bool check_min_latest_with_depot(rmm::device_uvector<i_t>& v_latest_time,
                                 i_t depot_earliest,
                                 rmm::cuda_stream_view stream_view)
{
  i_t min_latest;
  i_t* min_latest_ptr = thrust::min_element(
    rmm::exec_policy(stream_view), v_latest_time.begin() + 1, v_latest_time.end());
  raft::copy(&min_latest, min_latest_ptr, 1, stream_view.value());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));

  return min_latest >= depot_earliest;
}

/**
 * @brief Checks if biggest earliest time is smaller than depot latest time
 * @param v_earliest_time Earliest time vector
 * @param depot_latest Latest time of the depot
 * @param stream_view Stream view
 */
template <typename i_t>
bool check_max_earliest_with_depot(rmm::device_uvector<i_t>& v_earliest_time,
                                   i_t depot_latest,
                                   rmm::cuda_stream_view stream_view)
{
  i_t max_earliest;
  i_t* max_earliest_ptr = thrust::max_element(
    rmm::exec_policy(stream_view), v_earliest_time.begin() + 1, v_earliest_time.end());
  raft::copy(&max_earliest, max_earliest_ptr, 1, stream_view.value());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));

  return max_earliest <= depot_latest;
}

/**
 * @brief Checks if earliest time is smaller than latest time
 * @param v_earliest_time Earliest time vector
 * @param v_latest_time Latest time vector
 * @param stream_view Stream view
 */
template <typename i_t>
bool check_earliest_with_latest(rmm::device_uvector<i_t>& v_earliest_time,
                                rmm::device_uvector<i_t>& v_latest_time,
                                rmm::cuda_stream_view stream_view)
{
  return thrust::equal(rmm::exec_policy(stream_view),
                       v_earliest_time.begin(),
                       v_earliest_time.end(),
                       v_latest_time.begin(),
                       [] __device__(auto x, auto y) { return x <= y; });
}

/**
 * @brief Checks if a given array values are in between reference ranges
 * @param ptr Pointer to the array
 * @param size Size of the array
 * @param min_value Minimum limit
 * @param max_value Maximum limit
 * @param stream_view Stream view
 */
template <typename T, typename RefType>
bool check_min_max_values(const T* ptr,
                          size_t size,
                          const RefType min_value,
                          const RefType max_value,
                          rmm::cuda_stream_view stream_view)
{
  T min, max;
  thrust::pair<const T*, const T*> pair =
    thrust::minmax_element(rmm::exec_policy(stream_view), ptr, ptr + size);
  raft::copy(&min, pair.first, 1, stream_view.value());
  raft::copy(&max, pair.second, 1, stream_view.value());
  RAFT_CUDA_TRY(cudaStreamSynchronize(stream_view.value()));
  return (min >= static_cast<T>(min_value)) && (max <= static_cast<T>(max_value));
}

template <typename i_t>
void check_guess(i_t const* guess_id,
                 i_t const* truck_id,
                 i_t const* route,
                 size_t size,
                 i_t n_locations,
                 i_t n_guesses,
                 i_t fleet_size,
                 bool const* drop_return_trip,
                 bool const* skip_first_trip,
                 rmm::cuda_stream_view stream_view)
{
  cuopt_expects(check_min_max_values(truck_id, size, 0, fleet_size - 1, stream_view),
                error_type_t::ValidationError,
                "Route ids should be between 0 and fleet_size - 1");

  std::vector<i_t> h_guess_id(size);
  std::vector<i_t> h_truck_id(size);
  std::vector<i_t> h_drop_return_trip(fleet_size);
  std::vector<i_t> h_skip_first_trip(fleet_size);
  std::vector<i_t> route_truck_ids;
  std::unordered_set<i_t> guesses;
  cuda::std::identity id;
  rmm::device_uvector<i_t> d_int_drop_return_trip(fleet_size, stream_view);
  rmm::device_uvector<i_t> d_int_skip_first_trip(fleet_size, stream_view);
  thrust::transform(rmm::exec_policy(stream_view),
                    drop_return_trip,
                    drop_return_trip + fleet_size,
                    d_int_drop_return_trip.data(),
                    id);
  raft::update_host(
    h_drop_return_trip.data(), d_int_drop_return_trip.data(), fleet_size, stream_view.value());

  thrust::transform(rmm::exec_policy(stream_view),
                    skip_first_trip,
                    skip_first_trip + fleet_size,
                    d_int_skip_first_trip.data(),
                    id);
  raft::update_host(
    h_skip_first_trip.data(), d_int_skip_first_trip.data(), fleet_size, stream_view.value());

  raft::update_host(h_guess_id.data(), guess_id, size, stream_view);
  raft::update_host(h_truck_id.data(), truck_id, size, stream_view);

  i_t count_non_drop_return = 0;
  i_t count_non_skip_first  = 0;
  i_t prev_guess_id         = -1;
  i_t prev_truck_id         = -1;
  i_t n_routes              = 0;
  i_t curr_guess_routes     = 0;

  for (size_t i = 0; i < size; i++) {
    i_t guess_id = h_guess_id[i];
    i_t truck_id = h_truck_id[i];

    cuopt_expects(curr_guess_routes <= fleet_size,
                  error_type_t::ValidationError,
                  "Number of routes in guess exceeds fleet_size");

    if (prev_guess_id != guess_id) {
      cuopt_expects(guesses.find(guess_id) == guesses.end(),
                    error_type_t::ValidationError,
                    "Guess ids should be contiguous");
      guesses.insert(guess_id);
      curr_guess_routes = 0;
      route_truck_ids.clear();
    }

    // this if block detects the same truck ids on last truc id of previous solution
    // and first truck id of the current solution
    if (truck_id != prev_truck_id || prev_guess_id != guess_id) {
      cuopt_expects(
        std::unique(route_truck_ids.begin(), route_truck_ids.end()) == route_truck_ids.end(),
        error_type_t::ValidationError,
        "Found duplicate truck ids in solution");
      prev_truck_id = truck_id;
      prev_guess_id = guess_id;
      if (h_drop_return_trip[truck_id] == 0) count_non_drop_return++;
      if (h_skip_first_trip[truck_id] == 0) count_non_skip_first++;
      n_routes++;
      ++curr_guess_routes;
      route_truck_ids.push_back(truck_id);
    }
  }
  size_t node_amount = (n_locations - 1) * n_guesses + count_non_skip_first + count_non_drop_return;

  cuopt_expects(size == node_amount,
                error_type_t::ValidationError,
                "Route length not matching the number of nodes.");
  cuopt_expects(check_min_max_values(route, size, 0, n_locations - 1, stream_view),
                error_type_t::ValidationError,
                "Node ids need to be between 0 and n_locations - 1.");
}

/**
 * @brief Checks if there is a circular precedence reference in the given precedences arrays.
 *
 * @param node_id Node id that has precedence requirements
 * @param preceding_nodes Nodes that should precede this node id.
 * @param n_preceding_nodes Number of preceding nodes.
 * @param precedence Map that holds previously added precedences.
 * @param stream_view Stream view.
 * @return true If there is no circular reference.
 * @return false If there is a circular reference.
 */
template <typename i_t>
bool check_no_circular_precedence(i_t node_id,
                                  i_t const* preceding_nodes,
                                  i_t n_preceding_nodes,
                                  std::unordered_map<i_t, std::pair<i_t const*, i_t>> precedence,
                                  rmm::cuda_stream_view stream_view)
{
  for (const auto& pair : precedence) {
    auto other_node   = pair.first;
    auto other_prec   = pair.second;
    auto prec_ptr     = other_prec.first;
    auto n_prec_nodes = other_prec.second;
    auto end_ptr      = prec_ptr + n_prec_nodes;
    auto iter_end     = thrust::find(rmm::exec_policy(stream_view), prec_ptr, end_ptr, node_id);
    // if a precedence found for this node in other vec
    // check if there is a reference the other way
    if (iter_end != end_ptr) {
      end_ptr  = preceding_nodes + n_preceding_nodes;
      iter_end = thrust::find(rmm::exec_policy(stream_view), preceding_nodes, end_ptr, other_node);
      if (iter_end != end_ptr) { return false; }
    }
  }
  return true;
}

/**
 * @brief Checks if an item exists in the given device array.
 *
 * @tparam T Generic type.
 * @param item_id Item id to be checked
 * @param device_ptr Array to be searched.
 * @param n_items Item count in the array.
 * @param stream_view Stream view
 * @return bool Whether the item exists
 */
template <typename T>
bool check_exists(T item_id, T const* device_ptr, T n_items, rmm::cuda_stream_view stream_view)
{
  auto end_ptr  = device_ptr + n_items;
  auto iter_end = thrust::find(rmm::exec_policy(stream_view), device_ptr, end_ptr, item_id);
  return iter_end != end_ptr;
}

template bool check_min_max_values<uint8_t, int>(const uint8_t* ptr,
                                                 size_t size,
                                                 const int min_value,
                                                 const int max_value,
                                                 rmm::cuda_stream_view stream_view);

template bool check_min_max_values<int, int>(const int* ptr,
                                             size_t size,
                                             const int min_value,
                                             const int max_value,
                                             rmm::cuda_stream_view stream_view);

template bool check_min_max_values<int, int16_t>(const int* ptr,
                                                 size_t size,
                                                 const int16_t min_value,
                                                 const int16_t max_value,
                                                 rmm::cuda_stream_view stream_view);

template bool check_min_max_values<int, uint16_t>(const int* ptr,
                                                  size_t size,
                                                  const uint16_t min_value,
                                                  const uint16_t max_value,
                                                  rmm::cuda_stream_view stream_view);

template bool check_min_max_values<float, float>(const float* ptr,
                                                 size_t size,
                                                 const float min_value,
                                                 const float max_value,
                                                 rmm::cuda_stream_view stream_view);

template bool check_min_max_values<double, double>(const double* ptr,
                                                   size_t size,
                                                   const double min_value,
                                                   const double max_value,
                                                   rmm::cuda_stream_view stream_view);

template void transform_absolute<int>(rmm::device_uvector<int>& v,
                                      rmm::cuda_stream_view stream_view);

template bool check_no_circular_precedence<int>(
  int node_id,
  int const* preceding_nodes,
  int n_preceding_nodes,
  std::unordered_map<int, std::pair<int const*, int>> precedence,
  rmm::cuda_stream_view stream_view);

template bool check_exists<int>(int item_id,
                                int const* device_ptr,
                                int n_items,
                                rmm::cuda_stream_view stream_view);

template bool check_earliest_with_latest<int>(rmm::device_uvector<int>&,
                                              rmm::device_uvector<int>&,
                                              rmm::cuda_stream_view);
template bool check_max_earliest_with_depot<int>(rmm::device_uvector<int>&,
                                                 int,
                                                 rmm::cuda_stream_view);
template bool check_pickup_tw<int>(
  int const*, int const*, int const*, int const*, unsigned long, rmm::cuda_stream_view);
template bool check_pickup_demands<int>(
  int const*, int const*, int const*, unsigned long, rmm::cuda_stream_view);

template bool check_pdp_values<int, uint8_t>(
  int const*, int const*, uint8_t const*, unsigned long, rmm::cuda_stream_view);
template bool check_pdp_values<int, int>(
  int const*, int const*, int const*, unsigned long, rmm::cuda_stream_view);
template bool check_pdp_values<int, float>(
  int const*, int const*, float const*, unsigned long, rmm::cuda_stream_view);

template bool check_min_latest_with_depot<int>(rmm::device_uvector<int>&,
                                               int,
                                               rmm::cuda_stream_view);
template void check_guess<int>(int const*,
                               int const*,
                               int const*,
                               unsigned long,
                               int,
                               int,
                               int,
                               bool const*,
                               bool const*,
                               rmm::cuda_stream_view);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
