/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2021-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <raft/core/handle.hpp>
#include <rmm/device_uvector.hpp>
#include <rmm/exec_policy.hpp>

#include <unordered_map>

namespace cuopt {
namespace routing {
namespace detail {

template <typename i_t, typename f_t>
bool is_symmetric_matrix(f_t const* matrix, i_t width, raft::handle_t const* handle_ptr);

template <typename i_t>
bool check_min_latest_with_depot(rmm::device_uvector<i_t>& v_latest_time,
                                 i_t depot_earliest,
                                 rmm::cuda_stream_view stream_view);
template <typename i_t>
bool check_max_earliest_with_depot(rmm::device_uvector<i_t>& v_earliest_time,
                                   i_t depot_latest,
                                   rmm::cuda_stream_view stream_view);
template <typename i_t>
bool check_earliest_with_latest(rmm::device_uvector<i_t>& v_earliest_time,
                                rmm::device_uvector<i_t>& v_latest_time,
                                rmm::cuda_stream_view stream_view);
template <typename T, typename RefType>
bool check_min_max_values(const T* ptr,
                          size_t size,
                          const RefType min_value,
                          const RefType max_value,
                          rmm::cuda_stream_view stream_view);

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
                 rmm::cuda_stream_view stream_view);

template <typename i_t>
bool check_pickup_tw(const i_t* pickup_indices,
                     const i_t* delivery_indices,
                     const i_t* earliest_time,
                     const i_t* latest_time,
                     size_t n_requests,
                     rmm::cuda_stream_view stream_view);

template <typename i_t>
bool check_pickup_demands(const i_t* pickup_indices,
                          const i_t* delivery_indices,
                          const i_t* demands,
                          size_t n_requests,
                          rmm::cuda_stream_view stream_view);

template <typename i_t, typename v_t>
bool check_pdp_values(const i_t* pickup_indices,
                      const i_t* delivery_indices,
                      const v_t* values,
                      size_t n_requests,
                      rmm::cuda_stream_view stream_view);

template <typename i_t>
bool check_no_circular_precedence(i_t node_id,
                                  i_t const* preceding_nodes,
                                  i_t n_preceding_nodes,
                                  std::unordered_map<i_t, std::pair<i_t const*, i_t>> precedence,
                                  rmm::cuda_stream_view stream_view);

template <typename T>
bool check_exists(T item_id, T const* device_ptr, T n_items, rmm::cuda_stream_view stream_view);

template <typename T>
void transform_absolute(rmm::device_uvector<T>& v, rmm::cuda_stream_view stream_view);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
