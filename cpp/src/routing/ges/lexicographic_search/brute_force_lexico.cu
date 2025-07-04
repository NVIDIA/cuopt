/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights
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

#include "../../solution/solution.cuh"
#include "../ejection_pool.cuh"
#include "../guided_ejection_search.cuh"
#include "lexicographic_search.cuh"

#include <algorithm>

namespace cuopt {
namespace routing {
namespace detail {

constexpr int b_k_max = 3;

// in order to have trivially copyable
template <int size>
struct sequence_t {
  HD sequence_t()
  {
    for (int i = 0; i < size; ++i) {
      s[i] = std::numeric_limits<int>::max();
    }
  }
  int s[size];
};

template <typename i_t, int size>
DI void insertion_sort(i_t array[size], i_t n_ejections)
{
  i_t j, key;
  for (i_t i = 1; i < n_ejections; i++) {
    key = array[i];
    j   = i - 1;
    while (j >= 0 && array[j] > key) {
      array[j + 1] = array[j];
      j            = j - 1;
    }
    array[j + 1] = key;
  }
}

// this file and kernel is used for testing and validation
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void brute_force_lexico_kernel(
  sequence_t<b_k_max>* sequence,
  const typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const typename route_t<i_t, f_t, REQUEST>::view_t route,
  i_t n_ejections,
  const request_info_t<i_t, REQUEST>* request_id,
  uint32_t* __restrict__ global_min_p,
  i_t* __restrict__ global_sequence,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  const i_t* __restrict__ p_scores)
{
  extern __shared__ i_t shmem[];

  auto& route_node_map = solution.route_node_map;

  typename route_node_map_t<i_t>::view_t s_route_node_map;
  i_t* sh_ptr = shmem;
  thrust::tie(s_route_node_map, sh_ptr) =
    route_node_map_t<i_t>::view_t::create_shared(sh_ptr, route_node_map.size());
  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    sh_ptr, route, route.get_num_nodes() + 2);

  __syncthreads();
  s_route.copy_from(route);
  __syncthreads();
  auto request_node  = solution.get_request(request_id);
  auto pickup_node   = request_node.pickup;
  auto delivery_node = request_node.delivery;
  // the max p score that is allowed it the requests p score
  i_t min_p_score = p_scores[pickup_node.id()] - 1;

  for (i_t pickup_idx = 0; pickup_idx < route.get_num_nodes(); ++pickup_idx) {
    for (i_t delivery_idx = pickup_idx; delivery_idx < route.get_num_nodes(); ++delivery_idx) {
      i_t curr_p_score = 0;
      s_route.copy_from(route);
      s_route_node_map.copy_from(route_node_map);
      __syncthreads();
      if (threadIdx.x == 0) {
        request_id_t<REQUEST> request_locations(pickup_idx, delivery_idx);
        s_route.template insert_request<REQUEST>(
          request_locations, request_node, s_route_node_map, true);
        sequence_t<2 * b_k_max> sequence_including_delivery;
        i_t counter      = 0;
        bool pd_feasible = true;
        for (i_t pickup_ejection_idx = 0; pickup_ejection_idx < n_ejections;
             ++pickup_ejection_idx) {
          i_t pickup_intra_idx = sequence[blockIdx.x].s[pickup_ejection_idx];
          auto curr_node       = s_route.get_node(pickup_intra_idx);
          curr_p_score += p_scores[curr_node.id()];
          i_t delivery_intra_idx =
            s_route_node_map.get_intra_route_idx(s_route.requests().brother_info[pickup_intra_idx]);
          bool are_we_ejecting_the_current_request = curr_node.id() == pickup_node.id();
          if (!curr_node.request.is_pickup() || are_we_ejecting_the_current_request) {
            pd_feasible = false;
            break;
          }
          sequence_including_delivery.s[counter] = pickup_intra_idx;
          counter++;
          sequence_including_delivery.s[counter] = delivery_intra_idx;
          counter++;
        }

        if (!pd_feasible || curr_p_score > min_p_score) { continue; }
        cuopt_assert(2 * n_ejections == counter, "counter is wrong");
        insertion_sort<i_t, 2 * b_k_max>(sequence_including_delivery.s, counter);
        for (i_t ejection_idx = 0; ejection_idx < counter; ++ejection_idx) {
          i_t adjusted_idx = sequence_including_delivery.s[ejection_idx] - ejection_idx;
          assert(!s_route.get_node(adjusted_idx).node_info().is_break());
          s_route.eject_node(adjusted_idx, s_route_node_map, true);
        }
        route_t<i_t, f_t, REQUEST>::view_t::compute_forward(s_route);
        route_t<i_t, f_t, REQUEST>::view_t::compute_backward(s_route);
        bool is_feasible = true;
        for (i_t idx = 0; idx < s_route.get_num_nodes(); ++idx) {
          is_feasible =
            is_feasible && s_route.get_node(idx).forward_feasible(s_route.vehicle_info());
          is_feasible =
            is_feasible && s_route.get_node(idx).backward_feasible(s_route.vehicle_info());
        }
        if (is_feasible) {
          min_p_score = curr_p_score;
          atomicMin(global_min_p,
                    bit_cast<uint32_t, p_val_seq_t>(p_val_seq_t(curr_p_score, counter)));
          while (atomicExch(solution.lock, 1) != 0)
            ;  // acquire
          __threadfence();
          if (global_min_p[0] ==
              bit_cast<uint32_t, p_val_seq_t>(p_val_seq_t(curr_p_score, counter))) {
            cuopt_assert(curr_p_score > 0, "P score should be greater than 0 ");
            cuopt_assert(curr_p_score < p_scores[request_id->info.node()],
                         "P score should be smaller or equal than requests ");
            // Sad uncoallsced global writes of route then the thread found best sequence
            global_sequence[0] = counter;
            global_sequence[1] = curr_p_score;
            global_sequence[2] = pickup_idx;
            global_sequence[3] = delivery_idx;
            for (i_t i = 0; i < counter; ++i) {
              global_sequence[i + 4] = sequence_including_delivery.s[i];
            }
          }
          __threadfence();
          *(solution.lock) = 0;  // release
        }
      }
    }
  }
}

template <typename i_t>
std::vector<sequence_t<b_k_max>> comb(i_t n, i_t k)
{
  std::string bitmask(k, 1);  // K leading 1's
  bitmask.resize(n, 0);       // N-K trailing 0's
  std::vector<sequence_t<b_k_max>> combinations;
  // print integers and permute bitmask
  do {
    i_t idx = 0;
    sequence_t<b_k_max> seq;
    for (i_t i = 1; i < n; ++i)  // [1..N-1] integers
    {
      if (bitmask[i]) {
        seq.s[idx] = i;
        ++idx;
      }
    }
    if (idx == k) combinations.push_back(seq);
  } while (std::prev_permutation(bitmask.begin(), bitmask.end()));
  return combinations;
}

template <typename i_t, typename f_t, request_t REQUEST>
std::vector<i_t> guided_ejection_search_t<i_t, f_t, REQUEST>::brute_force_lexico(
  solution_t<i_t, f_t, REQUEST>& sol, request_info_t<i_t, REQUEST>* __restrict__ req)
{
  auto stream          = sol.sol_handle->get_stream();
  i_t TPB              = 32;
  const i_t zero       = 0;
  const auto value_max = std::numeric_limits<typename decltype(global_min_p_)::value_type>::max();
  sol.d_lock.set_value_async(zero, stream);
  rmm::device_uvector<i_t> global_sequence(2 * b_k_max + lexico_result_buffer_size, stream);
  rmm::device_scalar<uint32_t> global_min_p(value_max, stream);
  // for each route generate combinations and run a kernel
  for (i_t r_id = 0; r_id < sol.n_routes; ++r_id) {
    auto& route = sol.get_route(r_id);
    for (i_t n_ejections = 1; n_ejections <= b_k_max; ++n_ejections) {
      // generate combination of ejections
      auto combinations = comb(route.n_nodes.value(stream) + 2, n_ejections);
      rmm::device_uvector<sequence_t<b_k_max>> d_combinations(combinations.size(), stream);
      raft::copy(d_combinations.data(), combinations.data(), combinations.size(), stream);
      // there would be maximum 2 insertinos (request PD pair)
      size_t shared_size_for_route         = sol.get_temp_route_shared_size(2);
      size_t shared_size_for_intra_indices = (sol.get_num_orders()) * 2 * sizeof(i_t);
      size_t shared_size                   = shared_size_for_route + shared_size_for_intra_indices;
      i_t n_blocks                         = combinations.size();
      brute_force_lexico_kernel<i_t, f_t, REQUEST>
        <<<n_blocks, TPB, shared_size, stream>>>(d_combinations.data(),
                                                 sol.view(),
                                                 route.view(),
                                                 n_ejections,
                                                 req,
                                                 global_min_p.data(),
                                                 global_sequence.data(),
                                                 EP.view(),
                                                 p_scores_.data());
      // copy the best result and keep it here
      sol.sol_handle->sync_stream();
    }
  }
  if (global_min_p.value(stream) != value_max) {
    std::vector<i_t> sequence(global_sequence.element(0, stream) + 3);
    // copy including pickup and delivery
    raft::copy(sequence.data(), global_sequence.data() + 1, sequence.size(), stream);
    stream.synchronize();
    return sequence;
  }
  return std::vector<i_t>{};
}

template std::vector<int> guided_ejection_search_t<int, float, request_t::PDP>::brute_force_lexico(
  solution_t<int, float, request_t::PDP>& sol,
  request_info_t<int, request_t::PDP>* __restrict__ req);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
