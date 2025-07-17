/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#include <utilities/cuda_helpers.cuh>
#include "../solution/solution.cuh"

#include <routing/utilities/cuopt_utils.cuh>

#include <thrust/random.h>
#include <thrust/shuffle.h>

namespace cuopt {
namespace routing {
namespace detail {

template <typename T, typename i_t = int, typename f_t = float>
__device__ inline T shfl_sync(T val,
                              i_t srcLane,
                              i_t width = raft::WarpSize,
                              uint mask = 0xffffffffu)
{
  return __shfl_sync(mask, val, srcLane, width);
}

template <typename T, typename i_t = int>
DI void weighted_random_warp_reduce(raft::random::PCGenerator& rng, T& weight, i_t& idx)
{
#pragma unroll
  for (i_t offset = raft::WarpSize / 2; offset > 0; offset /= 2) {
    T tmp_weight = shfl_sync(weight, raft::laneId() + offset);
    i_t tmp_idx  = shfl_sync(idx, raft::laneId() + offset);
    T sum        = (tmp_weight + weight);
    weight       = sum;
    if (sum != 0) {
      i_t rnd_number = (rng.next_u32() % sum);
      if (rnd_number < tmp_weight) { idx = tmp_idx; }
    }
  }
}

template <typename T>
DI void sorted_insert_intra(T* array, T item, int curr_size)
{
  for (int i = curr_size - 1; i >= 0; --i) {
    if (array[i] < item + i) {
      array[i + 1] = item + i;
      return;
    } else {
      array[i + 1] = array[i];
    }
  }
  array[0] = item;
}

// greedily ejects requests until all of them are feasible
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void eject_until_feasible_kernel(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                            bool add_slack_to_sol,
                                            int64_t seed)
{
  extern __shared__ i_t shmem[];
  __shared__ i_t ejection_counter;
  // keep the currenlty ejected node ids here sorted
  __shared__ i_t ejected_intra_indices[10];
  // if single route is not given, it is for all solution

  i_t route_id = blockIdx.x;
  raft::random::PCGenerator thread_rng(
    seed + (threadIdx.x + blockIdx.x * blockDim.x),
    uint64_t(solution.solution_id * (threadIdx.x + blockIdx.x * blockDim.x)),
    0);
  auto route = solution.routes[route_id];
  init_shmem(ejection_counter, 0);
  auto sh_route =
    route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(shmem, route, route.get_num_nodes());
  __syncthreads();
  sh_route.copy_from(route);
  __syncthreads();
  i_t min_ejected_target            = 0;
  bool is_originally_feasible_route = sh_route.is_feasible();
  bool keep_sorted_ejected          = false;
  if (add_slack_to_sol) {
    const i_t n_infeasible_routes = *solution.n_infeasible_routes;
    keep_sorted_ejected           = n_infeasible_routes < 5;
    if (n_infeasible_routes == 1) {
      min_ejected_target = 4;
    } else if (n_infeasible_routes == 2) {
      min_ejected_target = 3;
    } else if (n_infeasible_routes == 3) {
      min_ejected_target = 2;
    } else if (n_infeasible_routes == 4) {
      min_ejected_target = 2;
    }

    // eject 1 request from a feasible route
    if (is_originally_feasible_route && blockIdx.x < n_infeasible_routes * 1.5) {
      min_ejected_target = 1;
    } else if (is_originally_feasible_route) {
      return;
    }
  } else {
    if (is_originally_feasible_route) { return; }
  }

  int num_breaks_in_route = sh_route.get_num_breaks();

  i_t min_nodes_per_route = add_slack_to_sol
                              ? 1 + num_breaks_in_route + request_info_t<i_t, REQUEST>::size()
                              : 1 + num_breaks_in_route;

  // eject until feasible
  while (!sh_route.is_feasible() ||
         ejection_counter < request_info_t<i_t, REQUEST>::size() * min_ejected_target) {
    // if we already achieved our ejection target, stop keeping the ejected
    keep_sorted_ejected =
      keep_sorted_ejected &&
      (ejection_counter < request_info_t<i_t, REQUEST>::size() * min_ejected_target);
    bool eject_previous_neighbors = !is_originally_feasible_route && sh_route.is_feasible();
    i_t n_nodes_route             = sh_route.get_num_nodes();
    i_t ejected_idx;
    // if only 1 request remained
    if (n_nodes_route <= min_nodes_per_route) break;
    if (eject_previous_neighbors) {
      if (threadIdx.x == 0) {
        // find one random previously ejected node from previously ejected nodes
        i_t random_ejected = thread_rng.next_u32() % ejection_counter;
        // choose an active neighbor from that
        i_t down    = thread_rng.next_u32() % 2;
        ejected_idx = ejected_intra_indices[random_ejected] - random_ejected + down;
        if (ejected_idx == 0) {
          ++ejected_idx;
        } else if (ejected_idx == n_nodes_route) {
          --ejected_idx;
        }
        // just a safe guard! remove this!
        if (ejected_idx < 1 || ejected_idx > n_nodes_route - 1) { ejected_idx = 1; }
      }
    } else {
      const double excess_of_route = sh_route.get_weighted_excess(d_default_weights);
      i_t min_node_per_thread      = -1;
      i_t counter_per_thread       = 1;
      // find the ejection that causes the best reduction in excess
      for (i_t i = threadIdx.x + 1; i < n_nodes_route; i += blockDim.x) {
        const auto prev_node = sh_route.get_node(i - 1);
        const auto next_node = sh_route.get_node(i + 1);
        if (sh_route.requests().node_info[i].is_break()) { continue; }
        f_t total_excess = node_t<i_t, f_t, REQUEST>::total_excess_of_combine(
          prev_node, next_node, sh_route.vehicle_info());
        if (total_excess < excess_of_route || is_originally_feasible_route) {
          if (thread_rng.next_u32() % counter_per_thread == 0) { min_node_per_thread = i; }
          ++counter_per_thread;
        }
      }

      i_t valid_thread = counter_per_thread - 1;
      ejected_idx      = min_node_per_thread;
      if (threadIdx.x < n_nodes_route - 1 && counter_per_thread > 1) { assert(ejected_idx >= 0); }
      weighted_random_warp_reduce(thread_rng, valid_thread, ejected_idx);
    }
    __syncthreads();
    if (threadIdx.x == 0) {
      // This can happen if only break nodes are infeasible, if break nodes
      // are infeasible, we will just remove all the nodes and then breaks
      if (ejected_idx <= 0) {
        for (int i = 1; i < n_nodes_route; ++i) {
          if (!sh_route.requests().node_info[i].is_break()) {
            ejected_idx = i;
            break;
          }
        }
      }

      cuopt_assert(ejected_idx > 0 && ejected_idx < sh_route.get_num_nodes(),
                   "Reduction idx error");

      const bool is_break_node = sh_route.node_info(ejected_idx).is_break();

      [[maybe_unused]] i_t brother_idx;
      if constexpr (REQUEST == request_t::PDP) {
        if (!is_break_node) {
          // get the brother node idx and eject both
          brother_idx = solution.route_node_map.get_intra_route_idx(
            sh_route.requests().brother_info[ejected_idx]);
          // if brother comes before(ie pickup), eject it first
          if (brother_idx < ejected_idx) { raft::swapVals(brother_idx, ejected_idx); }
        }
      }
      sh_route.eject_node(ejected_idx, solution.route_node_map);  // This can eject a break node
      if (keep_sorted_ejected) {
        sorted_insert_intra(ejected_intra_indices, ejected_idx, ejection_counter);
      }
      ++ejection_counter;
      if constexpr (REQUEST == request_t::PDP) {
        if (!is_break_node) {
          sh_route.eject_node(brother_idx - 1, solution.route_node_map);
          if (keep_sorted_ejected) {
            sorted_insert_intra(ejected_intra_indices, brother_idx - 1, ejection_counter);
          }
          ++ejection_counter;
        }
      }
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward(sh_route);
      solution.routes_to_copy[route_id] = 1;
      sh_route.compute_cost();
    }
    __syncthreads();
  }

  // this will be true only when there are breaks. Even after removing all nodes, if we are
  // infeasible, remove the break. When this kernel is called from make_feasible() function, we will
  // work without a break the local search operator will add the breaks later on. For initial
  // solution construction, we will call the add breaks operator again to insert feasible breaks
  if (!sh_route.is_feasible()) {
    if (threadIdx.x == 0) {
      while (sh_route.get_num_breaks() > 0) {
        for (int i = 1; i < sh_route.get_num_nodes(); ++i) {
          if (sh_route.requests().node_info[i].is_break()) {
            sh_route.eject_node(i, solution.route_node_map);
            route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
            route_t<i_t, f_t, REQUEST>::view_t::compute_backward(sh_route);
            solution.routes_to_copy[route_id] = 1;
            sh_route.compute_cost();
            break;
          }
        }
      }
    }
    __syncthreads();
  }

  route.copy_from(sh_route);
}

// single threaded kernel that populates the ep with unserved requests
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          int TPB,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__global__ void populate_ep_with_unserved_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t* ep_index_out)
{
  if (threadIdx.x == 0) {
    for (i_t i = 0; i < solution.get_num_requests(); ++i) {
      auto request_id = solution.get_request(i);
      i_t node_id     = request_id.id();
      if (!solution.route_node_map.is_node_served(node_id)) {
        EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, node_id));
      }
    }
    *ep_index_out = EP.index_;
  }
}

// single threaded kernel that populates the ep with unserved requests
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          int TPB,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__global__ void populate_ep_with_unserved_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t* ep_index_out)
{
  __shared__ i_t pickup_indices[TPB];
  __shared__ i_t delivery_indices[TPB];
  for (i_t offset = 0; offset < solution.get_num_requests(); offset += TPB) {
    i_t len = min(TPB, solution.get_num_requests() - offset);
    block_copy(pickup_indices, solution.problem.pickup_indices.data() + offset, (size_t)len);
    block_copy(delivery_indices, solution.problem.delivery_indices.data() + offset, (size_t)len);
    __syncthreads();

    if (threadIdx.x == 0) {
      for (i_t i = 0; i < len; ++i) {
        auto pickup_node = pickup_indices[i];
        if (!solution.route_node_map.is_node_served(pickup_node)) {
          request_id_t<REQUEST> request_id(pickup_node, delivery_indices[i]);
          EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, request_id));
        }
      }
      *ep_index_out = EP.index_;
    }
    __syncthreads();
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
__global__ void populate_ep_with_selected_unserved_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  raft::device_span<const i_t> unserviced_nodes,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t* ep_index_out,
  int64_t seed)
{
  if (threadIdx.x == 0) {
    for (i_t i = 0; i < unserviced_nodes.size(); ++i) {
      i_t node_id = unserviced_nodes[i];
      cuopt_assert(!solution.route_node_map.is_node_served(node_id), "node should not be served!");
      EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, node_id));
    }
    *ep_index_out = EP.index_;
    // shuffle EP as much as possible
    raft::random::PCGenerator thread_rng(
      seed + (threadIdx.x + blockDim.x * blockIdx.x), unserviced_nodes.size(), 0);
    for (i_t i = 0; i < unserviced_nodes.size() && EP.index_ > 0; ++i) {
      raft::swapVals(EP.stack_[EP.index_], EP.stack_[thread_rng.next_u32() % (EP.index_)]);
    }
  }
}

// single threaded kernel that populates the ep with unserved requests
template <typename i_t,
          typename f_t,
          request_t REQUEST,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
__global__ void populate_ep_with_selected_unserved_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  raft::device_span<const i_t> unserviced_nodes,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  i_t* ep_index_out,
  int64_t seed)
{
  auto& order_info = solution.problem.order_info;
  if (threadIdx.x == 0) {
    for (i_t i = 0; i < unserviced_nodes.size(); ++i) {
      i_t node_id = unserviced_nodes[i];
      cuopt_assert(!solution.route_node_map.is_node_served(node_id), "node should not be served!");
      if (order_info.is_pickup_index[node_id]) {
        request_id_t<REQUEST> request_id(node_id, order_info.pair_indices[node_id]);
        EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, request_id));
      }
    }
    *ep_index_out = EP.index_;

    i_t n_requests = unserviced_nodes.size() / 2;
    // shuffle EP as much as possible
    raft::random::PCGenerator thread_rng(
      seed + (threadIdx.x + blockDim.x * blockIdx.x), n_requests, 0);
    for (i_t i = 0; i < n_requests && EP.index_ > 0; ++i) {
      raft::swapVals(EP.stack_[EP.index_], EP.stack_[thread_rng.next_u32() % (EP.index_)]);
    }
  }
}

// greedly eject nodes until all are feasible, this will result with unserved requests
template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::eject_until_feasible(bool add_slack_to_sol)
{
  raft::common::nvtx::range fun_scope("eject_until_feasible");
  auto stream   = sol_handle->get_stream();
  const i_t TPB = 32;
  compute_max_active();
  size_t sh_size = get_temp_route_shared_size();
  bool is_set    = set_shmem_of_kernel(eject_until_feasible_kernel<i_t, f_t, REQUEST>, sh_size);
  cuopt_assert(is_set, "Not enough shared memory on device for get_all_feasible_insertion!");
  cuopt_expects(is_set, error_type_t::OutOfMemoryError, "Not enough shared memory on device");
  eject_until_feasible_kernel<i_t, f_t, REQUEST>
    <<<n_routes, TPB, sh_size, stream>>>(view(), add_slack_to_sol, seed_generator::get_seed());
  compute_cost();
  global_runtime_checks(false, true, "eject_until_feasible");
}

// finds unserved requests and populates the ejection pool with them
template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::populate_ep_with_unserved(
  ejection_pool_t<request_info_t<i_t, REQUEST>>& EP)
{
  raft::common::nvtx::range fun_scope("populate_ep_with_unserved");
  auto stream = sol_handle->get_stream();
  rmm::device_scalar<i_t> ep_index_out(EP.index_, stream);
  const i_t TPB = 256;
  populate_ep_with_unserved_kernel<i_t, f_t, REQUEST, TPB>
    <<<1, TPB, 0, stream>>>(view(), EP.view(), ep_index_out.data());
  EP.index_ = ep_index_out.value(stream);
  stream.synchronize();
  if (EP.size() > 1) {
    thrust::default_random_engine g(seed_generator::get_seed());
    thrust::shuffle(
      sol_handle->get_thrust_policy(), EP.stack_.begin(), EP.stack_.begin() + EP.size(), g);
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
void solution_t<i_t, f_t, REQUEST>::populate_ep_with_selected_unserved(
  ejection_pool_t<request_info_t<i_t, REQUEST>>& EP, const std::vector<i_t>& unserviced)
{
  raft::common::nvtx::range fun_scope("populate_ep_with_unserved");
  auto stream = sol_handle->get_stream();
  rmm::device_scalar<i_t> ep_index_out(EP.index_, stream);
  constexpr auto const TPB = 256;

  auto unserviced_device = cuopt::device_copy(unserviced, stream);
  auto unserviced_view =
    raft::device_span<i_t const>(unserviced_device.data(), unserviced_device.size());

  populate_ep_with_selected_unserved_kernel<i_t, f_t, REQUEST><<<1, TPB, 0, stream>>>(
    view(), unserviced_view, EP.view(), ep_index_out.data(), seed_generator::get_seed());
  RAFT_CHECK_CUDA(stream);
  EP.index_ = ep_index_out.value(stream);
  stream.synchronize();
}

template void solution_t<int, float, request_t::PDP>::eject_until_feasible(bool);
template void solution_t<int, float, request_t::PDP>::populate_ep_with_unserved(
  ejection_pool_t<request_info_t<int, request_t::PDP>>& EP);
template void solution_t<int, float, request_t::PDP>::populate_ep_with_selected_unserved(
  ejection_pool_t<request_info_t<int, request_t::PDP>>& EP, const std::vector<int>& unserviced);

template void solution_t<int, float, request_t::VRP>::eject_until_feasible(bool);
template void solution_t<int, float, request_t::VRP>::populate_ep_with_unserved(
  ejection_pool_t<request_info_t<int, request_t::VRP>>& EP);
template void solution_t<int, float, request_t::VRP>::populate_ep_with_selected_unserved(
  ejection_pool_t<request_info_t<int, request_t::VRP>>& EP, const std::vector<int>& unserviced);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
