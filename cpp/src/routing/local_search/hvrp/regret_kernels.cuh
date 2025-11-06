/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../../solution/solution.cuh"
#include "../../util_kernels/set_nodes_data.cuh"
#include "../../util_kernels/top_k.cuh"
#include "../move_candidates/move_candidates.cuh"
#include "routing/utilities/cuopt_utils.cuh"
#include "vehicle_assignment.cuh"

#include <routing/utilities/constants.hpp>

namespace cuopt {
namespace routing {
namespace detail {

auto constexpr write_diagonal   = false;
auto constexpr items_per_thread = 2;
double constexpr scaling_factor = 10000.;

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void reset_vehicle_availability_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t const sol,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  auto k                    = blockIdx.x;
  auto vehicle_availability = raft::device_span<i_t>(
    vehicle_assignment.vehicle_availability.data() + k * sol.problem.get_num_buckets(),
    sol.problem.get_num_buckets());

  for (auto i = threadIdx.x; i < vehicle_availability.size(); i += blockDim.x) {
    vehicle_availability[i] = sol.problem.fleet_info.vehicle_availability[i];
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_route_costs_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t const sol,
  typename move_candidates_t<i_t, f_t>::view_t const move_candidates,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  extern __shared__ i_t shmem[];
  auto const n_buckets     = sol.problem.get_num_buckets();
  auto route_id            = blockIdx.x / n_buckets;
  auto inserting_bucket_id = blockIdx.x % n_buckets;

  auto route     = sol.routes[route_id];
  auto bucket_id = sol.problem.fleet_info.buckets[route.get_vehicle_id()];

  if (bucket_id == inserting_bucket_id) {
    vehicle_assignment.route_costs_per_bucket[route_id * n_buckets + inserting_bucket_id] =
      route.get_cost(move_candidates.include_objective, move_candidates.weights);
    return;
  }

  auto sh_route =
    route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(shmem, route, route.get_num_nodes());
  __syncthreads();
  sh_route.copy_from(route);
  __syncthreads();

  auto inserting_vehicle_id = sol.problem.bucket_to_vehicle_id[inserting_bucket_id];
  reset_vehicle_id<i_t, f_t, REQUEST>(sol.problem, sh_route, inserting_vehicle_id);
  __syncthreads();

  if (threadIdx.x == 0) {
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
    sh_route.compute_cost();
    auto new_cost = sh_route.get_cost(move_candidates.include_objective, move_candidates.weights);
    vehicle_assignment.route_costs_per_bucket[route_id * n_buckets + inserting_bucket_id] =
      new_cost;
  }
}

template <typename i_t, typename f_t, request_t REQUEST, int TPB>
__global__ void compute_route_cost_differences_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t const sol,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  extern __shared__ i_t shmem[];

  auto sh_route_costs_per_bucket = raft::device_span<double>(
    (double*)shmem, max(sol.problem.get_num_buckets(), min_bucket_entries));
  auto top_costs = raft::device_span<double>(
    &sh_route_costs_per_bucket.data()[sh_route_costs_per_bucket.size()], min_bucket_entries);
  auto top_indices =
    raft::device_span<i_t>((i_t*)&top_costs.data()[top_costs.size()], min_bucket_entries);

  auto k_regrets = vehicle_assignment.get_k_regrets();
  auto k_iter    = k_regrets - 1;
  auto route_id  = blockIdx.x / k_iter;
  auto k_idx     = blockIdx.x % k_iter;
  auto k         = k_idx + k_min_regrets;

  auto vehicle_availability = raft::device_span<i_t>(
    vehicle_assignment.vehicle_availability.data() + k_idx * sol.problem.get_num_buckets(),
    sol.problem.get_num_buckets());

  if (!vehicle_assignment.run_sort[k_idx]) { return; }
  // Route is already assigned
  if (vehicle_assignment.assignments[k_idx * sol.n_routes + route_id] != -1) { return; }

  for (auto i = threadIdx.x; i < sh_route_costs_per_bucket.size(); i += blockDim.x) {
    sh_route_costs_per_bucket[i] = std::numeric_limits<double>::max();
    if (i < min_bucket_entries) { top_costs[i] = std::numeric_limits<double>::max(); }
  }
  __syncthreads();

  for (auto bucket = threadIdx.x; bucket < sol.problem.get_num_buckets(); bucket += blockDim.x) {
    if (vehicle_availability[bucket] == 0) {
      sh_route_costs_per_bucket[bucket] = std::numeric_limits<double>::max();
    } else {
      sh_route_costs_per_bucket[bucket] =
        vehicle_assignment
          .route_costs_per_bucket[route_id * sol.problem.get_num_buckets() + bucket] *
        scaling_factor;
    }
  }
  __syncthreads();

  raft::device_span<const double> route_bucket_costs = raft::device_span<const double>(
    reinterpret_cast<const double*>(sh_route_costs_per_bucket.data()),
    sh_route_costs_per_bucket.size());
  top_k_indices_per_row<i_t, double, min_bucket_entries, TPB, write_diagonal, items_per_thread>(
    route_id, route_bucket_costs, top_costs, top_indices);

  auto global_offset = k_idx * (sol.n_routes * k_iter) + route_id * k_iter;
  for (auto i = threadIdx.x; i < k_regrets - 1; i += blockDim.x) {
    cuopt_assert(top_costs[i] <= top_costs[i + 1], "Issue with sort");
    vehicle_assignment.cost_differences[global_offset + i] =
      __ddiv_rn(top_costs[i + 1] - top_costs[i], scaling_factor);
  }

  global_offset = k_idx * sol.n_routes;
  // We only need to store the best vehicle type of the route that has highest vehicle_assignment
  vehicle_assignment.top_bucket[global_offset + route_id] = top_indices[0];
  vehicle_assignment.top_cost[global_offset + route_id]   = __ddiv_rn(top_costs[0], scaling_factor);

  if (threadIdx.x == 0) { atomicExch(&vehicle_assignment.run_sort[k_idx], 0); }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void compute_route_vehicle_assignments_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t const sol,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  extern __shared__ i_t shmem[];

  auto k_regrets = vehicle_assignment.get_k_regrets();
  auto k_iter    = k_regrets - 1;
  auto route_id  = blockIdx.x / k_iter;
  auto k_idx     = blockIdx.x % k_iter;
  auto k         = k_idx + k_min_regrets;

  // Route is already assigned
  if (vehicle_assignment.assignments[k_idx * sol.n_routes + route_id] != -1) {
    vehicle_assignment.regret_score_per_route[k_idx * sol.n_routes + route_id] =
      -std::numeric_limits<double>::max();
    return;
  }

  auto global_offset = k_idx * (sol.n_routes * k_iter) + route_id * k_iter;

  double t_vehicle_assignment = 0.;
  for (i_t i = threadIdx.x; i < k - 1; i += blockDim.x) {
    t_vehicle_assignment = vehicle_assignment.cost_differences[global_offset + i];
  }
  auto regret_score = raft::blockReduce(t_vehicle_assignment, (char*)shmem);
  if (threadIdx.x == 0) {
    vehicle_assignment.regret_score_per_route[k_idx * sol.n_routes + route_id] = regret_score;
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void update_assignment_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t const sol,
  typename move_candidates_t<i_t, f_t>::view_t const move_candidates,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  extern __shared__ i_t shmem[];
  __shared__ i_t reduction_index;
  __shared__ double reduction_buf[2 * warp_size];
  __shared__ i_t sh_max_availability;
  __shared__ i_t sh_lock;
  __shared__ i_t sh_route_id;
  __shared__ i_t sh_best_bucket;

  if (threadIdx.x == 0) {
    sh_max_availability = -1;
    sh_route_id         = -1;
    sh_lock             = 0;
    sh_best_bucket      = -1;
  }
  __syncthreads();

  auto k_regrets  = vehicle_assignment.get_k_regrets();
  auto k_iter     = k_regrets - 1;
  auto k_idx      = blockIdx.x;
  auto assignment = raft::device_span<i_t>(
    vehicle_assignment.assignments.data() + k_idx * sol.n_routes, sol.n_routes);
  auto vehicle_availability = raft::device_span<i_t>(
    vehicle_assignment.vehicle_availability.data() + k_idx * sol.problem.get_num_buckets(),
    sol.problem.get_num_buckets());

  auto t_route_id       = -1;
  auto t_best_bucket    = -1;
  auto t_availability   = -1;
  double t_worst_regret = -std::numeric_limits<double>::max();

  for (i_t i = threadIdx.x; i < sol.n_routes; i += blockDim.x) {
    auto tmp_regret = vehicle_assignment.regret_score_per_route[k_idx * sol.n_routes + i];
    if (tmp_regret > t_worst_regret) {
      t_worst_regret = tmp_regret;
      t_route_id     = i;
    }
  }

  // Retrieve worst vehicle_assignments
  i_t curr_thread = threadIdx.x;

  // block_reduce_ranked does a min reduction
  t_worst_regret            = -t_worst_regret;
  double tmp_t_worst_regret = t_worst_regret;
  block_reduce_ranked(tmp_t_worst_regret, curr_thread, reduction_buf, &reduction_index);

  if (t_worst_regret != std::numeric_limits<double>::max() && t_worst_regret == reduction_buf[0]) {
    auto global_offset = k_idx * sol.n_routes;
    t_best_bucket      = vehicle_assignment.top_bucket[global_offset + t_route_id];
    t_availability     = vehicle_availability[t_best_bucket];
    cuopt_assert(t_availability >= 0, "Thread Vehicle bucket should be available");
    atomicMax_block(&sh_max_availability, t_availability);
    __threadfence_block();
  }
  __syncthreads();

  cuopt_assert(sh_max_availability >= 0, "Vehicle bucket should be available");

  // Pick bucket with worst vehicle_assignment and highest availability.
  // There could be multiple threads matching.
  if (t_worst_regret != std::numeric_limits<double>::max() && t_worst_regret == reduction_buf[0] &&
      t_availability == sh_max_availability) {
    if (try_acquire_lock_block(&sh_lock)) {
      assignment[t_route_id] = t_best_bucket;
      --vehicle_availability[t_best_bucket];
      sh_route_id    = t_route_id;
      sh_best_bucket = t_best_bucket;
    }
  }
  __syncthreads();

  cuopt_assert(sh_route_id >= 0, "At least one route should be picked");

  if (threadIdx.x == 0) {
    auto global_offset = k_idx * sol.n_routes;
    vehicle_assignment.assignment_costs[k_idx] +=
      vehicle_assignment.top_cost[global_offset + sh_route_id];
    // One of the vehicle types is no longer available. Top k is needed again.
    if (vehicle_availability[sh_best_bucket] == 0) { vehicle_assignment.run_sort[k_idx] = 1; }
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void find_best_assignment_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  __shared__ double reduction_buf[2 * warp_size];
  __shared__ i_t reduction_index;

  auto best_thread_cost = std::numeric_limits<double>::max();
  for (auto k = threadIdx.x; k < vehicle_assignment.assignment_costs.size(); k += blockDim.x) {
    auto tmp_cost = vehicle_assignment.assignment_costs[k];
    if (tmp_cost < best_thread_cost) { best_thread_cost = tmp_cost; }
  }

  i_t curr_thread = threadIdx.x;
  block_reduce_ranked(best_thread_cost, curr_thread, reduction_buf, &reduction_index);
  if (reduction_index == curr_thread) {
    cuopt_assert(reduction_buf[0] != std::numeric_limits<double>::max(),
                 "No vehicle_assignment solution was found");
    *vehicle_assignment.best_k    = reduction_index;
    *vehicle_assignment.best_cost = reduction_buf[0];
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void update_solution_kernel(
  typename solution_t<i_t, f_t, REQUEST>::view_t sol,
  typename move_candidates_t<i_t, f_t>::view_t const move_candidates,
  typename vehicle_assignment_t<i_t, f_t, REQUEST>::view_t vehicle_assignment)
{
  extern __shared__ i_t shmem[];
  __shared__ i_t sh_vehicle_id;

  // Load best_k solution and update the new solution
  auto k_idx      = *vehicle_assignment.best_k;
  auto assignment = raft::device_span<i_t>(
    vehicle_assignment.assignments.data() + k_idx * sol.n_routes, sol.n_routes);
  auto route_id = blockIdx.x;

  vehicle_assignment.pop_next_vehicle_id(route_id, assignment[route_id], sh_vehicle_id);
  __syncthreads();

  // Add cost of the new route
  auto route = sol.routes[route_id];
  auto sh_route =
    route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(shmem, route, route.get_num_nodes());
  __syncthreads();
  sh_route.copy_from(route);
  __syncthreads();

  reset_vehicle_id<i_t, f_t, REQUEST>(sol.problem, sh_route, sh_vehicle_id);

  if (threadIdx.x == 0) {
    route_t<i_t, f_t, REQUEST>::view_t::compute_forward(sh_route);
    route_t<i_t, f_t, REQUEST>::view_t::compute_backward(sh_route);
    sol.routes_to_copy[sh_route.get_id()]   = 1;
    sol.routes_to_search[sh_route.get_id()] = 1;
  }
  __syncthreads();
  route.copy_from(sh_route);
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
