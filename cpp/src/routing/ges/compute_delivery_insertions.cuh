/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../local_search/delivery_insertion.cuh"
#include "../node/node.cuh"
#include "../solution/solution.cuh"
#include "found_solution.cuh"

#include <cub/block/block_merge_sort.cuh>
#include <cub/block/block_scan.cuh>
#include <utilities/seed_generator.cuh>

namespace cuopt {
namespace routing {
namespace detail {

enum class insert_mode_t { SQUEEZE = 0, GES, LOCAL_SEARCH };

struct device_less_t {
  template <typename DataType>
  DI bool operator()(const DataType& lhs, const DataType& rhs)
  {
    return lhs < rhs;
  }
};

template <int BLOCK_SIZE, typename i_t, request_t REQUEST>
DI void sort_to_delete(i_t* to_delete, i_t fragment_size)
{
  // Suppose fragment_size * 2 will never exceed blockDim
  cuopt_assert((fragment_size * request_info_t<i_t, REQUEST>::size()) <= blockDim.x,
               "Support only fragment size up to blockDim");
  typedef cub::BlockMergeSort<i_t, BLOCK_SIZE, 1> BlockMergeSort;
  __shared__ typename BlockMergeSort::TempStorage temp_storage_shuffle;
  i_t thread_val[1];
  if (threadIdx.x < fragment_size * request_info_t<i_t, REQUEST>::size())
    thread_val[0] = to_delete[threadIdx.x];
  else
    thread_val[0] = std::numeric_limits<i_t>::max();  // So that it ends up at the end of the sort
  BlockMergeSort(temp_storage_shuffle).Sort(thread_val, device_less_t());
  __syncthreads();  // TODO Useful ?
  if (threadIdx.x < fragment_size * request_info_t<i_t, REQUEST>::size())
    to_delete[threadIdx.x] = thread_val[0];
  __syncthreads();
}

template <typename i_t>
DI bool is_sorted(i_t const* array, i_t size)
{
  cuopt_assert(size < blockDim.x, "Should be smaller than blockDim");
  __shared__ bool is_sorted;
  is_sorted = true;
  __syncthreads();
  if (threadIdx.x < size - 1) {
    if (array[threadIdx.x] >= array[threadIdx.x + 1]) { is_sorted = false; }
  }
  __syncthreads();
  return is_sorted;
}

template <typename i_t>
DI bool is_positive(i_t const* array, i_t size)
{
  cuopt_assert(size < blockDim.x, "Should be smaller than blockDim");
  __shared__ bool is_stricly_positive;
  is_stricly_positive = true;
  __syncthreads();
  if (threadIdx.x < size) {
    if (array[threadIdx.x] <= 0) { is_stricly_positive = false; }
  }
  __syncthreads();
  return is_stricly_positive;
}

template <typename i_t, int BLOCK_SIZE>
DI i_t binary_block_reduce(int val)
{
  static_assert(BLOCK_SIZE <= 1024);
  cuopt_assert(val == 0 || val == 1, "Binary block reduce only acceptes 0 or 1");
  __shared__ i_t shared[BLOCK_SIZE / raft::WarpSize];
  const uint32_t mask                 = __ballot_sync(~0, val);
  const uint32_t n_deletable_requests = __popc(mask);

  // Each first thread of the warp
  if (threadIdx.x % raft::WarpSize == 0)
    shared[threadIdx.x / raft::WarpSize] = n_deletable_requests;
  __syncthreads();

  val = (threadIdx.x < BLOCK_SIZE / raft::WarpSize) ? shared[threadIdx.x] : 0;

  // Warp reduce on shared array
  if (threadIdx.x < raft::WarpSize)
    return raft::warpReduce(val);
  else  // Only first warp gets the results
    return -1;
}

template <typename i_t, typename f_t, request_t REQUEST, insert_mode_t insert_mode>
DI bool find_node_insertion(const typename route_t<i_t, f_t, REQUEST>::view_t& curr_route,
                            request_node_t<i_t, f_t, REQUEST>& request_node,
                            i_t node_insertion_idx,
                            infeasible_cost_t const& weights,
                            double excess_limit)
{
  const i_t route_length = curr_route.get_num_nodes();
  auto& node             = request_node.node();

  if (node_insertion_idx < route_length) {
    node_t<i_t, f_t, REQUEST> current = curr_route.get_node(node_insertion_idx);
    // Insert node after node_insertion_idx
    current.calculate_forward_all(node, curr_route.vehicle_info());
    if constexpr (insert_mode == insert_mode_t::SQUEEZE) { return true; }

    if constexpr (insert_mode == insert_mode_t::GES) {
      if (node.forward_feasible(curr_route.vehicle_info()) &&
          node_t<i_t, f_t, REQUEST>::feasible_time_combine(
            node, curr_route.get_node(node_insertion_idx + 1), curr_route.vehicle_info())) {
        return true;
      }
    }

    if constexpr (insert_mode == insert_mode_t::LOCAL_SEARCH) {
      if (node.time_dim.forward_feasible(
            curr_route.vehicle_info(), weights[dim_t::TIME], excess_limit) &&
          node_t<i_t, f_t, REQUEST>::time_combine(node,
                                                  curr_route.get_node(node_insertion_idx + 1),
                                                  curr_route.vehicle_info(),
                                                  weights,
                                                  excess_limit)) {
        return true;
      }
    }
  }
  return false;
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          pick_mode_t pick_mode,
          insert_mode_t insert_mode,
          std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
DI void find_request_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& view,
                               const typename route_t<i_t, f_t, REQUEST>::view_t& curr_route,
                               request_node_t<i_t, f_t, REQUEST>& request_node,
                               i_t node_insertion_idx,
                               bool include_objective,
                               infeasible_cost_t const& weights,
                               double excess_limit,
                               cand_t* feasible_move,
                               raft::random::PCGenerator* rng = nullptr)
{
  auto insert_node = find_node_insertion<i_t, f_t, REQUEST, insert_mode>(
    curr_route, request_node, node_insertion_idx, weights, excess_limit);

  if (insert_node) {
    auto node = request_node.node();
    // Combine 2 fragments
    node_t<i_t, f_t, REQUEST> current = curr_route.get_node(node_insertion_idx + 1);
    bool fragment_combine_feasible    = node_t<i_t, f_t, REQUEST>::combine(
      node, current, curr_route.vehicle_info(), weights, excess_limit);

    const i_t route_id = curr_route.get_id();

    const auto old_objective_cost     = view.routes[route_id].get_objective_cost();
    const auto old_infeasibility_cost = view.routes[route_id].get_infeasibility_cost();
    if (fragment_combine_feasible) {
      // Feasible insertion found
      if constexpr (pick_mode == pick_mode_t::COST_DELTA) {
        double insertion_cost_delta =
          node.calculate_forward_all_and_delta(current,
                                               curr_route.vehicle_info(),
                                               include_objective,
                                               weights,
                                               old_objective_cost,
                                               old_infeasibility_cost);
        // atomically update the feasible_move for this request
        update_best_cand<i_t, f_t>(
          insertion_cost_delta, feasible_move, node_insertion_idx, node_insertion_idx);
      } else if constexpr (pick_mode == pick_mode_t::PROBABILITY) {
        update_random_cand<i_t, f_t>(rng, feasible_move, node_insertion_idx, node_insertion_idx);
      }
    }
  }
}

template <typename i_t,
          typename f_t,
          request_t REQUEST,
          pick_mode_t pick_mode,
          insert_mode_t insert_mode,
          std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
DI void find_request_insertion(typename solution_t<i_t, f_t, REQUEST>::view_t& view,
                               const typename route_t<i_t, f_t, REQUEST>::view_t& curr_route,
                               request_node_t<i_t, f_t, REQUEST>& request_node,
                               i_t node_insertion_idx,
                               bool include_objective,
                               infeasible_cost_t const& weights,
                               double excess_limit,
                               cand_t* feasible_move,
                               raft::random::PCGenerator* rng = nullptr)
{
  auto insert_node = find_node_insertion<i_t, f_t, REQUEST, insert_mode>(
    curr_route, request_node, node_insertion_idx, weights, excess_limit);
  if (insert_node) {
    auto pickup_node   = request_node.pickup;
    auto delivery_node = request_node.delivery;
    find_delivery_insertion<i_t, f_t, REQUEST, pick_mode>(view,
                                                          pickup_node,
                                                          delivery_node,
                                                          node_insertion_idx,
                                                          curr_route,
                                                          include_objective,
                                                          weights,
                                                          excess_limit,
                                                          feasible_move,
                                                          rng);
  }
}

template <int BLOCK_SIZE, bool store_all, typename i_t, typename f_t, request_t REQUEST>
DI void find_all_delivery_insertions(typename solution_t<i_t, f_t, REQUEST>::view_t& view,
                                     const typename route_t<i_t, f_t, REQUEST>::view_t& curr_route,
                                     const request_info_t<i_t, REQUEST>* request_id,
                                     feasible_move_t feasible_candidates,
                                     int64_t seed,
                                     [[maybe_unused]] i_t p_score        = 0,
                                     [[maybe_unused]] i_t frag_to_delete = 0,
                                     [[maybe_unused]] i_t frag_step      = 0)
{
  raft::random::PCGenerator thread_rng(
    seed + (threadIdx.x + blockIdx.x * blockDim.x),
    uint64_t(view.solution_id * (threadIdx.x + blockIdx.x * blockDim.x)),
    0);
  [[maybe_unused]] __shared__ i_t atomic_min_random_counter;
  if (threadIdx.x == 0) { atomic_min_random_counter = 1; }
  __syncthreads();

  const i_t route_length = curr_route.get_num_nodes();
  cuopt_assert(curr_route.is_valid(), "Invalid route");
  auto request_node = view.get_request(request_id);

  // If multiple are found per thread they are selected through a conditional probability
  // Block stride loop is handled internally to allow reduce block wise (using i_t i = threadIdx.x
  // would make us loose threads)
  for (i_t i = 0; i < route_length; i += blockDim.x) {
    cand_t feasible_move = cand_t{0, 0, static_cast<uint64_t>(1)};

    const i_t pickup_insertion_idx = threadIdx.x + i;
    const bool include_objective   = true;
    find_request_insertion<i_t, f_t, REQUEST, pick_mode_t::PROBABILITY, insert_mode_t::GES>(
      view,
      curr_route,
      request_node,
      pickup_insertion_idx,
      // set default weights and 0 excess
      include_objective,
      d_default_weights,
      std::numeric_limits<f_t>::epsilon(),
      &feasible_move,
      &thread_rng);

    i_t insertion_1, insertion_2, insertion_3, insertion_4;
    double cost_delta;
    move_candidates_t<i_t, f_t>::get_candidate(
      feasible_move, insertion_1, insertion_2, insertion_3, insertion_4, cost_delta);

    bool th_move_found = feasible_move.cost_counter.counter > 1;
    i_t insertion_pos  = insertion_2;
    if constexpr (store_all) {
      // Increase size in global struct
      i_t number_of_feasible_moves = binary_block_reduce<i_t, BLOCK_SIZE>(th_move_found);
      if (threadIdx.x == 0 && number_of_feasible_moves > 0) {
        atomicAdd(feasible_candidates.size_, number_of_feasible_moves);
      }

      // Write to global
      if (th_move_found) {
        cuopt_assert(insertion_pos < std::numeric_limits<uint16_t>::max(),
                     "Delivery location value should be < 65536");
        cuopt_assert(curr_route.get_id() < view.n_routes,
                     "Can't write a route id greater than number of routes");

        auto found_sol                  = found_sol_t(0,
                                     static_cast<uint16_t>(curr_route.get_id()),
                                     static_cast<uint16_t>(pickup_insertion_idx),
                                     static_cast<uint16_t>(insertion_pos));
        auto pickup_insertion_node_info = curr_route.requests().node_info[pickup_insertion_idx];
        feasible_candidates.record(
          pickup_insertion_node_info, curr_route.get_id(), insertion_pos, found_sol);
      }
    } else {
      // Scan to count the number of found feasible insertion + elect the correct thread (with its
      // locally found indices) to do the atomicMin
      __shared__ int selected_position;

      i_t thread_data = th_move_found;
      cuopt_assert(thread_data == 0 || thread_data == 1, "Thread data value should only be 0 or 1");

      // Inclusive scan
      typedef cub::BlockScan<i_t, BLOCK_SIZE> BlockScan;
      __shared__ typename BlockScan::TempStorage temp_storage;
      BlockScan(temp_storage).InclusiveSum(thread_data, thread_data);
      // TODO : needed ?
      __syncthreads();
      cuopt_assert(thread_data <= blockDim.x, "Scan value can only be smaller than block size");

      // Last thread picks a random number (to be sure it has the last accumulated value)
      if (threadIdx.x == BLOCK_SIZE - 1) {
        // Handles undifined % 0, if not move was found, next if will not trigger and no atomic
        // will be performed
        if (thread_data != 0) {
          selected_position = thread_rng.next_u32() % thread_data;
        } else {
          selected_position = -1;
        }
      }
      __syncthreads();
      // Concerned node do the atomic alone (&& mandatory since scan can make the value spans)
      if (thread_data == selected_position && th_move_found) {
        static_assert(sizeof(uint64_t) == sizeof(unsigned long long int));
        // Conditional probability to perform the atomicMin in case the route is bigger than
        // blockDim.x (because else, since it's a atomicMin it will always be the first one
        // because of data representation)
        if (thread_rng.next_u32() % atomic_min_random_counter == 0) {
          cuopt_assert(insertion_pos < curr_route.get_num_nodes(),
                       "Found feasible move can only be in route");
          // Special offset if
          // intra pickup id is depot because depot doesn't have a global id
          cuopt_assert(frag_step <= 4, "Only frag size of size 4 max supported");
          uint16_t block_id_frag_to_delete =
            (((frag_to_delete - 1) % frag_step) << 14);  // Use first 2 most signficant bits
          cuopt_assert(blockIdx.x / frag_step < 1 << 14,
                       "Overflow write to block_id_frag_to_delete");
          block_id_frag_to_delete += blockIdx.x / frag_step;
          atomicMin(reinterpret_cast<unsigned long long int*>(&(feasible_candidates.data_[0])),
                    bit_cast<unsigned long long int, found_sol_t>(found_sol_t(
                      p_score, block_id_frag_to_delete, pickup_insertion_idx, insertion_pos)));
        }
        ++atomic_min_random_counter;
      }
    }
  }
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
