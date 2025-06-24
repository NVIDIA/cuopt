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

#include "lexicographic_search.cuh"
#include "node_stack.cuh"

#include "../found_solution.cuh"
#include "../guided_ejection_search.cuh"
#include "lexicographic_search.cuh"

#include <routing/utilities/cuopt_utils.cuh>
#include <utilities/seed_generator.cuh>

#include "raft/core/span.hpp"
#include "raft/random/device/sample.cuh"

namespace cuopt {
namespace routing {
namespace detail {

constexpr int threads_per_block_lexico = 64;

// this is only used during run-time tests
template <typename i_t, typename f_t, request_t REQUEST>
bool compare_lexico_results(guided_ejection_search_t<i_t, f_t, REQUEST>& ges,
                            solution_t<i_t, f_t, REQUEST>& solution,
                            request_info_t<i_t, REQUEST>* __restrict__ request_id,
                            ejection_pool_t<request_info_t<i_t, REQUEST>>& EP,
                            i_t k_max)
{
  // currently only k_max 3 supported, previously different k_max has been tested
  if (k_max == 3) {
    auto brute_force_sequence = ges.brute_force_lexico(solution, request_id);
    cuopt_assert(brute_force_sequence.size() != 0, "the brute force didn't find anything");

    auto stream           = ges.solution_ptr->sol_handle->get_stream();
    uint32_t h_global_min = ges.global_min_p_.value(stream);

    std::vector<i_t> lexico_sequence(2 * k_max + 1);
    raft::update_host(
      lexico_sequence.data(), ges.global_sequence_.data() + 2, 2 * k_max + 1, stream);
    stream.synchronize();
    p_val_seq_t p_val(0, 0);
    memcpy((uint32_t*)&p_val, &h_global_min, sizeof(uint32_t));
    cuopt_assert(p_val.p_val == brute_force_sequence[0], "p scores don't match");
    i_t delivery_insertion_idx_in_permutation = ges.global_sequence_.element(0, stream);
    i_t delivery_insertion_position = lexico_sequence[delivery_insertion_idx_in_permutation];
    i_t sequence_size               = ges.global_sequence_.element(1, stream);

    i_t brute_pickup_insertion   = brute_force_sequence[1];
    i_t brute_delivery_insertion = brute_force_sequence[2] + 1;
    i_t brute_force_counter      = 0;
    for (i_t i = 0; i < sequence_size; ++i) {
      if (i != delivery_insertion_idx_in_permutation) {
        i_t curr = lexico_sequence[i];
        if (curr > delivery_insertion_position) curr++;
        if (curr != brute_force_sequence[brute_force_counter + 3]) {
          printf(
            "warning: the lexico sequence don't match. This is expected in early iterations!\n");
        }
        brute_force_counter++;
      }
    }
  }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI void get_route_id_and_insertion_idx(
  i_t& route_id,
  i_t& pickup_insert_idx,
  i_t block_idx,
  const typename solution_t<i_t, f_t, REQUEST>::view_t& solution)
{
  // + 1 to skip depot in node selection, n_routes last blocks handles beginning of each route
  // node that this is not the intra_route_idx but the node_id after which we want to insert
  // for every set of blocks get the node id after which the pickup is inserted
  bool depot_included = solution.problem.order_info.depot_included;
  const i_t pickup_insertion_position =
    (block_idx % (solution.get_num_depot_excluded_orders() + solution.n_routes)) +
    (i_t)depot_included;
  cuopt_assert(pickup_insertion_position <= solution.get_num_orders() + solution.n_routes,
               "Number of launched block should just be enough for each order + route");
  // Find concerned route
  if (pickup_insertion_position < solution.get_num_orders())
    route_id = solution.route_node_map.get_route_id(pickup_insertion_position);
  else  // Should round up (in the modulo sense) to the n first routes indices
    route_id = pickup_insertion_position % solution.get_num_orders();
  cuopt_assert(route_id >= 0 || route_id == -1,
               "Route id from route_id_per_node should be positive or flagged");
  cuopt_assert(route_id < solution.n_routes, "Route id should be smaller than number of routes");
  if (route_id == -1) return;
  // Find concerend intra_route_idx
  if (pickup_insertion_position < solution.get_num_orders()) {
    pickup_insert_idx = solution.route_node_map.get_intra_route_idx(pickup_insertion_position);
  } else {
    // For the ending blocks they are all handling insertion at the beginning of each route
    pickup_insert_idx = 0;
  }
  cuopt_assert(pickup_insert_idx >= 0, "Intra pickup id must be positive");
}

/*
 * @brief Each block handle one pickup position
 * For delivery, they are all tried in the lexicographic order
 * @param global_min_p All blocks atomicMin there
 * @param global_sequence Global array of dimension [gridDimx.x * 2k_max + 1 + 2] (last + 2 is for
 * route_id + delivery insertion index)
 */
template <typename i_t, typename f_t, request_t REQUEST>
__global__ void lexicographic_search(typename solution_t<i_t, f_t, REQUEST>::view_t solution,
                                     i_t k_max,
                                     const request_info_t<i_t, REQUEST>* __restrict__ request_id,
                                     const i_t* __restrict__ p_scores,
                                     uint32_t* __restrict__ global_min_p,
                                     i_t* __restrict__ global_sequence,
                                     i_t* global_random_counter)
{
  cuopt_assert(request_id != nullptr, "Request id should not be nullptr");
  cuopt_assert(request_id->is_valid(solution.problem.order_info.depot_included),
               "Request id should be positive");
  cuopt_assert(solution.get_num_orders() > 0, "Number of orders must be stricly positive");

  // the name is very verbose not to confuse this with node_ids or intra_route_index
  // this is specifying which position in the 2k + 1 items is the delivery
  // every set of blocks handle a delivery location in 2k+1
  i_t delivery_insertion_idx_in_permutation =
    blockIdx.x / (solution.get_num_depot_excluded_orders() + solution.n_routes);

  i_t route_id, pickup_insert_idx;
  // Find concerned route and pickup insertion index
  get_route_id_and_insertion_idx<i_t, f_t, REQUEST>(
    route_id, pickup_insert_idx, blockIdx.x, solution);
  // Discard deleted requests
  if (route_id == -1) { return; }

  // Get route and copy it to shared memory
  auto& route       = solution.routes[route_id];
  auto route_length = route.get_num_nodes() + 1; /* second +1 is for the inserted pickup node*/
  cuopt_assert(pickup_insert_idx < route_length,
               "Intra pickup id must be inferior to route length");

  extern __shared__ i_t shmem[];
  const auto& dimensions_info = solution.problem.dimensions_info;
  auto request_node           = solution.get_request(request_id);
  auto pickup_node            = request_node.node();
  auto delivery_node          = pickup_node;
  if constexpr (REQUEST == request_t::PDP) { delivery_node = request_node.delivery; }

  /*
   * Take an early exit when there is vehicle order mismatch for the request and the current route.
   * lexicographic is used only for GES and it only works for feasible cases, so this is correct.
   * This avoids storing unnecessary dim_between for mismatch dimension
   */
  if (route.dimensions_info().has_dimension(dim_t::MISMATCH) &&
      !route.vehicle_info().order_match[pickup_node.id()]) {
    return;
  }

  node_stack_t<i_t, f_t, REQUEST> node_stack{shmem,
                                             k_max,
                                             delivery_insertion_idx_in_permutation,
                                             p_scores[request_id->info.node()],
                                             delivery_node,
                                             route_length,
                                             route};

  // TODO after raft changes stabilize try to use mdarrays/mdspans when they are more mature

  node_stack.insert_node_and_update_data(
    p_scores, pickup_insert_idx, pickup_node, solution.route_node_map);

  // for now only parallelize on the start index
  // later parallelize on the first 2 start indices
  for (i_t start_idx = threadIdx.x + 1; start_idx < node_stack.route_length;
       start_idx += blockDim.x) {
    node_stack.reset();
    node_t<i_t, f_t, REQUEST> temp_node(dimensions_info);

    temp_node = node_stack.s_route.get_node(start_idx);
    ++node_stack.stack_top;
    if (!node_stack.template is_stack_top_insertion<REQUEST>()) {
      if (!node_stack.s_route.get_node(start_idx - 1)
             .forward_feasible(node_stack.s_route.vehicle_info())) {
        continue;
      }
      copy_forward_data(temp_node, node_stack.s_route.get_node(start_idx - 1));
      if constexpr (REQUEST == request_t::PDP) {
        if (temp_node.request.is_pickup()) {
          node_stack.current_p_score += node_stack.p_scores[start_idx];
          node_stack.n_ejected_pickups++;
        }
        // first one cannot be delivery ejection
        else {
          continue;
        }
      } else {
        node_stack.current_p_score += node_stack.p_scores[start_idx];
        node_stack.n_ejected_pickups++;
      }
    }
    // push intra index, from_index and the node with forward data of previous node
    node_stack.top().intra_idx = start_idx;
    node_stack.top().from_idx  = start_idx - 1;
    node_stack.top()           = temp_node;

    // if the first item we are traversing is the inserted pickup node and we are trying to eject it
    // (i.e the first item in permutation is not a delivery) continue to next thread
    if (start_idx == (pickup_insert_idx + 1) &&
        !node_stack.template is_stack_top_insertion<REQUEST>()) {
      continue;
    }

    // if the delivery insertion is the first item and it comes before the pickup insertion
    if (node_stack.template is_stack_top_insertion<REQUEST>() && start_idx <= pickup_insert_idx) {
      continue;
    }

    if (node_stack.template is_stack_top_insertion<REQUEST>()) {
      node_stack.calculate_forward_to_delivery(start_idx, temp_node);
      cuopt_assert(node_stack.check_dim_to_delivery(start_idx, temp_node), "Mismatch");
      if (!temp_node.forward_feasible(node_stack.s_route.vehicle_info()) ||
          !node_stack.delivery_node.forward_feasible(node_stack.s_route.vehicle_info())) {
        continue;
      }
    }

    // if the first item has a bigger p score, continue
    if (node_stack.current_p_score > node_stack.min_p_score) { continue; }
    cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
    cuopt_assert(node_stack.p_score_check(), "");

    bool advance = false;
    while (true) {
      cuopt_assert(node_stack.top().intra_idx < route_length,
                   "curr_idx in lexico loop should be smaller that route_length");
      if (!advance) {
        f_t time_between;
        node_t<i_t, f_t, REQUEST> from_node(dimensions_info);
        // if the top is ejection but the hole behind it starts with delivery or the top is a
        // delivery
        if (REQUEST == request_t::PDP && (node_stack.top().from_idx == node_stack.route_length ||
                                          node_stack.template is_stack_top_insertion<REQUEST>())) {
          time_between =
            node_stack.template get_dim_from_delivery<dim_t::TIME>(node_stack.top().intra_idx + 1);

          from_node = node_stack.delivery_node;

          cuopt_assert(node_stack.check_dim_from_delivery(
                         node_stack.top().intra_idx + 1,
                         node_stack.s_route.get_node(node_stack.top().intra_idx + 1)),
                       "dim buffer mismatch");
        } else {
          time_between = node_stack.template get_dim_between<dim_t::TIME>(
            node_stack.top().from_idx, node_stack.top().intra_idx + 1);
          from_node = node_stack.s_route.get_node(node_stack.top().from_idx);
          copy_forward_data(from_node, node_stack.top());
          cuopt_assert(node_stack.check_dim_between(
                         node_stack.top().from_idx,
                         node_stack.top().intra_idx + 1,
                         node_stack.s_route.requests().node_info[node_stack.top().from_idx],
                         node_stack.s_route.requests().node_info[node_stack.top().intra_idx + 1]),
                       "dim buffer mismatch");
        }
        // feasible combine checks whether the top of the stack is feasible to combine with next
        // node in the route
        if (node_t<i_t, f_t, REQUEST>::feasible_combine(
              from_node,
              node_stack.s_route.get_node(node_stack.top().intra_idx + 1),
              node_stack.s_route.vehicle_info(),
              time_between)) {
          node_stack.template update_best_sequence<REQUEST>();
        }
      }

      // if the top of the stack is the last item in the route
      if (node_stack.top().intra_idx == node_stack.route_length - 1) {
        advance = true;
        // if that is an ejection handle the scores
        if (!node_stack.template is_stack_top_insertion<REQUEST>()) {
          node_stack.current_p_score -= node_stack.p_scores[node_stack.top().intra_idx];
          if constexpr (REQUEST == request_t::PDP) {
            node_stack.n_ejected_pickups -=
              i_t(node_stack.s_route.requests().is_pickup_node(node_stack.top().intra_idx));
          } else {
            node_stack.n_ejected_pickups -= 1;
          }
        }
        // if one item is left on the stack
        if (--node_stack.stack_top <= 1) { break; }
        cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
        cuopt_assert(node_stack.p_score_check(), "");
      }

      // if stack is full, advance
      advance = advance || (node_stack.stack_top == max_neighbors<i_t, REQUEST>(k_max));
      // advance logic
      if (advance) {
        bool is_forward_feasible = true;
        if (node_stack.template is_stack_top_insertion<REQUEST>()) {
          // we compute the next node because that will be in the route and we will insert after
          // this
          temp_node = node_stack.s_route.get_node(node_stack.top().intra_idx + 1);
          // compute forward data for the top of the stack and write it to the temp node, this
          // forward data is the from_node's data (i.e current stack top)
          is_forward_feasible = node_stack.advance_insertion(temp_node);
          if (is_forward_feasible) {
            bool is_delivery_time_dist_feasible =
              node_stack.delivery_node.time_dim.forward_feasible(
                node_stack.s_route.vehicle_info()) &&
              node_stack.delivery_node.distance_dim.forward_feasible(
                node_stack.s_route.vehicle_info());
            if (!is_delivery_time_dist_feasible) {
              if (--node_stack.stack_top <= 1) { break; }
              cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
              cuopt_assert(node_stack.p_score_check(), "");
              continue;
            } else if (!node_stack.delivery_node.capacity_dim.forward_feasible(
                         node_stack.s_route.vehicle_info())) {
              // skip the pop later, and just advance
              node_stack.top().from_idx = node_stack.top().intra_idx;
              node_stack.top()          = temp_node;
              node_stack.top().intra_idx++;
              advance = false;
              cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
              cuopt_assert(node_stack.p_score_check(), "");
              continue;
            }
          } else {
            if (--node_stack.stack_top <= 1) { break; }
            cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
            cuopt_assert(node_stack.p_score_check(), "");
            continue;
          }
        } else {
          // compute forward data for the top of the stack and write it to the temp node, this
          // forward data is the from_node's data (i.e current stack top) we compute the top here,
          // because that will go back into the route
          temp_node           = node_stack.s_route.get_node(node_stack.top().intra_idx);
          is_forward_feasible = node_stack.template advance_ejection<REQUEST>(temp_node);
          node_stack.current_p_score -= node_stack.p_scores[node_stack.top().intra_idx];

          if constexpr (REQUEST == request_t::PDP) {
            node_stack.n_ejected_pickups -=
              i_t(node_stack.s_route.requests().is_pickup_node(node_stack.top().intra_idx));
          } else {
            node_stack.n_ejected_pickups -= 1;
          }

          if constexpr (REQUEST == request_t::PDP) {
            if (!temp_node.request.is_pickup()) {
              is_forward_feasible =
                is_forward_feasible &&
                !node_stack.check_paired_pickup_is_ejected(temp_node.request.brother_id());
            }
          }
        }

        // if the top of the stack(the forward data is currently in temp_node) is not forward
        // feasible(including forward capacity!) or the paired pickup is ejected
        if (!is_forward_feasible) {
          // if one item is left on the stack
          if (--node_stack.stack_top <= 1) { break; }
          cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
          cuopt_assert(node_stack.p_score_check(), "");
          continue;
        }
        node_stack.top().from_idx = node_stack.top().intra_idx;
        node_stack.top()          = node_stack.s_route.get_node(node_stack.top().intra_idx + 1);
        copy_forward_data(node_stack.top(), temp_node);
        node_stack.top().intra_idx++;

      }
      // expand logic
      else {
        temp_node = node_stack.s_route.get_node(node_stack.top().intra_idx + 1);
        node_stack.get_stack_item(node_stack.stack_top) = node_stack.top();
        node_stack.stack_top++;
        // keep delivery node updated for forward data
        // if the forward data of the delivery is not feasible
        // advance the delivery node
        if (node_stack.template is_stack_top_insertion<REQUEST>()) {
          bool forward_feasible     = node_stack.expand_insertion(temp_node);
          node_stack.top().from_idx = node_stack.top().intra_idx;
          node_stack.top().intra_idx++;
          // the data on temp is actual
          node_stack.top() = temp_node;
          if (forward_feasible) {
            bool is_delivery_time_dist_feasible =
              node_stack.delivery_node.time_dim.forward_feasible(
                node_stack.s_route.vehicle_info()) &&
              node_stack.delivery_node.distance_dim.forward_feasible(
                node_stack.s_route.vehicle_info());
            if (!is_delivery_time_dist_feasible) {
              if (--node_stack.stack_top <= 1) { break; }
              advance = true;
              cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
              cuopt_assert(node_stack.p_score_check(), "");
              continue;
            } else if (!node_stack.delivery_node.capacity_dim.forward_feasible(
                         node_stack.s_route.vehicle_info())) {
              advance = true;
              cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
              cuopt_assert(node_stack.p_score_check(), "");
              continue;
            }
          } else {
            if (--node_stack.stack_top <= 1) { break; }
            advance = true;
            cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
            cuopt_assert(node_stack.p_score_check(), "");
            continue;
          }
        } else {
          // copy the stack data forward as we are keeping from which node we have computed this
          // but keep the node fixed data (node ids brother id etc here)
          node_stack.top() = temp_node;
          // set special index for the from index if delivery comes before the hole
          if (REQUEST == request_t::PDP &&
              node_stack.stack_top - 2 == node_stack.delivery_insertion_idx_in_permutation) {
            node_stack.top().from_idx = route_length;
            copy_forward_data(node_stack.top(), node_stack.delivery_node);
          } else {
            copy_forward_data(node_stack.top(),
                              node_stack.get_stack_item(node_stack.stack_top - 2));
          }
          node_stack.top().intra_idx++;
        }
      }
      advance = false;
      // filter logic
      bool is_current_pickup = true;
      if constexpr (REQUEST == request_t::PDP) {
        is_current_pickup =
          node_stack.s_route.requests().is_pickup_node(node_stack.top().intra_idx);
      }
      if (!node_stack.template is_stack_top_insertion<REQUEST>()) {
        if constexpr (REQUEST == request_t::PDP) {
          node_stack.n_ejected_pickups += i_t(is_current_pickup);
        } else {
          node_stack.n_ejected_pickups += 1;
        }
        node_stack.current_p_score += node_stack.p_scores[node_stack.top().intra_idx];
        cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
        cuopt_assert(node_stack.p_score_check(), "");
        // advance here means that we will not continue with the lower part of the lexicographical
        // search p_score filter
        advance = advance || node_stack.current_p_score > node_stack.min_p_score;
        // pickup ejection filter
        advance = advance || node_stack.n_ejected_pickups > k_max;
        // check whether we are ejecting pickup node
        advance = advance || node_stack.top().intra_idx == pickup_insert_idx + 1;
        // the pickup node of the top of the stack is not ejected

        if constexpr (REQUEST == request_t::PDP) {
          if (!is_current_pickup)
            advance =
              advance || !node_stack.check_paired_pickup_is_ejected(
                           node_stack.s_route.requests().brother_id(node_stack.top().intra_idx));
        }
      } else {
        cuopt_assert(node_stack.template k_max_ejection_check<REQUEST>(), "");
        cuopt_assert(node_stack.p_score_check(), "");
        advance = advance || (node_stack.top().intra_idx <= pickup_insert_idx);

        if constexpr (REQUEST == request_t::PDP) {
          // if the pickup node of the node before the delivery is ejected we cannot continue
          if (!is_current_pickup &&
              node_stack.check_paired_pickup_is_ejected(
                node_stack.s_route.requests().brother_id(node_stack.top().intra_idx))) {
            // if one item is left on the stack
            if (--node_stack.stack_top <= 1) { break; }
            advance = true;
          }
        }
      }
      node_stack.sanity_checks(advance);
    }
  }
  // block_random_sample uses, warp_size * 2 * (i_t) memory
  __shared__ i_t reusable_shmem[2 * raft::WarpSize];
  __syncthreads();
  // there might not be any time or capacity dimension so it is safer to use static shmem
  i_t thread_p_score = node_stack.best_sequence_size != std::numeric_limits<i_t>::max()
                         ? node_stack.min_p_score
                         : std::numeric_limits<i_t>::max();
  // Record move at block level and write to global (to call at the end):
  // FIXME: do not use raft reduce yet. there is a bug in min/max reduction
  block_reduce(thread_p_score, reusable_shmem);
  // reusable_shmem[0] contains the min value
  __syncthreads();
  i_t reduction_val = reusable_shmem[0];
  // if no valid move has been found
  // or only delivery insertion is found, we have single insertion kernel that handles this case
  if (reduction_val >= p_scores[request_id->info.node()] || reduction_val == 0) return;
  __syncthreads();
  // Filter and select only one of the thread for final global write
  bool pred       = thread_p_score == reduction_val;
  i_t thread_data = pred ? node_stack.random_counter : 0;
  i_t selected_thread =
    raft::random::device::block_random_sample(node_stack.thread_rng, reusable_shmem, (i_t)pred);
  // also reduce the total random counter
  i_t block_random_counter = raft::blockReduce(node_stack.random_counter, (char*)reusable_shmem);
  __syncthreads();
  if (threadIdx.x == 0) {
    reusable_shmem[2 * raft::WarpSize - 1] = block_random_counter;
    reusable_shmem[0]                      = selected_thread;
  }
  __syncthreads();
  // reuse shmem from the time buffers
  if (threadIdx.x == reusable_shmem[0]) {
    cuopt_assert(thread_p_score == reduction_val, "Elected thread should have same smallest val");
    uint32_t block_p_val_seq =
      bit_cast<uint32_t, p_val_seq_t>(p_val_seq_t(thread_p_score, node_stack.best_sequence_size));
    // we need to keep track of it to prevent global bias
    // reusable_shmem last item keeps the scan result
    i_t total_random_counter = reusable_shmem[2 * raft::WarpSize - 1];
    cuopt_assert(total_random_counter > 0, "total_random_counter should be greater than 0");
    uint32_t old_p_val_seq = atomicMin(global_min_p, block_p_val_seq);
    while (atomicExch(solution.lock, 1) != 0)
      ;  // acquire
    __threadfence();
    if (global_min_p[0] == block_p_val_seq) {
      bool update = true;
      // if it is the same value, use conditional probability to update
      if (old_p_val_seq == block_p_val_seq) {
        i_t old_counter        = *global_random_counter;
        i_t new_counter        = old_counter + total_random_counter;
        *global_random_counter = old_counter + total_random_counter;
        raft::random::PCGenerator global_rng(2829, old_counter, 0);
        update = (global_rng.next_u32() % new_counter) >= old_counter;
      }
      // otherwise just update and reset the counter
      else {
        *global_random_counter = total_random_counter;
      }

      if (update) {
        // Sad uncoallsced global writes of route then the thread found best sequence
        global_sequence[0] = blockIdx.x;
        if constexpr (REQUEST == request_t::PDP) {
          cuopt_assert(
            node_stack.best_sequence_size > node_stack.delivery_insertion_idx_in_permutation,
            "Sequence size should be bigger than delivery_insertion_idx_in_permutation");
        }
        cuopt_assert((node_stack.best_sequence_size <= max_neighbors<i_t, REQUEST>(k_max)),
                     "Sequence size should be smaller than 2*k_max +1 ");
        global_sequence[1] = node_stack.best_sequence_size;
        for (i_t i = 0; i < node_stack.best_sequence_size; ++i) {
          global_sequence[i + 2] = node_stack.get_best_sequnce(i);
        }
      }
      // if we are at the min_p_score and best sequence size, conditionally update the global
      // values according to global counter
      cuopt_assert(node_stack.min_p_score > 0,
                   "P score should be greater than 0 when sequence is greater than 1");
      cuopt_assert(node_stack.min_p_score < p_scores[request_id->info.node()],
                   "P score should be smaller than requests ");
    }
    __threadfence();
    *(solution.lock) = 0;  // release
  }
}

template <typename i_t, typename f_t, request_t REQUEST>
__global__ void execute_lexico_move(
  typename solution_t<i_t, f_t, REQUEST>::view_t solution,
  const request_info_t<i_t, REQUEST>* __restrict__ request_id,
  uint32_t* __restrict__ global_min_p,
  i_t* __restrict__ global_sequence,
  typename ejection_pool_t<request_info_t<i_t, REQUEST>>::view_t EP,
  const i_t* __restrict__ p_scores)

{
  extern __shared__ i_t shmem[];
  cuopt_assert(request_id != nullptr, "Request id should not be nullptr");
  cuopt_assert(request_id->is_valid(solution.problem.order_info.depot_included),
               "Request id should be positive");
  p_val_seq_t p_val_seq  = bit_cast<p_val_seq_t, uint32_t>(*global_min_p);
  i_t original_block_idx = global_sequence[0];
  i_t route_id, pickup_insert_idx;
  get_route_id_and_insertion_idx<i_t, f_t, REQUEST>(
    route_id, pickup_insert_idx, original_block_idx, solution);
  cuopt_assert(route_id >= 0, "Route id cannot be negative");
  cuopt_assert(route_id < solution.n_routes, "Route id should be smaller than n_nodes");
  i_t delivery_insertion_idx_in_permutation =
    original_block_idx / (solution.get_num_depot_excluded_orders() + solution.n_routes);
  i_t sequence_size = global_sequence[1];
  cuopt_assert(p_val_seq.sequence_size == sequence_size,
               "Global sequence size and p_val_seq doesn't match");
  auto& route  = solution.routes[route_id];
  auto s_route = route_t<i_t, f_t, REQUEST>::view_t::create_shared_route(
    shmem, route, route.get_num_nodes() + sequence_size + 1);
  __syncthreads();
  s_route.copy_from(route);
  auto request_node = solution.get_request(request_id);
  auto pickup_node  = request_node.node();
  __syncthreads();
  // subract - 1 because the delivery index is considered after the pickup is inserted
  i_t delivery_insertion_idx = 0;
  if constexpr (REQUEST == request_t::PDP) {
    delivery_insertion_idx = global_sequence[delivery_insertion_idx_in_permutation + 2];
  }
  // insert the pickup at the respective position to compute the forward backward data
  if (threadIdx.x == 0) {
    request_id_t<REQUEST> request_locations;
    if constexpr (REQUEST == request_t::PDP) {
      request_locations = request_id_t<REQUEST>(pickup_insert_idx, delivery_insertion_idx - 1);
    } else {
      request_locations = request_id_t<REQUEST>(pickup_insert_idx);
    }
    // insert request
    s_route.template insert_request<REQUEST>(
      request_locations, request_node, solution.route_node_map, true);
    i_t n_ejections_executed = 0;
    for (i_t i = 0; i < sequence_size; ++i) {
      bool eject = true;
      if constexpr (REQUEST == request_t::PDP) {
        eject = delivery_insertion_idx_in_permutation != i;
      }
      // eject
      if (eject) {
        // ejection indices were considered after the request pickup was inserted
        i_t ejection_idx = global_sequence[i + 2];
        cuopt_assert(pickup_insert_idx + 1 != ejection_idx, "Cannot eject the inserted pickup ");

        // adjust the ejection index
        // the ejection index was computed before we inserted the delivery
        // every ejection index after the request delivery insertion should be incremented by 1
        if constexpr (REQUEST == request_t::PDP) {
          if (ejection_idx > delivery_insertion_idx) ++ejection_idx;
        }
        // we need to adjust the ejection_idx by the n_ejections executed before
        ejection_idx -= n_ejections_executed;
        const auto ejected_node = s_route.get_node(ejection_idx);
        assert(!s_route.get_node(ejection_idx).node_info().is_break());
        s_route.eject_node(ejection_idx, solution.route_node_map, true);
        ++n_ejections_executed;

        if constexpr (REQUEST == request_t::PDP) {
          if (ejected_node.request.is_pickup()) {
            request_id_t<REQUEST> ejected_request_id(ejected_node.id(),
                                                     ejected_node.request.brother_id());
            EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, ejected_request_id));
          }
        } else {
          request_id_t<REQUEST> ejected_request_id(ejected_node.id());
          EP.push(create_request<i_t, f_t, REQUEST>(solution.problem, ejected_request_id));
        }
      }
      route_t<i_t, f_t, REQUEST>::view_t::compute_forward(s_route);
      route_t<i_t, f_t, REQUEST>::view_t::compute_backward(s_route);
    }
  }
  __syncthreads();
  cuopt_assert(s_route.is_feasible(),
               "The route after the lexicographical move should be feasible!");
  route.copy_from(s_route);
}

// runs and executes lexicographical search and returns whether the move is executed
template <typename i_t, typename f_t, request_t REQUEST>
bool guided_ejection_search_t<i_t, f_t, REQUEST>::run_lexicographic_search(
  request_info_t<i_t, REQUEST>* __restrict__ request_id)
{
  auto stream = solution_ptr->sol_handle->get_stream();
  RAFT_CHECK_CUDA(stream);

  i_t average_route_size = solution_ptr->get_num_orders() / solution_ptr->n_routes;

  i_t const_1, const_2, const_3;
  const_1   = 30;
  const_2   = 50;
  const_3   = 100;
  i_t k_max = 5;
  if (average_route_size > const_3) {
    k_max = 2;
  } else if (average_route_size > const_2) {
    k_max = 3;
  } else if (average_route_size > const_1) {
    k_max = 4;
  }

  // As we fix the insertion we don't need to try k_max permutations of insertion for cvrptw
  i_t n_blocks_lexico = (solution_ptr->get_num_orders() + solution_ptr->get_n_routes() -
                         solution_ptr->problem_ptr->order_info.depot_included_);
  if constexpr (REQUEST == request_t::PDP) {
    n_blocks_lexico *= max_neighbors<i_t, REQUEST>(k_max);
  }

  size_t sh_size = 0;
  bool is_set    = false;
  while (k_max > 1) {
    sh_size = node_stack_t<i_t, f_t, REQUEST>::get_shared_size(
      solution_ptr, 1, k_max, threads_per_block_lexico);
    if (set_shmem_of_kernel(lexicographic_search<i_t, f_t, REQUEST>, sh_size)) {
      is_set = true;
      break;
    }
    k_max--;
  }

  if (k_max == 1 || !is_set) { return false; }
  // Init global min before call to lexicographic
  const auto max = std::numeric_limits<typename decltype(global_min_p_)::value_type>::max();
  const i_t zero = 0;
  global_min_p_.set_value_async(max, stream);
  solution_ptr->d_lock.set_value_async(zero, stream);
  global_random_counter_.set_value_async(zero, stream);
  lexicographic_search<i_t, f_t>
    <<<n_blocks_lexico, threads_per_block_lexico, sh_size, stream>>>(solution_ptr->view(),
                                                                     k_max,
                                                                     request_id,
                                                                     p_scores_.data(),
                                                                     global_min_p_.data(),
                                                                     global_sequence_.data(),
                                                                     global_random_counter_.data());
  solution_ptr->sol_handle->sync_stream();
  RAFT_CHECK_CUDA(stream);
  // If global_min_p_ != max do the move
  if (global_min_p_.value(stream) != max) {
    // cuopt_assert(compare_lexico_results(*this, solution, request_id, EP, k_max), "");
    const auto shared_for_tmp_route =
      solution_ptr->get_temp_route_shared_size(max_neighbors<i_t, REQUEST>(k_max) + 1);
    if (!set_shmem_of_kernel(execute_lexico_move<i_t, f_t, REQUEST>, shared_for_tmp_route)) {
      return false;
    }
    execute_lexico_move<i_t, f_t, REQUEST>
      <<<1, threads_per_block_lexico, shared_for_tmp_route, stream>>>(solution_ptr->view(),
                                                                      request_id,
                                                                      global_min_p_.data(),
                                                                      global_sequence_.data(),
                                                                      EP.view(),
                                                                      p_scores_.data());
    RAFT_CHECK_CUDA(stream);
    i_t removed_size = global_sequence_.element(1, stream);
    if constexpr (REQUEST == request_t::PDP) { removed_size = (removed_size - 1) / 2; }
    EP.index_ += removed_size;
    return true;
  }
  return false;
}

template bool guided_ejection_search_t<int, float, request_t::PDP>::run_lexicographic_search(
  request_info_t<int, request_t::PDP>* __restrict__ request_id);

template bool guided_ejection_search_t<int, float, request_t::VRP>::run_lexicographic_search(
  request_info_t<int, request_t::VRP>* __restrict__ request_id);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
