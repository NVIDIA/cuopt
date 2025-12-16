/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2023-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/seed_generator.cuh>
#include "../node/node.cuh"
#include "../route/route.cuh"
#include "../routing_helpers.cuh"
#include "../solution/solution.cuh"

constexpr int max_permutation_intra = 5;

namespace cuopt {
namespace routing {
namespace detail {

template <typename T>
DI void random_shuffle(T* array, int size, raft::random::PCGenerator& rng)
{
  if (size <= 1) return;
  for (int i = size - 1; i > 0; i--) {
    const auto j = rng.next_u32() % (i + 1);
    raft::swapVals(array[i], array[j]);
  }
}

template <request_t REQUEST, std::enable_if_t<REQUEST == request_t::PDP, bool> = true>
static constexpr int min_permutations()
{
  return 1;
}

template <request_t REQUEST, std::enable_if_t<REQUEST == request_t::VRP, bool> = true>
static constexpr int min_permutations()
{
  return 1;
}

// Helper function to simply the sumation of factorials
template <typename i_t, int index, int array_size>
HDI static constexpr int sum_factorial_array(
  const std::pair<i_t, i_t> permutation_array[array_size])
{
  static_assert(index >= 0 && index < array_size, "Invalid sum factorial value");
  if (index == 0) return factorial(permutation_array[index].second);
  return factorial(permutation_array[index].second) +
         sum_factorial_array<index - 1, array_size>(permutation_array);
}

template <int N>
HDI static constexpr std::array<int, N> get_sequence()
{
  std::array<int, N> arr;
  constexpr_for<0, N, 1>([&](auto I) { arr[I] = I; });
  return arr;
}

HDI static constexpr int factorial(int n)
{
  cuopt_assert((n >= 1), "Invalid factorial value");
  if (n == 1) return 1;
  return n * factorial(n - 1);
}

template <request_t REQUEST>
HDI static constexpr int sum_factorial(int n)
{
  cuopt_assert(n >= min_permutations<REQUEST>(), "Invalid sum factorial value");
  if (n == min_permutations<REQUEST>()) return factorial(min_permutations<REQUEST>());
  return factorial(n) + sum_factorial<REQUEST>(n - 1);
}

// Yields the nth permutation of array "values" in lexicographic order
template <typename i_t, int size, request_t REQUEST>
__device__ void get_nth_permutation(i_t res[size], int index)
{
  // Swap those two case to have the last trailing threads prioritizing the reverse order which is
  // usually better
  if (index == 1) index = factorial(size) - 1;
  if (index == factorial(size) - 1 && size > 1) index = 1;

  cuopt_assert(index < factorial(size) && index >= 0, "Invalid index");

  auto values = get_sequence<size>();

  i_t res_id = 0;

  // Stack
  i_t stack[size];
  int s_id = -1;

  // Loop to generate the factroid of the sequence
  for (int i = 1; i < size + 1; ++i) {
    stack[++s_id] = index % i;
    index /= i;
  }

  // Loop to generate nth permutation
  for (int i = 0; i < size; ++i) {
    const i_t a   = stack[s_id];
    res[res_id++] = values[a];

    // Shift / Remove one value per iteration
    for (int j = a; j < size - 1; ++j)
      values[j] = values[j + 1];
    --s_id;
  }
}

// Same as get_nth_permutation but to handle non-constexpr case
template <typename i_t, request_t REQUEST>
__device__ void get_nth_permutation(i_t* res, int index, i_t* values, int size)
{
  // Swap those two case to have the last trailing threads prioritizing the reverse order which is
  // usually better
  if (index == 1) index = factorial(size) - 1;
  if (index == factorial(size) - 1 && size > 1) index = 1;

  cuopt_assert(index < factorial(size) && index >= 0, "Invalid index");

  i_t res_id = 0;

  // Stack
  i_t stack[max_permutation_intra + 1];
  int s_id = -1;

  // Loop to generate the factroid of the sequence
  for (int i = 1; i < size + 1; ++i) {
    stack[++s_id] = index % i;
    index /= i;
  }

  // Loop to generate nth permutation
  for (int i = 0; i < size; ++i) {
    const i_t a   = stack[s_id];
    res[res_id++] = values[a];

    // Shift / Remove one value per iteration
    for (int j = a; j < size - 1; ++j)
      values[j] = values[j + 1];
    --s_id;
  }
}

// Count the amount of brothers of nodes inside the window
// Or returns -1 if a delivery is before a pickup
template <typename i_t, typename f_t, request_t REQUEST>
DI i_t
invalid_window_permutation_check(const typename solution_t<i_t, f_t, REQUEST>::view_t& solution,
                                 const NodeInfo<>* node_infos,
                                 const NodeInfo<>* brother_infos,
                                 i_t window_size)
{
  i_t brother_inside_count = 0;
  for (i_t i = 0; i < window_size; ++i) {
    const auto node_info = node_infos[i];
    if (node_info.is_pickup()) {
      const auto delivery_info = brother_infos[i];
      for (i_t j = 0; j < window_size; ++j) {
        if (node_infos[j].node() == delivery_info.node())  // Found matching brother
        {
          if (i < j)  // Delivery is after pickup
            ++brother_inside_count;
          else if (i > j)  // Delivery is before pickup
            return -1;
        }
      }
    }
  }
  return brother_inside_count;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI int invalid_window_permutation_check(const node_t<i_t, f_t, REQUEST>* nodes, i_t window_size)
{
  i_t brother_inside_count = 0;
  for (i_t i = 0; i < window_size; ++i) {
    const auto node = nodes[i];
    // don't allow fragments with breaks in it
    if (!node.request.info.is_service_node()) { return -1; }

    if (node.request.is_pickup()) {
      const i_t delivery_id = node.request.brother_id();
      for (i_t j = 0; j < window_size; ++j) {
        if (nodes[j].id() == delivery_id)  // Found matching brother
        {
          if (i < j)  // Delivery is after pickup
            ++brother_inside_count;
          else if (i > j)  // Delivery is before pickup
            return -1;
        }
      }
    }
  }
  return brother_inside_count;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool forward_fragment_update(const node_t<i_t, f_t, REQUEST>& curr_node,
                                const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                                node_t<i_t, f_t, REQUEST>* fragment,
                                i_t fragment_size,
                                const infeasible_cost_t& weights,
                                double excess_limit)
{
  cuopt_assert(fragment_size != 0, "Fragment size cannot be zero!");
  curr_node.calculate_forward_all(fragment[0], s_route.vehicle_info());
  if (s_route.dimensions_info().has_dimension(dim_t::TIME) &&
      !fragment[0].time_dim.forward_feasible(
        s_route.vehicle_info(), weights[dim_t::TIME], excess_limit)) {
    return false;
  }

  // Update window forward info
  for (int j = 0; j < fragment_size - 1; ++j) {
    auto& _curr_node = fragment[j];
    auto& _next_node = fragment[j + 1];
    _curr_node.calculate_forward_all(_next_node, s_route.vehicle_info());
    if (s_route.dimensions_info().has_dimension(dim_t::TIME) &&
        !_next_node.time_dim.forward_feasible(
          s_route.vehicle_info(), weights[dim_t::TIME], excess_limit)) {
      return false;
    }
  }
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool forward_fragment_update_cvrp(const node_t<i_t, f_t, REQUEST>& curr_node,
                                     const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                                     node_t<i_t, f_t, REQUEST>* fragment,
                                     i_t fragment_size,
                                     f_t fragment_dist,
                                     f_t fragment_demand,
                                     const infeasible_cost_t& weights,
                                     double excess_limit)
{
  cuopt_assert(fragment_size != 0, "Fragment size cannot be zero!");

  f_t arc_value = get_arc_of_dimension<i_t, f_t, dim_t::DIST, true>(
    curr_node.request.info, fragment[0].request.info, s_route.vehicle_info());
  fragment[fragment_size - 1].distance_dim.distance_forward =
    curr_node.distance_dim.distance_forward + arc_value + fragment_dist;
  fragment[fragment_size - 1].capacity_dim.gathered[0] =
    curr_node.capacity_dim.gathered[0] + fragment_demand;
  fragment[fragment_size - 1].capacity_dim.max_to_node[0] =
    fragment[fragment_size - 1].capacity_dim.gathered[0];
  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool backward_fragment_update(const node_t<i_t, f_t, REQUEST>& curr_node,
                                 const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                                 node_t<i_t, f_t, REQUEST>* fragment,
                                 i_t fragment_size,
                                 const infeasible_cost_t& weights,
                                 double excess_limit)
{
  curr_node.calculate_backward_all(fragment[fragment_size - 1], s_route.vehicle_info());

  if (s_route.dimensions_info().has_dimension(dim_t::TIME) &&
      !fragment[fragment_size - 1].time_dim.backward_feasible(
        s_route.vehicle_info(), weights[dim_t::TIME], excess_limit)) {
    return false;
  }

  // Update window backward info

  for (int j = fragment_size - 1; j > 0; --j) {
    auto& _curr_node = fragment[j];
    auto& _prev_node = fragment[j - 1];
    _curr_node.calculate_backward_all(_prev_node, s_route.vehicle_info());
    if (s_route.dimensions_info().has_dimension(dim_t::TIME) &&
        !_prev_node.time_dim.backward_feasible(
          s_route.vehicle_info(), weights[dim_t::TIME], excess_limit)) {
      return false;
    }
  }

  return true;
}

template <typename i_t, typename f_t, request_t REQUEST>
DI bool backward_fragment_update_cvrp(const node_t<i_t, f_t, REQUEST>& curr_node,
                                      const typename route_t<i_t, f_t, REQUEST>::view_t& s_route,
                                      node_t<i_t, f_t, REQUEST>* fragment,
                                      i_t fragment_size,
                                      f_t fragment_dist,
                                      f_t fragment_demand,
                                      const infeasible_cost_t& weights,
                                      double excess_limit)
{
  f_t arc_value = get_arc_of_dimension<i_t, f_t, dim_t::DIST, true>(
    fragment[fragment_size - 1].request.info, curr_node.request.info, s_route.vehicle_info());
  fragment[0].distance_dim.distance_backward =
    curr_node.distance_dim.distance_backward + arc_value + fragment_dist;

  fragment[0].capacity_dim.max_after[0] = curr_node.capacity_dim.max_after[0] + fragment_demand;

  return true;
}

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
