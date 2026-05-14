/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <utilities/cuda_helpers.cuh>
#include "../routing_helpers.cuh"

#include <routing/fleet_info.hpp>
#include <routing/routing_details.hpp>
#include <routing/structures.hpp>

#include <cstdint>

namespace cuopt {
namespace routing {
namespace detail {

//! Maximum number of orders tracked in the backward lateness arrays per route position.
//! Limits the node struct size (bwd_lateness_b/w[K] live in registers during backward propagation)
//! and shared memory (execute_two_opt_moves allocates s_route + fragment, both carrying K arrays).
//! Keep K small enough so that shared_route_size + size_of_frag stays within the device shmem
//! limit. With K=32, routes with more than 32 orders silently drop the excess from the backward
//! arrays (the oldest/leftmost orders), giving an approximate lateness penalty for those routes.
static constexpr int MAX_SOFT_TIME_ROUTE_SIZE = 48;

/**
 * @brief Per-node state for the SOFT_TIME dimension.
 *
 * Tracks weighted completion time (WCT) for soft time scheduling, and optionally
 * lateness penalty for orders that have a due time measured from t=0.
 *
 * WCT = sum_k ( order_weight[k] * completion_time[k] )
 * where completion_time[k] = time when the tool finishes processing order k.
 *
 * Forward state:
 *   actual_start[k]   = max(fwd_completion[k-1] + transit_{k-1,k}, earliest_time[k])
 *   fwd_completion[k] = actual_start[k] + service_time[k]
 *   fwd_wct[k]        = fwd_wct[k-1] + order_weight[k] * fwd_completion[k]
 *   fwd_lateness[k]   = fwd_lateness[k-1]
 *                       + max(0, actual_start[k] - due_time[k]) * order_weight[k]
 *                       (exact prefix sum; only when due_time[k] > 0)
 *
 * Backward WCT state accumulates weight sums and relative WCT (assuming zero
 * waiting due to earliest_time) so that combine() is O(1) at any split point.
 *
 * Backward lateness state: sorted array of (b_j, w_j) pairs where
 *   b_j = due_time_j - delta_j^k  (effective breakpoint from split k)
 *   delta_j^k = cumulative service+transit from position k to position j
 * Sorted ascending by b_j. Updated each backward step by shifting all b_j
 * by -(service[k] + transit[k,k+1]) and inserting node k at b_k = due_time[k].
 *
 * At get_cost for split k with handoff h (= actual_start[k]):
 *   P_suffix(h) = sum_j  max(0, h - b_j) * w_j   (O(m) scan, m <= MAX_SOFT_TIME_ROUTE_SIZE)
 *   total_lateness = prev.fwd_lateness + P_suffix(h)
 *
 * SOFT_TIME has no infeasibility constraints — purely objective.
 * combine() always returns 0.
 *
 * Arc value for this dimension = travel_time(from, to) only.
 * Service time is fetched from vehicle_info.order_service_times at propagation time.
 *
 * All time/weight fields use integer arithmetic (int32_t/int64_t).
 * Transit times are rounded to the nearest integer on use.
 * Conversion to double happens only in get_cost() when writing to obj_cost.
 */
template <typename i_t, typename f_t>
class soft_time_node_t {
 public:
  // ---- Fixed data (set once from problem input) ----
  int32_t order_weight = 0;   //! Importance weight of this order; 0 for non-service nodes
  NodeInfo<i_t> node_info{};  //! Node identity: type (DEPOT/DELIVERY/BREAK) + index

  //! Earliest allowable start time (order arrival time). 0 = no constraint.
  int32_t earliest_time = 0;
  //! Due time deadline: order must start processing by this time (t=0 reference). 0 = no
  //! constraint.
  int32_t due_time = 0;

  // ---- Forward WCT state ----
  //! Completion time of this order = actual_start + service_time
  int32_t fwd_completion = 0;
  //! Accumulated WCT: sum_{i=0}^{k} order_weight[i] * completion[i]
  int32_t fwd_wct = 0;

  // ---- Backward WCT state ----
  //! Sum of order_weights from this position to end of route
  int32_t bwd_weight_sum = 0;
  //! Relative WCT of suffix [k..n] assuming handoff time = 0 and no earliest_time waits
  int32_t bwd_wct_rel = 0;

  // ---- Forward lateness objective state ----
  //! Exact prefix sum: sum_{i=0}^{k} max(0, actual_start[i] - due_time[i]) * order_weight[i]
  //! (only for nodes with due_time > 0). Accumulated from forward propagation.
  int32_t fwd_lateness = 0;

  // ---- Backward lateness objective state ----
  //! Sorted (ascending by b_j) array of effective breakpoints b_j = due_time_j - delta_j^k.
  //! At get_cost: P_suffix(h) = sum_j max(0, h - b_j) * w_j.
  int32_t bwd_lateness_b[MAX_SOFT_TIME_ROUTE_SIZE] = {};
  //! Corresponding order_weight values for each breakpoint entry.
  int32_t bwd_lateness_w[MAX_SOFT_TIME_ROUTE_SIZE] = {};
  //! Number of valid entries in bwd_lateness_b / bwd_lateness_w.
  int bwd_lateness_n = 0;

  // -----------------------------------------------------------------------
  // Service-time helper: mirrors get_transit_time's logic in arc_value.hpp.
  // DEPOT → 0, BREAK → break_durations[node()], SERVICE → order_service_times[node()].
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  HDI static int32_t get_service_time(const soft_time_node_t& n,
                                      const VehicleInfo<f_t, is_device>& vehicle_info) noexcept
  {
    if (n.node_info.is_depot()) { return 0; }
    if (n.node_info.is_break()) {
      return vehicle_info.break_durations.empty()
               ? 0
               : static_cast<int32_t>(vehicle_info.break_durations[n.node_info.node()]);
    }
    return vehicle_info.order_service_times.empty()
             ? 0
             : static_cast<int32_t>(vehicle_info.order_service_times[n.node_info.node()]);
  }

  // -----------------------------------------------------------------------
  // calculate_forward
  // Called as:  this.calculate_forward(next, travel_time, vehicle_info)
  // arc = travel_time(this -> next)
  //
  //   actual_start        = max(fwd_completion + transit, earliest_time[next])
  //   next.fwd_completion = actual_start + service[next]
  //   next.fwd_wct        = fwd_wct + order_weight[next] * fwd_completion[next]
  //   next.fwd_lateness   = fwd_lateness
  //                         + max(0, actual_start - due_time[next]) * order_weight[next]
  //                         (only if due_time[next] > 0)
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_forward(soft_time_node_t& next,
                             double travel_time,
                             const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    int32_t service_time_next = get_service_time<is_device>(next, vehicle_info);
    int32_t transit           = static_cast<int32_t>(travel_time);

    int32_t arrival_real = fwd_completion + transit;
    int32_t actual_start =
      (next.earliest_time > 0) ? max(arrival_real, next.earliest_time) : arrival_real;
    next.fwd_completion = actual_start + service_time_next;
    next.fwd_wct        = fwd_wct + next.order_weight * next.fwd_completion;

    // Exact prefix sum — uses max(), so backward can't be O(1); handled by bwd arrays.
    next.fwd_lateness =
      fwd_lateness +
      (next.due_time > 0 ? max(0, actual_start - next.due_time) * next.order_weight : 0);
  }

  // -----------------------------------------------------------------------
  // calculate_backward
  // Called as:  this.calculate_backward(prev, travel_time, vehicle_info)
  // arc = travel_time(prev -> this)
  //
  // WCT backward recurrence (same approximation as TIME dimension):
  //   prev.bwd_weight_sum = prev.order_weight + this.bwd_weight_sum
  //   prev.bwd_wct_rel    = prev.bwd_weight_sum * service_time[prev]
  //                       + this.bwd_wct_rel + this.bwd_weight_sum * travel_time
  //
  // Lateness backward:
  //   step = service_time[prev] + travel_time
  //   1. Copy this.bwd_lateness_b/w arrays into prev, shifting all b_j by -step.
  //      (Uniform shift preserves ascending sort order.)
  //   2. If prev has due_time > 0, insert (due_time[prev], order_weight[prev])
  //      at b_prev = due_time[prev] (delta_prev^prev = 0) in sorted position.
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  void HDI calculate_backward(soft_time_node_t& prev,
                              double travel_time,
                              const VehicleInfo<f_t, is_device>& vehicle_info) const noexcept
  {
    int32_t service_time_prev = get_service_time<is_device>(prev, vehicle_info);
    int32_t transit           = static_cast<int32_t>(travel_time);

    // --- WCT backward ---
    prev.bwd_weight_sum = prev.order_weight + bwd_weight_sum;
    prev.bwd_wct_rel =
      prev.bwd_weight_sum * service_time_prev + bwd_wct_rel + bwd_weight_sum * transit;

    // --- Exact lateness backward ---
    int32_t step        = service_time_prev + transit;
    int n               = bwd_lateness_n;
    prev.bwd_lateness_n = n;
    // Shift all existing breakpoints by -step (uniform; preserves ascending order)
    for (int j = 0; j < n; j++) {
      prev.bwd_lateness_b[j] = bwd_lateness_b[j] - step;
      prev.bwd_lateness_w[j] = bwd_lateness_w[j];
    }
    // Insert prev node if it has a due time constraint and space remains
    if (prev.due_time > 0 && n < MAX_SOFT_TIME_ROUTE_SIZE) {
      int32_t new_b = prev.due_time;  // b_prev^prev = due_time[prev] (delta = 0)
      int32_t new_w = prev.order_weight;
      // Find insertion point to keep ascending order
      int ins = 0;
      while (ins < n && prev.bwd_lateness_b[ins] <= new_b) {
        ins++;
      }
      // Shift entries right to make room
      for (int j = n; j > ins; j--) {
        prev.bwd_lateness_b[j] = prev.bwd_lateness_b[j - 1];
        prev.bwd_lateness_w[j] = prev.bwd_lateness_w[j - 1];
      }
      prev.bwd_lateness_b[ins] = new_b;
      prev.bwd_lateness_w[ins] = new_w;
      prev.bwd_lateness_n      = n + 1;
    }
  }

  // -----------------------------------------------------------------------
  // combine
  // SOFT_TIME has no constraints (purely objective). Always returns 0.
  // -----------------------------------------------------------------------
  static HDI double combine([[maybe_unused]] const soft_time_node_t& prev,
                            [[maybe_unused]] const soft_time_node_t& next,
                            [[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double travel_time) noexcept
  {
    return 0.;
  }

  // -----------------------------------------------------------------------
  // forward/backward excess — always 0 (purely objective, no constraints)
  // -----------------------------------------------------------------------
  HDI double forward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI double backward_excess([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info) const noexcept
  {
    return 0.;
  }

  HDI bool forward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                            [[maybe_unused]] double weight       = 1.0,
                            [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return true;
  }

  HDI bool backward_feasible([[maybe_unused]] const VehicleInfo<f_t>& vehicle_info,
                             [[maybe_unused]] double weight       = 1.0,
                             [[maybe_unused]] double excess_limit = 0.) const noexcept
  {
    return true;
  }

  // -----------------------------------------------------------------------
  // get_cost
  // Called with prev_node = node[k-1] (fwd already propagated into `this`).
  //
  // WCT: total_wct = prev.fwd_wct + handoff_time * bwd_weight_sum + bwd_wct_rel
  //   handoff_time = actual_start[this] = fwd_completion - service_time[this]
  //
  // LATENESS (if has_lateness):
  //   P_suffix(h) = sum_j max(0, h - b_j) * w_j   over bwd_lateness_b/w arrays
  //   total = prev.fwd_lateness + P_suffix(h)
  // -----------------------------------------------------------------------
  template <bool is_device = true>
  HDI void get_cost(const soft_time_node_t& prev_node,
                    const VehicleInfo<f_t, is_device>& vehicle_info,
                    const soft_time_dimension_info_t& dim_info,
                    objective_cost_t& obj_cost,
                    [[maybe_unused]] infeasible_cost_t& inf_cost) const noexcept
  {
    int32_t service_time = get_service_time<is_device>(*this, vehicle_info);
    int32_t handoff_time = fwd_completion - service_time;  // = actual_start[this]
    obj_cost[objective_t::WEIGHTED_COMPLETION_TIME] =
      static_cast<double>(prev_node.fwd_wct + handoff_time * bwd_weight_sum + bwd_wct_rel);

    if (dim_info.has_lateness) {
      int32_t penalty = 0;
      for (int j = 0; j < bwd_lateness_n; j++) {
        int32_t excess = handoff_time - bwd_lateness_b[j];
        if (excess > 0) { penalty += excess * bwd_lateness_w[j]; }
      }
      obj_cost[objective_t::LATENESS] = static_cast<double>(prev_node.fwd_lateness + penalty);
    }
  }
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
