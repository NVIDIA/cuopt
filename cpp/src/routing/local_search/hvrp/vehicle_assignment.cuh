/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "../../solution/solution.cuh"
#include "../move_candidates/move_candidates.cuh"

namespace cuopt {
namespace routing {
namespace detail {

constexpr int const k_max_regrets      = 2;
constexpr int const k_min_regrets      = 2;
constexpr int const min_bucket_entries = 64;

/**
 * @brief
 * In the homogenous/no vehicle type limits environment the BF split is optimal, so adding it to the
 * LS we should see some improvement (and it covers the split operator) In the restricted
 * environment the heuristic is greedy so we might get a split that is worse than the original one.
 * But if we already have some split (which is true everywhere outside OX) we could use an easy
 * regret heuristic: For each route calculate the cost difference between assigning the second best
 * and the best vehicle. Lets call this regret. Now sort all the regrets and choose the route with
 * the highest regret. Assign greedily the vehicle to to this route (any ties should be resolved by
 * assigning vehicle that is more available). Continue this process until we have assigned vehicles
 * to all the routes. (At each iteration we have to recalculate the regret for the routes where
 * after the previous step some vehicle type among best/second best were used) This could be
 * generalized to k-regret: the sum of differences between assigning k-th best up to second best.
 * Actually we could try to loop this heuristic 4 times: for 2-regret, 3-regret,...,5-regret and
 * choose the best assignment that comes out of them.
 * @tparam i_t
 * @tparam f_t
 * @tparam REQUEST
 */
template <typename i_t, typename f_t, request_t REQUEST>
struct vehicle_assignment_t {
  vehicle_assignment_t(solution_handle_t<i_t, f_t> const* sol_handle_)
    : route_costs_per_bucket(0, sol_handle_->get_stream()),
      cost_differences(0, sol_handle_->get_stream()),
      regret_score_per_route(0, sol_handle_->get_stream()),
      assignments(0, sol_handle_->get_stream()),
      top_bucket(0, sol_handle_->get_stream()),
      top_cost(0, sol_handle_->get_stream()),
      vehicle_availability(0, sol_handle_->get_stream()),
      vehicle_buckets(0, sol_handle_->get_stream()),
      bucket_offsets(0, sol_handle_->get_stream()),
      run_sort(0, sol_handle_->get_stream()),
      assignment_costs(0, sol_handle_->get_stream()),
      best_cost(sol_handle_->get_stream()),
      best_k(sol_handle_->get_stream()),
      gl_lock(sol_handle_->get_stream())

  {
    gl_lock.set_value_to_zero_async(sol_handle_->get_stream());
  }

  void resize(i_t n_routes, i_t n_buckets, rmm::cuda_stream_view stream_view)
  {
    k_regrets   = std::min(n_buckets, k_max_regrets);
    auto k_iter = k_regrets - 1;
    route_costs_per_bucket.resize(n_routes * n_buckets, stream_view);
    cost_differences.resize(k_iter * n_routes * k_iter, stream_view);
    regret_score_per_route.resize(k_iter * n_routes, stream_view);
    assignments.resize(k_iter * n_routes, stream_view);
    top_bucket.resize(k_iter * n_routes, stream_view);
    top_cost.resize(k_iter * n_routes, stream_view);
    vehicle_availability.resize(k_iter * n_buckets, stream_view);
    run_sort.resize(k_iter, stream_view);
    assignment_costs.resize(k_iter, stream_view);
  }

  i_t get_k_regrets() const { return k_regrets; }

  struct view_t {
    constexpr void pop_next_vehicle_id(i_t route_id, i_t bucket, i_t& vehicle_id)
    {
      if (threadIdx.x == 0) {
        acquire_lock(gl_lock);
        auto n_available = vehicle_availability[bucket];
        cuopt_assert(n_available > 0, "Infeasible use of buckets");
        auto bucket_offset = bucket_offsets[bucket];
        vehicle_id         = vehicle_buckets[bucket_offset];
        raft::swapVals(vehicle_buckets[bucket_offset],
                       vehicle_buckets[bucket_offset + n_available - 1]);
        --vehicle_availability[bucket];
        release_lock(gl_lock);
      }
    }

    constexpr i_t get_k_regrets() const { return k_regrets; }

    raft::device_span<double> route_costs_per_bucket;
    raft::device_span<double> cost_differences;
    raft::device_span<double> regret_score_per_route;
    raft::device_span<i_t> assignments;
    raft::device_span<i_t> top_bucket;
    raft::device_span<double> top_cost;
    raft::device_span<i_t> vehicle_availability;
    raft::device_span<i_t> vehicle_buckets;
    raft::device_span<i_t> bucket_offsets;
    raft::device_span<i_t> run_sort;
    raft::device_span<double> assignment_costs;
    double* best_cost;
    i_t* best_k;
    i_t* gl_lock;
    i_t k_regrets;
  };

  view_t view()
  {
    view_t v;
    v.route_costs_per_bucket = cuopt::make_span(route_costs_per_bucket);
    v.cost_differences       = cuopt::make_span(cost_differences);
    v.regret_score_per_route = cuopt::make_span(regret_score_per_route);
    v.assignments            = cuopt::make_span(assignments);
    v.top_bucket             = cuopt::make_span(top_bucket);
    v.top_cost               = cuopt::make_span(top_cost);
    v.vehicle_availability   = cuopt::make_span(vehicle_availability);
    v.vehicle_buckets        = cuopt::make_span(vehicle_buckets);
    v.bucket_offsets         = cuopt::make_span(bucket_offsets);
    v.run_sort               = cuopt::make_span(run_sort);
    v.assignment_costs       = cuopt::make_span(assignment_costs);
    v.best_cost              = best_cost.data();
    v.best_k                 = best_k.data();
    v.gl_lock                = gl_lock.data();
    v.k_regrets              = k_regrets;
    return v;
  }

  rmm::device_uvector<double> route_costs_per_bucket;
  rmm::device_uvector<double> regret_score_per_route;
  rmm::device_uvector<double> cost_differences;
  rmm::device_uvector<i_t> assignments;
  rmm::device_uvector<i_t> top_bucket;
  rmm::device_uvector<double> top_cost;
  rmm::device_uvector<i_t> vehicle_availability;
  rmm::device_uvector<i_t> vehicle_buckets;
  rmm::device_uvector<i_t> bucket_offsets;
  rmm::device_uvector<i_t> run_sort;
  rmm::device_uvector<double> assignment_costs;
  rmm::device_scalar<double> best_cost;
  rmm::device_scalar<i_t> best_k;
  rmm::device_scalar<i_t> gl_lock;
  i_t k_regrets;
};

template <typename i_t, typename f_t, request_t REQUEST>
bool run_vehicle_assignment(solution_t<i_t, f_t, REQUEST>& sol,
                            move_candidates_t<i_t, f_t>& move_candidates,
                            vehicle_assignment_t<i_t, f_t, REQUEST>& vehicle_assignment);

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
