/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2022-2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include "solution.cuh"

#include "../ges/guided_ejection_search.cuh"
#include "../local_search/local_search.cuh"
#include "../problem/problem.cuh"
#include "../routing_helpers.cuh"

namespace cuopt {
namespace routing {
namespace detail {

// this class keeps the modifier and generator object
template <typename i_t, typename f_t, typename Solution, typename Problem>
class routing_resource_t {
 public:
  explicit routing_resource_t(solution_handle_t<i_t, f_t>* sol_handle,
                              const Problem* problem_,
                              Solution& dummy_sol)
    : ls(sol_handle,
         problem_->get_num_orders(),
         problem_->get_fleet_size(),
         problem_->order_info.depot_included_,
         problem_->viables),
      ges(dummy_sol, &ls)  // the ls will be dangled as this object will be moved to the shared pool
  {
    raft::common::nvtx::range fun_scope("routing_resource_t");
  }

  local_search_t<i_t, f_t, Solution::request_type> ls;
  guided_ejection_search_t<i_t, f_t, Solution::request_type> ges;
};

template <typename i_t, typename f_t, typename Solution, typename Problem>
class pool_allocator_t {
 public:
  pool_allocator_t(const Problem& problem_, i_t n_solutions_, i_t desired_n_routes = -1)
    : stream_pool(n_solutions_), problem(problem_)
  {
    raft::common::nvtx::range fun_scope("pool_allocator_t");
    // FIXME:: This is temporary, we should let the diversity manager decide this
    std::vector<i_t> desired_vehicle_ids;
    if (desired_n_routes > 0) {
      desired_vehicle_ids.resize(desired_n_routes);
      std::iota(desired_vehicle_ids.begin(), desired_vehicle_ids.end(), 0);
    }
    sol_handles.reserve(n_solutions_);
    for (i_t i = 0; i < n_solutions_; ++i) {
      sol_handles.emplace_back(
        std::make_unique<solution_handle_t<i_t, f_t>>(stream_pool.get_stream(i)));
    }
    Solution dummy_sol{problem_, 0, sol_handles[0].get()};
    resource_pool =
      std::make_unique<shared_pool_t<routing_resource_t<i_t, f_t, Solution, Problem>>>(
        sol_handles[0].get(), &problem, dummy_sol);
    // unfortunately this is needed as the ls ptr in ges is dangling after the move
    // emplace_back needs a move ctr in the resource pool that's why we can't avoid this
    for (auto& res : resource_pool->shared_resources) {
      res.ges.local_search_ptr_ = &res.ls;
    }
  }

  void sync_all_streams() const
  {
    for (size_t i = 0; i < stream_pool.get_pool_size(); ++i) {
      stream_pool.get_stream(i).synchronize();
    }
  }

  // a stream pool that will be used to execute different solutions on
  // we are currently not using raft handles stream pool as it is constructed in python layer
  // TODO: later consider using raft stream pool and construct it on python layer
  // however that pushes some internal logic to the higher levels which we want to avoid
  // rmm::cuda_stream_pool is non-movable as it contains an atomic variables
  // KEEP THIS MEMBER ABOVE OTHER MEMBERS, so that it is destructed the last
  rmm::cuda_stream_pool stream_pool;

  // problem description
  const Problem& problem;
  std::vector<std::unique_ptr<solution_handle_t<i_t, f_t>>> sol_handles;
  // keep a thread safe pool of local search and ges objects that can be reused
  std::unique_ptr<shared_pool_t<routing_resource_t<i_t, f_t, Solution, Problem>>> resource_pool;
};

}  // namespace detail
}  // namespace routing
}  // namespace cuopt
