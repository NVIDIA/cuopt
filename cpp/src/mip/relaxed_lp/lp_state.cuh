/*
 * SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
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

#pragma once

#include <raft/util/cudart_utils.hpp>
#include <rmm/device_uvector.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class lp_state_t {
 public:
  lp_state_t(problem_t<i_t, f_t>& problem, rmm::cuda_stream_view stream)
    : prev_primal(problem.n_variables, stream), prev_dual(problem.n_constraints, stream)
  {
    thrust::fill(problem.handle_ptr->get_thrust_policy(),
                 prev_primal.data(),
                 prev_primal.data() + problem.n_variables,
                 0);
    thrust::fill(problem.handle_ptr->get_thrust_policy(),
                 prev_dual.data(),
                 prev_dual.data() + problem.n_constraints,
                 0);
  }

  lp_state_t(problem_t<i_t, f_t>& problem) : lp_state_t(problem, problem.handle_ptr->get_stream())
  {
  }

  lp_state_t(const lp_state_t<i_t, f_t>& other)
    : prev_primal(other.prev_primal, other.prev_primal.stream()),
      prev_dual(other.prev_dual, other.prev_dual.stream())
  {
  }

  lp_state_t(lp_state_t<i_t, f_t>&& other) noexcept            = default;
  lp_state_t& operator=(lp_state_t<i_t, f_t>&& other) noexcept = default;

  void resize(problem_t<i_t, f_t>& problem, rmm::cuda_stream_view stream)
  {
    prev_primal.resize(problem.n_variables, stream);
    prev_dual.resize(problem.n_constraints, stream);
  }

  void set_state(const rmm::device_uvector<f_t>& primal_solution,
                 const rmm::device_uvector<f_t>& dual_solution)
  {
    cuopt_assert(primal_solution.size() == prev_primal.size(),
                 "The size of the primal solution must match the previous size!");
    cuopt_assert(dual_solution.size() == prev_dual.size(),
                 "The size of the dual solution must match the previous size!");
    raft::copy(
      prev_primal.data(), primal_solution.data(), primal_solution.size(), prev_primal.stream());
    raft::copy(prev_dual.data(), dual_solution.data(), dual_solution.size(), prev_dual.stream());
  }
  rmm::device_uvector<f_t> prev_primal;
  rmm::device_uvector<f_t> prev_dual;
};

}  // namespace cuopt::linear_programming::detail
