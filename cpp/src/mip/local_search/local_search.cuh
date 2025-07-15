/*
 * SPDX-FileCopyrightText: Copyright (c) 2024-2025 NVIDIA CORPORATION & AFFILIATES. All rights
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

#pragma once

#include <mip/local_search/feasibility_pump/feasibility_pump.cuh>
#include <mip/local_search/line_segment_search/line_segment_search.cuh>
#include <mip/solution/solution.cuh>
#include <mip/solver.cuh>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class local_search_t {
 public:
  local_search_t() = delete;
  local_search_t(mip_solver_context_t<i_t, f_t>& context,
                 rmm::device_uvector<f_t>& lp_optimal_solution_);
  void generate_fast_solution(solution_t<i_t, f_t>& solution, timer_t timer);
  bool generate_solution(solution_t<i_t, f_t>& solution,
                         bool perturb,
                         bool& early_exit,
                         f_t time_limit = 300.);
  bool run_fj_until_timer(solution_t<i_t, f_t>& solution,
                          const weight_t<i_t, f_t>& weights,
                          timer_t timer);
  bool run_local_search(solution_t<i_t, f_t>& solution,
                        const weight_t<i_t, f_t>& weights,
                        timer_t timer,
                        f_t baseline_objective            = std::numeric_limits<f_t>::lowest(),
                        bool at_least_one_parent_feasible = true);
  bool run_fj_annealing(solution_t<i_t, f_t>& solution,
                        timer_t timer,
                        f_t baseline_objective = std::numeric_limits<f_t>::lowest());
  bool run_fj_line_segment(solution_t<i_t, f_t>& solution, timer_t timer);
  bool run_fj_on_zero(solution_t<i_t, f_t>& solution, timer_t timer);
  bool check_fj_on_lp_optimal(solution_t<i_t, f_t>& solution, bool perturb, timer_t timer);
  bool run_staged_fp(solution_t<i_t, f_t>& solution, timer_t timer, bool& early_exit);
  bool run_fp(solution_t<i_t, f_t>& solution, timer_t timer);
  void resize_vectors(problem_t<i_t, f_t>& problem, const raft::handle_t* handle_ptr);

  mip_solver_context_t<i_t, f_t>& context;
  rmm::device_uvector<f_t>& lp_optimal_solution;
  bool lp_optimal_exists{false};
  rmm::device_uvector<f_t> fj_sol_on_lp_opt;
  fj_t<i_t, f_t> fj;
  // fj_tree_t<i_t, f_t> fj_tree;
  constraint_prop_t<i_t, f_t> constraint_prop;
  lb_constraint_prop_t<i_t, f_t> lb_constraint_prop;
  line_segment_search_t<i_t, f_t> line_segment_search;
  feasibility_pump_t<i_t, f_t> fp;
  std::mt19937 rng;
};

}  // namespace cuopt::linear_programming::detail
