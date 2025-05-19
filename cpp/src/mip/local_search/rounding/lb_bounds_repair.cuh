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

#include <mip/presolve/load_balanced_bounds_presolve.cuh>
#include <mip/problem/load_balanced_problem.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>
#include "bounds_repair.cuh"

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
struct lb_bounds_t {
  lb_bounds_t(const raft::handle_t* handle_ptr) : bounds(0, handle_ptr->get_stream()) {}
  void resize(i_t var_size, const raft::handle_t* handle_ptr)
  {
    bounds.resize(2 * var_size, handle_ptr->get_stream());
  }
  void update_from(const load_balanced_problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr)
  {
    cuopt_assert(bounds.size() == pb.variable_bounds.size(), "");
    raft::copy(bounds.data(), pb.variable_bounds.data(), bounds.size(), handle_ptr->get_stream());
  };
  void update_to(load_balanced_problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr)
  {
    cuopt_assert(bounds.size() == pb.variable_bounds.size(), "");
    raft::copy(pb.variable_bounds.data(), bounds.data(), bounds.size(), handle_ptr->get_stream());
  };
  rmm::device_uvector<f_t> bounds;
};

template <typename i_t, typename f_t>
class lb_bounds_repair_t {
 public:
  lb_bounds_repair_t(const raft::handle_t* handle_ptr);
  void resize(const load_balanced_problem_t<i_t, f_t>& problem);
  void reset();
  std::tuple<f_t, i_t> get_ii_violation(
    load_balanced_problem_t<i_t, f_t>& problem,
    load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve);
  i_t get_random_cstr();
  bool detect_cycle(i_t cstr_idx);
  i_t compute_best_shift(load_balanced_problem_t<i_t, f_t>& problem,
                         problem_t<i_t, f_t>& original_problem,
                         i_t curr_cstr);
  void compute_damages(load_balanced_problem_t<i_t, f_t>& problem,
                       load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve,
                       problem_t<i_t, f_t>& original_problem,
                       i_t n_candidates);
  bool repair_problem(load_balanced_problem_t<i_t, f_t>* problem,
                      load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve,
                      problem_t<i_t, f_t>& original_problem,
                      timer_t timer_,
                      const raft::handle_t* handle_ptr_);
  void apply_move(load_balanced_problem_t<i_t, f_t>* problem,
                  problem_t<i_t, f_t>& original_problem,
                  i_t move_idx);
  i_t get_random_idx(i_t size);
  i_t find_cutoff_index(const candidates_t<i_t, f_t>& candidates,
                        i_t best_cstr_delta,
                        f_t best_damage,
                        i_t n_candidates);

  // load_balanced_bounds_presolve_t<i_t, f_t>& lb_bound_presolve;
  candidates_t<i_t, f_t> candidates;
  lb_bounds_t<i_t, f_t> best_bounds;
  rmm::device_uvector<f_t> cstr_violations_up;
  rmm::device_uvector<f_t> cstr_violations_down;
  rmm::device_uvector<i_t> violated_constraints;
  rmm::device_uvector<i_t> violated_cstr_map;
  rmm::device_scalar<f_t> total_vio;
  f_t best_violation;
  f_t curr_violation;
  i_t h_n_violated_cstr;
  const raft::handle_t* handle_ptr;
  std::mt19937 gen;
  timer_t timer{0.};
  std::vector<i_t> cycle_vector;
  i_t cycle_write_pos = 0;
};

}  // namespace cuopt::linear_programming::detail
