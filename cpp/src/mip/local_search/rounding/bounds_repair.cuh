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

#include <mip/presolve/bounds_presolve.cuh>
#include <mip/problem/problem.cuh>
#include <mip/solution/solution.cuh>
#include <utilities/copy_helpers.hpp>
#include <utilities/timer.hpp>

namespace cuopt::linear_programming::detail {

// from the paper, probability of choosing random candidate= noise parameter
constexpr double p                  = 0.75;
constexpr double ROUNDOFF_TOLERANCE = 1e-7;
constexpr int MAX_CYCLE_SEQUENCE    = 5;

template <typename i_t, typename f_t>
struct bounds_t {
  bounds_t(const raft::handle_t* handle_ptr)
    : lb(0, handle_ptr->get_stream()), ub(0, handle_ptr->get_stream())
  {
  }
  void resize(i_t var_size, const raft::handle_t* handle_ptr)
  {
    lb.resize(var_size, handle_ptr->get_stream());
    ub.resize(var_size, handle_ptr->get_stream());
  }
  void update_from(const problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr)
  {
    cuopt_assert(lb.size() == pb.variable_bounds.size(), "");
    cuopt_assert(ub.size() == pb.variable_bounds.size(), "");
    thrust::transform(
      handle_ptr->get_thrust_policy(),
      pb.variable_bounds.begin(),
      pb.variable_bounds.end(),
      thrust::make_zip_iterator(thrust::make_tuple(lb.begin(), ub.begin())),
      [] __device__(auto i) { return thrust::make_tuple(get_lower(i), get_upper(i)); });
  };
  void update_to(problem_t<i_t, f_t>& pb, const raft::handle_t* handle_ptr)
  {
    cuopt_assert(lb.size() == pb.variable_bounds.size(), "");
    cuopt_assert(ub.size() == pb.variable_bounds.size(), "");
    using f_t2 = typename type_2<f_t>::type;
    thrust::transform(handle_ptr->get_thrust_policy(),
                      thrust::make_zip_iterator(thrust::make_tuple(lb.begin(), ub.begin())),
                      thrust::make_zip_iterator(thrust::make_tuple(lb.end(), ub.end())),
                      pb.variable_bounds.begin(),
                      [] __device__(auto i) { return f_t2{thrust::get<0>(i), thrust::get<1>(i)}; });
  };
  rmm::device_uvector<f_t> lb;
  rmm::device_uvector<f_t> ub;
};

template <typename i_t, typename f_t>
struct candidates_t {
  candidates_t(const raft::handle_t* handle_ptr)
    : variable_index(0, handle_ptr->get_stream()),
      bound_shift(0, handle_ptr->get_stream()),
      damage(0, handle_ptr->get_stream()),
      cstr_delta(0, handle_ptr->get_stream()),
      n_candidates(handle_ptr->get_stream()),
      at_least_one_singleton_moved(handle_ptr->get_stream())
  {
  }
  void resize(i_t max_candidates, const raft::handle_t* handle_ptr)
  {
    variable_index.resize(max_candidates, handle_ptr->get_stream());
    bound_shift.resize(max_candidates, handle_ptr->get_stream());
    damage.resize(max_candidates, handle_ptr->get_stream());
    cstr_delta.resize(max_candidates, handle_ptr->get_stream());
    at_least_one_singleton_moved.set_value_to_zero_async(handle_ptr->get_stream());
  }

  struct view_t {
    raft::device_span<i_t> variable_index;
    raft::device_span<f_t> bound_shift;
    raft::device_span<f_t> damage;
    raft::device_span<i_t> cstr_delta;
    i_t* n_candidates;
    i_t* at_least_one_singleton_moved;
  };

  view_t view()
  {
    view_t v;
    v.variable_index               = make_span(variable_index);
    v.bound_shift                  = make_span(bound_shift);
    v.damage                       = make_span(damage);
    v.cstr_delta                   = make_span(cstr_delta);
    v.n_candidates                 = n_candidates.data();
    v.at_least_one_singleton_moved = at_least_one_singleton_moved.data();
    return v;
  }
  rmm::device_uvector<i_t> variable_index;
  rmm::device_uvector<f_t> bound_shift;
  rmm::device_uvector<f_t> damage;
  rmm::device_uvector<i_t> cstr_delta;
  rmm::device_scalar<i_t> n_candidates;
  rmm::device_scalar<i_t> at_least_one_singleton_moved;
};

template <typename i_t, typename f_t>
class bounds_repair_t {
 public:
  bounds_repair_t(const problem_t<i_t, f_t>& p, bound_presolve_t<i_t, f_t>& bound_presolve);
  void resize(const problem_t<i_t, f_t>& problem);
  void reset();
  f_t get_ii_violation(problem_t<i_t, f_t>& problem);
  i_t get_random_cstr();
  bool detect_cycle(i_t cstr_idx);
  i_t compute_best_shift(problem_t<i_t, f_t>& problem,
                         problem_t<i_t, f_t>& original_problem,
                         i_t curr_cstr);
  void compute_damages(problem_t<i_t, f_t>& problem, i_t n_candidates);
  bool repair_problem(problem_t<i_t, f_t>& problem,
                      problem_t<i_t, f_t>& original_problem,
                      timer_t timer_,
                      const raft::handle_t* handle_ptr_);
  void apply_move(problem_t<i_t, f_t>& problem,
                  problem_t<i_t, f_t>& original_problem,
                  i_t move_idx);
  i_t get_random_idx(i_t size);
  i_t find_cutoff_index(const candidates_t<i_t, f_t>& candidates,
                        i_t best_cstr_delta,
                        f_t best_damage,
                        i_t n_candidates);

  bound_presolve_t<i_t, f_t>& bound_presolve;
  candidates_t<i_t, f_t> candidates;
  bounds_t<i_t, f_t> best_bounds;
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
