/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#include <branch_and_bound/branch_and_bound.hpp>
#include <branch_and_bound/mip_node.hpp>
#include <branch_and_bound/pseudo_costs.hpp>

#include <mip_heuristics/problem/problem.cuh>

#include <cuts/cuts.hpp>
#include <dual_simplex/initial_basis.hpp>
#include <dual_simplex/presolve.hpp>
#include <dual_simplex/user_problem.hpp>

namespace cuopt::linear_programming::dual_simplex {

namespace {
template <typename i_t, typename f_t>
void full_variable_types(const user_problem_t<i_t, f_t>& original_problem,
                         const lp_problem_t<i_t, f_t>& original_lp,
                         std::vector<variable_type_t>& var_types)
{
  var_types = original_problem.var_types;
  if (original_lp.num_cols > original_problem.num_cols) {
    var_types.resize(original_lp.num_cols);
    for (i_t k = original_problem.num_cols; k < original_lp.num_cols; k++) {
      var_types[k] = variable_type_t::CONTINUOUS;
    }
  }
}
}  // anonymous namespace

template <typename i_t, typename f_t>
branch_and_bound_t<i_t, f_t>::branch_and_bound_t(
  cuopt::linear_programming::detail::problem_t<i_t, f_t>* mip_problem_ptr,
  const simplex_solver_settings_t<i_t, f_t>& solver_settings,
  f_t start_time,
  i_t num_gpus)
  : original_problem_(mip_problem_ptr->handle_ptr),
    settings_(solver_settings),
    original_lp_(mip_problem_ptr->handle_ptr, 1, 1, 1),
    Arow_(1, 1, 0),
    incumbent_(1),
    root_relax_soln_(1, 1),
    pc_(1),
    solver_status_(mip_status_t::UNSET),
    mip_problem_ptr_(mip_problem_ptr),
    pdlp_root_num_gpus_(num_gpus)
{
  exploration_stats_.start_time = start_time;
  mip_problem_ptr->recompute_objective_integrality();
  original_problem_.objective_is_integral = mip_problem_ptr->is_objective_integral();
  mip_problem_ptr->get_host_user_problem(original_problem_);

#ifdef PRINT_CONSTRAINT_MATRIX
  settings_.log.printf("A");
  original_problem_.A.print_matrix();
#endif

  dualize_info_t<i_t, f_t> dualize_info;
  convert_user_problem(original_problem_, settings_, original_lp_, new_slacks_, dualize_info);
  full_variable_types(original_problem_, original_lp_, var_types_);

#ifdef CHECK_SLACKS
  assert(new_slacks_.size() == original_lp_.num_rows);
  for (i_t slack : new_slacks_) {
    const i_t col_start = original_lp_.A.col_start[slack];
    const i_t col_end   = original_lp_.A.col_start[slack + 1];
    const i_t col_len   = col_end - col_start;
    if (col_len != 1) {
      settings_.log.printf("Slack %d has %d nzs\n", slack, col_len);
      assert(col_len == 1);
    }
    const i_t i = original_lp_.A.i[col_start];
    const f_t x = original_lp_.A.x[col_start];
    if (std::abs(x) != 1.0) {
      settings_.log.printf("Slack %d row %d has non-unit coefficient %e\n", slack, i, x);
      assert(std::abs(x) == 1.0);
    }
  }
#endif

  upper_bound_    = inf;
  root_objective_ = std::numeric_limits<f_t>::quiet_NaN();
}

template branch_and_bound_t<int, double>::branch_and_bound_t(
  cuopt::linear_programming::detail::problem_t<int, double>*,
  const simplex_solver_settings_t<int, double>&,
  double,
  int);

#ifdef MIP_INSTANTIATION_FLOAT
template branch_and_bound_t<int, float>::branch_and_bound_t(
  cuopt::linear_programming::detail::problem_t<int, float>*,
  const simplex_solver_settings_t<int, float>&,
  float,
  int);
#endif

}  // namespace cuopt::linear_programming::dual_simplex
