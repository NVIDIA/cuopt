/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <branch_and_bound/diving_heuristics.hpp>

namespace cuopt::mathematical_optimization::mip {

struct submip_stats_t {
  omp_atomic_t<int> total_success             = 0;
  omp_atomic_t<double> success_fixrate_sum    = 0;
  omp_atomic_t<int> total_infeasible          = 0;
  omp_atomic_t<double> infeasible_fixrate_sum = 0;
  omp_atomic_t<int> total_calls               = 0;
  omp_atomic_t<int> total_empty               = 0;

  void save_success(double fixrate)
  {
    ++total_success;
    success_fixrate_sum += fixrate;
  }

  void save_infeasible(double fixrate)
  {
    ++total_infeasible;
    infeasible_fixrate_sum += fixrate;
  }

  void save_empty() { ++total_empty; }
  double average_infeasible_fixrate() const { return infeasible_fixrate_sum / total_infeasible; }
  double average_success_fixrate() const { return success_fixrate_sum / total_success; }
};

template <typename i_t, typename f_t>
void solve_submip(diving_worker_t<i_t, f_t>* worker,
                  branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
                  const simplex::user_problem_t<i_t, f_t>& original_problem,
                  submip_stats_t& submip_stats,
                  f_t fixrate,
                  int64_t simplex_iter_used,
                  simplex::simplex_solver_settings_t<i_t, f_t> submip_settings);

template <typename i_t, typename f_t>
void mutation(diving_worker_t<i_t, f_t>* worker,
              branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
              const simplex::user_problem_t<i_t, f_t>& original_problem,
              submip_stats_t& submip_stats,
              simplex::simplex_solver_settings_t<i_t, f_t> settings);

template <typename i_t, typename f_t>
void rins(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const simplex::user_problem_t<i_t, f_t>& original_problem,
          submip_stats_t& submip_stats,
          simplex::simplex_solver_settings_t<i_t, f_t> submip_settings);

template <typename i_t, typename f_t>
void rens(diving_worker_t<i_t, f_t>* worker,
          branch_and_bound_stats_t<i_t, f_t>& bnb_stats,
          const simplex::user_problem_t<i_t, f_t>& original_problem,
          submip_stats_t& submip_stats,
          simplex::simplex_solver_settings_t<i_t, f_t> submip_settings);

}  // namespace cuopt::mathematical_optimization::mip
