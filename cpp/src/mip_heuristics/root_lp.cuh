/* clang-format off */
/*
 * SPDX-FileCopyrightText: Copyright (c) 2025-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 */
/* clang-format on */

#pragma once

#include <cuopt/linear_programming/pdlp/solver_settings.hpp>
#include <dual_simplex/crossover.hpp>
#include <dual_simplex/types.hpp>

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>

namespace cuopt::linear_programming::detail {

template <typename i_t, typename f_t>
class problem_t;

/** Run PDLP or Barrier for root LP. Uses concurrent_halt to stop; does not set it. Crossover done
 * by caller. */
template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>
run_solver_for_root_lp(problem_t<i_t, f_t>* problem,
                       f_t time_limit,
                       std::atomic<int>* concurrent_halt,
                       i_t num_gpus,
                       method_t method);

/**
 * Run crush + crossover on a root LP solution and optionally store as winner (first to finish).
 * Used by B&B when running PDLP and Barrier concurrently; both paths call this after their solver
 * returns.
 */
template <typename i_t, typename f_t>
cuopt::linear_programming::dual_simplex::crossover_status_t run_crush_crossover_and_maybe_win(
  const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>& result,
  const cuopt::linear_programming::dual_simplex::user_problem_t<i_t, f_t>& original_problem,
  const cuopt::linear_programming::dual_simplex::lp_problem_t<i_t, f_t>& original_lp,
  const std::vector<i_t>& new_slacks,
  const cuopt::linear_programming::dual_simplex::simplex_solver_settings_t<i_t, f_t>&
    crossover_settings,
  f_t start_time,
  std::atomic<int>* concurrent_halt,
  std::function<void()> set_halter,
  std::function<void(
    const cuopt::linear_programming::dual_simplex::root_relaxation_first_solution_t<i_t, f_t>&)>
    on_first_lp_solution,
  std::mutex* first_solver_mutex,
  bool* first_solver_callback_done,
  std::mutex* first_result_mutex,
  std::atomic<int>* winner,
  int winner_id,
  cuopt::linear_programming::dual_simplex::crossover_status_t* first_crossover_status_out,
  cuopt::linear_programming::dual_simplex::lp_solution_t<i_t, f_t>* winner_crossover_soln,
  std::vector<cuopt::linear_programming::dual_simplex::variable_status_t>* winner_crossover_vstatus,
  f_t* winner_root_objective,
  const char* this_solver_name,
  std::string* winner_solver_name_out);

}  // namespace cuopt::linear_programming::detail
